"""
Weighted score wagers implementation: each model has its own router that outputs a single scalar wager.
Wagers are normalized via softmax to sum to 1.

The key difference from decentralized wagers is in the update mechanism:
- Uses weighted score wagering with strictly proper scoring rules
- Net payout: w_i * (s(p_i, omega) - weighted_avg_score)
- Where s(p_i, omega) is a strictly proper scoring rule (e.g., logarithmic scoring rule)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple

from .base import WageringMethod
from lm_polygraph.utils.model import WhiteboxModel


from .utils import compute_scoring_rule

class BrRegretWagers(WageringMethod):
    """
    Wagering method where each model has its own router that outputs a single scalar wager.
    Uses weighted score wagering mechanism for training.
    
    Architecture:
    - Each model's hidden state is projected to a common dimension
    - Each model has its own router (MLP) that outputs a single scalar (raw wager logit)
    - Raw wagers are collected and normalized via softmax to sum to 1
    
    Training:
    - Uses weighted score wagering with strictly proper scoring rules
    - Net payout for model i: w_i * (s(p_i, omega) - weighted_avg_score)
    - Where s(p_i, omega) is a strictly proper scoring rule on outcome omega
    - Loss is the negative sum of net payouts (to maximize payouts)
    
    Router Architecture:
    - Input: Single model's projected hidden state [common_hidden_dim]
    - Output: Single scalar (raw wager logit)
    - Structure: MLP(hidden_dim → hidden_layers → 1)
    """
    
    def __init__(self, num_models: int, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the weighted score wagers method.
        
        Args:
            num_models: Number of LLMs in the ensemble
            config: Configuration dictionary with:
                - common_hidden_dim: Common dimension for projected hidden states (default: 4096)
                - hidden_layers: List of hidden layer sizes (default: [512, 256])
                - learning_rate: Learning rate for optimizer (default: 1e-5)
                - temperature: Temperature for softmax (default: 2.0)
                - grad_clip_norm: Gradient clipping norm (default: 1.0)
                - normalize_hidden_states: Whether to L2 normalize hidden states (default: True)
                - scoring_rule: Scoring rule to use - 'brier' (default: 'brier')
                  Supported scoring rules: 'logarithmic', 'brier'
                - device: Device to run on (default: 'cuda' if available, else 'cpu')
        """
        super().__init__(num_models, config)
        
        # Get configuration (ensure proper types)
        self.common_hidden_dim = int(config.get("common_hidden_dim", 4096))
        self.hidden_layers = list(config.get("hidden_layers", [512, 256]))
        self.learning_rate = float(config.get("learning_rate", 1e-5))
        self.temperature = float(config.get("temperature", 2.0))
        self.grad_clip_norm = float(config.get("grad_clip_norm", 1.0))
        self.normalize_hidden_states = config.get("normalize_hidden_states", True)
        self.scoring_rule = str(config.get("scoring_rule", "brier")).lower()
        self.device_str = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.device = torch.device(self.device_str)
        
        # Validate scoring rule
        if self.scoring_rule not in ["logarithmic", "brier"]:
            raise ValueError(
                f"Unknown scoring rule: {self.scoring_rule}. "
                "Supported: 'logarithmic', 'brier'"
            )
        
        # Build per-model projection layers to handle variable hidden dimensions
        self.model_projections = nn.ModuleDict()
        self._model_hidden_dims = {}
        
        # Build per-model routers
        self.routers = nn.ModuleList()
        for i in range(num_models):
            router = self._build_router().to(self.device)
            self.routers.append(router)
        
        # Collect all parameters (routers + projections) for optimizer
        all_params = list(self.routers.parameters())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(all_params, lr=self.learning_rate)
        
        # Training mode flag
        self._training = True
        
        # Cache for computed values during training (to avoid recomputation in update())
        self._cached_wagers: Optional[torch.Tensor] = None
        self._cached_projected_states: Optional[List[torch.Tensor]] = None
        
        # Batch counter for alternating router updates over 5 consecutive batches
        self._batch_counter = 0
        self._update_cycle_length = 50
    
    def _build_router(self) -> nn.Module:
        """
        Build a single router MLP.
        
        Returns:
            nn.Sequential: Router network that maps [common_hidden_dim] -> [1]
        """
        layers = []
        prev_dim = self.common_hidden_dim
        
        # Hidden layers
        for hidden_dim in self.hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        # Output layer: outputs single scalar
        layers.append(nn.Linear(prev_dim, 1))
        
        return nn.Sequential(*layers)
    
    def _get_router_indices_to_update(self) -> List[int]:
        """
        Determine which router(s) to update in the current batch.
        
        Each router is updated for 5 consecutive batches before moving to the next router.
        For example with 3 models:
        - Batch 0-4: Update router 0
        - Batch 5-9: Update router 1
        - Batch 10-14: Update router 2
        - Batch 15-19: Update router 0 (cycle repeats)
        
        Returns:
            List[int]: Indices of routers to update in this batch
        """
        router_idx = (self._batch_counter // self._update_cycle_length) % self.num_models
        return [0,1]
    
    def _compute_wagers_from_hidden_states(
        self,
        hidden_states: np.ndarray,
        return_projected_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Internal helper method to compute wagers from hidden states.
        
        Args:
            hidden_states: np.ndarray of shape [num_models, hidden_dim] or object array with variable shapes
            return_projected_states: If True, also return the projected states
            
        Returns:
            wagers: torch.Tensor of shape [num_models] with probabilities (sum to 1)
            projected_states: Optional list of projected states (if return_projected_states=True)
        """
        # Handle variable hidden state dimensions
        if isinstance(hidden_states, np.ndarray) and hidden_states.dtype == object:
            num_models_in = len(hidden_states)
            model_hidden_dims = [hs.shape[0] if len(hs.shape) == 1 else hs.shape[-1] for hs in hidden_states]
            hidden_states_list = [torch.as_tensor(hs, dtype=torch.float32).to(self.device) for hs in hidden_states]
        else:
            hidden_states_tensor = torch.as_tensor(hidden_states, dtype=torch.float32).to(self.device)
            num_models_in, model_hidden_dim = hidden_states_tensor.shape
            model_hidden_dims = [model_hidden_dim] * num_models_in
            hidden_states_list = [hidden_states_tensor[i] for i in range(num_models_in)]
        
        if num_models_in != self.num_models:
            raise ValueError(
                f"Expected {self.num_models} models, got {num_models_in} in hidden_states"
            )
        
        # Step 1: Project each model's hidden state to common_hidden_dim
        projected_states = []
        for i in range(num_models_in):
            model_hs = hidden_states_list[i]
            model_hidden_dim = model_hidden_dims[i]
            
            # Flatten if needed
            if model_hs.dim() > 1:
                model_hs = model_hs.flatten()
                model_hidden_dim = model_hs.shape[0]
            
            # Normalize hidden state
            if self.normalize_hidden_states:
                norm = torch.norm(model_hs)
                if norm > 0:
                    model_hs = model_hs / (norm + 1e-8)
            
            # Create projection layer if needed
            proj_key = f"proj_{model_hidden_dim}"
            if proj_key not in self.model_projections:
                projection = nn.Linear(model_hidden_dim, self.common_hidden_dim).to(self.device)
                self.model_projections[proj_key] = projection
                if self._training:
                    self.optimizer.add_param_group({'params': projection.parameters()})
            
            projection = self.model_projections[proj_key]
            projected = projection(model_hs.unsqueeze(0)).squeeze(0)
            projected_states.append(projected)
        
        # Step 2: Route each model independently
        raw_wagers = []
        for router in self.routers:
            router.eval() if not self._training else router.train()
        with torch.set_grad_enabled(self._training):
            for i in range(num_models_in):
                projected_hs = projected_states[i]
                router_i = self.routers[i]
                raw_wager_i = router_i(projected_hs.unsqueeze(0))
                raw_wagers.append(raw_wager_i.squeeze())
        
        # Step 3: Apply sigmoid to individual raw wagers and then normalize
        raw_wagers_tensor = torch.stack(raw_wagers)
        sigmoid_wagers = torch.sigmoid(raw_wagers_tensor)
        wagers = sigmoid_wagers / (torch.sum(sigmoid_wagers))
        
        if return_projected_states:
            return wagers, projected_states
        return wagers, None

    
    def compute_wagers(
        self,
        hidden_states_list: List[np.ndarray],
        model_logits: Optional[np.ndarray] = None,
        gold_label: Optional[np.ndarray] = None,
        **kwargs
    ) ->    Dict[str, Any]:
        """
        Compute wagers for heterogeneous batch (variable hidden dimensions per model).
        
        Args:
            hidden_states_list: List of [num_models] where each is [batch_size, hidden_dim_i]
            model_logits: Optional, not used
            gold_label: Optional, ground-truth labels (used by methods that compute label-aware diagnostics)
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary with:
                "wagers": np.ndarray of shape [batch_size, num_models] with non-negative weights
                "nash_gap": Optional float or np.ndarray of shape [batch_size] (if computed)
        """
        if len(hidden_states_list) != self.num_models:
            raise ValueError(f"Expected {self.num_models} models, got {len(hidden_states_list)}")
        
        batch_size = hidden_states_list[0].shape[0]
        batch_wagers = []
        
        for b in range(batch_size):
            sample_hidden_states_list = [hs[b] for hs in hidden_states_list]
            # Create numpy array from list of samples
            sample_hidden_states = np.array(sample_hidden_states_list, dtype=object)
            sample_wagers, _ = self._compute_wagers_from_hidden_states(sample_hidden_states)
            batch_wagers.append(sample_wagers)
        
        batch_wagers_tensor = torch.stack(batch_wagers, dim=0)
        
        if self._training:
            self._cached_wagers = batch_wagers_tensor
        
        wagers_np = batch_wagers_tensor.detach().cpu().numpy()
        return {"wagers": wagers_np}
    
    def update(
        self,
        aggregated_probs: np.ndarray,
        aggregated_pred: np.ndarray,
        gold_label: np.ndarray,
        model_probs: np.ndarray,
        model_logits: np.ndarray,
        question: Optional[str] = None,
        hidden_states: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Update for a batch of samples."""
        batch_size = model_logits.shape[0]
        
        # Reuse cached values from compute_wagers() if available
        if self._cached_wagers is not None:
            batch_wagers = self._cached_wagers  # [batch_size, num_models]
        elif hidden_states is not None:
            raise NotImplementedError("Update with hidden_states not implemented yet.")
        
        # Clear cache after use
        self._cached_wagers = None
        if hasattr(self, '_cached_projected_states'):
            self._cached_projected_states = None
        
        # Convert to tensors
        model_probs_tensor = torch.as_tensor(model_probs, dtype=torch.float32).to(self.device)
        gold_label_tensor = torch.as_tensor(gold_label, dtype=torch.long).to(self.device)
        
        # Compute scoring rules for batch: [batch_size, num_models] (vectorized)
        batch_model_scores = compute_scoring_rule(
            model_probs_tensor, gold_label_tensor, scoring_rule=self.scoring_rule
        )  # [batch_size, num_models]
        
        # Compute weighted average scores for each sample: [batch_size]
        batch_weighted_avg_scores = torch.sum(batch_model_scores * batch_wagers, dim=1)
        
        # Compute net payouts for batch: [batch_size, num_models]
        batch_net_payouts = batch_wagers * (batch_model_scores - batch_weighted_avg_scores.unsqueeze(1))
        
        # Compute BR wagers for batch
        batch_br_wagers = 0.5 * (1 - (batch_weighted_avg_scores.unsqueeze(1) - batch_wagers * batch_model_scores) / batch_model_scores)
        batch_br_wagers = torch.clamp(batch_br_wagers, min=0.0, max=1.0)
        
        # Loss: MSE between br_wagers and wagers, averaged over batch
        loss = torch.nn.functional.mse_loss(batch_br_wagers.detach(), batch_wagers)
        
        # Backward pass with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        
        # Get router indices to update
        routers_to_update = self._get_router_indices_to_update()
        
        # Zero out gradients for routers NOT being updated
        for i, router in enumerate(self.routers):
            if i not in routers_to_update:
                for param in router.parameters():
                    if param.grad is not None:
                        param.grad.zero_()
        
        # Clip gradients only for routers being updated
        if routers_to_update:
            updated_router_params = [p for i, router in enumerate(self.routers) 
                                    if i in routers_to_update 
                                    for p in router.parameters()]
            if updated_router_params:
                torch.nn.utils.clip_grad_norm_(updated_router_params, self.grad_clip_norm)
        
        if len(self.model_projections) > 0:
            proj_params = [p for proj in self.model_projections.values() for p in proj.parameters()]
            torch.nn.utils.clip_grad_norm_(proj_params, self.grad_clip_norm)
        
        self.optimizer.step()
        
        # Increment batch counter
        self._batch_counter += 1
        
        # Compute batch metrics
        batch_correct = (np.argmax(aggregated_probs, axis=1) == gold_label)
        batch_accuracy = float(np.mean(batch_correct))
        
        return {
            "loss": float(loss.item()),
            "weighted_avg_score_mean": float(torch.mean(batch_weighted_avg_scores).item()),
            "net_payout_sum_mean": float(torch.mean(torch.sum(batch_net_payouts, dim=1)).item()),
            "model_scores_mean": float(torch.mean(batch_model_scores).item()),
            "batch_accuracy": batch_accuracy,
            "routers_updated_in_batch": routers_to_update,
            "batch_counter": int(self._batch_counter),
            "batch_size": batch_size,
        }
    
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get list of trainable parameters (all routers + projections)."""
        params = list(self.routers.parameters())
        for proj in self.model_projections.values():
            params.extend(proj.parameters())
        return params
    
    def train_mode(self):
        """Set the method to training mode."""
        for router in self.routers:
            router.train()
        self._training = True
        self._cached_wagers = None
        self._cached_projected_states = None
        self._batch_counter = 0  # Reset batch counter when entering training mode
    
    def eval_mode(self):
        """Set the method to evaluation mode."""
        for router in self.routers:
            router.eval()
        self._training = False
        self._cached_wagers = None
        self._cached_projected_states = None
        self._batch_counter = 0  # Reset batch counter when entering eval mode
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for checkpointing."""
        routers_state_dict = {f"router_{i}": router.state_dict() for i, router in enumerate(self.routers)}
        return {
            "routers_state_dict": routers_state_dict,
            "model_projections_state_dict": {k: v.state_dict() for k, v in self.model_projections.items()},
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": {
                "common_hidden_dim": self.common_hidden_dim,
                "hidden_layers": self.hidden_layers,
                "learning_rate": self.learning_rate,
                "temperature": self.temperature,
                "grad_clip_norm": self.grad_clip_norm,
                "normalize_hidden_states": self.normalize_hidden_states,
                "scoring_rule": self.scoring_rule,
                "device": self.device_str,
            },
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dictionary from checkpoint."""
        if "routers_state_dict" in state_dict:
            routers_state_dict = state_dict["routers_state_dict"]
            for i, router in enumerate(self.routers):
                key = f"router_{i}"
                if key in routers_state_dict:
                    router.load_state_dict(routers_state_dict[key])
        if "model_projections_state_dict" in state_dict:
            for key, proj_state in state_dict["model_projections_state_dict"].items():
                if key not in self.model_projections:
                    in_features = proj_state["weight"].shape[1]
                    out_features = proj_state["weight"].shape[0]
                    projection = nn.Linear(in_features, out_features).to(self.device)
                    self.model_projections[key] = projection
                    if self._training:
                        self.optimizer.add_param_group({'params': projection.parameters()})
                self.model_projections[key].load_state_dict(proj_state)
        if "optimizer_state_dict" in state_dict:
            try:
                self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            except (ValueError, KeyError) as e:
                import logging
                log = logging.getLogger("lm_polygraph")
                log.warning(
                    f"Could not load optimizer state dict (parameter mismatch): {e}. "
                    "Continuing with fresh optimizer state."
                )
