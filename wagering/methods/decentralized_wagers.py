"""
Decentralized wagers implementation: each model has its own router that outputs a single scalar wager.
Wagers are normalized via softmax to sum to 1.
"""

import numpy as np
import torch
import torch.nn as nn
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add src/ to path for lm_polygraph imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from .base import WageringMethod
from lm_polygraph.utils.model import WhiteboxModel
from wagering.aggregation.linear_pooling import LinearPooling


class DecentralizedWagers(WageringMethod):
    """
    Wagering method where each model has its own router that outputs a single scalar wager.
    
    Architecture:
    - Each model's hidden state is projected to a common dimension
    - Each model has its own router (MLP) that outputs a single scalar (raw wager logit)
    - Raw wagers are collected and normalized via softmax to sum to 1
    
    Router Architecture:
    - Input: Single model's projected hidden state [common_hidden_dim]
    - Output: Single scalar (raw wager logit)
    - Structure: MLP(hidden_dim → hidden_layers → 1)
    """
    
    def __init__(self, num_models: int, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the decentralized wagers method.
        
        Args:
            num_models: Number of LLMs in the ensemble
            config: Configuration dictionary with:
                - common_hidden_dim: Common dimension for projected hidden states (default: 4096)
                - hidden_layers: List of hidden layer sizes (default: [512, 256])
                - learning_rate: Learning rate for optimizer (default: 1e-4)
                - device: Device to run on (default: 'cuda' if available, else 'cpu')
        """
        super().__init__(num_models, config)
        
        # Get configuration (ensure proper types)
        self.common_hidden_dim = int(config.get("common_hidden_dim", 4096))
        self.hidden_layers = list(config.get("hidden_layers", [512, 256]))
        self.learning_rate = float(config.get("learning_rate", 1e-4))  # Lower default LR: 1e-5 instead of 1e-4
        self.temperature = float(config.get("temperature", 2.0))  # Temperature for softmax (higher = softer)
        self.grad_clip_norm = float(config.get("grad_clip_norm", 1.0))  # Gradient clipping norm
        self.normalize_hidden_states = config.get("normalize_hidden_states", True)  # L2 normalize hidden states
        self.device_str = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.device = torch.device(self.device_str)
        
        # Build per-model projection layers to handle variable hidden dimensions
        # These will be created dynamically when we first see the hidden states
        # Shared across all routers (projection is model-specific, not router-specific)
        self.model_projections = nn.ModuleDict()
        self._model_hidden_dims = {}  # Track each model's hidden dim
        
        # Build per-model routers
        # Each router takes a single model's projected hidden state and outputs a scalar
        self.routers = nn.ModuleList()
        for i in range(num_models):
            router = self._build_router().to(self.device)
            self.routers.append(router)
        
        # Collect all parameters (routers + projections) for optimizer
        all_params = list(self.routers.parameters())
        # Projections will be added to optimizer when created
        
        # Optimizer
        self.optimizer = torch.optim.Adam(all_params, lr=self.learning_rate)
        
        # Training mode flag
        self._training = True
        
        # Cache for computed values during training (to avoid recomputation in update())
        # These are set by compute_wagers() and cleared by update()
        self._cached_wagers: Optional[torch.Tensor] = None
        self._cached_projected_states: Optional[List[torch.Tensor]] = None
    
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

    def compute_wagers(
        self,
        hidden_states_list: List[np.ndarray],
        model_logits: Optional[np.ndarray] = None,
        gold_label: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
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
        projected_batch_list = []
        
        # Project each model's batch
        for i in range(self.num_models):
            model_hs_batch = hidden_states_list[i]
            model_hidden_dim = model_hs_batch.shape[-1]
            
            proj_key = f"proj_{i}"
            if proj_key not in self.model_projections:
                projection = nn.Linear(model_hidden_dim, self.common_hidden_dim).to(self.device)
                self.model_projections[proj_key] = projection
                if self._training:
                    self.optimizer.add_param_group({'params': projection.parameters()})
            
            projection = self.model_projections[proj_key]
            model_hs_tensor = torch.as_tensor(model_hs_batch, dtype=torch.float32).to(self.device)
            
            if self.normalize_hidden_states:
                norms = torch.norm(model_hs_tensor, dim=1, keepdim=True)
                model_hs_tensor = model_hs_tensor / (norms + 1e-8)
            
            with torch.set_grad_enabled(self._training):
                projected_batch = projection(model_hs_tensor)
            
            projected_batch_list.append(projected_batch)
        
        # Route
        raw_wagers_list = []
        for router in self.routers:
            router.eval() if not self._training else router.train()
        
        with torch.set_grad_enabled(self._training):
            for i in range(self.num_models):
                model_projected = projected_batch_list[i]
                router_i = self.routers[i]
                raw_wager_i = router_i(model_projected)
                raw_wagers_list.append(raw_wager_i)
        
        # Normalize
        raw_wagers_tensor = torch.cat(raw_wagers_list, dim=1)
        # wagers = torch.softmax(raw_wagers_tensor / self.temperature, dim=1)
        sigmoid_wagers = torch.sigmoid(raw_wagers_tensor/self.temperature)
        wagers = sigmoid_wagers / (torch.sum(sigmoid_wagers, dim=1, keepdim=True))
        
        if self._training:
            self._cached_wagers = wagers
            projected_concatenated = torch.cat(projected_batch_list, dim=1)
            self._cached_projected_states = projected_concatenated
        
        return {"wagers": wagers.detach().cpu().numpy()}
    
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
        """
        Update the routers using cross-entropy loss.
        
        Reuses cached wagers from compute_wagers() if available.
        Supports single samples ([num_models, hidden_dim]) and batches ([batch_size, num_models, ...]).
        
        Args:
            aggregated_probs: [batch_size, num_options] or [num_options]
            aggregated_pred: [batch_size] or scalar
            gold_label: [batch_size] or scalar
            model_probs: [batch_size, num_models, num_options] or [num_models, num_options]
            model_logits: [batch_size, num_models, num_options] or [num_models, num_options]
            question: Optional
            hidden_states: [batch_size, num_models, hidden_dim], [num_models, hidden_dim], or list
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with loss
        """
        if not self._training:
            return {}
        
        # Use cached wagers or recompute
        if self._cached_wagers is not None:
            wagers = self._cached_wagers
        else:
            raise ValueError(
                "DecentralizedWagers.update() requires cached wagers from compute_wagers(). "
                "Please ensure compute_wagers() is called before update()."
            )
        
        # Clear cache
        self._cached_wagers = None
        self._cached_projected_states = None
        
        # Normalize dimensions
        is_batch = model_logits.ndim == 3
        if not is_batch:
            model_logits = model_logits[np.newaxis, :, :]
            gold_label = np.array([gold_label])
            wagers = wagers[np.newaxis, :]
        
        batch_size = model_logits.shape[0]
        model_logits_tensor = torch.as_tensor(model_logits, dtype=torch.float32).to(self.device)
        gold_label_tensor = torch.as_tensor(gold_label, dtype=torch.long).to(self.device)
        
        # Aggregate and compute loss (batch processing)
        aggregated_probs_torch = LinearPooling.aggregate_torch(
            model_logits_tensor, wagers
        )  # [batch_size, num_options]
        
        losses = -torch.log(aggregated_probs_torch[torch.arange(batch_size), gold_label_tensor] + 1e-10)
        loss = losses.mean()
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.routers.parameters(), self.grad_clip_norm)
        if len(self.model_projections) > 0:
            proj_params = [p for proj in self.model_projections.values() for p in proj.parameters()]
            torch.nn.utils.clip_grad_norm_(proj_params, self.grad_clip_norm)
        self.optimizer.step()
        
        return {"loss": float(loss.detach().cpu().numpy())}

    
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
        # Clear cache when switching modes
        self._cached_wagers = None
        self._cached_projected_states = None
    
    def eval_mode(self):
        """Set the method to evaluation mode."""
        for router in self.routers:
            router.eval()
        self._training = False
        # Clear cache when switching modes
        self._cached_wagers = None
        self._cached_projected_states = None
    
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
            # Create projections from checkpoint if they don't exist yet
            # (projections are normally created dynamically when compute_wagers() is first called)
            for key, proj_state in state_dict["model_projections_state_dict"].items():
                if key not in self.model_projections:
                    # Extract hidden dimension from projection weight shape
                    # proj_state is a state dict with 'weight' and 'bias' keys
                    # weight shape is [common_hidden_dim, model_hidden_dim]
                    model_hidden_dim = proj_state['weight'].shape[1]
                    projection = nn.Linear(model_hidden_dim, self.common_hidden_dim).to(self.device)
                    self.model_projections[key] = projection
                    # Add to optimizer (will be used if resuming training, ignored for eval)
                    if self._training:
                        self.optimizer.add_param_group({'params': projection.parameters()})
                
                # Now load the state
                self.model_projections[key].load_state_dict(proj_state)
        if "optimizer_state_dict" in state_dict:
            try:
                self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            except (ValueError, KeyError) as e:
                # Optimizer state dict may not match if projection layers were added/removed
                # This is acceptable - we'll continue with a fresh optimizer
                import logging
                log = logging.getLogger("lm_polygraph")
                log.warning(
                    f"Could not load optimizer state dict (parameter mismatch): {e}. "
                    "Continuing with fresh optimizer state."
                )

