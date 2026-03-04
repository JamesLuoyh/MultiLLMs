"""
MADDPG-based Kelly Wagers implementation: Centralized training, decentralized execution.

Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm for learning wagers.
- Per-model actors (routers): Output wagers from hidden states (decentralized execution)
- Per-model centralized critics: Estimate value from all hidden states and wagers
- Training with reward defined by logarithmic utility
- Actor updates via policy gradient
- Critic updates via MSE loss minimization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple

from .base import WageringMethod
from lm_polygraph.utils.model import WhiteboxModel
from .utils import compute_scoring_rule


class KellyWagers(WageringMethod):
    """
    MADDPG-based wagering method for ensemble learning.
    
    Architecture:
    - Per-model Actor: Takes model's hidden state, outputs wager (deterministic policy)
    - Per-model Critic: Takes all hidden states + all wagers, outputs value estimate
    
    Training:
    - Reward: r_i = log(1 + w_i * (s(p_i, omega) - weighted_avg_score))
    - Critic Loss: MSE between predicted Q and reward
    - Actor Loss: Policy gradient
    
    Actors and Critics:
    - Input: Single model's projected hidden state [common_hidden_dim] for actors
    - Input: Concatenated all hidden states and wagers for critics
    - Actor Output: Single scalar (wager value)
    - Critic Output: Single scalar (value estimate)
    """
    
    def __init__(self, num_models: int, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the MADDPG-based wagers method.
        
        Args:
            num_models: Number of LLMs in the ensemble
            config: Configuration dictionary with:
                - common_hidden_dim: Common dimension for projected hidden states (default: 4096)
                - actor_hidden_layers: List of hidden layer sizes for actors (default: [512, 256])
                - critic_hidden_layers: List of hidden layer sizes for critics (default: [1024, 512])
                - actor_learning_rate: Actor learning rate (default: 1e-5)
                - critic_learning_rate: Critic learning rate (default: 1e-5)
                - temperature: Temperature for softmax wager normalization (default: 1.0)
                - grad_clip_norm: Gradient clipping norm (default: 1.0)
                - normalize_hidden_states: Whether to L2 normalize hidden states (default: True)
                - scoring_rule: Scoring rule - 'logarithmic' (default: 'logarithmic')
                - entropy_coeff: Entropy coefficient for exploration (default: 1e-2)
                - device: Device to run on (default: 'cuda' if available, else 'cpu')
        """
        super().__init__(num_models, config)
        
        # Configuration
        self.common_hidden_dim = int(config.get("common_hidden_dim", 4096))
        self.actor_hidden_layers = list(config.get("actor_hidden_layers", [512, 256]))
        self.critic_hidden_layers = list(config.get("critic_hidden_layers", [1024, 512]))
        self.actor_learning_rate = float(config.get("actor_learning_rate", 1e-5))
        self.critic_learning_rate = float(config.get("critic_learning_rate", 1e-5))
        self.temperature = float(config.get("temperature", 1.0))
        self.grad_clip_norm = float(config.get("grad_clip_norm", 1.0))
        self.normalize_hidden_states = config.get("normalize_hidden_states", True)
        self.scoring_rule = str(config.get("scoring_rule", "logarithmic")).lower()
        self.entropy_coeff = float(config.get("entropy_coeff", 1e-2))
        self.device_str = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.device = torch.device(self.device_str)
        
        # Validate scoring rule
        if self.scoring_rule not in ["logarithmic"]:
            raise ValueError(
                f"Unknown scoring rule: {self.scoring_rule}. "
                "Supported: 'logarithmic'"
            )
        
        # Build per-model projection layers
        self.model_projections = nn.ModuleDict()
        
        # Build per-model actors and critics
        self.actors = nn.ModuleList()
        self.critics = nn.ModuleList()
        
        for i in range(num_models):
            actor = self._build_actor().to(self.device)
            critic = self._build_critic().to(self.device)
            self.actors.append(actor)
            self.critics.append(critic)
        
        # Optimizers
        actor_params = [p for actor in self.actors for p in actor.parameters()]
        self.actor_optimizer = torch.optim.Adam(actor_params, lr=self.actor_learning_rate)
        
        critic_params = [p for critic in self.critics for p in critic.parameters()]
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=self.critic_learning_rate)
        
        # Training mode flag
        self._training = True
        
        self._old_wagers: Optional[torch.Tensor] = None
        self._cached_wagers: Optional[torch.Tensor] = None
        self._cached_projected_states: Optional[List[torch.Tensor]] = None
    
    def _build_actor(self) -> nn.Module:
        """
        Build a single actor network (policy network).
        
        Returns:
            nn.Sequential: Actor network that maps [common_hidden_dim] -> [1]
        """
        layers = []
        prev_dim = self.common_hidden_dim
        
        # Hidden layers
        for hidden_dim in self.actor_hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        # Output layer: outputs single scalar (wager logit)
        layers.append(nn.Linear(prev_dim, 1))
        
        return nn.Sequential(*layers)
    
    def _build_critic(self) -> nn.Module:
        """
        Build a single critic network (value network).
        
        Input: Concatenation of all projected hidden states and all wagers
        Critic i takes: [hidden_1, hidden_2, ..., hidden_M, w_1, w_2, ..., w_M]
        Output: Single scalar (value estimate)
        
        Returns:
            nn.Sequential: Critic network
        """
        # Input dimension: num_models * common_hidden_dim + num_models (for wagers)
        input_dim = self.num_models * self.common_hidden_dim + self.num_models
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in self.critic_hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        # Output layer: outputs single scalar (value)
        layers.append(nn.Linear(prev_dim, 1))
        
        return nn.Sequential(*layers)
    

    def _compute_wagers_from_hidden_states(
        self,
        hidden_states: np.ndarray,
        return_projected_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Compute wagers from hidden states using per-model actors.
        
        Args:
            hidden_states: np.ndarray of shape [num_models, hidden_dim] or object array with variable shapes
            return_projected_states: If True, also return the projected states
            
        Returns:
            wagers: torch.Tensor of shape [num_models] with wager values (normalized via softmax)
            projected_states: Optional list of projected states
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
                # Add to actor optimizer if in training mode
                if self._training:
                    self.actor_optimizer.add_param_group({'params': projection.parameters()})
            
            projection = self.model_projections[proj_key]
            projected = projection(model_hs.unsqueeze(0)).squeeze(0)
            projected_states.append(projected)
        
        # Step 2: Compute wagers via per-model actors
        raw_wagers = []
        for actor in self.actors:
            actor.eval() if not self._training else actor.train()
        
        with torch.set_grad_enabled(self._training):
            for i in range(num_models_in):
                projected_hs = projected_states[i]
                actor_i = self.actors[i]
                raw_wager_i = actor_i(projected_hs.unsqueeze(0))
                raw_wagers.append(raw_wager_i.squeeze())
        
        # Step 3: Normalize wagers via sigmoid and normalization
        raw_wagers_tensor = torch.stack(raw_wagers)
        wagers = torch.sigmoid(raw_wagers_tensor)
        wagers = wagers / torch.sum(wagers)
        if return_projected_states:
            return wagers, projected_states
        return wagers, None
    
    def compute_wagers(
        self,
        hidden_states_list: List[np.ndarray],
        model_logits: Optional[np.ndarray] = None,
        gold_label: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute wagers for a batch of samples with heterogeneous (variable) hidden dimensions per model.
        
        This method processes each model's hidden states in batch through its projection layer,
        then aggregates back, maximizing batch processing efficiency.
        
        Args:
            hidden_states_list: List of [num_models] where each element is np.ndarray of shape [batch_size, hidden_dim_i]
            model_logits: Optional, not used in this computation
            gold_label: Optional, ground-truth labels (used by methods that compute label-aware diagnostics)
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary with:
                "wagers": np.ndarray of shape [batch_size, num_models] with probabilities (sum to 1 for each row)   
        """
        if len(hidden_states_list) != self.num_models:
            raise ValueError(
                f"Expected {self.num_models} models, got {len(hidden_states_list)} in hidden_states_list"
            )
        
        batch_size = hidden_states_list[0].shape[0]
        
        # Step 1: Project each model's hidden states in batch
        # Result: projected_batch_list[i] has shape [batch_size, common_hidden_dim]
        projected_batch_list = []
        
        for i in range(self.num_models):
            model_hs_batch = hidden_states_list[i]  # [batch_size, hidden_dim_i]
            model_hidden_dim = model_hs_batch.shape[-1]
            
            # Create projection layer if needed
            proj_key = f"proj_{model_hidden_dim}"
            if proj_key not in self.model_projections:
                projection = nn.Linear(model_hidden_dim, self.common_hidden_dim).to(self.device)
                self.model_projections[proj_key] = projection
                # Add to actor optimizer if in training mode
                if self._training:
                    self.actor_optimizer.add_param_group({'params': projection.parameters()})
            
            projection = self.model_projections[proj_key]
            
            # Convert to tensor and project entire batch at once
            model_hs_tensor = torch.as_tensor(model_hs_batch, dtype=torch.float32).to(self.device)
            
            # Normalize if needed
            if self.normalize_hidden_states:
                norms = torch.norm(model_hs_tensor, dim=1, keepdim=True)
                model_hs_tensor = model_hs_tensor / (norms + 1e-8)
            
            # Project in batch: [batch_size, hidden_dim_i] -> [batch_size, common_hidden_dim]
            with torch.set_grad_enabled(self._training):
                projected_batch = projection(model_hs_tensor)  # [batch_size, common_hidden_dim]
            
            projected_batch_list.append(projected_batch)
        
        # Step 2: Compute wagers via per-model actors (still per-sample because wagers are sample-specific)
        # but with projected states already computed efficiently in batch
        batch_wagers = []
        
        for b in range(batch_size):
            sample_raw_wagers = []
            
            with torch.set_grad_enabled(self._training):
                for i in range(self.num_models):
                    projected_hs = projected_batch_list[i][b]  # [common_hidden_dim]
                    actor_i = self.actors[i]
                    actor_i.eval() if not self._training else actor_i.train()
                    raw_wager_i = actor_i(projected_hs.unsqueeze(0))  # [1, 1]
                    sample_raw_wagers.append(raw_wager_i.squeeze())
            
            # Normalize wagers via sigmoid and softmax
            sample_raw_wagers_tensor = torch.stack(sample_raw_wagers)
            sample_wagers = torch.sigmoid(sample_raw_wagers_tensor)
            sample_wagers = sample_wagers / torch.sum(sample_wagers)
            batch_wagers.append(sample_wagers)
        
        # Stack all into batch: [batch_size, num_models]
        batch_wagers_tensor = torch.stack(batch_wagers, dim=0)
        return {"wagers": batch_wagers_tensor.detach().cpu().numpy()}
    
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
        Update both critics and actors using MADDPG algorithm.
        
        Supports both batch and single-sample modes. Auto-detects based on input shape.
        
        1. Compute reward: r_i = log(1 + w_i * (s(p_i, omega) - weighted_avg_score))
        2. Update critics: minimize MSE(r_i - Q^phi_i(states, wagers))
        3. Update actors: maximize gradient of Q wrt wagers
        
        Args:
            aggregated_probs: Aggregated probabilities, shape [num_options] (single)
                            or [batch_size, num_options] (batch)
            aggregated_pred: Predicted class index
            gold_label: Ground truth class index (outcome omega), int (single) or np.ndarray (batch)
            model_probs: Per-model probabilities, shape [num_models, num_options] (single)
                        or [batch_size, num_models, num_options] (batch)
            model_logits: Per-model logits, shape [num_models, num_options] (single)
                         or [batch_size, num_models, num_options] (batch)
            question: The input question/prompt (optional)
            hidden_states: np.ndarray of shape [num_models, hidden_dim] (single)
                          or [batch_size, num_models, hidden_dim] (batch)
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary with loss and update information (averaged over batch if batch mode)
        """
        if not self._training:
            return {}
        
        # Reuse cached wagers if available, otherwise recompute
        if self._cached_wagers is not None:
            batch_wagers = self._cached_wagers
        elif hidden_states is not None:
            # Detect mode and compute wagers
            if isinstance(hidden_states, list):
                # Heterogeneous: use compute_wagers_batch_heterogeneous
                batch_wagers_np = self.compute_wagers(hidden_states)
                batch_wagers = torch.as_tensor(batch_wagers_np, dtype=torch.float32).to(self.device)
    
        batch_size = model_logits.shape[0]
        
        # Wagers already computed and passed in
        
        # Convert to tensors
        model_probs_tensor = torch.as_tensor(model_probs, dtype=torch.float32).to(self.device)
        gold_label_tensor = torch.as_tensor(gold_label, dtype=torch.long).to(self.device)
        
        # Compute scoring rules for batch: [batch_size, num_models]
        batch_model_scores = []
        for b in range(batch_size):
            scores = compute_scoring_rule(
                model_probs_tensor[b], gold_label_tensor[b], scoring_rule=self.scoring_rule
            )
            batch_model_scores.append(scores)
        batch_model_scores = torch.stack(batch_model_scores)  # [batch_size, num_models]
        
        # Compute weighted average scores for each sample: [batch_size]
        batch_weighted_avg_scores = torch.sum(batch_model_scores * batch_wagers, dim=1)
        
        # Compute rewards for batch: [batch_size, num_models]
        batch_utility_terms = batch_wagers * (batch_model_scores - batch_weighted_avg_scores.unsqueeze(1))
        batch_rewards = torch.log(torch.clamp(1.0 + batch_utility_terms, min=1e-10))
        
        # ============================================================
        # CRITIC UPDATE: Minimize MSE over entire batch
        # ============================================================
        mse_loss = 0.0
        for i in range(self.num_models):
            critic_i = self.critics[i]
            # Process each sample in batch through critic
            for b in range(batch_size):
                # Project hidden states for this sample and recompute critic input
                sample_hidden_states = hidden_states[b]  # [num_models, hidden_dim]
                _, sample_projected_states = self._compute_wagers_from_hidden_states(
                    sample_hidden_states, return_projected_states=True
                )
                sample_wagers = batch_wagers[b]  # [num_models]
                critic_input = torch.cat(sample_projected_states + [sample_wagers], dim=0).unsqueeze(0)
                
                q_value = critic_i(critic_input).squeeze()
                target = batch_rewards[b, i].detach()
                loss_i_b = F.mse_loss(q_value, target)
                mse_loss += loss_i_b
        
        # Average over all critic-sample pairs
        mse_loss = mse_loss / (self.num_models * batch_size)
        
        # Add cross entropy loss on aggregated output
        aggregated_probs_tensor = torch.as_tensor(aggregated_probs, dtype=torch.float32).to(self.device)
        ce_loss = F.cross_entropy(aggregated_probs_tensor, gold_label_tensor)
        
        # Total critic loss
        critic_loss = mse_loss + ce_loss
        
        # Backward pass for critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(
            [p for critic in self.critics for p in critic.parameters()],
            self.grad_clip_norm
        )
        self.critic_optimizer.step()
        
        # ============================================================
        # ACTOR UPDATE: Maximize policy gradient over batch (simplified for efficiency)
        # ============================================================
        # For efficiency in batch mode, we update actors with a simplified loss
        # that doesn't require recomputing critic values for all combinations
        actor_loss = 0.0
        entropy_loss = 0.0
        
        for b in range(batch_size):
            sample_hidden_states = hidden_states[b]  # [num_models, hidden_dim]
            _, sample_projected_states = self._compute_wagers_from_hidden_states(
                sample_hidden_states, return_projected_states=True
            )
            sample_wagers = batch_wagers[b]  # [num_models]
            
            # Compute entropy for this sample
            if self.entropy_coeff > 0:
                entropy = -torch.sum(sample_wagers * torch.log(sample_wagers + 1e-10))
                entropy_loss += -self.entropy_coeff * entropy
        
        # Average entropy loss over batch
        entropy_loss = entropy_loss / batch_size if batch_size > 0 else entropy_loss
        
        # For actor loss, use the batch rewards directly (simplified MADDPG)
        actor_loss = -torch.mean(batch_rewards)  # Maximize rewards
        
        # Backward pass for actors
        self.actor_optimizer.zero_grad()
        total_actor_loss = actor_loss + entropy_loss
        total_actor_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(
            [p for actor in self.actors for p in actor.parameters()],
            self.grad_clip_norm
        )
        if len(self.model_projections) > 0:
            proj_params = [p for proj in self.model_projections.values() for p in proj.parameters()]
            torch.nn.utils.clip_grad_norm_(proj_params, self.grad_clip_norm)
        
        self.actor_optimizer.step()
        
        # Compute batch metrics
        batch_correct = (np.argmax(aggregated_probs, axis=1) == gold_label)
        batch_accuracy = float(np.mean(batch_correct))
        
        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "mse_loss": float(mse_loss.item()),
            "ce_loss": float(ce_loss.item()),
            "entropy_loss": float(entropy_loss.item()),
            "batch_accuracy": batch_accuracy,
            "batch_size": batch_size,
        }
    
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get list of trainable parameters (actors, critics, projections)."""
        params = []
        for actor in self.actors:
            params.extend(actor.parameters())
        for critic in self.critics:
            params.extend(critic.parameters())
        for proj in self.model_projections.values():
            params.extend(proj.parameters())
        return params
    
    def train_mode(self):
        """Set the method to training mode."""
        for actor in self.actors:
            actor.train()
        for critic in self.critics:
            critic.train()
        self._training = True
    
    def eval_mode(self):
        """Set the method to evaluation mode."""
        for actor in self.actors:
            actor.eval()
        for critic in self.critics:
            critic.eval()
        self._training = False
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for checkpointing."""
        actors_state_dict = {f"actor_{i}": actor.state_dict() for i, actor in enumerate(self.actors)}
        critics_state_dict = {f"critic_{i}": critic.state_dict() for i, critic in enumerate(self.critics)}
        
        return {
            "actors_state_dict": actors_state_dict,
            "critics_state_dict": critics_state_dict,
            "model_projections_state_dict": {k: v.state_dict() for k, v in self.model_projections.items()},
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "config": {
                "common_hidden_dim": self.common_hidden_dim,
                "actor_hidden_layers": self.actor_hidden_layers,
                "critic_hidden_layers": self.critic_hidden_layers,
                "actor_learning_rate": self.actor_learning_rate,
                "critic_learning_rate": self.critic_learning_rate,
                "temperature": self.temperature,
                "grad_clip_norm": self.grad_clip_norm,
                "normalize_hidden_states": self.normalize_hidden_states,
                "scoring_rule": self.scoring_rule,
                "entropy_coeff": self.entropy_coeff,
                "device": self.device_str,
            },
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dictionary from checkpoint."""
        if "actors_state_dict" in state_dict:
            actors_state_dict = state_dict["actors_state_dict"]
            for i, actor in enumerate(self.actors):
                key = f"actor_{i}"
                if key in actors_state_dict:
                    actor.load_state_dict(actors_state_dict[key])
        
        if "critics_state_dict" in state_dict:
            critics_state_dict = state_dict["critics_state_dict"]
            for i, critic in enumerate(self.critics):
                key = f"critic_{i}"
                if key in critics_state_dict:
                    critic.load_state_dict(critics_state_dict[key])
        
        if "model_projections_state_dict" in state_dict:
            for key, proj_state in state_dict["model_projections_state_dict"].items():
                if key not in self.model_projections:
                    in_features = proj_state["weight"].shape[1]
                    out_features = proj_state["weight"].shape[0]
                    projection = nn.Linear(in_features, out_features).to(self.device)
                    self.model_projections[key] = projection
                    if self._training:
                        self.actor_optimizer.add_param_group({'params': projection.parameters()})
                self.model_projections[key].load_state_dict(proj_state)
        
        if "actor_optimizer_state_dict" in state_dict:
            try:
                self.actor_optimizer.load_state_dict(state_dict["actor_optimizer_state_dict"])
            except (ValueError, KeyError) as e:
                import logging
                log = logging.getLogger("lm_polygraph")
                log.warning(
                    f"Could not load actor optimizer state dict (parameter mismatch): {e}. "
                    "Continuing with fresh optimizer state."
                )
        
        if "critic_optimizer_state_dict" in state_dict:
            try:
                self.critic_optimizer.load_state_dict(state_dict["critic_optimizer_state_dict"])
            except (ValueError, KeyError) as e:
                import logging
                log = logging.getLogger("lm_polygraph")
                log.warning(
                    f"Could not load critic optimizer state dict (parameter mismatch): {e}. "
                    "Continuing with fresh optimizer state."
                )
