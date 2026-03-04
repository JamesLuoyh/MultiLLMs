"""
Base class for wagering methods that generate weights for LLMs.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np
import torch
import sys
from pathlib import Path

# Add src/ to path for lm_polygraph imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.utils.dataset import Dataset


class WageringMethod(ABC):
    """
    Base class for wagering methods that generate weights/wagers for each LLM.
    
    Each wagering method can optionally have trainable parameters (e.g., a router network).
    The method generates wagers based on the question, the models, or both.
    """
    
    def __init__(self, num_models: int, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the wagering method.
        
        Args:
            num_models: Number of LLMs in the ensemble
            config: Optional configuration dictionary
        """
        self.num_models = num_models
        self.config = config or {}
    
    @abstractmethod
    def compute_wagers(
        self,
        question: Optional[str] = None,
        models: Optional[List[WhiteboxModel]] = None,
        model_logits: Optional[np.ndarray] = None,
        gold_label: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute wagers (weights) for each LLM.
        
        Args:
            question: The input question/prompt (optional, not used in batch mode)
            models: List of LLM models (optional)
            model_logits: Pre-computed logits from models
                Shape [batch_size, num_models, num_options] for batch mode
                Shape [num_models, num_options] for single sample (backwards compatibility)
            gold_label: Optional ground truth labels
                Shape [batch_size] for batch mode
                Scalar (int) for single sample (backwards compatibility)
            **kwargs: Additional keyword arguments
                hidden_states: np.ndarray of shape [batch_size, num_models, hidden_dim] or list of arrays
                
        Returns:
            Dictionary with:
                "wagers": np.ndarray of shape [batch_size, num_models] with non-negative weights
                For single samples (backwards compatibility): shape [num_models]
                "nash_gap": Optional float or np.ndarray of shape [batch_size] (if computed)
        """
        pass
    
    @abstractmethod
    def update(
        self,
        aggregated_probs: np.ndarray,
        aggregated_pred: np.ndarray,
        gold_label: np.ndarray,
        model_probs: np.ndarray,
        model_logits: np.ndarray,
        question: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Update the wagering method based on the aggregated prediction and gold label.
        
        This method is called during training to update any trainable parameters
        (e.g., router networks) or internal state.
        
        Args:
            aggregated_probs: Aggregated probabilities
                Shape [batch_size, num_options] for batch mode
                Shape [num_options] for single sample (backwards compatibility)
            aggregated_pred: Predicted class indices
                Shape [batch_size] for batch mode
                Scalar (int) for single sample (backwards compatibility)
            gold_label: Ground truth class indices
                Shape [batch_size] for batch mode
                Scalar (int) for single sample (backwards compatibility)
            model_probs: Per-model probabilities
                Shape [batch_size, num_models, num_options] for batch mode
                Shape [num_models, num_options] for single sample
            model_logits: Per-model logits
                Shape [batch_size, num_models, num_options] for batch mode
                Shape [num_models, num_options] for single sample
            question: The input question/prompt (optional, not used in batch mode)
            **kwargs: Additional keyword arguments
                hidden_states: np.ndarray of shape [batch_size, num_models, hidden_dim] or list
                
        Returns:
            Dictionary with update information (loss, metrics, etc.)
            Should be aggregated per batch (e.g., average loss, not per-sample)
        """
        pass
    
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """
        Get list of trainable parameters (if any).
        
        Returns:
            List of PyTorch parameters that should be optimized
        """
        return []
    
    def train_mode(self):
        """Set the method to training mode (if applicable)."""
        pass
    
    def eval_mode(self):
        """Set the method to evaluation mode (if applicable)."""
        pass
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Get state dictionary for checkpointing.
        
        Returns:
            Dictionary containing the state of the wagering method
        """
        return {}
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load state dictionary from checkpoint.
        
        Args:
            state_dict: Dictionary containing the state to load
        """
        pass
    
    def save_pretrained(self, save_directory: str):
        """
        Save the wagering method to disk.
        
        Args:
            save_directory: Directory to save to
        """
        import os
        os.makedirs(save_directory, exist_ok=True)
        # Default implementation: save state dict
        import torch
        torch.save(self.state_dict(), os.path.join(save_directory, "wagering_state.pt"))
    
    def load_pretrained(self, save_directory: str):
        """
        Load the wagering method from disk.
        
        Args:
            save_directory: Directory to load from
        """
        import os
        import torch
        state_path = os.path.join(save_directory, "wagering_state.pt")
        if os.path.exists(state_path):
            state_dict = torch.load(state_path, map_location="cpu")
            self.load_state_dict(state_dict)


