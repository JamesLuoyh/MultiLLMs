"""
Zero-one wagers implementation: assigns zero or one weights to all LLMs.
"""

import numpy as np
from typing import Optional, Dict, Any, List

from .base import WageringMethod
from lm_polygraph.utils.model import WhiteboxModel


class ZeroOneWagers(WageringMethod):
    """
    Simple wagering method that assigns zero or one weights to all LLMs.
    
    This is the baseline method with no trainable parameters.
    """
    
    def compute_wagers(
        self,
        question: Optional[str] = None,
        models: Optional[List[WhiteboxModel]] = None,
        model_logits: Optional[np.ndarray] = None,
        gold_label: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Return zero-one wagers for all models.
        Supports both batch and single-sample modes.
        
        Args:
            question: Ignored (kept for interface compatibility)
            models: Ignored (kept for interface compatibility)
            model_logits: np.ndarray of shape [batch_size, num_models, num_options] (batch)
                or [num_models, num_options] (single sample)
            gold_label: Ignored (kept for interface compatibility)
            **kwargs: Ignored
            
        Returns:
            wagers: np.ndarray of shape [batch_size, num_models] (batch)
                or [num_models] (single sample)
        """
        base_wagers = np.concatenate([
            np.zeros(self.num_models // 2, dtype=np.float32),
            np.ones(self.num_models - self.num_models // 2, dtype=np.float32)
        ])
        
        # Detect batch mode from model_logits if provided
        if model_logits is not None and model_logits.ndim == 3:
            batch_size = model_logits.shape[0]
            # Return [batch_size, num_models] by repeating base_wagers
            return {"wagers": np.tile(base_wagers, (batch_size, 1))}
        else:
            # Single sample mode: return [num_models]
            return {"wagers": base_wagers}
    
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
        No-op update (zero-one wagers don't change).
        Supports both batch and single-sample modes.
        
        Args:
            aggregated_probs: Ignored
            aggregated_pred: Ignored (can be int or array)
            gold_label: Ignored (can be int or array)
            model_probs: Ignored
            model_logits: Ignored
            question: Ignored
            **kwargs: Ignored
            
        Returns:
            Empty dictionary
        """
        return {}


