"""
Linear pooling aggregation: weighted average of probabilities.
"""

import numpy as np
import torch
from typing import Tuple

from .base import AggregationFunction


class LinearPooling(AggregationFunction):
    """
    Linear pooling: weighted average of probabilities from each model.
    
    Linear pooling aggregates probabilities directly: A = sum_i w_i * P_i(H|E_i)
    where sum_i w_i = 1.
    """
    
    def aggregate(
        self,
        model_logits: np.ndarray,
        wagers: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aggregate using linear pooling (weighted average in probability space).
        
        Args:
            model_logits: Shape [batch_size, num_models, num_options] or [num_models, num_options]
            wagers: Shape [batch_size, num_models] or [num_models]
            
        Returns:
            aggregated_log_probs: Log-probabilities after aggregation
            aggregated_probs: Normalized probabilities after aggregation
        """
        model_logits = np.asarray(model_logits, dtype=np.float32)
        wagers = np.asarray(wagers, dtype=np.float32)
        
        if not np.allclose(np.sum(wagers, axis=-1), 1.0):
            raise ValueError("Wagers must sum to 1")
        
        # Batch mode
        if model_logits.ndim == 3 and wagers.ndim == 2:
            batch_size, num_models, num_options = model_logits.shape
            
            if wagers.shape != (batch_size, num_models):
                raise ValueError(
                    f"Wagers shape mismatch: expected [{batch_size}, {num_models}], "
                    f"got {wagers.shape}"
                )
            
            # Softmax to get probabilities
            max_logits = np.max(model_logits, axis=2, keepdims=True)
            stabilized = model_logits - max_logits
            exp_stabilized = np.exp(stabilized)
            probs = exp_stabilized / exp_stabilized.sum(axis=2, keepdims=True)
            
            # Weighted average
            aggregated_probs = (wagers[:, :, None] * probs).sum(axis=1)
            
            if np.any(np.isnan(aggregated_probs)) or np.any(np.isinf(aggregated_probs)):
                raise ValueError("Invalid aggregated probabilities (NaN or inf detected)")
            
            # Normalize
            probs_sum = aggregated_probs.sum(axis=1, keepdims=True)
            if np.any(probs_sum < 1e-10):
                raise ValueError("Aggregated probabilities sum to near-zero")
            
            aggregated_probs = aggregated_probs / probs_sum
            aggregated_probs = np.clip(aggregated_probs, 0.0, 1.0)
            aggregated_probs = aggregated_probs / aggregated_probs.sum(axis=1, keepdims=True)
            
            # Validate
            if not np.all(aggregated_probs >= 0):
                raise ValueError("Probabilities must be non-negative")
            if not np.allclose(aggregated_probs.sum(axis=1), 1.0, atol=1e-6):
                raise ValueError("Probabilities must sum to 1.0")
            
            # Log probabilities
            epsilon = 1e-10
            aggregated_log_probs = np.log(np.clip(aggregated_probs, epsilon, 1.0))
            
            return aggregated_log_probs, aggregated_probs
        
        # Single sample mode
        elif model_logits.ndim == 2 and wagers.ndim == 1:
            if wagers.shape[0] != model_logits.shape[0]:
                raise ValueError("Wagers shape must match number of models")
            
            if not np.isclose(np.sum(wagers), 1.0, atol=1e-6):
                raise ValueError(f"Wagers must sum to 1.0, got {np.sum(wagers)}")
            
            # Softmax to get probabilities
            max_logits = np.max(model_logits, axis=1, keepdims=True)
            stabilized = model_logits - max_logits
            exp_stabilized = np.exp(stabilized)
            probs = exp_stabilized / exp_stabilized.sum(axis=1, keepdims=True)
            
            # Weighted average
            pooled_probs = (wagers[:, None] * probs).sum(axis=0)
            pooled_probs = pooled_probs / pooled_probs.sum()
            
            # Validate
            if not np.all(pooled_probs >= 0):
                raise ValueError("Probabilities must be non-negative")
            if not np.isclose(pooled_probs.sum(), 1.0, atol=1e-6):
                raise ValueError("Probabilities must sum to 1.0")
            
            # Log probabilities
            epsilon = 1e-10
            pooled_log_probs = np.log(np.clip(pooled_probs, epsilon, 1.0))
            
            return pooled_log_probs, pooled_probs
        
        else:
            raise ValueError(
                f"Invalid shapes: model_logits={model_logits.shape}, wagers={wagers.shape}"
            )
    
    @staticmethod
    def aggregate_torch(
        model_logits: torch.Tensor,
        wagers: torch.Tensor,
    ) -> torch.Tensor:
        """
        PyTorch version of linear pooling (supports gradients).
        
        Args:
            model_logits: Shape [batch_size, num_models, num_options] or [num_models, num_options]
            wagers: Shape [batch_size, num_models] or [num_models]
            
        Returns:
            aggregated_probs: Aggregated probabilities
        """
        # Batch mode
        if model_logits.ndim == 3 and wagers.ndim == 2:
            model_probs = torch.softmax(model_logits, dim=2)
            aggregated_probs = (wagers.unsqueeze(2) * model_probs).sum(dim=1)
            aggregated_probs = aggregated_probs / aggregated_probs.sum(dim=1, keepdim=True)
            return aggregated_probs
        
        # Single sample mode
        elif model_logits.ndim == 2 and wagers.ndim == 1:
            model_probs = torch.softmax(model_logits, dim=1)
            aggregated_probs = (wagers.unsqueeze(1) * model_probs).sum(dim=0)
            aggregated_probs = aggregated_probs / aggregated_probs.sum()
            return aggregated_probs
        
        else:
            raise ValueError(
                f"Invalid shapes: model_logits={model_logits.shape}, wagers={wagers.shape}"
            )
