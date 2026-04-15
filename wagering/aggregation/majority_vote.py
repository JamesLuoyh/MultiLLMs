"""
Majority vote aggregation: convert each model prediction to an argmax vote,
then use vote fractions as option probabilities.
"""

from typing import Tuple

import numpy as np

from .base import AggregationFunction


class MajorityVote(AggregationFunction):
    """
    Majority vote aggregation over per-model argmax predictions.

    For each example, each model contributes one vote to its argmax option.
    The aggregated probability of an option is the fraction of votes it receives.
    """

    def aggregate(
        self,
        model_logits: np.ndarray,
        wagers: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aggregate using majority vote over argmax predictions.

        Args:
            model_logits: Shape [batch_size, num_models, num_options] or [num_models, num_options]
            wagers: Shape [batch_size, num_models] or [num_models] (unused for voting,
                but validated for shape compatibility with caller interfaces)

        Returns:
            aggregated_log_probs: Log-probabilities after aggregation
            aggregated_probs: Vote-fraction probabilities after aggregation
        """
        model_logits = np.asarray(model_logits, dtype=np.float32)
        wagers = np.asarray(wagers, dtype=np.float32)

        # Batch mode
        if model_logits.ndim == 3 and wagers.ndim == 2:
            batch_size, num_models, num_options = model_logits.shape

            if wagers.shape != (batch_size, num_models):
                raise ValueError(
                    f"Wagers shape mismatch: expected [{batch_size}, {num_models}], got {wagers.shape}"
                )

            model_preds = np.argmax(model_logits, axis=2)
            vote_one_hot = np.eye(num_options, dtype=np.float32)[model_preds]
            aggregated_probs = vote_one_hot.mean(axis=1)

            epsilon = 1e-10
            aggregated_log_probs = np.log(np.clip(aggregated_probs, epsilon, 1.0))
            return aggregated_log_probs, aggregated_probs

        # Single sample mode
        if model_logits.ndim == 2 and wagers.ndim == 1:
            num_models, num_options = model_logits.shape

            if wagers.shape != (num_models,):
                raise ValueError(
                    f"Wagers shape mismatch: expected [{num_models}], got {wagers.shape}"
                )

            model_preds = np.argmax(model_logits, axis=1)
            counts = np.bincount(model_preds, minlength=num_options).astype(np.float32)
            aggregated_probs = counts / float(num_models)

            epsilon = 1e-10
            aggregated_log_probs = np.log(np.clip(aggregated_probs, epsilon, 1.0))
            return aggregated_log_probs, aggregated_probs

        raise ValueError(
            f"Invalid shapes: model_logits={model_logits.shape}, wagers={wagers.shape}"
        )
