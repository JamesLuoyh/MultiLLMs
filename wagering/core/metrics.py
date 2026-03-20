from typing import List

import numpy as np
from sklearn.preprocessing import MinMaxScaler


class ECE:
    """Expected Calibration Error for confidence-style estimators."""

    def __init__(self, normalize: bool = False, n_bins: int = 20):
        self.normalize = normalize
        self.n_bins = n_bins

    def __str__(self) -> str:
        return "ece"

    @staticmethod
    def normalize_scores(scores: List[float]) -> List[float]:
        scores_array = np.asarray(scores).reshape(-1, 1)
        return MinMaxScaler().fit_transform(scores_array).flatten()

    def __call__(self, estimator: List[float], target: List[float]) -> float:
        if len(estimator) != len(target):
            raise ValueError("Estimator and target must have the same length.")

        estimator_array = np.asarray(estimator)
        target_array = np.asarray(target)

        # ECE expects confidence, not uncertainty.
        confidences = -estimator_array

        if self.normalize:
            confidences = self.normalize_scores(confidences)
            target_array = self.normalize_scores(target_array)

        bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        ece, n_total = 0.0, len(confidences)

        for i in range(self.n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            in_bin = (
                (confidences > lo) & (confidences <= hi)
                if i > 0
                else (confidences >= lo) & (confidences <= hi)
            )
            if not np.any(in_bin):
                continue

            acc_bin = np.mean(target_array[in_bin])
            conf_bin = np.mean(confidences[in_bin])
            ece += (np.sum(in_bin) / n_total) * abs(acc_bin - conf_bin)

        return float(ece)
