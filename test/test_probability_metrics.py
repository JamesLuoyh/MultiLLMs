"""Tests for probabilistic Bernoulli metrics."""

import math

import pytest

from wagering.core.metrics import bernoulli_kl_divergence, bernoulli_tv_distance
from wagering.core.dataset import Dataset


def test_bernoulli_tv_distance_basic() -> None:
    pred = [0.2, 0.7, 0.9]
    target = [0.1, 0.4, 0.8]
    expected = (0.1 + 0.3 + 0.1) / 3.0
    assert bernoulli_tv_distance(pred, target) == pytest.approx(expected)


def test_bernoulli_tv_distance_zero_when_equal() -> None:
    pred = [0.1, 0.5, 0.9]
    assert bernoulli_tv_distance(pred, pred) == pytest.approx(0.0)


def test_bernoulli_kl_divergence_zero_when_equal() -> None:
    probs = [0.2, 0.5, 0.8]
    assert bernoulli_kl_divergence(pred_probs=probs, target_probs=probs) == pytest.approx(0.0)


def test_bernoulli_kl_divergence_manual_case() -> None:
    # Single-example sanity check against closed-form Bernoulli KL.
    pred = [0.8]
    target = [0.6]
    expected = 0.6 * math.log(0.6 / 0.8) + 0.4 * math.log(0.4 / 0.2)
    assert bernoulli_kl_divergence(pred_probs=pred, target_probs=target) == pytest.approx(expected)


def test_bernoulli_metrics_validate_ranges() -> None:
    with pytest.raises(ValueError):
        bernoulli_tv_distance([1.2], [0.5])
    with pytest.raises(ValueError):
        bernoulli_kl_divergence([0.5], [-0.1])


def test_dataset_select_keeps_probabilistic_labels_aligned() -> None:
    dataset = Dataset(
        x=["q0", "q1", "q2"],
        y=["1", "0", "1"],
        batch_size=2,
    )
    dataset.probabilistic_labels = [0.9, 0.2, 0.8]

    dataset.select([2, 0])

    assert dataset.x == ["q2", "q0"]
    assert dataset.y == ["1", "1"]
    assert dataset.probabilistic_labels == [0.8, 0.9]
