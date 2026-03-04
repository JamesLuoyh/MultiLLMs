import numpy as np

from lm_polygraph.utils.multi_llm_ensemble import (
    aggregate_logits_log_pooling,
    run_online_ensemble,
)


def test_aggregate_logits_log_pooling_shapes_and_normalization():
    # Two models, three options
    llm_logits = np.array(
        [
            [1.0, 0.0, -1.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=np.float32,
    )
    wagers = np.array([0.6, 0.4], dtype=np.float32)

    log_probs, probs = aggregate_logits_log_pooling(llm_logits, wagers)

    assert log_probs.shape == (3,)
    assert probs.shape == (3,)
    # Probabilities should be non-negative and sum to 1 (up to numerical tolerance).
    assert np.all(probs >= 0.0)
    assert np.isclose(probs.sum(), 1.0, atol=1e-6)


def test_run_online_ensemble_wagers_history_shape_and_update():
    # Synthetic logits for 2 models, 5 questions, 3 options
    num_models = 2
    num_examples = 5
    num_options = 3

    rng = np.random.default_rng(0)
    all_model_logits = [
        rng.normal(size=(num_examples, num_options)).astype(np.float32)
        for _ in range(num_models)
    ]
    # Random labels in [0, num_options)
    labels = rng.integers(low=0, high=num_options, size=(num_examples,), dtype=np.int32)

    result = run_online_ensemble(
        all_model_logits=all_model_logits,
        labels=labels,
        initial_wagers=None,
    )

    pooled_probs = result["pooled_probs"]
    pooled_pred = result["pooled_pred"]
    labels_out = result["labels"]
    wagers_history = result["wagers_history"]

    # Basic shape checks
    assert pooled_probs.shape == (num_examples, num_options)
    assert pooled_pred.shape == (num_examples,)
    assert labels_out.shape == (num_examples,)
    assert wagers_history.shape == (num_examples + 1, num_models)

    # Each row of pooled_probs should form a valid probability distribution
    assert np.all(pooled_probs >= 0.0)
    assert np.allclose(pooled_probs.sum(axis=1), 1.0, atol=1e-5)

    # Initial wagers are uniform when initial_wagers is None
    assert np.allclose(
        wagers_history[0],
        np.ones(num_models, dtype=np.float32) / float(num_models),
        atol=1e-6,
    )






