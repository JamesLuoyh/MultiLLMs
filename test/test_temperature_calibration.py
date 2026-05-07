import numpy as np
import pytest

from wagering.core.dataset import Dataset
from wagering.aggregation import LinearPooling
from wagering.calibration import AdaptiveTemperatureCalibrator
from wagering.inference import WageringEvaluator
from wagering.methods import EqualWagers
from wagering.utils.config_utils import load_and_merge_configs


def test_load_and_merge_configs_supports_calibration_include(tmp_path):
    models_dir = tmp_path / "models"
    datasets_dir = tmp_path / "datasets"
    calibration_dir = tmp_path / "calibration"
    models_dir.mkdir()
    datasets_dir.mkdir()
    calibration_dir.mkdir()

    (models_dir / "model_a.yaml").write_text("path: org/model-a\n", encoding="utf-8")
    (datasets_dir / "dataset_a.yaml").write_text("name: org/dataset-a\n", encoding="utf-8")
    (calibration_dir / "ats.yaml").write_text(
        "\n".join(
            [
                "name: adaptive_temperature_scaling",
                "_include_datasets:",
                "  - ../datasets/dataset_a.yaml",
                "learning_rate: 1e-4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    main_config = tmp_path / "main.yaml"
    main_config.write_text(
        "\n".join(
            [
                "_include_models:",
                "  - models/model_a.yaml",
                "calibrated: false",
                "_include_calibration: calibration/ats.yaml",
                "wagering_method:",
                "  name: equal_wagers",
                "  config: {}",
                "aggregation:",
                "  name: weighted_linear_pooling",
                "  config: {}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    merged = load_and_merge_configs(main_config)

    assert merged["calibrated"] is True
    assert merged["calibration"]["name"] == "adaptive_temperature_scaling"
    assert merged["calibration"]["datasets"][0]["name"] == "org/dataset-a"


def test_adaptive_temperature_calibrator_round_trip(tmp_path):
    rng = np.random.RandomState(0)
    logits = np.stack(
        [
            rng.randn(32, 4).astype(np.float32),
            rng.randn(32, 4).astype(np.float32),
        ],
        axis=0,
    )
    hidden_states = [
        rng.randn(32, 3).astype(np.float32),
        rng.randn(32, 5).astype(np.float32),
    ]
    labels = np.argmax(logits[0] + logits[1], axis=1).astype(np.int64)

    calibrator = AdaptiveTemperatureCalibrator(
        model_paths=["model_a", "model_b"],
        input_dims=[3, 5],
        config={
            "device": "cpu",
            "head_hidden_layers": [8],
            "num_epochs": 3,
            "batch_size": 8,
            "validation_split_ratio": 0.2,
            "early_stopping_patience": 2,
            "min_temperature": 0.1,
            "max_temperature": 5.0,
            "shuffle_seed": 7,
        },
    )

    metrics = calibrator.fit(logits, hidden_states, labels)
    scaled_logits = calibrator.apply_to_stacked_logits(logits, hidden_states)
    temperatures = calibrator.predict_temperatures(hidden_states[0], model_index=0)

    assert len(metrics["model_metrics"]) == 2
    assert scaled_logits.shape == logits.shape
    assert np.all(temperatures >= 0.1)
    assert np.all(temperatures <= 5.0)

    artifact_dir = tmp_path / "calibrator"
    calibrator.save_pretrained(artifact_dir)
    loaded = AdaptiveTemperatureCalibrator.load_pretrained(artifact_dir, device="cpu")
    reloaded_logits = loaded.apply_to_stacked_logits(logits, hidden_states)

    assert np.allclose(scaled_logits, reloaded_logits, atol=1e-5)


def test_equal_wagers_evaluator_uses_calibrator_hidden_states(monkeypatch):
    dataset = Dataset(["question-1", "question-2"], ["A", "B"], batch_size=2)
    cached_logits = np.array(
        [
            [4.0, 1.0, 0.5, 0.0],
            [0.2, 0.3, 0.4, 5.0],
        ],
        dtype=np.float32,
    )
    cached_hidden_states = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
        dtype=np.float32,
    )
    cached_labels = np.array([0, 3], dtype=np.int32)

    class MockCalibrator:
        def __init__(self):
            self.called = False

        def apply_to_stacked_logits(self, all_model_logits, all_hidden_states):
            self.called = True
            assert all_hidden_states is not None
            return np.asarray(all_model_logits, dtype=np.float32) / 2.0

    calibrator = MockCalibrator()

    monkeypatch.setattr(
        "wagering.inference.evaluator.get_cached_logits_and_hidden_states_for_model",
        lambda model_path, ds, option_tokens, prompt_variant=None, model_index=None: (
            cached_logits,
            cached_hidden_states,
            cached_labels,
        ),
    )

    evaluator = WageringEvaluator(
        models=["model_a"],
        wagering_method=EqualWagers(num_models=1),
        aggregation_function=LinearPooling(),
        logit_calibrator=calibrator,
    )

    results = evaluator.evaluate(dataset, dataset_name="cached_eval")

    assert calibrator.called is True
    assert results["num_examples"] == 2
    assert results["accuracy"] == pytest.approx(1.0)