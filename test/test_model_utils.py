"""Tests for model-loading utilities."""

from unittest.mock import patch

from wagering.utils.model_utils import (
    load_models_from_config,
    should_load_prompt_perplexity_models_sequentially,
)


class _DummyLoadedModel:
    def __init__(self, model_path: str):
        self.model_path = model_path


@patch("wagering.utils.model_utils.torch.cuda.is_available", return_value=False)
@patch("wagering.utils.model_utils.load_api_keys", return_value={})
@patch("wagering.utils.model_utils.WhiteboxModel.from_pretrained")
def test_load_models_reuses_identical_configs(
    mock_from_pretrained,
    _mock_load_api_keys,
    _mock_cuda_available,
):
    """Repeated identical model configs should load once and reuse the same object."""

    def _make_dummy(model_path, **_kwargs):
        return _DummyLoadedModel(model_path)

    mock_from_pretrained.side_effect = _make_dummy

    model_cfgs = [
        {"path": "meta-llama/Llama-3-8B-Instruct", "instruct": True},
        {"path": "meta-llama/Llama-3-8B-Instruct", "instruct": True},
        {"path": "meta-llama/Llama-3-8B-Instruct", "instruct": True},
    ]

    models, model_names = load_models_from_config(model_cfgs)

    assert len(models) == 3
    assert models[0] is models[1]
    assert models[1] is models[2]
    assert mock_from_pretrained.call_count == 1
    assert model_names == [
        "meta-llama_Llama-3-8B-Instruct",
        "meta-llama_Llama-3-8B-Instruct",
        "meta-llama_Llama-3-8B-Instruct",
    ]


@patch("wagering.utils.model_utils.torch.cuda.is_available", return_value=False)
@patch("wagering.utils.model_utils.load_api_keys", return_value={})
@patch("wagering.utils.model_utils.WhiteboxModel.from_pretrained")
def test_load_models_can_disable_sharing(
    mock_from_pretrained,
    _mock_load_api_keys,
    _mock_cuda_available,
):
    """Opting out of shared loading should preserve one-load-per-slot behavior."""

    def _make_dummy(model_path, **_kwargs):
        return _DummyLoadedModel(model_path)

    mock_from_pretrained.side_effect = _make_dummy

    model_cfgs = [
        {"path": "meta-llama/Llama-3-8B-Instruct", "instruct": True},
        {"path": "meta-llama/Llama-3-8B-Instruct", "instruct": True},
    ]

    models, _ = load_models_from_config(model_cfgs, share_identical_models=False)

    assert len(models) == 2
    assert models[0] is not models[1]
    assert mock_from_pretrained.call_count == 2


@patch("wagering.utils.model_utils.torch.cuda.device_count", return_value=1)
@patch("wagering.utils.model_utils.torch.cuda.is_available", return_value=True)
def test_should_load_prompt_perplexity_sequentially_multi_on_one_gpu(
    _mock_cuda_available,
    _mock_device_count,
):
    assert should_load_prompt_perplexity_models_sequentially(4) is True


@patch("wagering.utils.model_utils.torch.cuda.device_count", return_value=4)
@patch("wagering.utils.model_utils.torch.cuda.is_available", return_value=True)
def test_should_load_prompt_perplexity_sequentially_fits_on_gpus(
    _mock_cuda_available,
    _mock_device_count,
):
    assert should_load_prompt_perplexity_models_sequentially(4) is False
