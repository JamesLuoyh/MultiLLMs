import torch
from transformers import GenerationConfig

from wagering.core.generation_parameters import GenerationParameters
from wagering.core.model import WhiteboxModel


class _FakeTokenizer:
    def decode(self, *_args, **_kwargs):
        return ""


class _FakeGenerationOutput:
    def __init__(self):
        self.scores = []


class _FakeModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.generation_config = GenerationConfig(max_length=4096)
        self.last_generate_kwargs = None

    def generate(self, **kwargs):
        self.last_generate_kwargs = kwargs
        return _FakeGenerationOutput()


def _base_batch():
    return {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        "return_dict_in_generate": True,
    }


def test_generate_forces_max_length_none_when_max_new_tokens_present():
    base_model = _FakeModel()
    wrapper = WhiteboxModel(
        model=base_model,
        tokenizer=_FakeTokenizer(),
        generation_parameters=GenerationParameters(max_new_tokens=16),
    )

    wrapper.generate(**_base_batch(), max_new_tokens=1)

    assert base_model.last_generate_kwargs is not None
    assert base_model.last_generate_kwargs["max_new_tokens"] == 1
    assert "max_length" in base_model.last_generate_kwargs
    assert base_model.last_generate_kwargs["max_length"] is None
    assert "generation_config" not in base_model.last_generate_kwargs


def test_generate_sanitizes_user_generation_config_without_mutating_input():
    base_model = _FakeModel()
    wrapper = WhiteboxModel(
        model=base_model,
        tokenizer=_FakeTokenizer(),
        generation_parameters=GenerationParameters(max_new_tokens=16),
    )

    user_config = GenerationConfig(max_length=4096)
    wrapper.generate(**_base_batch(), max_new_tokens=1, generation_config=user_config)

    assert base_model.last_generate_kwargs is not None
    sent_config = base_model.last_generate_kwargs["generation_config"]
    assert sent_config is not user_config
    assert sent_config.max_new_tokens == 1
    assert sent_config.max_length is None
    assert user_config.max_length == 4096
    assert "max_new_tokens" not in base_model.last_generate_kwargs
