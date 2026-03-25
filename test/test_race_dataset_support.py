import pytest
from datasets import Dataset as HFDataset

from wagering.core.dataset import Dataset as WageringDataset
from wagering.utils.dataset_utils import load_datasets_from_config
from wagering.utils.multi_llm_ensemble import (
    assign_pubmedqa_context_models,
    get_model_prompt_variant,
    get_model_specific_prompts,
)


def test_race_uses_first_question_per_article_and_builds_prompt_variants(monkeypatch):
    hf_rows = [
        {
            "article": "Article one context.",
            "problems": [
                {
                    "question": "First question article one?",
                    "options": ["A1", "B1", "C1", "D1"],
                    "answer": "B",
                },
                {
                    "question": "Second question article one?",
                    "options": ["A2", "B2", "C2", "D2"],
                    "answer": "D",
                },
            ],
        },
        {
            "article": "Article two context.",
            "problems": [
                {
                    "question": "First question article two?",
                    "options": ["A3", "B3", "C3", "D3"],
                    "answer": "C",
                },
                {
                    "question": "Second question article two?",
                    "options": ["A4", "B4", "C4", "D4"],
                    "answer": "A",
                },
            ],
        },
    ]
    hf_dataset = HFDataset.from_list(hf_rows)

    def _fake_load_dataset(path, split=None, **kwargs):
        _ = (path, split, kwargs)
        return hf_dataset

    monkeypatch.setattr("wagering.core.dataset.load_dataset", _fake_load_dataset)

    ds = WageringDataset.from_datasets(
        "EleutherAI/race",
        x_column="article",
        y_column="answer",
        batch_size=2,
        prompt=(
            "Article:\n{article}\n\n"
            "Question: {question}\n"
            "Choices:\n{choices}\n"
            "Answer:"
        ),
        prompt_without_context=(
            "Question: {question}\n"
            "Choices:\n{choices}\n"
            "Answer:"
        ),
        split="test",
    )

    assert len(ds.x) == 2
    assert ds.y == ["B", "C"]

    # First-question-only behavior per article.
    assert "First question article one?" in ds.x[0]
    assert "Second question article one?" not in ds.x[0]
    assert "First question article two?" in ds.x[1]
    assert "Second question article two?" not in ds.x[1]

    assert getattr(ds, "race_prompt_strategy", None) == "article_context_mixed"
    assert len(ds.race_with_context_x) == 2
    assert len(ds.race_without_context_x) == 2


def test_race_model_specific_article_context_assignment(monkeypatch):
    hf_rows = [
        {
            "article": "A1",
            "problems": [{"question": "Q1", "options": ["1", "2", "3", "4"], "answer": "A"}],
        },
        {
            "article": "A2",
            "problems": [{"question": "Q2", "options": ["1", "2", "3", "4"], "answer": "B"}],
        },
        {
            "article": "A3",
            "problems": [{"question": "Q3", "options": ["1", "2", "3", "4"], "answer": "C"}],
        },
    ]
    hf_dataset = HFDataset.from_list(hf_rows)

    monkeypatch.setattr("wagering.core.dataset.load_dataset", lambda *args, **kwargs: hf_dataset)

    ds = WageringDataset.from_datasets(
        "EleutherAI/race",
        x_column="article",
        y_column="answer",
        batch_size=2,
        split="test",
    )

    assignments = assign_pubmedqa_context_models(
        [ds],
        ["model/a", "model/b", "model/c"],
        random_seed=123,
    )

    assert 0 in assignments
    assert assignments[0]["dataset_type"] == "race"

    per_example = ds.race_context_assignment_by_example
    assert len(per_example) == len(ds.x)

    for model_idx in range(3):
        prompts = get_model_specific_prompts(ds, model_idx)
        for ex_idx in range(len(ds.x)):
            if per_example[ex_idx] == model_idx:
                assert prompts[ex_idx] == ds.race_with_context_x[ex_idx]
            else:
                assert prompts[ex_idx] == ds.race_without_context_x[ex_idx]

        variant = get_model_prompt_variant(ds, model_idx)
        assert isinstance(variant, str)
        assert variant.startswith(f"article_random_context_m{model_idx}_")


def test_race_single_source_split_and_serialized_problems(monkeypatch):
    # Match EleutherAI/race grouped-by-article style where `problems` is serialized text.
    rows = []
    for idx in range(10):
        rows.append(
            {
                "article": f"Article {idx}",
                "problems": str(
                    [
                        {
                            "question": f"Q{idx}-0",
                            "options": [f"A{idx}", f"B{idx}", f"C{idx}", f"D{idx}"],
                            "answer": "A" if idx % 2 == 0 else "B",
                        },
                        {
                            "question": f"Q{idx}-1",
                            "options": ["x", "y", "z", "w"],
                            "answer": "C",
                        },
                    ]
                ),
            }
        )

    hf_dataset = HFDataset.from_list(rows)
    monkeypatch.setattr("wagering.core.dataset.load_dataset", lambda *args, **kwargs: hf_dataset)

    cfg = {
        "name": "EleutherAI/race",
        "display_name": "race",
        "text_column": "article",
        "label_column": "answer",
        "batch_size": 2,
        "prompt": "Article:\n{article}\n\nQuestion: {question}\nChoices:\n{choices}\nAnswer:",
        "prompt_without_context": "Question: {question}\nChoices:\n{choices}\nAnswer:",
        "race_source_split": "test",
        "race_train_target_split": "train",
        "race_eval_target_split": "test",
        "race_split_ratios": [0.6, 0.2, 0.2],
        "split_seed": 123,
    }

    train_datasets, _ = load_datasets_from_config([cfg], split="train", random_seed=42)
    test_datasets, _ = load_datasets_from_config([cfg], split="test", random_seed=42)

    train_ds = train_datasets[0]
    test_ds = test_datasets[0]

    # For 10 examples with 0.6/0.2/0.2 split we expect 6 train and 2 test examples.
    assert len(train_ds.x) == 6
    assert len(test_ds.x) == 2

    # Ensure serialized `problems` were parsed and only first question is used.
    assert all("Q" in prompt for prompt in train_ds.x)
    assert all("-1" not in prompt for prompt in train_ds.x)
