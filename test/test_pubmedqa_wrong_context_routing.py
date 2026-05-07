import pytest
from datasets import Dataset as HFDataset

from wagering.core.dataset import Dataset as WageringDataset
from wagering.utils.multi_llm_ensemble import (
    assign_pubmedqa_context_models,
    get_model_specific_prompts,
)


def test_pubmedqa_wrong_context_routing_assigns_right_and_wrong_and_rest_without_context(monkeypatch):
    # Two examples with clearly distinguishable context strings.
    hf_rows = [
        {
            "question": "Q0?",
            "long_answer": "LA0",
            "context": {"contexts": ["CTX0"], "labels": ["METHODS"]},
            "final_decision": "yes",
        },
        {
            "question": "Q1?",
            "long_answer": "LA1",
            "context": {"contexts": ["CTX1"], "labels": ["RESULTS"]},
            "final_decision": "no",
        },
    ]
    hf_dataset = HFDataset.from_list(hf_rows)

    monkeypatch.setattr("wagering.core.dataset.load_dataset", lambda *args, **kwargs: hf_dataset)

    ds = WageringDataset.from_datasets(
        "qiaojin/PubMedQA",
        x_column="question",
        y_column="final_decision",
        batch_size=2,
        prompt="Question:\n{question}\nContext:\n{context}\nAnswer with YES or NO. Answer:",
        prompt_without_context="Question:\n{question}\nAnswer with YES or NO. Answer:",
        split="train",
    )

    # Enable the new behavior (normally set by dataset_utils.load_datasets_from_config).
    ds.pubmedqa_wrong_context_routing = True

    num_models = 4
    model_paths = [f"model/{i}" for i in range(num_models)]
    assign_pubmedqa_context_models([ds], model_paths, random_seed=123)

    right = getattr(ds, "pubmedqa_context_assignment_by_example", None)
    wrong = getattr(ds, "pubmedqa_wrong_context_assignment_by_example", None)
    assert isinstance(right, list) and len(right) == len(ds.x)
    assert isinstance(wrong, list) and len(wrong) == len(ds.x)
    assert all(int(r) != int(w) for r, w in zip(right, wrong))

    # For each model, ensure exactly one of the three prompt types is chosen per example.
    for model_idx in range(num_models):
        prompts = get_model_specific_prompts(ds, model_idx)
        assert len(prompts) == len(ds.x)
        for ex_idx in range(len(ds.x)):
            if int(right[ex_idx]) == model_idx:
                assert prompts[ex_idx] == ds.pubmedqa_with_context_x[ex_idx]
            elif int(wrong[ex_idx]) == model_idx:
                # Wrong context should differ from the original context for that example.
                assert prompts[ex_idx] != ds.pubmedqa_with_context_x[ex_idx]
                # But it should still be context-bearing (contain "Context:").
                assert "Context:" in prompts[ex_idx]
            else:
                assert prompts[ex_idx] == ds.pubmedqa_without_context_x[ex_idx]


def test_pubmedqa_wrong_context_routing_two_models_forces_other_model(monkeypatch):
    hf_rows = [
        {
            "question": "Q0?",
            "long_answer": "LA0",
            "context": {"contexts": ["CTX0"], "labels": ["METHODS"]},
            "final_decision": "yes",
        },
        {
            "question": "Q1?",
            "long_answer": "LA1",
            "context": {"contexts": ["CTX1"], "labels": ["RESULTS"]},
            "final_decision": "no",
        },
    ]
    hf_dataset = HFDataset.from_list(hf_rows)
    monkeypatch.setattr("wagering.core.dataset.load_dataset", lambda *args, **kwargs: hf_dataset)

    ds = WageringDataset.from_datasets(
        "qiaojin/PubMedQA",
        x_column="question",
        y_column="final_decision",
        batch_size=2,
        prompt="Question:\n{question}\nContext:\n{context}\nAnswer with YES or NO. Answer:",
        prompt_without_context="Question:\n{question}\nAnswer with YES or NO. Answer:",
        split="train",
    )
    ds.pubmedqa_wrong_context_routing = True

    model_paths = ["model/a", "model/b"]
    assign_pubmedqa_context_models([ds], model_paths, random_seed=7)

    right = ds.pubmedqa_context_assignment_by_example
    wrong = ds.pubmedqa_wrong_context_assignment_by_example
    assert all(int(r) in (0, 1) for r in right)
    assert all(int(w) in (0, 1) for w in wrong)
    assert all(int(r) != int(w) for r, w in zip(right, wrong))


def test_pubmedqa_wrong_context_routing_all_others(monkeypatch):
    hf_rows = [
        {
            "question": "Q0?",
            "long_answer": "LA0",
            "context": {"contexts": ["CTX0"], "labels": ["METHODS"]},
            "final_decision": "yes",
        },
        {
            "question": "Q1?",
            "long_answer": "LA1",
            "context": {"contexts": ["CTX1"], "labels": ["RESULTS"]},
            "final_decision": "no",
        },
        {
            "question": "Q2?",
            "long_answer": "LA2",
            "context": {"contexts": ["CTX2"], "labels": ["INTRO"]},
            "final_decision": "yes",
        },
        {
            "question": "Q3?",
            "long_answer": "LA3",
            "context": {"contexts": ["CTX3"], "labels": ["DISCUSSION"]},
            "final_decision": "no",
        },
    ]
    hf_dataset = HFDataset.from_list(hf_rows)
    monkeypatch.setattr("wagering.core.dataset.load_dataset", lambda *args, **kwargs: hf_dataset)

    ds = WageringDataset.from_datasets(
        "qiaojin/PubMedQA",
        x_column="question",
        y_column="final_decision",
        batch_size=2,
        prompt="Question:\n{question}\nContext:\n{context}\nAnswer with YES or NO. Answer:",
        prompt_without_context="Question:\n{question}\nAnswer with YES or NO. Answer:",
        split="train",
    )
    ds.pubmedqa_wrong_context_routing = True
    ds.pubmedqa_wrong_context_all_others = True

    num_models = 4
    model_paths = [f"model/{i}" for i in range(num_models)]
    assign_pubmedqa_context_models([ds], model_paths, random_seed=123)

    right = ds.pubmedqa_context_assignment_by_example
    assert isinstance(right, list) and len(right) == len(ds.x)

    # For each example, exactly one model gets the correct context; all other models get wrong context.
    for ex_idx in range(len(ds.x)):
        right_model = int(right[ex_idx])
        wrong_texts = []
        for model_idx in range(num_models):
            prompts = get_model_specific_prompts(ds, model_idx)
            if model_idx == right_model:
                assert prompts[ex_idx] == ds.pubmedqa_with_context_x[ex_idx]
            else:
                assert prompts[ex_idx] != ds.pubmedqa_with_context_x[ex_idx]
                assert "Context:" in prompts[ex_idx]
                assert prompts[ex_idx] != ds.pubmedqa_without_context_x[ex_idx]
                wrong_texts.append(prompts[ex_idx])

        # With 4 examples, there are 3 candidate wrong contexts per example; ensure wrong prompts
        # are not all identical across models.
        assert len(set(wrong_texts)) >= 2
