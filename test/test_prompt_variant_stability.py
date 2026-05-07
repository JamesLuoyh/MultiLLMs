import copy

import numpy as np

from wagering.core.dataset import Dataset
from wagering.utils.multi_llm_ensemble import assign_pubmedqa_context_models, get_model_prompt_variant, get_model_specific_prompts


def _make_minimal_pubmedqa_like_dataset(n: int = 20) -> Dataset:
    x = [f"Question:\nq{i}\nAnswer:" for i in range(n)]
    y = ["YES" if i % 2 == 0 else "NO" for i in range(n)]
    ds = Dataset(x, y, batch_size=4)

    # Minimal fields used by wrong-context builder.
    ds.pubmedqa_questions = [f"q{i}" for i in range(n)]
    ds.pubmedqa_long_answers = [f"a{i}" for i in range(n)]
    ds.pubmedqa_context_texts = [f"ctx{i}" for i in range(n)]
    ds.pubmedqa_prompt_template_with_context = "Question:\n{question}\nContext:\n{context}\nLong Answer:\n{long_answer}\nAnswer:"

    # Mixed-context prompt variants
    ds.pubmedqa_with_context_x = [f"WITH {i}" for i in range(n)]
    ds.pubmedqa_without_context_x = [f"WITHOUT {i}" for i in range(n)]

    return ds


def test_prompt_variant_stable_all_others_true() -> None:
    ds = _make_minimal_pubmedqa_like_dataset()
    ds.pubmedqa_prompt_strategy = "mixed_context"
    ds.pubmedqa_wrong_context_routing = True
    ds.pubmedqa_wrong_context_all_others = True

    model_paths = ["m0", "m1", "m2", "m3"]
    assign_pubmedqa_context_models([ds], model_paths, random_seed=123)

    before = [get_model_prompt_variant(ds, i) for i in range(len(model_paths))]
    # Trigger prompt materialization (historically where lazy rebuild happened).
    _ = [get_model_specific_prompts(ds, i) for i in range(len(model_paths))]
    after = [get_model_prompt_variant(ds, i) for i in range(len(model_paths))]

    assert before == after
    assert all(isinstance(v, str) and "all_wrong" in v for v in before)


def test_prompt_variant_stable_all_others_false() -> None:
    ds = _make_minimal_pubmedqa_like_dataset()
    ds.pubmedqa_prompt_strategy = "mixed_context"
    ds.pubmedqa_wrong_context_routing = True
    ds.pubmedqa_wrong_context_all_others = False

    model_paths = ["m0", "m1", "m2", "m3"]
    assign_pubmedqa_context_models([ds], model_paths, random_seed=7)

    before = [get_model_prompt_variant(ds, i) for i in range(len(model_paths))]
    _ = [get_model_specific_prompts(ds, i) for i in range(len(model_paths))]
    after = [get_model_prompt_variant(ds, i) for i in range(len(model_paths))]

    assert before == after
    assert all(isinstance(v, str) and "balanced_random_context" in v for v in before)


def test_prompt_variant_deterministic_across_fresh_dataset_objects() -> None:
    ds1 = _make_minimal_pubmedqa_like_dataset()
    ds2 = _make_minimal_pubmedqa_like_dataset()
    for ds in (ds1, ds2):
        ds.pubmedqa_prompt_strategy = "mixed_context"
        ds.pubmedqa_wrong_context_routing = True
        ds.pubmedqa_wrong_context_all_others = True

    model_paths = ["m0", "m1", "m2", "m3"]
    assign_pubmedqa_context_models([ds1], model_paths, random_seed=123)
    assign_pubmedqa_context_models([ds2], model_paths, random_seed=123)

    pv1 = [get_model_prompt_variant(ds1, i) for i in range(len(model_paths))]
    pv2 = [get_model_prompt_variant(ds2, i) for i in range(len(model_paths))]
    assert pv1 == pv2

