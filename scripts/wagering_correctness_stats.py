#!/usr/bin/env python3
"""
Compute per-question correctness patterns across models using cached logits.

This script is intentionally cache-first: it does not run model inference.
It loads datasets using the same config resolution as training/evaluation,
then reads per-model cached logits/labels from the shared disk cache used by
`scripts/wagering_train.py` and `scripts/wagering_eval.py`.

Outputs (by default under workdir/correctness_stats/<config_stem>/):
  - train_summary.json / test_summary.json
  - train_<dataset>.per_question.csv / test_<dataset>.per_question.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

import sys

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wagering.utils import load_and_merge_configs, load_datasets_from_config
from wagering.utils.multi_llm_ensemble import (
    assign_pubmedqa_context_models,
    configure_wagering_cache_dir,
    get_cached_logits_and_hidden_states_for_model,
    get_model_prompt_variant,
)

log = logging.getLogger("wagering")


def _slugify(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_").lower()


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _mask_to_pattern(mask: int, model_names: List[str]) -> str:
    if mask == 0:
        return "none"
    chosen = [model_names[i] for i in range(len(model_names)) if (mask >> i) & 1]
    if len(chosen) == 1:
        return f"only:{chosen[0]}"
    return "and:(" + ",".join(chosen) + ")"


def _mask_to_pattern_names(mask: int, names: List[str]) -> str:
    """Like _mask_to_pattern, but uses an explicit names list (already role-qualified)."""
    if mask == 0:
        return "none"
    chosen = [names[i] for i in range(len(names)) if (mask >> i) & 1]
    if len(chosen) == 1:
        return f"only:{chosen[0]}"
    return "and:(" + ",".join(chosen) + ")"


def _compute_correctness_for_dataset(
    dataset,
    dataset_name: str,
    model_paths: List[str],
    option_tokens: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    num_models = len(model_paths)
    if num_models <= 0:
        raise ValueError("Config has no models")

    labels_ref: Optional[np.ndarray] = None
    per_model_pred: List[np.ndarray] = []
    per_model_correct: List[np.ndarray] = []

    for model_idx, model_path in enumerate(model_paths):
        prompt_variant = get_model_prompt_variant(dataset, model_index=model_idx)
        cached_logits, _, cached_labels = get_cached_logits_and_hidden_states_for_model(
            model_path,
            dataset,
            option_tokens,
            prompt_variant=prompt_variant,
            model_index=model_idx,
            hidden_state_layers=None,
        )
        if cached_logits is None or cached_labels is None:
            raise RuntimeError(
                "Cache miss for correctness stats. "
                f"dataset={dataset_name}, model_index={model_idx}, model={model_path}, prompt_variant={prompt_variant}. "
                "Run wagering_train/eval first (or any pipeline that collects logits) to populate cache."
            )

        logits = np.asarray(cached_logits, dtype=np.float32)
        labels = np.asarray(cached_labels, dtype=np.int32)
        if logits.ndim != 2 or logits.shape[0] != labels.shape[0]:
            raise RuntimeError(
                f"Cached logits/labels shape mismatch for dataset={dataset_name}, model={model_path}: "
                f"logits={logits.shape}, labels={labels.shape}"
            )

        if labels_ref is None:
            labels_ref = labels
        else:
            if labels_ref.shape != labels.shape or not np.array_equal(labels_ref, labels):
                raise RuntimeError(
                    f"Cached labels mismatch across models for dataset={dataset_name}. "
                    f"First_model={model_paths[0]} vs model={model_path}. "
                    "This should not happen; caches may have been produced with inconsistent dataset configs."
                )

        pred = np.argmax(logits, axis=1).astype(np.int32)
        correct = (pred == labels).astype(bool)
        per_model_pred.append(pred)
        per_model_correct.append(correct)

    assert labels_ref is not None
    correct_matrix = np.stack(per_model_correct, axis=1)  # [N, M]
    preds_matrix = np.stack(per_model_pred, axis=1)  # [N, M]

    # For mixed-context datasets (PubMedQA/RACE), each example chooses one "context model index".
    # We'll emit additional stats that treat the chosen model as "with_context" and all others as
    # "without_context". This matches the dataset's prompt routing behavior and avoids any
    # counterfactual "two+ models with context" cases.
    mixed_context_assignment: Optional[np.ndarray] = None
    for attr in ("pubmedqa_context_assignment_by_example", "race_context_assignment_by_example"):
        raw = getattr(dataset, attr, None)
        if isinstance(raw, list) and len(raw) == correct_matrix.shape[0]:
            try:
                mixed_context_assignment = np.asarray(raw, dtype=np.int32)
            except Exception:
                mixed_context_assignment = None
            break

    # Build bitmask per example: bit i set iff model i correct.
    bit_weights = (1 << np.arange(num_models, dtype=np.int64)).astype(np.int64)
    masks = (correct_matrix.astype(np.int64) * bit_weights[None, :]).sum(axis=1).astype(np.int64)

    per_question = pd.DataFrame(
        {
            "example_index": np.arange(len(masks), dtype=np.int64),
            "label": labels_ref.astype(np.int32),
            "correct_mask": masks,
            "correct_pattern": [_mask_to_pattern(int(m), model_paths) for m in masks.tolist()],
        }
    )
    if mixed_context_assignment is not None:
        per_question["context_model_index"] = mixed_context_assignment.astype(np.int32)
        per_question["context_model_path"] = [
            model_paths[int(i)] if 0 <= int(i) < num_models else ""
            for i in mixed_context_assignment.tolist()
        ]

    for model_idx, model_path in enumerate(model_paths):
        safe = _slugify(model_path)
        per_question[f"pred__{safe}"] = preds_matrix[:, model_idx].astype(np.int32)
        per_question[f"correct__{safe}"] = correct_matrix[:, model_idx].astype(bool)

    # Summary counts by mask/pattern.
    unique_masks, counts = np.unique(masks, return_counts=True)
    mask_counts: Dict[str, int] = {str(int(m)): int(c) for m, c in zip(unique_masks.tolist(), counts.tolist())}
    pattern_counts: Dict[str, int] = {}
    for m, c in zip(unique_masks.tolist(), counts.tolist()):
        pattern_counts[_mask_to_pattern(int(m), model_paths)] = int(c)

    summary: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "num_examples": int(len(masks)),
        "num_models": int(num_models),
        "models": list(model_paths),
        "mask_counts": mask_counts,
        "pattern_counts": dict(sorted(pattern_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "none_correct": int(mask_counts.get("0", 0)),
    }
    # Convenience: only-one-model counts
    for i, model_path in enumerate(model_paths):
        only_mask = str(1 << i)
        summary[f"only_correct__{_slugify(model_path)}"] = int(mask_counts.get(only_mask, 0))

    # Convenience: only-one-model incorrect counts (all other models correct)
    all_correct_mask = (1 << num_models) - 1
    for i, model_path in enumerate(model_paths):
        only_incorrect_mask = str(all_correct_mask ^ (1 << i))
        summary[f"only_incorrect__{_slugify(model_path)}"] = int(mask_counts.get(only_incorrect_mask, 0))

    if mixed_context_assignment is not None:
        # Mixed-context breakdown:
        # - exactly one model is "with_context" per example (the assigned context model)
        # - all other models are "without_context"
        # We report:
        # - per-model correctness rates split by role (with_context vs without_context)
        # - counts for role-aware correctness patterns (none, only_context, only_wo_model_i, etc.)
        assignment = mixed_context_assignment
        if assignment.shape[0] != correct_matrix.shape[0]:
            raise RuntimeError(
                f"Context assignment length mismatch for dataset={dataset_name}: "
                f"assignments={assignment.shape[0]}, examples={correct_matrix.shape[0]}"
            )
        if np.any((assignment < 0) | (assignment >= num_models)):
            raise RuntimeError(
                f"Invalid context assignment indices for dataset={dataset_name}: "
                f"expected in [0, {num_models - 1}]"
            )

        role_counts: Dict[str, int] = {
            "none_correct": 0,
            "all_correct": 0,
            "only_context_correct": 0,
            "only_without_context_correct": 0,
            "all_without_context_correct": 0,  # all 3 wo correct and context wrong
            "two_without_context_only": 0,
            "one_without_context_only": 0,
            "one_without_context_plus_context": 0,
            "two_without_context_plus_context": 0,
        }
        only_wo_by_model = {str(i): 0 for i in range(num_models)}
        only_wc_by_model = {str(i): 0 for i in range(num_models)}

        with_ctx_total = np.zeros((num_models,), dtype=np.int64)
        with_ctx_correct = np.zeros((num_models,), dtype=np.int64)
        without_ctx_total = np.zeros((num_models,), dtype=np.int64)
        without_ctx_correct = np.zeros((num_models,), dtype=np.int64)

        # Iterate once; dataset sizes here are small/moderate (1k-ish), so Python loop is OK.
        for ex_idx in range(correct_matrix.shape[0]):
            c = int(assignment[ex_idx])
            row = correct_matrix[ex_idx, :].astype(bool)
            c_ok = bool(row[c])
            wo_idx = [i for i in range(num_models) if i != c]
            wo_ok = row[wo_idx]
            n_ok = int(row.sum())
            n_wo_ok = int(np.sum(wo_ok))

            with_ctx_total[c] += 1
            with_ctx_correct[c] += int(c_ok)
            for i in wo_idx:
                without_ctx_total[i] += 1
                without_ctx_correct[i] += int(row[i])

            if n_ok == 0:
                role_counts["none_correct"] += 1
                continue
            if n_ok == num_models:
                role_counts["all_correct"] += 1
                continue

            if c_ok and n_wo_ok == 0:
                role_counts["only_context_correct"] += 1
                only_wc_by_model[str(c)] += 1
                continue
            if (not c_ok) and n_wo_ok == 1:
                role_counts["only_without_context_correct"] += 1
                # Find which model (must be one of wo_idx).
                for i in wo_idx:
                    if row[i]:
                        only_wo_by_model[str(i)] += 1
                        break
                continue
            if (not c_ok) and n_wo_ok == 3:
                role_counts["all_without_context_correct"] += 1
                continue

            if (not c_ok) and n_wo_ok == 2:
                role_counts["two_without_context_only"] += 1
                continue
            if (not c_ok) and n_wo_ok == 1:
                role_counts["one_without_context_only"] += 1
                continue
            if c_ok and n_wo_ok == 1:
                role_counts["one_without_context_plus_context"] += 1
                continue
            if c_ok and n_wo_ok == 2:
                role_counts["two_without_context_plus_context"] += 1
                continue

        per_model_role = []
        for i, path in enumerate(model_paths):
            per_model_role.append(
                {
                    "model_index": int(i),
                    "model_path": str(path),
                    "with_context": {
                        "num_examples": int(with_ctx_total[i]),
                        "num_correct": int(with_ctx_correct[i]),
                        "accuracy": float(with_ctx_correct[i] / max(1, with_ctx_total[i])),
                    },
                    "without_context": {
                        "num_examples": int(without_ctx_total[i]),
                        "num_correct": int(without_ctx_correct[i]),
                        "accuracy": float(without_ctx_correct[i] / max(1, without_ctx_total[i])),
                    },
                    "only_correct_counts": {
                        "only_correct_when_with_context": int(only_wc_by_model[str(i)]),
                        "only_correct_when_without_context": int(only_wo_by_model[str(i)]),
                    },
                }
            )

        summary["mixed_context_breakdown"] = {
            "context_assignment_attr": (
                "pubmedqa_context_assignment_by_example"
                if hasattr(dataset, "pubmedqa_context_assignment_by_example")
                else "race_context_assignment_by_example"
            ),
            "role_pattern_counts": role_counts,
            "only_without_context_correct_by_model_index": only_wo_by_model,
            "only_with_context_correct_by_model_index": only_wc_by_model,
            "per_model_role_accuracy": per_model_role,
        }

        # Treat each model-with-context / model-without-context as distinct "variants".
        # There are 2*M variant names:
        #   - "<model_path>|with_context"
        #   - "<model_path>|without_context"
        # For each example, exactly one model contributes to the with_context group and
        # the remaining M-1 contribute to without_context.
        variant_names: List[str] = []
        for p in model_paths:
            variant_names.append(f"{p}|with_context")
        for p in model_paths:
            variant_names.append(f"{p}|without_context")

        variant_masks = np.zeros((correct_matrix.shape[0],), dtype=np.int64)
        for ex_idx in range(correct_matrix.shape[0]):
            c = int(assignment[ex_idx])
            row = correct_matrix[ex_idx, :].astype(bool)
            m = 0
            # with_context bit for context model only
            if row[c]:
                m |= 1 << c
            # without_context bits for all other models
            for i in range(num_models):
                if i == c:
                    continue
                if row[i]:
                    m |= 1 << (num_models + i)
            variant_masks[ex_idx] = int(m)

        uniq_v, cnt_v = np.unique(variant_masks, return_counts=True)
        variant_pattern_counts: Dict[str, int] = {}
        for m, c in zip(uniq_v.tolist(), cnt_v.tolist()):
            variant_pattern_counts[_mask_to_pattern_names(int(m), variant_names)] = int(c)
        summary["mixed_context_breakdown"]["variant_pattern_counts"] = dict(
            sorted(variant_pattern_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        )

        # Also provide a higher-level aggregation that matches your requested buckets.
        bucket_counts: Dict[str, int] = {"none": 0, "all_correct": 0}
        for i, p in enumerate(model_paths):
            bucket_counts[f"only:{p}|with_context"] = 0
            bucket_counts[f"only:{p}|without_context"] = 0
            bucket_counts[f"all_without_context_correct__context_is:{p}"] = 0  # 3 wo correct, wc wrong

        for ex_idx in range(correct_matrix.shape[0]):
            c = int(assignment[ex_idx])
            row = correct_matrix[ex_idx, :].astype(bool)
            c_ok = bool(row[c])
            wo_idx = [i for i in range(num_models) if i != c]
            n_wo_ok = int(np.sum(row[wo_idx]))
            n_ok = int(np.sum(row))

            if n_ok == 0:
                bucket_counts["none"] += 1
            elif n_ok == num_models:
                bucket_counts["all_correct"] += 1
            elif c_ok and n_wo_ok == 0:
                bucket_counts[f"only:{model_paths[c]}|with_context"] += 1
            elif (not c_ok) and n_wo_ok == 1:
                for i in wo_idx:
                    if row[i]:
                        bucket_counts[f"only:{model_paths[i]}|without_context"] += 1
                        break
            elif (not c_ok) and n_wo_ok == (num_models - 1):
                bucket_counts[f"all_without_context_correct__context_is:{model_paths[c]}"] += 1

        summary["mixed_context_breakdown"]["role_aware_buckets"] = bucket_counts

        # Additionally, break everything down by which model was assigned context.
        # This yields M sub-summaries, one per context model index.
        by_context_model: Dict[str, Any] = {}
        for ctx_i in range(num_models):
            idxs = np.where(assignment == ctx_i)[0]
            ctx_n = int(idxs.shape[0])
            ctx_payload: Dict[str, Any] = {
                "context_model_index": int(ctx_i),
                "context_model_path": str(model_paths[ctx_i]),
                "num_examples": ctx_n,
            }
            if ctx_n == 0:
                by_context_model[str(ctx_i)] = ctx_payload
                continue

            # Recompute role-aware buckets restricted to this context model.
            # Keep only keys that can be non-zero for this slice:
            # - exactly one with_context model (ctx_i)
            # - the remaining models are without_context
            # - only one all_without_context_correct key (context_is == ctx_i)
            ctx_bucket_counts: Dict[str, int] = {
                "none": 0,
                "all_correct": 0,
                f"only:{model_paths[ctx_i]}|with_context": 0,
                f"all_without_context_correct__context_is:{model_paths[ctx_i]}": 0,
            }
            for i in range(num_models):
                if i == ctx_i:
                    continue
                ctx_bucket_counts[f"only:{model_paths[i]}|without_context"] = 0

            ctx_variant_masks = np.zeros((ctx_n,), dtype=np.int64)
            for j, ex_idx in enumerate(idxs.tolist()):
                row = correct_matrix[ex_idx, :].astype(bool)
                c_ok = bool(row[ctx_i])
                wo_idx = [i for i in range(num_models) if i != ctx_i]
                n_wo_ok = int(np.sum(row[wo_idx]))
                n_ok = int(np.sum(row))

                if n_ok == 0:
                    ctx_bucket_counts["none"] += 1
                elif n_ok == num_models:
                    ctx_bucket_counts["all_correct"] += 1
                elif c_ok and n_wo_ok == 0:
                    ctx_bucket_counts[f"only:{model_paths[ctx_i]}|with_context"] += 1
                elif (not c_ok) and n_wo_ok == 1:
                    for i in wo_idx:
                        if row[i]:
                            ctx_bucket_counts[f"only:{model_paths[i]}|without_context"] += 1
                            break
                elif (not c_ok) and n_wo_ok == (num_models - 1):
                    ctx_bucket_counts[f"all_without_context_correct__context_is:{model_paths[ctx_i]}"] += 1

                # Variant mask restricted to this context model (same naming/bit scheme).
                m = 0
                if row[ctx_i]:
                    m |= 1 << ctx_i
                for i in range(num_models):
                    if i == ctx_i:
                        continue
                    if row[i]:
                        m |= 1 << (num_models + i)
                ctx_variant_masks[j] = int(m)

            uniq_ctx_v, cnt_ctx_v = np.unique(ctx_variant_masks, return_counts=True)
            ctx_variant_pattern_counts: Dict[str, int] = {}
            for m, c in zip(uniq_ctx_v.tolist(), cnt_ctx_v.tolist()):
                ctx_variant_pattern_counts[_mask_to_pattern_names(int(m), variant_names)] = int(c)

            ctx_payload["role_aware_buckets"] = ctx_bucket_counts
            ctx_payload["variant_pattern_counts"] = dict(
                sorted(ctx_variant_pattern_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            )
            by_context_model[str(ctx_i)] = ctx_payload

        summary["mixed_context_breakdown"]["by_context_model_index"] = by_context_model

    return per_question, summary


def _load_split_datasets(
    cfg: Dict[str, Any],
    split: str,
    *,
    random_seed: int,
) -> List[Tuple[Any, str]]:
    if split == "train":
        configs = cfg.get("datasets") or []
    elif split == "test":
        configs = cfg.get("test_datasets") or []
    else:
        raise ValueError(f"Unsupported split={split}")

    if not configs:
        return []

    sst = bool(cfg.get("shared_source_tripartition", False))
    peer = cfg.get("test_datasets") if split == "train" else cfg.get("datasets")
    datasets, names = load_datasets_from_config(
        configs,
        split=split,
        random_seed=random_seed,
        shared_source_tripartition=sst,
        tripartition_peer_dataset_configs=peer,
        infer_eval_split_train_without_peer=False,
        force_shared_source_tripartition=bool(sst and split == "test"),
    )
    return list(zip(datasets, names))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)


def run_correctness_stats(
    config_path: Path,
    output_dir: Path,
    *,
    splits: List[str],
) -> Dict[str, Any]:
    cfg = load_and_merge_configs(config_path)

    # Ensure we read caches from the same directory as training/eval/pipeline.
    # `cache_path` is a user-provided root; the ensemble cache lives in a stable subdir.
    cache_root = cfg.get("cache_path")
    cache_dir = configure_wagering_cache_dir(str(cache_root) if cache_root else None)
    log.info("Using wagering logits/hidden-states cache dir: %s", cache_dir)

    model_cfgs = cfg.get("models") or []
    model_paths = [str(m["path"]) for m in model_cfgs if isinstance(m, dict) and m.get("path")]
    if not model_paths:
        raise ValueError("No models found in config (expected config['models'][i]['path']).")

    option_tokens = cfg.get("option_tokens", ["A", "B", "C", "D"])
    option_tokens = [str(t) for t in option_tokens]

    dataset_split_seed = int(cfg.get("dataset_split_seed", 42))

    out: Dict[str, Any] = {
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "models": model_paths,
        "option_tokens": option_tokens,
        "dataset_split_seed": dataset_split_seed,
        "splits": {},
    }

    for split in splits:
        split_jobs = _load_split_datasets(cfg, split, random_seed=dataset_split_seed)
        if not split_jobs:
            log.info("No datasets configured for split=%s. Skipping.", split)
            continue

        # Ensure mixed-context routing is configured before cache lookups.
        assign_pubmedqa_context_models(
            [ds for ds, _ in split_jobs],
            model_paths,
            random_seed=dataset_split_seed,
        )

        split_dir = _ensure_dir(output_dir / split)
        split_summaries: List[Dict[str, Any]] = []

        for dataset, dataset_name in split_jobs:
            per_question, summary = _compute_correctness_for_dataset(
                dataset=dataset,
                dataset_name=dataset_name,
                model_paths=model_paths,
                option_tokens=option_tokens,
            )
            split_summaries.append(summary)

            csv_name = f"{split}_{_slugify(dataset_name)}.per_question.csv"
            per_question.to_csv(split_dir / csv_name, index=False)

            _write_json(split_dir / f"{split}_{_slugify(dataset_name)}.summary.json", summary)

        split_summary_payload = {
            "split": split,
            "num_datasets": int(len(split_summaries)),
            "datasets": split_summaries,
        }
        _write_json(split_dir / f"{split}_summary.json", split_summary_payload)
        out["splits"][split] = split_summary_payload

    _write_json(output_dir / "summary.json", out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-question correctness stats from cached logits.")
    parser.add_argument("config", type=str, help="Path to config YAML (same as wagering_train/eval).")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory (default: workdir/correctness_stats/<config_stem>).",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,test",
        help="Comma-separated list of splits to compute: train,test (default: train,test).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    allowed = {"train", "test"}
    bad = [s for s in splits if s not in allowed]
    if bad:
        raise ValueError(f"Unsupported splits {bad}. Allowed: {sorted(allowed)}")

    if args.out:
        output_dir = Path(args.out)
    else:
        output_dir = PROJECT_ROOT / "workdir" / "correctness_stats" / config_path.stem

    run_correctness_stats(
        config_path=config_path,
        output_dir=_ensure_dir(output_dir),
        splits=splits,
    )

    log.info("Done. Outputs written to %s", output_dir)


if __name__ == "__main__":
    main()

