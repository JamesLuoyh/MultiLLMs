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

