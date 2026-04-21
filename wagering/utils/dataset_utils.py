"""
Dataset loading utilities.

Simplified version with strict error handling.
"""

import logging
import sys
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Sequence

import numpy as np

# Ensure local project modules are importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from wagering.core.dataset import Dataset

log = logging.getLogger(__name__)


def _stable_cache_value(value: Any) -> Any:
    """Normalize nested values to JSON-stable structures for cache signatures."""
    if isinstance(value, dict):
        return {str(k): _stable_cache_value(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_stable_cache_value(v) for v in value]
    return value


def _local_path_fingerprint(value: Any) -> Any:
    """Return a stable fingerprint for existing local file paths in nested values.

    This helps invalidate logits/hidden-state caches when a dataset file is
    regenerated at the same path (for example, CSV content changes while the
    filename stays constant).
    """
    if isinstance(value, dict):
        return {str(k): _local_path_fingerprint(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_local_path_fingerprint(v) for v in value]
    if isinstance(value, (str, Path)):
        raw_path = str(value)
        try:
            path_obj = Path(raw_path).expanduser()
        except Exception:
            return {"path": raw_path, "exists": False}

        if path_obj.exists() and path_obj.is_file():
            stat = path_obj.stat()
            return {
                "path": str(path_obj),
                "exists": True,
                "size": int(stat.st_size),
                "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))),
            }
        return {"path": raw_path, "exists": False}
    return value


def _build_dataset_cache_config_signature(
    dataset_cfg: Dict[str, Any],
    *,
    dataset_name: str,
    load_split: str,
    resolved_path: Any,
    resolved_split: str,
    resolved_config_name: Optional[str],
    dataset_target_split: Optional[str],
    random_seed: Optional[int],
) -> Dict[str, Any]:
    """Build a deterministic dataset configuration signature for cache keys."""
    cache_cfg = dict(dataset_cfg)
    # Debug-only subsetting should not force cache misses. Keep cache identity tied
    # to source data and split config, not post-load evaluation caps.
    cache_cfg.pop("size", None)
    cache_cfg.pop("sample_size", None)

    signature_payload: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "load_split": load_split,
        "resolved_path": resolved_path,
        "resolved_path_fingerprint": _local_path_fingerprint(resolved_path),
        "resolved_split": resolved_split,
        "resolved_config_name": resolved_config_name,
        "dataset_target_split": dataset_target_split,
        # Keep random seed in signature only if explicitly pinned in dataset config.
        "split_seed": dataset_cfg.get("split_seed") if "split_seed" in dataset_cfg else None,
        "dataset_config": cache_cfg,
        "dataset_files_fingerprint": _local_path_fingerprint({
            "name": dataset_cfg.get("name"),
            "data_files": dataset_cfg.get("data_files"),
            "data_dir": dataset_cfg.get("data_dir"),
        }),
    }
    if "split_seed" not in dataset_cfg:
        # Ignore global run-time seed to maximize cache reuse across shuffle sweeps.
        signature_payload["runtime_random_seed"] = None
    else:
        signature_payload["runtime_random_seed"] = random_seed

    normalized_payload = _stable_cache_value(signature_payload)
    serialized = json.dumps(normalized_payload, sort_keys=True, separators=(",", ":"), default=str)
    return {
        "schema_version": 2,
        "payload": normalized_payload,
        "signature": hashlib.md5(serialized.encode("utf-8")).hexdigest(),
    }


def _is_pubmedqa_dataset_config(dataset_cfg: Dict[str, Any]) -> bool:
    """Return True when a dataset config targets PubMedQA."""
    fields = [
        dataset_cfg.get("name", ""),
        dataset_cfg.get("display_name", ""),
        dataset_cfg.get("config_name", ""),
        dataset_cfg.get("train_config_name", ""),
        dataset_cfg.get("eval_config_name", ""),
        dataset_cfg.get("test_config_name", ""),
        dataset_cfg.get("source_config_name", ""),
    ]
    normalized = " ".join(str(field).lower() for field in fields if field is not None)
    return "pubmedqa" in normalized or "pubmed_qa" in normalized


def _is_race_dataset_config(dataset_cfg: Dict[str, Any]) -> bool:
    """Return True when a dataset config targets RACE."""
    fields = [
        dataset_cfg.get("name", ""),
        dataset_cfg.get("display_name", ""),
        dataset_cfg.get("config_name", ""),
        dataset_cfg.get("train_config_name", ""),
        dataset_cfg.get("eval_config_name", ""),
        dataset_cfg.get("test_config_name", ""),
    ]
    normalized = " ".join(str(field).lower() for field in fields if field is not None)
    if "eleutherai/race" in normalized:
        return True
    padded = f" {normalized} "
    return " race " in padded


def calibration_dataset_configs_include_pubmedqa(dataset_configs: Sequence[Dict[str, Any]]) -> bool:
    """True if any calibration dataset uses mixed-context routing (PubMedQA or RACE)."""
    return any(
        _is_pubmedqa_dataset_config(cfg) or _is_race_dataset_config(cfg)
        for cfg in dataset_configs
    )


def _normalize_split_ratios(raw_ratios: Any) -> Tuple[float, float, float]:
    """Normalize split ratios to a valid (train, val, test) tuple."""
    default_ratios = (0.6, 0.2, 0.2)
    if raw_ratios is None:
        return default_ratios

    if not isinstance(raw_ratios, Sequence) or len(raw_ratios) != 3:
        log.warning(
            "Invalid split_ratios=%s. Falling back to default ratios %s.",
            raw_ratios,
            default_ratios,
        )
        return default_ratios

    try:
        ratio_array = np.array([float(v) for v in raw_ratios], dtype=np.float64)
    except (TypeError, ValueError):
        log.warning(
            "Could not parse split_ratios=%s. Falling back to default ratios %s.",
            raw_ratios,
            default_ratios,
        )
        return default_ratios

    if np.any(ratio_array < 0) or not np.any(ratio_array > 0):
        log.warning(
            "Non-positive split_ratios=%s. Falling back to default ratios %s.",
            raw_ratios,
            default_ratios,
        )
        return default_ratios

    ratio_array = ratio_array / ratio_array.sum()
    return tuple(float(v) for v in ratio_array.tolist())


def _normalize_race_split_ratios(raw_ratios: Any) -> Tuple[float, float, float]:
    """Normalize RACE split ratios to a valid (train, val, test) tuple."""
    default_ratios = (0.6, 0.2, 0.2)
    if raw_ratios is None:
        return default_ratios

    if not isinstance(raw_ratios, Sequence) or len(raw_ratios) != 3:
        log.warning(
            "Invalid race_split_ratios=%s. Falling back to default ratios %s.",
            raw_ratios,
            default_ratios,
        )
        return default_ratios

    try:
        ratio_array = np.array([float(v) for v in raw_ratios], dtype=np.float64)
    except (TypeError, ValueError):
        log.warning(
            "Could not parse race_split_ratios=%s. Falling back to default ratios %s.",
            raw_ratios,
            default_ratios,
        )
        return default_ratios

    if np.any(ratio_array < 0) or not np.any(ratio_array > 0):
        log.warning(
            "Non-positive race_split_ratios=%s. Falling back to default ratios %s.",
            raw_ratios,
            default_ratios,
        )
        return default_ratios

    ratio_array = ratio_array / ratio_array.sum()
    return tuple(float(v) for v in ratio_array.tolist())


def _subset_pubmedqa_dataset(dataset: Dataset, indices: np.ndarray) -> Dataset:
    """Apply an index subset while keeping PubMedQA prompt variants aligned."""
    index_list = [int(i) for i in indices.tolist()]
    with_context_prompts = getattr(dataset, "pubmedqa_with_context_x", None)
    without_context_prompts = getattr(dataset, "pubmedqa_without_context_x", None)

    dataset.select(index_list)

    if isinstance(with_context_prompts, list):
        dataset.pubmedqa_with_context_x = [with_context_prompts[i] for i in index_list]
    if isinstance(without_context_prompts, list):
        dataset.pubmedqa_without_context_x = [without_context_prompts[i] for i in index_list]

    return dataset


def _subset_race_dataset(dataset: Dataset, indices: np.ndarray) -> Dataset:
    """Apply an index subset while keeping RACE prompt variants aligned."""
    index_list = [int(i) for i in indices.tolist()]
    with_context_prompts = getattr(dataset, "race_with_context_x", None)
    without_context_prompts = getattr(dataset, "race_without_context_x", None)

    dataset.select(index_list)

    if isinstance(with_context_prompts, list):
        dataset.race_with_context_x = [with_context_prompts[i] for i in index_list]
    if isinstance(without_context_prompts, list):
        dataset.race_without_context_x = [without_context_prompts[i] for i in index_list]

    return dataset


def _apply_race_split(
    dataset: Dataset,
    dataset_name: str,
    target_split: str,
    split_seed: int,
    split_ratios: Tuple[float, float, float],
    requested_size: Optional[int],
) -> Dataset:
    """Deterministically split a single-source RACE split into train/val/test partitions."""
    split_aliases = {
        "train": "train",
        "val": "validation",
        "validation": "validation",
        "test": "test",
        "train_val": "train_val",
        "train+val": "train_val",
    }
    normalized_target = split_aliases.get(str(target_split).strip().lower())
    if normalized_target is None:
        raise ValueError(
            f"Unsupported RACE target split '{target_split}'. "
            "Use one of: train, validation, test, train_val."
        )

    total_examples = len(dataset.x)
    if total_examples <= 0:
        raise ValueError(f"RACE dataset '{dataset_name}' is empty before splitting")

    rng = np.random.RandomState(int(split_seed))
    all_indices = np.arange(total_examples, dtype=np.int64)
    rng.shuffle(all_indices)

    ratio_array = np.array(split_ratios, dtype=np.float64)
    raw_counts = ratio_array * float(total_examples)
    split_counts = np.floor(raw_counts).astype(np.int64)
    remainder = int(total_examples - split_counts.sum())
    if remainder > 0:
        residual_order = np.argsort(-(raw_counts - split_counts))
        for idx in residual_order[:remainder]:
            split_counts[idx] += 1

    train_count, val_count, test_count = [int(v) for v in split_counts.tolist()]

    train_indices = np.array(all_indices[:train_count], copy=True)
    val_start = train_count
    val_end = train_count + val_count
    val_indices = np.array(all_indices[val_start:val_end], copy=True)
    test_indices = np.array(all_indices[val_end:], copy=True)

    split_indices_map = {
        "train": train_indices,
        "validation": val_indices,
        "test": test_indices,
        "train_val": np.concatenate([train_indices, val_indices]),
    }
    selected_indices = np.array(split_indices_map[normalized_target], copy=True)
    rng.shuffle(selected_indices)

    if requested_size is not None:
        requested_size_int = int(requested_size)
        if requested_size_int <= 0:
            raise ValueError(
                f"Invalid size={requested_size_int} for RACE dataset '{dataset_name}'. "
                "Expected a positive integer."
            )
        if requested_size_int < selected_indices.shape[0]:
            selected_indices = selected_indices[:requested_size_int]

    dataset = _subset_race_dataset(dataset, selected_indices)

    dataset.race_split_source = "single_source_split"
    dataset.race_balanced_split = normalized_target
    dataset.race_split_seed = int(split_seed)
    dataset.race_split_ratios = tuple(float(v) for v in ratio_array.tolist())
    dataset.race_split_counts = {
        "source_examples": int(total_examples),
        "train_examples": train_count,
        "validation_examples": val_count,
        "test_examples": test_count,
        "selected_examples": int(len(dataset.x)),
    }

    log.info(
        "RACE deterministic split for %s: split=%s, seed=%d, source=%d, "
        "counts(train/val/test)=(%d/%d/%d), selected=%d",
        dataset_name,
        normalized_target,
        int(split_seed),
        int(total_examples),
        train_count,
        val_count,
        test_count,
        int(len(dataset.x)),
    )

    return dataset


def _apply_pubmedqa_balanced_split(
    dataset: Dataset,
    dataset_name: str,
    target_split: str,
    split_seed: int,
    split_ratios: Tuple[float, float, float],
    requested_size: Optional[int],
) -> Dataset:
    """
    Balance YES/NO labels, perform deterministic 6:2:2-style splitting, then subset.

    The split is performed on balanced labels first, then the target partition is selected.
    """
    split_aliases = {
        "train": "train",
        "val": "validation",
        "validation": "validation",
        "test": "test",
        "train_val": "train_val",
        "train+val": "train_val",
    }
    normalized_target = split_aliases.get(str(target_split).strip().lower())
    if normalized_target is None:
        raise ValueError(
            f"Unsupported PubMedQA target split '{target_split}'. "
            "Use one of: train, validation, test, train_val."
        )

    labels = np.array([str(label).strip().upper() for label in dataset.y], dtype=object)
    yes_indices = np.where(labels == "YES")[0]
    no_indices = np.where(labels == "NO")[0]

    if yes_indices.size == 0 or no_indices.size == 0:
        raise ValueError(
            f"PubMedQA dataset '{dataset_name}' must contain both YES and NO labels "
            f"(found YES={yes_indices.size}, NO={no_indices.size})."
        )

    min_class_count = int(min(yes_indices.size, no_indices.size))
    rng = np.random.RandomState(int(split_seed))

    yes_pool = np.array(yes_indices, copy=True)
    no_pool = np.array(no_indices, copy=True)
    rng.shuffle(yes_pool)
    rng.shuffle(no_pool)
    yes_pool = yes_pool[:min_class_count]
    no_pool = no_pool[:min_class_count]

    ratio_array = np.array(split_ratios, dtype=np.float64)
    raw_counts = ratio_array * float(min_class_count)
    split_counts = np.floor(raw_counts).astype(np.int64)
    remainder = int(min_class_count - split_counts.sum())
    if remainder > 0:
        residual_order = np.argsort(-(raw_counts - split_counts))
        for idx in residual_order[:remainder]:
            split_counts[idx] += 1

    train_count, val_count, test_count = [int(v) for v in split_counts.tolist()]

    train_yes = yes_pool[:train_count]
    val_yes = yes_pool[train_count: train_count + val_count]
    test_yes = yes_pool[train_count + val_count:]

    train_no = no_pool[:train_count]
    val_no = no_pool[train_count: train_count + val_count]
    test_no = no_pool[train_count + val_count:]

    train_indices = np.concatenate([train_yes, train_no])
    val_indices = np.concatenate([val_yes, val_no])
    test_indices = np.concatenate([test_yes, test_no])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    split_indices_map = {
        "train": train_indices,
        "validation": val_indices,
        "test": test_indices,
        "train_val": np.concatenate([train_indices, val_indices]),
    }
    selected_indices = np.array(split_indices_map[normalized_target], copy=True)
    rng.shuffle(selected_indices)

    if requested_size is not None:
        requested_size_int = int(requested_size)
        if requested_size_int <= 0:
            raise ValueError(
                f"Invalid size={requested_size_int} for PubMedQA dataset '{dataset_name}'. "
                "Expected a positive integer."
            )

        if requested_size_int < selected_indices.shape[0]:
            selected_labels = labels[selected_indices]
            split_yes_indices = selected_indices[selected_labels == "YES"]
            split_no_indices = selected_indices[selected_labels == "NO"]
            per_class_limit = min(
                split_yes_indices.shape[0],
                split_no_indices.shape[0],
                requested_size_int // 2,
            )

            if per_class_limit <= 0:
                raise ValueError(
                    f"Requested size={requested_size_int} is too small to keep YES/NO balance "
                    f"for PubMedQA dataset '{dataset_name}'."
                )

            rng.shuffle(split_yes_indices)
            rng.shuffle(split_no_indices)

            selected_indices = np.concatenate(
                [split_yes_indices[:per_class_limit], split_no_indices[:per_class_limit]]
            )
            rng.shuffle(selected_indices)

            balanced_size = int(selected_indices.shape[0])
            if balanced_size < requested_size_int:
                log.warning(
                    "Requested size=%d for PubMedQA dataset '%s' adjusted to %d to preserve YES/NO balance.",
                    requested_size_int,
                    dataset_name,
                    balanced_size,
                )

    dataset = _subset_pubmedqa_dataset(dataset, selected_indices)

    dataset.pubmedqa_balanced_source = "pqa_artificial"
    dataset.pubmedqa_balanced_split = normalized_target
    dataset.pubmedqa_split_seed = int(split_seed)
    dataset.pubmedqa_split_ratios = tuple(float(v) for v in ratio_array.tolist())
    dataset.pubmedqa_balanced_counts = {
        "source_yes": int(yes_indices.size),
        "source_no": int(no_indices.size),
        "balanced_per_label": int(min_class_count),
        "train_per_label": train_count,
        "validation_per_label": val_count,
        "test_per_label": test_count,
        "selected_examples": int(len(dataset.x)),
    }

    if normalized_target == "train_val":
        train_val_size = train_count + val_count
        if train_val_size > 0:
            dataset.pubmedqa_train_val_split_ratio = float(val_count / train_val_size)

    log.info(
        "PubMedQA balanced split for %s: split=%s, seed=%d, source_yes=%d, source_no=%d, "
        "balanced_per_label=%d, per_label_counts(train/val/test)=(%d/%d/%d), selected=%d",
        dataset_name,
        normalized_target,
        int(split_seed),
        int(yes_indices.size),
        int(no_indices.size),
        int(min_class_count),
        train_count,
        val_count,
        test_count,
        int(len(dataset.x)),
    )

    return dataset


def _apply_balanced_binary_split(
    dataset: Dataset,
    dataset_name: str,
    target_split: str,
    split_seed: int,
    split_ratios: Tuple[float, float, float],
    requested_size: Optional[int],
    positive_label: Optional[str] = None,
) -> Dataset:
    """Balance binary labels, perform deterministic split, then subset.

    This is a PubMedQA-like split policy for non-PubMed binary datasets.
    """
    split_aliases = {
        "train": "train",
        "val": "validation",
        "validation": "validation",
        "test": "test",
        "train_val": "train_val",
        "train+val": "train_val",
    }
    normalized_target = split_aliases.get(str(target_split).strip().lower())
    if normalized_target is None:
        raise ValueError(
            f"Unsupported balanced-binary target split '{target_split}'. "
            "Use one of: train, validation, test, train_val."
        )

    raw_labels = np.array([str(label).strip().upper() for label in dataset.y], dtype=object)
    unique_labels = sorted(set(raw_labels.tolist()))
    if len(unique_labels) != 2:
        raise ValueError(
            f"Dataset '{dataset_name}' must be binary for balanced split "
            f"(found labels={unique_labels})."
        )

    if positive_label is None:
        positive_candidates = ["1", "YES", "TRUE", "POSITIVE"]
        chosen_positive = next((v for v in positive_candidates if v in unique_labels), unique_labels[1])
    else:
        chosen_positive = str(positive_label).strip().upper()
        if chosen_positive not in unique_labels:
            raise ValueError(
                f"positive_label='{positive_label}' not found in dataset labels {unique_labels}"
            )

    negative_label = next(lbl for lbl in unique_labels if lbl != chosen_positive)

    pos_indices = np.where(raw_labels == chosen_positive)[0]
    neg_indices = np.where(raw_labels == negative_label)[0]
    if pos_indices.size == 0 or neg_indices.size == 0:
        raise ValueError(
            f"Dataset '{dataset_name}' must contain both classes for balanced split "
            f"(positive={pos_indices.size}, negative={neg_indices.size})."
        )

    min_class_count = int(min(pos_indices.size, neg_indices.size))
    rng = np.random.RandomState(int(split_seed))

    pos_pool = np.array(pos_indices, copy=True)
    neg_pool = np.array(neg_indices, copy=True)
    rng.shuffle(pos_pool)
    rng.shuffle(neg_pool)
    pos_pool = pos_pool[:min_class_count]
    neg_pool = neg_pool[:min_class_count]

    ratio_array = np.array(split_ratios, dtype=np.float64)
    raw_counts = ratio_array * float(min_class_count)
    split_counts = np.floor(raw_counts).astype(np.int64)
    remainder = int(min_class_count - split_counts.sum())
    if remainder > 0:
        residual_order = np.argsort(-(raw_counts - split_counts))
        for idx in residual_order[:remainder]:
            split_counts[idx] += 1

    train_count, val_count, test_count = [int(v) for v in split_counts.tolist()]

    train_pos = pos_pool[:train_count]
    val_pos = pos_pool[train_count: train_count + val_count]
    test_pos = pos_pool[train_count + val_count:]

    train_neg = neg_pool[:train_count]
    val_neg = neg_pool[train_count: train_count + val_count]
    test_neg = neg_pool[train_count + val_count:]

    train_indices = np.concatenate([train_pos, train_neg])
    val_indices = np.concatenate([val_pos, val_neg])
    test_indices = np.concatenate([test_pos, test_neg])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    split_indices_map = {
        "train": train_indices,
        "validation": val_indices,
        "test": test_indices,
        "train_val": np.concatenate([train_indices, val_indices]),
    }
    selected_indices = np.array(split_indices_map[normalized_target], copy=True)
    rng.shuffle(selected_indices)

    if requested_size is not None:
        requested_size_int = int(requested_size)
        if requested_size_int <= 0:
            raise ValueError(
                f"Invalid size={requested_size_int} for dataset '{dataset_name}'. "
                "Expected a positive integer."
            )

        if requested_size_int < selected_indices.shape[0]:
            selected_labels = raw_labels[selected_indices]
            split_pos_indices = selected_indices[selected_labels == chosen_positive]
            split_neg_indices = selected_indices[selected_labels == negative_label]
            per_class_limit = min(
                split_pos_indices.shape[0],
                split_neg_indices.shape[0],
                requested_size_int // 2,
            )

            if per_class_limit <= 0:
                # Debug fallback: allow very small subsets (e.g., size=1) even if
                # exact class balance is impossible.
                selected_indices = selected_indices[:requested_size_int]
                log.warning(
                    "Requested size=%d for dataset '%s' is too small to preserve class "
                    "balance; using unbalanced debug subset of size %d.",
                    requested_size_int,
                    dataset_name,
                    int(selected_indices.shape[0]),
                )
                per_class_limit = 0

            if per_class_limit > 0:
                rng.shuffle(split_pos_indices)
                rng.shuffle(split_neg_indices)

                selected_indices = np.concatenate(
                    [split_pos_indices[:per_class_limit], split_neg_indices[:per_class_limit]]
                )
                rng.shuffle(selected_indices)

                balanced_size = int(selected_indices.shape[0])
                if balanced_size < requested_size_int:
                    log.warning(
                        "Requested size=%d for dataset '%s' adjusted to %d to preserve class balance.",
                        requested_size_int,
                        dataset_name,
                        balanced_size,
                    )

    # Keep any mixed-context prompt variants aligned with the selected subset.
    dataset_type_hint = str(getattr(dataset, "pubmedqa_prompt_strategy", "")).strip().lower()
    if dataset_type_hint == "mixed_context":
        dataset = _subset_pubmedqa_dataset(dataset, selected_indices)
    else:
        dataset.select([int(i) for i in selected_indices.tolist()])

    dataset.binary_balanced_split = normalized_target
    dataset.binary_split_seed = int(split_seed)
    dataset.binary_split_ratios = tuple(float(v) for v in ratio_array.tolist())
    dataset.binary_positive_label = chosen_positive
    dataset.binary_negative_label = negative_label
    dataset.binary_balanced_counts = {
        "source_positive": int(pos_indices.size),
        "source_negative": int(neg_indices.size),
        "balanced_per_label": int(min_class_count),
        "train_per_label": train_count,
        "validation_per_label": val_count,
        "test_per_label": test_count,
        "selected_examples": int(len(dataset.x)),
    }

    if normalized_target == "train_val":
        train_val_size = train_count + val_count
        if train_val_size > 0:
            dataset.binary_train_val_split_ratio = float(val_count / train_val_size)

    log.info(
        "Balanced binary split for %s: split=%s, seed=%d, source_pos=%d, source_neg=%d, "
        "balanced_per_label=%d, per_label_counts(train/val/test)=(%d/%d/%d), selected=%d",
        dataset_name,
        normalized_target,
        int(split_seed),
        int(pos_indices.size),
        int(neg_indices.size),
        int(min_class_count),
        train_count,
        val_count,
        test_count,
        int(len(dataset.x)),
    )

    return dataset


def load_datasets_from_config(
    dataset_configs: List[Dict[str, Any]],
    split: str = "train",
    random_seed: Optional[int] = None,
) -> Tuple[List[Dataset], List[str]]:
    """
    Load multiple datasets from configuration.
    
    Args:
        dataset_configs: List of dataset configuration dictionaries, each containing:
            - name: Dataset name (REQUIRED)
            - display_name: Display name for logging (optional, defaults to name)
            - text_column: Text column name (optional, default "input")
            - label_column: Label column name (optional, default "output")
            - batch_size: Batch size (optional, default 8)
            - prompt: Prompt template (optional, default "")
            - description: Dataset description (optional, default "")
            - n_shot: Number of few-shot examples (optional, default 0)
            - few_shot_split: Split to use for few-shot (optional, default "train")
            - few_shot_prompt: Few-shot prompt template (optional)
            - instruct: Whether to use instruct format (optional, default False)
            - train_split: Split name for training (optional, default "train")
            - eval_split: Split name for evaluation (optional, default "test")
            - train_config_name: HF subset/config for training mode (optional)
            - eval_config_name: HF subset/config for evaluation mode (optional)
            - test_config_name: Alias for eval_config_name (optional)
            - config_name: HF subset/config fallback for all modes (optional)
                        - source_config_name: source subset/config for split-sensitive datasets
                        - source_split: source split for split-sensitive datasets (default "train")
                        - train_target_split: target split for training phase (default "train_val")
                        - eval_target_split: target split for evaluation phase (default "test")
                        - split_ratios: train/val/test ratios (default [0.6, 0.2, 0.2])
                        - split_seed: Seed for deterministic PubMedQA splitting (optional)
            - size: Number of examples to load (optional, loads all if None)
                        - source_size: Source cap before PubMedQA split (optional)
            - load_from_disk: Whether to load from disk (optional, default False)
            - trust_remote_code: Whether to trust remote code (optional, default False)
        split: Split to load ("train" or "test")
                random_seed: Optional global seed used by split-sensitive dataset loaders.
        
    Returns:
        Tuple of (list of Dataset instances, list of dataset names)
        
    Raises:
        ValueError: If dataset config is invalid
    """
    if not dataset_configs:
        raise ValueError("Must provide at least one dataset config")
    
    datasets = []
    dataset_names = []
    
    for i, dataset_cfg in enumerate(dataset_configs):
        if "name" not in dataset_cfg:
            raise ValueError(f"Dataset config {i} missing required 'name' field: {dataset_cfg}")
        
        dataset_path = dataset_cfg["name"]
        is_pubmedqa = _is_pubmedqa_dataset_config(dataset_cfg)
        is_race = _is_race_dataset_config(dataset_cfg)
        generic_split_keys_present = any(
            key in dataset_cfg
            for key in ("split_ratios", "train_target_split", "eval_target_split")
        )
        use_balanced_binary_split = (not is_pubmedqa and not is_race and generic_split_keys_present)
        # Backward-compatible alias: some configs use sample_size for dataset caps.
        requested_size = dataset_cfg.get("size", dataset_cfg.get("sample_size", None))

        if is_pubmedqa:
            config_name = dataset_cfg.get(
                "source_config_name",
                dataset_cfg.get(
                    "train_config_name",
                    dataset_cfg.get(
                        "eval_config_name",
                        dataset_cfg.get("config_name", "pqa_artificial"),
                    ),
                ),
            )
            actual_split = dataset_cfg.get("source_split", "train")
            if split == "train":
                dataset_target_split = dataset_cfg.get("train_target_split", "train_val")
            elif split == "test":
                dataset_target_split = dataset_cfg.get("eval_target_split", "test")
            else:
                dataset_target_split = dataset_cfg.get("validation_target_split", split)
        elif is_race:
            config_name = dataset_cfg.get(
                "race_source_config_name",
                dataset_cfg.get(
                    "train_config_name",
                    dataset_cfg.get(
                        "eval_config_name",
                        dataset_cfg.get("config_name"),
                    ),
                ),
            )
            actual_split = dataset_cfg.get(
                "race_source_split",
                dataset_cfg.get("train_split", dataset_cfg.get("eval_split", "test")),
            )
            if split == "train":
                dataset_target_split = dataset_cfg.get("race_train_target_split", "train")
            elif split == "test":
                dataset_target_split = dataset_cfg.get("race_eval_target_split", "test")
            else:
                dataset_target_split = dataset_cfg.get("race_validation_target_split", split)
        elif split == "train":
            config_name = dataset_cfg.get("train_config_name", dataset_cfg.get("config_name"))
            split_key = "train_split"
            actual_split = dataset_cfg.get(split_key, split)
            dataset_target_split = None
        else:
            config_name = dataset_cfg.get(
                "eval_config_name",
                dataset_cfg.get("test_config_name", dataset_cfg.get("config_name")),
            )
            split_key = "eval_split"
            actual_split = dataset_cfg.get(split_key, split)
            dataset_target_split = None

        if config_name and isinstance(dataset_path, str):
            dataset_path = [dataset_path, config_name]
        dataset_name = dataset_cfg.get(
            "display_name",
            str(dataset_path).replace("/", "_").replace("[", "").replace("]", "")
        )

        source_size = dataset_cfg.get("source_size", None) if (is_pubmedqa or is_race or use_balanced_binary_split) else requested_size
        
        # Load dataset
        try:
            requested_local_files_only = dataset_cfg.get("local_files_only", True)
            if requested_local_files_only is not True:
                log.warning(
                    "Dataset '%s' requested local_files_only=%r; enforcing local-first loading with fallback.",
                    dataset_name,
                    requested_local_files_only,
                )
            dataset_load_kwargs = {
                "batch_size": dataset_cfg.get("batch_size", 8),
                "prompt": dataset_cfg.get("prompt", ""),
                "description": dataset_cfg.get("description", ""),
                "n_shot": dataset_cfg.get("n_shot", 0),
                "few_shot_split": dataset_cfg.get("few_shot_split", "train"),
                "few_shot_prompt": dataset_cfg.get("few_shot_prompt", None),
                "instruct": dataset_cfg.get("instruct", False),
                "split": actual_split,
                "size": source_size,
                "load_from_disk": dataset_cfg.get("load_from_disk", False),
                "trust_remote_code": dataset_cfg.get("trust_remote_code", False),
                "local_files_only": True,
                "allow_remote_fallback": dataset_cfg.get("allow_remote_fallback", True),
            }
            # Forward optional dataset-loader keys (e.g. JSON data_files/field).
            for optional_key in (
                "data_files",
                "data_dir",
                "field",
                "dataset_format",
                "probability_label_column",
                "positive_label",
                "mixed_context_routing",
                "prompt_helper",
                "prompt_without_context_helper",
            ):
                if optional_key in dataset_cfg and dataset_cfg.get(optional_key) is not None:
                    dataset_load_kwargs[optional_key] = dataset_cfg.get(optional_key)
            if "prompt_without_context" in dataset_cfg:
                dataset_load_kwargs["prompt_without_context"] = dataset_cfg["prompt_without_context"]
            if "pubmedqa_context_model_path" in dataset_cfg:
                dataset_load_kwargs["pubmedqa_context_model_path"] = dataset_cfg["pubmedqa_context_model_path"]

            dataset = Dataset.load(
                dataset_path,
                dataset_cfg.get("text_column", "input"),
                dataset_cfg.get("label_column", "output"),
                **dataset_load_kwargs,
            )

            if is_pubmedqa:
                seed_candidate = dataset_cfg.get("split_seed", random_seed if random_seed is not None else 42)
                try:
                    split_seed = int(seed_candidate)
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"Invalid split_seed for PubMedQA dataset '{dataset_name}': {seed_candidate}"
                    ) from e

                split_ratios = _normalize_split_ratios(dataset_cfg.get("split_ratios"))
                dataset = _apply_pubmedqa_balanced_split(
                    dataset=dataset,
                    dataset_name=dataset_name,
                    target_split=dataset_target_split,
                    split_seed=split_seed,
                    split_ratios=split_ratios,
                    requested_size=requested_size,
                )
            elif is_race:
                seed_candidate = dataset_cfg.get("split_seed", random_seed if random_seed is not None else 42)
                try:
                    split_seed = int(seed_candidate)
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"Invalid split_seed for RACE dataset '{dataset_name}': {seed_candidate}"
                    ) from e

                split_ratios = _normalize_race_split_ratios(dataset_cfg.get("race_split_ratios"))
                dataset = _apply_race_split(
                    dataset=dataset,
                    dataset_name=dataset_name,
                    target_split=dataset_target_split,
                    split_seed=split_seed,
                    split_ratios=split_ratios,
                    requested_size=requested_size,
                )
            elif use_balanced_binary_split:
                seed_candidate = dataset_cfg.get("split_seed", random_seed if random_seed is not None else 42)
                try:
                    split_seed = int(seed_candidate)
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"Invalid split_seed for balanced-binary dataset '{dataset_name}': {seed_candidate}"
                    ) from e

                split_ratios = _normalize_split_ratios(dataset_cfg.get("split_ratios"))
                if split == "train":
                    target_split = dataset_cfg.get("train_target_split", "train_val")
                elif split == "test":
                    target_split = dataset_cfg.get("eval_target_split", "test")
                else:
                    target_split = dataset_cfg.get("validation_target_split", split)
                dataset = _apply_balanced_binary_split(
                    dataset=dataset,
                    dataset_name=dataset_name,
                    target_split=target_split,
                    split_seed=split_seed,
                    split_ratios=split_ratios,
                    requested_size=requested_size,
                    positive_label=dataset_cfg.get("positive_label"),
                )

            dataset.cache_dataset_config = _build_dataset_cache_config_signature(
                dataset_cfg=dataset_cfg,
                dataset_name=dataset_name,
                load_split=split,
                resolved_path=dataset_path,
                resolved_split=actual_split,
                resolved_config_name=config_name,
                dataset_target_split=dataset_target_split,
                random_seed=random_seed,
            )
            dataset.cache_dataset_name = dataset_name
            dataset.cache_dataset_split = actual_split
        except Exception as e:
            raise RuntimeError(
                f"Failed to load dataset {i} (name: {dataset_path}, split: {actual_split}): {e}"
            ) from e
        
        if len(dataset.x) == 0:
            raise ValueError(
                f"Dataset {dataset_name} loaded but has 0 examples "
                f"(split: {actual_split})"
            )
        
        datasets.append(dataset)
        dataset_names.append(dataset_name)
        if is_pubmedqa:
            log.info(
                f"Loaded dataset {i+1}/{len(dataset_configs)}: {dataset_name} "
                f"(source_split: {actual_split}, target_split: {dataset_target_split}, size: {len(dataset.x)})"
            )
        elif is_race:
            log.info(
                f"Loaded dataset {i+1}/{len(dataset_configs)}: {dataset_name} "
                f"(source_split: {actual_split}, target_split: {dataset_target_split}, size: {len(dataset.x)})"
            )
        else:
            log.info(
                f"Loaded dataset {i+1}/{len(dataset_configs)}: {dataset_name} "
                f"(split: {actual_split}, size: {len(dataset.x)})"
            )
    
    return datasets, dataset_names
