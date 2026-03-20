"""
Dataset loading utilities.

Simplified version with strict error handling.
"""

import logging
import sys
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


def _is_pubmedqa_dataset_config(dataset_cfg: Dict[str, Any]) -> bool:
    """Return True when a dataset config targets PubMedQA."""
    fields = [
        dataset_cfg.get("name", ""),
        dataset_cfg.get("display_name", ""),
        dataset_cfg.get("config_name", ""),
        dataset_cfg.get("train_config_name", ""),
        dataset_cfg.get("eval_config_name", ""),
        dataset_cfg.get("test_config_name", ""),
        dataset_cfg.get("pubmedqa_source_config_name", ""),
    ]
    normalized = " ".join(str(field).lower() for field in fields if field is not None)
    return "pubmedqa" in normalized or "pubmed_qa" in normalized


def _normalize_pubmedqa_split_ratios(raw_ratios: Any) -> Tuple[float, float, float]:
    """Normalize PubMedQA split ratios to a valid (train, val, test) tuple."""
    default_ratios = (0.6, 0.2, 0.2)
    if raw_ratios is None:
        return default_ratios

    if not isinstance(raw_ratios, Sequence) or len(raw_ratios) != 3:
        log.warning(
            "Invalid pubmedqa_split_ratios=%s. Falling back to default ratios %s.",
            raw_ratios,
            default_ratios,
        )
        return default_ratios

    try:
        ratio_array = np.array([float(v) for v in raw_ratios], dtype=np.float64)
    except (TypeError, ValueError):
        log.warning(
            "Could not parse pubmedqa_split_ratios=%s. Falling back to default ratios %s.",
            raw_ratios,
            default_ratios,
        )
        return default_ratios

    if np.any(ratio_array < 0) or not np.any(ratio_array > 0):
        log.warning(
            "Non-positive pubmedqa_split_ratios=%s. Falling back to default ratios %s.",
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
                        - pubmedqa_source_config_name: PubMedQA source subset (default "pqa_artificial")
                        - pubmedqa_source_split: PubMedQA source split (default "train")
                        - pubmedqa_train_target_split: PubMedQA target split for training phase
                            (default "train_val")
                        - pubmedqa_eval_target_split: PubMedQA target split for evaluation phase
                            (default "test")
                        - pubmedqa_split_ratios: PubMedQA train/val/test ratios (default [0.6, 0.2, 0.2])
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
        requested_size = dataset_cfg.get("size", None)

        if is_pubmedqa:
            config_name = dataset_cfg.get(
                "pubmedqa_source_config_name",
                dataset_cfg.get(
                    "train_config_name",
                    dataset_cfg.get(
                        "eval_config_name",
                        dataset_cfg.get("config_name", "pqa_artificial"),
                    ),
                ),
            )
            actual_split = dataset_cfg.get("pubmedqa_source_split", "train")
            if split == "train":
                pubmedqa_target_split = dataset_cfg.get("pubmedqa_train_target_split", "train_val")
            elif split == "test":
                pubmedqa_target_split = dataset_cfg.get("pubmedqa_eval_target_split", "test")
            else:
                pubmedqa_target_split = dataset_cfg.get("pubmedqa_validation_target_split", split)
        elif split == "train":
            config_name = dataset_cfg.get("train_config_name", dataset_cfg.get("config_name"))
            split_key = "train_split"
            actual_split = dataset_cfg.get(split_key, split)
            pubmedqa_target_split = None
        else:
            config_name = dataset_cfg.get(
                "eval_config_name",
                dataset_cfg.get("test_config_name", dataset_cfg.get("config_name")),
            )
            split_key = "eval_split"
            actual_split = dataset_cfg.get(split_key, split)
            pubmedqa_target_split = None

        if config_name and isinstance(dataset_path, str):
            dataset_path = [dataset_path, config_name]
        dataset_name = dataset_cfg.get(
            "display_name",
            str(dataset_path).replace("/", "_").replace("[", "").replace("]", "")
        )

        source_size = dataset_cfg.get("source_size", None) if is_pubmedqa else requested_size
        
        # Load dataset
        try:
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
            }
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

                split_ratios = _normalize_pubmedqa_split_ratios(dataset_cfg.get("pubmedqa_split_ratios"))
                dataset = _apply_pubmedqa_balanced_split(
                    dataset=dataset,
                    dataset_name=dataset_name,
                    target_split=pubmedqa_target_split,
                    split_seed=split_seed,
                    split_ratios=split_ratios,
                    requested_size=requested_size,
                )
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
                f"(source_split: {actual_split}, target_split: {pubmedqa_target_split}, size: {len(dataset.x)})"
            )
        else:
            log.info(
                f"Loaded dataset {i+1}/{len(dataset_configs)}: {dataset_name} "
                f"(split: {actual_split}, size: {len(dataset.x)})"
            )
    
    return datasets, dataset_names
