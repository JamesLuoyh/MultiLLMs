"""
Dataset loading utilities.

Simplified version with strict error handling.
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add src/ to path for lm_polygraph imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from lm_polygraph.utils.dataset import Dataset

log = logging.getLogger(__name__)


def load_datasets_from_config(
    dataset_configs: List[Dict[str, Any]],
    split: str = "train",
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
            - size: Number of examples to load (optional, loads all if None)
            - load_from_disk: Whether to load from disk (optional, default False)
            - trust_remote_code: Whether to trust remote code (optional, default False)
        split: Split to load ("train" or "test")
        
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
        dataset_name = dataset_cfg.get(
            "display_name",
            str(dataset_path).replace("/", "_").replace("[", "").replace("]", "")
        )
        
        # Determine split key (train_split vs eval_split)
        split_key = f"{split}_split" if split == "train" else "eval_split"
        actual_split = dataset_cfg.get(split_key, split)
        
        # Load dataset
        try:
            dataset = Dataset.load(
                dataset_path,
                dataset_cfg.get("text_column", "input"),
                dataset_cfg.get("label_column", "output"),
                batch_size=dataset_cfg.get("batch_size", 8),
                prompt=dataset_cfg.get("prompt", ""),
                description=dataset_cfg.get("description", ""),
                n_shot=dataset_cfg.get("n_shot", 0),
                few_shot_split=dataset_cfg.get("few_shot_split", "train"),
                few_shot_prompt=dataset_cfg.get("few_shot_prompt", None),
                instruct=dataset_cfg.get("instruct", False),
                split=actual_split,
                size=dataset_cfg.get("size", None),
                load_from_disk=dataset_cfg.get("load_from_disk", False),
                trust_remote_code=dataset_cfg.get("trust_remote_code", False),
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
        log.info(
            f"Loaded dataset {i+1}/{len(dataset_configs)}: {dataset_name} "
            f"(split: {actual_split}, size: {len(dataset.x)})"
        )
    
    return datasets, dataset_names
