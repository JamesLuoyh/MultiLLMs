"""
Utility functions for loading and merging YAML config files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Union, Optional


def load_yaml_file(file_path: Path) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dictionary."""
    with open(file_path, "r") as f:
        return yaml.safe_load(f) or {}


def resolve_config_path(path: Union[str, Path], base_dir: Path) -> Path:
    """
    Resolve a config path (can be relative or absolute).
    
    Args:
        path: Path to config file (relative or absolute)
        base_dir: Base directory for resolving relative paths
        
    Returns:
        Resolved Path object
    """
    path = Path(path)
    if path.is_absolute():
        return path
    return base_dir / path


def load_and_merge_configs(
    main_config_path: Path,
    base_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Load main config and merge referenced config files.
    
    Supports:
    - `_include_models`: List of model config file paths (relative to base_dir)
    - `_include_datasets`: List of dataset config file paths (relative to base_dir)
    
    Args:
        main_config_path: Path to main config file
        base_dir: Base directory for resolving relative paths (defaults to main_config_path.parent)
        
    Returns:
        Merged configuration dictionary
    """
    if base_dir is None:
        base_dir = main_config_path.parent
    
    # Load main config
    config = load_yaml_file(main_config_path)
    
    # Load and merge model configs if specified
    if "_include_models" in config:
        model_configs = []
        for model_path in config["_include_models"]:
            model_file = resolve_config_path(model_path, base_dir)
            if not model_file.exists():
                # Try in models subdirectory
                model_file = base_dir / "models" / model_path
            if not model_file.exists():
                # Try with just the filename (strip models/ prefix if present)
                model_filename = Path(model_path).name
                model_file = base_dir / "models" / model_filename
            if not model_file.exists():
                # Try going up one directory level (for sweep_results case)
                parent_base_dir = base_dir.parent if base_dir.name.startswith("sweep_results") else base_dir
                model_file = resolve_config_path(model_path, parent_base_dir)
                if not model_file.exists():
                    # Also try parent/models subdirectory
                    model_file = parent_base_dir / "models" / Path(model_path).name
            if model_file.exists():
                model_config = load_yaml_file(model_file)
                model_configs.append(model_config)
            else:
                raise FileNotFoundError(f"Model config not found: {model_path}")
        config["models"] = model_configs
        del config["_include_models"]
    
    # Load and merge dataset configs if specified
    if "_include_datasets" in config:
        dataset_configs = []
        override_configs = config.get("datasets", [])
        for idx, dataset_path in enumerate(config["_include_datasets"]):
            dataset_file = resolve_config_path(dataset_path, base_dir)
            if not dataset_file.exists():
                # Try in datasets subdirectory
                dataset_file = base_dir / "datasets" / dataset_path
            if not dataset_file.exists():
                # Try with just the filename (strip datasets/ prefix if present)
                dataset_filename = Path(dataset_path).name
                dataset_file = base_dir / "datasets" / dataset_filename
            if not dataset_file.exists():
                # Try going up one directory level (for sweep_results case)
                parent_base_dir = base_dir.parent if base_dir.name.startswith("sweep_results") else base_dir
                dataset_file = resolve_config_path(dataset_path, parent_base_dir)
                if not dataset_file.exists():
                    # Also try parent/datasets subdirectory
                    dataset_file = parent_base_dir / "datasets" / Path(dataset_path).name
            if dataset_file.exists():
                dataset_config = load_yaml_file(dataset_file)
                # Merge with override if provided
                if idx < len(override_configs):
                    dataset_config.update(override_configs[idx])
                dataset_configs.append(dataset_config)
            else:
                raise FileNotFoundError(f"Dataset config not found: {dataset_path}")
        config["datasets"] = dataset_configs
        del config["_include_datasets"]
        # Remove override configs if they were used
        if "datasets" in config and isinstance(config["datasets"], list) and len(config["datasets"]) == len(dataset_configs):
            # Already merged, but check if there are extra overrides
            pass
    
    # Handle test_datasets similarly
    if "_include_test_datasets" in config:
        test_dataset_configs = []
        override_configs = config.get("test_datasets", [])
        for idx, dataset_path in enumerate(config["_include_test_datasets"]):
            dataset_file = resolve_config_path(dataset_path, base_dir)
            if not dataset_file.exists():
                dataset_file = base_dir / "datasets" / dataset_path
            if not dataset_file.exists():
                # Try with just the filename (strip datasets/ prefix if present)
                dataset_filename = Path(dataset_path).name
                dataset_file = base_dir / "datasets" / dataset_filename
            if not dataset_file.exists():
                # Try going up one directory level (for sweep_results case)
                parent_base_dir = base_dir.parent if base_dir.name.startswith("sweep_results") else base_dir
                dataset_file = resolve_config_path(dataset_path, parent_base_dir)
                if not dataset_file.exists():
                    # Also try parent/datasets subdirectory
                    dataset_file = parent_base_dir / "datasets" / Path(dataset_path).name
            if dataset_file.exists():
                dataset_config = load_yaml_file(dataset_file)
                # Merge with override if provided
                if idx < len(override_configs):
                    dataset_config.update(override_configs[idx])
                test_dataset_configs.append(dataset_config)
            else:
                raise FileNotFoundError(f"Test dataset config not found: {dataset_path}")
        config["test_datasets"] = test_dataset_configs
        del config["_include_test_datasets"]
    
    # Handle ood_dataset
    if "_include_ood_dataset" in config:
        ood_path = config["_include_ood_dataset"]
        dataset_file = resolve_config_path(ood_path, base_dir)
        if not dataset_file.exists():
            dataset_file = base_dir / "datasets" / ood_path
        if not dataset_file.exists():
            # Try with just the filename (strip datasets/ prefix if present)
            ood_filename = Path(ood_path).name
            dataset_file = base_dir / "datasets" / ood_filename
        if not dataset_file.exists():
            # Try going up one directory level (for sweep_results case)
            parent_base_dir = base_dir.parent if base_dir.name.startswith("sweep_results") else base_dir
            dataset_file = resolve_config_path(ood_path, parent_base_dir)
            if not dataset_file.exists():
                # Also try parent/datasets subdirectory
                dataset_file = parent_base_dir / "datasets" / Path(ood_path).name
        if dataset_file.exists():
            ood_config = load_yaml_file(dataset_file)
            # Merge with override if provided
            if "ood_dataset" in config and isinstance(config["ood_dataset"], dict):
                ood_config.update(config["ood_dataset"])
            config["ood_dataset"] = ood_config
        else:
            raise FileNotFoundError(f"OOD dataset config not found: {ood_path}")
        del config["_include_ood_dataset"]
    
    return config
