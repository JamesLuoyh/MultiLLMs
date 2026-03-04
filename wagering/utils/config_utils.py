"""
Configuration loading utilities.

Simplified version that loads configs without complex fallback logic.
All paths should be explicit and errors are raised immediately if files are not found.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_yaml_file(file_path: Path) -> Dict[str, Any]:
    """
    Load a YAML file and return its contents as a dictionary.
    
    Raises:
        FileNotFoundError: If file does not exist
        yaml.YAMLError: If file is not valid YAML
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")
    
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
        if config is None:
            raise ValueError(f"Config file is empty: {file_path}")
        return config


def resolve_config_path(path: str | Path, base_dir: Path) -> Path:
    """
    Resolve a config path (can be relative or absolute).
    
    Args:
        path: Path to config file (relative or absolute)
        base_dir: Base directory for resolving relative paths
        
    Returns:
        Resolved Path object
        
    Raises:
        FileNotFoundError: If resolved path does not exist
    """
    path = Path(path)
    
    if path.is_absolute():
        resolved = path
    else:
        resolved = base_dir / path
    
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved} (base_dir: {base_dir})")
    
    return resolved


def load_and_merge_configs(
    main_config_path: Path,
    base_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Load main config and merge referenced config files.
    
    Supports:
    - `_include_models`: List of model config file paths (relative to base_dir)
    - `_include_datasets`: List of dataset config file paths (relative to base_dir)
    - `_include_test_datasets`: List of test dataset config file paths
    
    Args:
        main_config_path: Path to main config file
        base_dir: Base directory for resolving relative paths (defaults to main_config_path.parent)
        
    Returns:
        Merged configuration dictionary
        
    Raises:
        FileNotFoundError: If any referenced config file is not found
        ValueError: If config is invalid
    """
    main_config_path = Path(main_config_path)
    if base_dir is None:
        base_dir = main_config_path.parent
    
    # Load main config
    config = load_yaml_file(main_config_path)
    
    # Load and merge model configs
    if "_include_models" in config:
        if not isinstance(config["_include_models"], list):
            raise ValueError(f"_include_models must be a list, got {type(config['_include_models'])}")
        
        model_configs = []
        for model_path in config["_include_models"]:
            model_file = resolve_config_path(model_path, base_dir)
            model_config = load_yaml_file(model_file)
            model_configs.append(model_config)
        
        config["models"] = model_configs
        del config["_include_models"]
    
    # Load and merge dataset configs
    if "_include_datasets" in config:
        if not isinstance(config["_include_datasets"], list):
            raise ValueError(f"_include_datasets must be a list, got {type(config['_include_datasets'])}")
        
        # Get any override configs specified in the main config
        override_configs = config.get("datasets", [])
        
        dataset_configs = []
        for idx, dataset_path in enumerate(config["_include_datasets"]):
            dataset_file = resolve_config_path(dataset_path, base_dir)
            dataset_config = load_yaml_file(dataset_file)
            
            # Merge with override if provided
            if idx < len(override_configs) and isinstance(override_configs[idx], dict):
                dataset_config.update(override_configs[idx])
            
            dataset_configs.append(dataset_config)
        
        config["datasets"] = dataset_configs
        del config["_include_datasets"]
    
    # Load and merge test dataset configs
    if "_include_test_datasets" in config:
        if not isinstance(config["_include_test_datasets"], list):
            raise ValueError(f"_include_test_datasets must be a list, got {type(config['_include_test_datasets'])}")
        
        # Get any override configs specified in the main config
        override_configs = config.get("test_datasets", [])
        
        test_dataset_configs = []
        for idx, dataset_path in enumerate(config["_include_test_datasets"]):
            dataset_file = resolve_config_path(dataset_path, base_dir)
            dataset_config = load_yaml_file(dataset_file)
            
            # Merge with override if provided
            if idx < len(override_configs) and isinstance(override_configs[idx], dict):
                dataset_config.update(override_configs[idx])
            
            test_dataset_configs.append(dataset_config)
        
        config["test_datasets"] = test_dataset_configs
        del config["_include_test_datasets"]

    # Load and merge OOD dataset config
    if "_include_ood_dataset" in config:
        ood_path = config["_include_ood_dataset"]
        dataset_file = resolve_config_path(ood_path, base_dir)
        ood_config = load_yaml_file(dataset_file)

        # Merge with override if provided
        if "ood_dataset" in config and isinstance(config["ood_dataset"], dict):
            ood_config.update(config["ood_dataset"])

        config["ood_dataset"] = ood_config
        del config["_include_ood_dataset"]
    
    # Validate required keys
    if "models" not in config or not config["models"]:
        raise ValueError("Config must specify models")
    if "wagering_method" not in config:
        raise ValueError("Config must specify wagering_method")
    if "aggregation" not in config:
        raise ValueError("Config must specify aggregation")
    
    return config
