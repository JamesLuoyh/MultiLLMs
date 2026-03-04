"""
Checkpoint directory utilities.

Simplified version with direct, predictable naming.
"""

import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any


def sanitize_name(name: str, max_length: int = 20) -> str:
    """
    Sanitize a name for use in file paths.
    
    Args:
        name: Name to sanitize
        max_length: Maximum length of the sanitized name
        
    Returns:
        Sanitized name safe for file paths
    """
    # Remove special characters, keep alphanumeric, dash, underscore
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    # Replace multiple underscores with single
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    if not sanitized:
        raise ValueError(f"Cannot sanitize name '{name}' - results in empty string")
    
    return sanitized


def get_model_name(model_path: str) -> str:
    """
    Extract a short model name from a model path.
    
    Args:
        model_path: Model path (e.g., "meta-llama/Llama-3.2-1B")
        
    Returns:
        Sanitized short name
    """
    parts = model_path.replace("/", "_").split("_")
    # Take last 2-3 parts for brevity
    if len(parts) >= 2:
        return sanitize_name("_".join(parts[-2:]), max_length=25)
    return sanitize_name(model_path, max_length=25)


def get_dataset_name(dataset_config: Dict[str, Any]) -> str:
    """
    Extract a short dataset name from dataset config.
    
    Args:
        dataset_config: Dataset configuration dict
        
    Returns:
        Sanitized short name
        
    Raises:
        ValueError: If dataset name cannot be determined
    """
    name = dataset_config.get("display_name") or dataset_config.get("name")
    
    if not name:
        raise ValueError(f"Dataset config missing 'name' or 'display_name': {dataset_config}")
    
    if isinstance(name, list):
        # Handle ['org/dataset', 'config'] format
        name = name[0] if name else None
    
    if not name:
        raise ValueError(f"Invalid dataset name in config: {dataset_config}")
    
    # Extract short name
    parts = str(name).replace("/", "_").split("_")
    if len(parts) >= 1:
        return sanitize_name(parts[-1], max_length=20)
    return sanitize_name(str(name), max_length=20)


def generate_checkpoint_dir(
    base_dir: Path,
    models: List[Dict[str, Any]],
    datasets: List[Dict[str, Any]],
    wagering_method: Dict[str, Any],
    aggregation: Dict[str, Any],
    create_hash: bool = True,
) -> Path:
    """
    Generate a unique checkpoint directory name based on configuration.
    
    Args:
        base_dir: Base directory for checkpoints
        models: List of model configs
        datasets: List of dataset configs
        wagering_method: Wagering method config
        aggregation: Aggregation function config
        create_hash: If True, append a hash for uniqueness
        
    Returns:
        Path to checkpoint directory
        
    Raises:
        ValueError: If any config is invalid
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate inputs
    if not models:
        raise ValueError("Must provide at least one model")
    if not datasets:
        raise ValueError("Must provide at least one dataset")
    if not wagering_method or "name" not in wagering_method:
        raise ValueError("wagering_method must have 'name'")
    if not aggregation or "name" not in aggregation:
        raise ValueError("aggregation must have 'name'")
    
    # Extract components
    model_names = [get_model_name(m["path"]) for m in models if "path" in m]
    if not model_names:
        raise ValueError("No valid model paths found in model configs")
    
    dataset_names = [get_dataset_name(d) for d in datasets]
    if not dataset_names:
        raise ValueError("No valid dataset names found in dataset configs")
    
    wagering_name = sanitize_name(wagering_method["name"], max_length=20)
    aggregation_name = sanitize_name(aggregation["name"], max_length=20)
    
    # Build directory name components
    components = []
    
    # Models (sorted for consistency)
    models_str = "_".join(sorted(set(model_names)))
    components.append(f"models_{models_str}")
    
    # Datasets (sorted for consistency)
    datasets_str = "_".join(sorted(set(dataset_names)))
    components.append(f"datasets_{datasets_str}")
    
    # Wagering method
    components.append(f"wagering_{wagering_name}")
    
    # Aggregation
    components.append(f"agg_{aggregation_name}")
    
    # Join components
    dir_name = "_".join(components)
    
    # Add hash for uniqueness if requested
    if create_hash:
        # Create hash from full config for uniqueness
        config_str = f"{models}_{datasets}_{wagering_method}_{aggregation}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        dir_name = f"{dir_name}_{config_hash}"
    
    return base_dir / dir_name


def get_checkpoint_metadata(
    models: List[Dict[str, Any]],
    datasets: List[Dict[str, Any]],
    wagering_method: Dict[str, Any],
    aggregation: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Get metadata dictionary for logging/analytics.
    
    Args:
        models: List of model configs
        datasets: List of dataset configs
        wagering_method: Wagering method config
        aggregation: Aggregation function config
        
    Returns:
        Dictionary with metadata for analytics
        
    Raises:
        ValueError: If any config is invalid
    """
    if not models:
        raise ValueError("Must provide at least one model")
    if not datasets:
        raise ValueError("Must provide at least one dataset")
    
    model_names = [m.get("path", "unknown") for m in models]
    dataset_names = [get_dataset_name(d) for d in datasets]
    
    return {
        "models": model_names,
        "model_count": len(models),
        "datasets": dataset_names,
        "dataset_count": len(datasets),
        "wagering_method": wagering_method.get("name", "unknown"),
        "wagering_config": wagering_method.get("config", {}),
        "aggregation_method": aggregation.get("name", "unknown"),
        "aggregation_config": aggregation.get("config", {}),
    }
