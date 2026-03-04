"""
Utility functions for wagering training and inference pipelines.
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.utils.dataset import Dataset
from lm_polygraph.utils.multi_llm_ensemble import (
    collect_option_logits_for_model,
    filter_dataset_by_token_length,
)

log = logging.getLogger("lm_polygraph")


def load_models_from_config(
    model_configs: List[Dict[str, Any]],
    cache_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[WhiteboxModel], List[str]]:
    """
    Load multiple whitebox models from configuration.
    
    Args:
        model_configs: List of model configuration dictionaries, each containing:
            - path: Model path (required)
            - path_to_load_script: Path to load script (optional, for custom loading)
            - load_model_args: Arguments for model loading (optional)
            - load_tokenizer_args: Arguments for tokenizer loading (optional)
            - instruct: Whether model is instruction-tuned (optional)
        cache_kwargs: Optional cache kwargs for model loading
        
    Returns:
        Tuple of (list of WhiteboxModel instances, list of model names)
    """
    from lm_polygraph.utils.common import load_external_module
    from lm_polygraph.utils.generation_parameters import GenerationParametersFactory
    
    cache_kwargs = cache_kwargs or {}
    models = []
    model_names = []
    
    # Get original cwd for config resolution (try hydra first, fallback to current dir)
    try:
        from hydra.utils import get_original_cwd
        from hydra.core.global_hydra import GlobalHydra
        # Only use hydra if it's initialized
        if GlobalHydra.instance().is_initialized():
            original_cwd = Path(get_original_cwd())
        else:
            original_cwd = Path(os.getcwd())
    except (RuntimeError, AttributeError, ImportError, ValueError):
        original_cwd = Path(os.getcwd())
    
    # Try to load API keys for HF token
    def load_api_keys_from_config():
        api_keys_path = original_cwd / ".api_keys.yaml"
        if api_keys_path.exists():
            import yaml
            try:
                with open(api_keys_path, "r") as f:
                    config = yaml.safe_load(f)
                    if config:
                        filtered = {}
                        for k, v in config.items():
                            if v is None or v == "null" or v == "":
                                continue
                            if isinstance(v, str) and ("your-" in v.lower() and "-here" in v.lower()):
                                continue
                            filtered[k] = v
                        return filtered
            except Exception as e:
                log.debug(f"Could not load API keys config: {e}")
        return {}
    
    api_keys = load_api_keys_from_config()
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if hf_token is None:
        hf_token = api_keys.get("hf_token") or api_keys.get("huggingface_token")
    
    for model_cfg in model_configs:
        model_path = model_cfg["path"]
        model_names.append(model_path.replace("/", "_"))
        
        # Use custom load script if provided
        if model_cfg.get("path_to_load_script"):
            load_script_path = model_cfg["path_to_load_script"]
            if not os.path.isabs(load_script_path):
                # Try relative to examples/configs/model/ first, then original_cwd
                # Handle both "model/default_causal.py" and "default_causal.py" formats
                if "/" in load_script_path:
                    # Has directory component like "model/default_causal.py"
                    examples_path = original_cwd / "examples" / "configs" / load_script_path
                else:
                    # Just filename, assume it's in model/
                    examples_path = original_cwd / "examples" / "configs" / "model" / load_script_path
                
                if examples_path.exists():
                    load_script_path = str(examples_path)
                else:
                    # Fallback to original_cwd
                    load_script_path = str(original_cwd / load_script_path)
            
            load_module = load_external_module(str(load_script_path))
            
            load_model_args = {"model_path": model_path}
            load_model_args.update(model_cfg.get("load_model_args", {}))
            if hf_token is not None:
                load_model_args["token"] = hf_token
            
            base_model = load_module.load_model(**load_model_args)
            
            load_tok_args = {"model_path": model_path}
            load_tok_args.update(model_cfg.get("load_tokenizer_args", {}))
            if hf_token is not None:
                load_tok_args["token"] = hf_token
            tokenizer = load_module.load_tokenizer(**load_tok_args)
            
            # Set pad_token_id
            if tokenizer.pad_token_id is not None:
                if hasattr(base_model, 'generation_config'):
                    base_model.generation_config.pad_token_id = tokenizer.pad_token_id
                if hasattr(base_model.config, 'pad_token_id'):
                    base_model.config.pad_token_id = tokenizer.pad_token_id
            
            generation_params = GenerationParametersFactory.from_params(
                yaml_config=model_cfg.get("generation_params", {}),
                native_config=base_model.generation_config.to_dict()
            )
            
            instruct = model_cfg.get("instruct", False)
            model = WhiteboxModel(
                base_model,
                tokenizer,
                model_path,
                model_cfg.get("type", "CausalLM"),
                generation_params,
                instruct=instruct,
            )
        else:
            # Use default loading
            instruct = model_cfg.get("instruct", False)
            model = WhiteboxModel.from_pretrained(
                model_path,
                generation_params=model_cfg.get("generation_params", {}),
                device_map=model_cfg.get("load_model_args", {}).get("device_map", "auto"),
                add_bos_token=model_cfg.get("add_bos_token", True),
                instruct=instruct,
                **cache_kwargs,
            )
        
        models.append(model)
        log.info(f"Loaded model: {model_path}")
    
    return models, model_names


def load_datasets_from_config(
    dataset_configs: List[Dict[str, Any]],
    split: str = "train",
) -> Tuple[List[Dataset], List[str]]:
    """
    Load multiple datasets from configuration.
    
    Args:
        dataset_configs: List of dataset configuration dictionaries
        split: Split to load ("train" or "test")
        
    Returns:
        Tuple of (list of Dataset instances, list of dataset names)
    """
    datasets = []
    dataset_names = []
    
    for dataset_cfg in dataset_configs:
        dataset_path = dataset_cfg["name"]
        dataset_name = dataset_cfg.get("display_name", str(dataset_path).replace("/", "_").replace("[", "").replace("]", ""))
        
        # Determine split key (train_split vs eval_split)
        split_key = f"{split}_split" if split == "train" else "eval_split"
        actual_split = dataset_cfg.get(split_key, split)
        
        # Use Dataset.load for loading
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
        
        # Filter by token length if specified
        max_prompt_tokens = dataset_cfg.get("max_prompt_tokens")
        if max_prompt_tokens:
            # Filtering will be handled by the training script if needed
            pass
        
        datasets.append(dataset)
        dataset_names.append(dataset_name)
        log.info(f"Loaded dataset: {dataset_name} (split: {actual_split}, size: {len(dataset.x)})")
    
    return datasets, dataset_names


def collect_all_model_logits(
    models: List[WhiteboxModel],
    datasets: List[Dataset],
    option_tokens: List[str] = ["A", "B", "C", "D"],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Collect logits from all models on all datasets.
    
    Args:
        models: List of WhiteboxModel instances
        datasets: List of Dataset instances (will be concatenated)
        option_tokens: List of option tokens (e.g., ["A", "B", "C", "D"])
        
    Returns:
        Tuple of:
            - all_logits: np.ndarray of shape [num_models, num_examples, num_options]
            - labels: np.ndarray of shape [num_examples]
            - model_names: List of model names
    """
    # Concatenate all datasets
    all_x = []
    all_y = []
    for dataset in datasets:
        all_x.extend(dataset.x)
        all_y.extend(dataset.y)
    
    combined_dataset = Dataset(all_x, all_y, batch_size=datasets[0].batch_size)
    
    # Collect logits from each model
    all_model_logits = []
    model_names = []
    
    for model in models:
        logits, labels = collect_option_logits_for_model(
            model, combined_dataset, option_tokens
        )
        all_model_logits.append(logits)
        model_names.append(model.model_path.replace("/", "_"))
        log.info(f"Collected logits from {model.model_path}: shape {logits.shape}")
    
    # Stack logits: [num_models, num_examples, num_options]
    all_logits = np.stack(all_model_logits, axis=0)
    
    return all_logits, np.array(all_y, dtype=np.int32), model_names
