"""
Model loading utilities.

Simplified version with strict error handling.
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add src/ to path for lm_polygraph imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.utils.common import load_external_module
from lm_polygraph.utils.generation_parameters import GenerationParametersFactory

log = logging.getLogger(__name__)


def load_api_keys() -> Dict[str, str]:
    """
    Load API keys from .api_keys.yaml file if it exists.
    
    Returns:
        Dictionary of API keys
    """
    api_keys_path = PROJECT_ROOT / ".api_keys.yaml"
    if not api_keys_path.exists():
        return {}
    
    try:
        import yaml
        with open(api_keys_path, "r") as f:
            config = yaml.safe_load(f)
            if not config:
                return {}
            
            # Filter out invalid values
            filtered = {}
            for k, v in config.items():
                if v is None or v == "null" or v == "":
                    continue
                if isinstance(v, str) and ("your-" in v.lower() and "-here" in v.lower()):
                    continue
                filtered[k] = v
            return filtered
    except Exception as e:
        log.warning(f"Could not load API keys: {e}")
        return {}


def load_models_from_config(
    model_configs: List[Dict[str, Any]],
    cache_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[WhiteboxModel], List[str]]:
    """
    Load multiple whitebox models from configuration.
    
    Args:
        model_configs: List of model configuration dictionaries, each containing:
            - path: Model path (REQUIRED)
            - path_to_load_script: Path to load script (optional, for custom loading)
            - load_model_args: Arguments for model loading (optional)
            - load_tokenizer_args: Arguments for tokenizer loading (optional)
            - instruct: Whether model is instruction-tuned (optional, default False)
            - generation_params: Generation parameters (optional)
        cache_kwargs: Optional cache kwargs for model loading
        
    Returns:
        Tuple of (list of WhiteboxModel instances, list of model names)
        
    Raises:
        ValueError: If model config is invalid
        FileNotFoundError: If load script not found
    """
    if not model_configs:
        raise ValueError("Must provide at least one model config")
    
    cache_kwargs = cache_kwargs or {}
    models = []
    model_names = []
    
    # Load API keys for HF token
    api_keys = load_api_keys()
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if hf_token is None:
        hf_token = api_keys.get("hf_token") or api_keys.get("huggingface_token")
    
    for i, model_cfg in enumerate(model_configs):
        if "path" not in model_cfg:
            raise ValueError(f"Model config {i} missing required 'path' field: {model_cfg}")
        
        model_path = model_cfg["path"]
        model_names.append(model_path.replace("/", "_"))
        
        # Use custom load script if provided
        if model_cfg.get("path_to_load_script"):
            load_script_path = model_cfg["path_to_load_script"]
            
            # Resolve load script path
            if not os.path.isabs(load_script_path):
                # Try relative to examples/configs/
                examples_path = PROJECT_ROOT / "examples" / "configs" / load_script_path
                if examples_path.exists():
                    load_script_path = str(examples_path)
                else:
                    raise FileNotFoundError(
                        f"Load script not found: {load_script_path} "
                        f"(tried {examples_path})"
                    )
            
            if not Path(load_script_path).exists():
                raise FileNotFoundError(f"Load script not found: {load_script_path}")
            
            # Load the module
            load_module = load_external_module(str(load_script_path))
            
            # Load model
            load_model_args = {"model_path": model_path}
            load_model_args.update(model_cfg.get("load_model_args", {}))
            if hf_token is not None:
                load_model_args["token"] = hf_token
            
            base_model = load_module.load_model(**load_model_args)
            
            # Load tokenizer
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
            
            # Generation params
            generation_params = GenerationParametersFactory.from_params(
                yaml_config=model_cfg.get("generation_params", {}),
                native_config=base_model.generation_config.to_dict()
            )
            
            # Create WhiteboxModel
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
            device_map = model_cfg.get("load_model_args", {}).get("device_map", "auto")
            
            model = WhiteboxModel.from_pretrained(
                model_path,
                generation_params=model_cfg.get("generation_params", {}),
                device_map=device_map,
                add_bos_token=model_cfg.get("add_bos_token", True),
                instruct=instruct,
                **cache_kwargs,
            )
        
        models.append(model)
        log.info(f"Loaded model {i+1}/{len(model_configs)}: {model_path}")
    
    return models, model_names
