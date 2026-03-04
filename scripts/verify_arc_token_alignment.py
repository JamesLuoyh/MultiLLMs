#!/usr/bin/env python3
"""
Verification script to check if ARC-Easy token IDs align with bayesian-peft format.

This script:
1. Loads ARC-Easy dataset with the new prompt format
2. Resolves token IDs for answer keys (A, B, C, D)
3. Displays the token IDs and token names for verification
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lm_polygraph.utils.dataset import Dataset
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.utils.multi_llm_ensemble import _resolve_option_token_ids
from lm_polygraph.utils.config_loader import load_and_merge_configs

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("lm_polygraph")


def verify_token_alignment(config_path: str, model_path: str = None):
    """
    Verify that token IDs for ARC-Easy answer keys are correctly resolved.
    
    Args:
        config_path: Path to evaluation config file
        model_path: Optional model path (if not in config)
    """
    log.info(f"Loading config from {config_path}")
    config_path_obj = Path(config_path)
    config = load_and_merge_configs(config_path_obj)
    
    # Load OOD dataset (ARC-Easy)
    # The config loader processes _include_ood_dataset, but we need to check if ood_dataset exists
    # If not, we need to load it manually from the original config file
    
    base_dir = config_path_obj.parent
    
    if "ood_dataset" in config:
        # Config loader already processed it, but we need the base config
        # Load the original config to get _include_ood_dataset
        import yaml
        with open(config_path_obj, 'r') as f:
            raw_config = yaml.safe_load(f)
        
        if "_include_ood_dataset" in raw_config:
            ood_dataset_path = base_dir / raw_config["_include_ood_dataset"]
            if not ood_dataset_path.exists():
                ood_dataset_path = base_dir / "datasets" / Path(raw_config["_include_ood_dataset"]).name
            
            log.info(f"Loading OOD dataset config from {ood_dataset_path}")
            with open(ood_dataset_path, 'r') as f:
                ood_dataset_cfg = yaml.safe_load(f)
            
            # Merge with any overrides from main config
            ood_dataset_cfg.update(config["ood_dataset"])
        else:
            # Use ood_dataset directly
            ood_dataset_cfg = config["ood_dataset"]
        
        # Load the dataset using load_datasets_from_config
        from lm_polygraph.utils.wagering_utils import load_datasets_from_config
        datasets, dataset_names = load_datasets_from_config([ood_dataset_cfg], split="test")
        arc_dataset = datasets[0] if datasets else None
    else:
        log.error("No OOD dataset found in config (looking for 'ood_dataset')")
        log.info(f"Available config keys: {list(config.keys())}")
        return
    
    if arc_dataset is None or len(arc_dataset.x) == 0:
        log.error("ARC-Easy dataset not found or empty")
        return
    
    log.info(f"Found ARC-Easy dataset: {dataset_names[0] if dataset_names else 'unknown'} with {len(arc_dataset.x)} examples")
    
    # Load a model (use first model from config)
    from lm_polygraph.utils.wagering_utils import load_models_from_config
    
    # The config loader processes _include_models and creates config["models"] as a list of model config dicts
    # load_models_from_config expects model_configs: List[Dict[str, Any]] directly
    if "models" in config and isinstance(config["models"], list) and len(config["models"]) > 0:
        # Pass the list directly - load_models_from_config expects List[Dict]
        models = load_models_from_config(config["models"])
    else:
        log.error("No models found in config")
        log.info(f"Config keys: {list(config.keys())}")
        log.info(f"Config models: {config.get('models', 'NOT FOUND')}")
        return
    
    if not models:
        log.error("Failed to load models")
        return
    
    # load_models_from_config returns (models, model_names) tuple
    if isinstance(models, tuple) and len(models) == 2:
        model_list, model_names = models
        model = model_list[0]
        log.info(f"Using model: {model_names[0] if model_names else 'unknown'}")
    else:
        # Assume it's just a list
        model = models[0]
        log.info(f"Using model: {getattr(model, 'model_path', getattr(model, 'name', 'unknown'))}")
    
    # Get a sample prompt
    if len(arc_dataset.x) == 0:
        log.error("ARC-Easy dataset is empty")
        return
    
    sample_prompt = arc_dataset.x[0]
    log.info(f"\nSample prompt (first 500 chars):\n{sample_prompt[:500]}...")
    
    # Resolve token IDs
    option_tokens = ["A", "B", "C", "D"]
    log.info(f"\nResolving token IDs for options: {option_tokens}")
    log.info(f"Using prompt context: '{sample_prompt.rsplit('Answer:', 1)[0] + 'Answer: '}'")
    
    token_ids = _resolve_option_token_ids(model, option_tokens, sample_prompt=sample_prompt)
    
    # Display results
    log.info("\n" + "="*60)
    log.info("Token ID Resolution Results:")
    log.info("="*60)
    
    for opt, tid in zip(option_tokens, token_ids):
        token_name = model.tokenizer.convert_ids_to_tokens([tid])[0]
        # Also check what "Answer: " + opt tokenizes to
        test_str = f"Answer: {opt}"
        test_ids = model.tokenizer.encode(test_str, add_special_tokens=False)
        log.info(f"\nOption: {opt}")
        log.info(f"  Resolved token ID: {tid}")
        log.info(f"  Token name: {token_name}")
        log.info(f"  Test encoding 'Answer: {opt}': {test_ids}")
        if len(test_ids) > 0:
            log.info(f"  Last token ID from 'Answer: {opt}': {test_ids[-1]}")
            if test_ids[-1] == tid:
                log.info(f"  ✓ Token ID matches 'Answer: {opt}' context")
            else:
                log.warning(f"  ✗ Token ID does NOT match 'Answer: {opt}' context")
    
    log.info("\n" + "="*60)
    log.info("Verification complete!")
    log.info("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify ARC-Easy token ID alignment")
    parser.add_argument(
        "config",
        type=str,
        help="Path to evaluation config file (e.g., examples/configs/wagering_training/eval_mmlu_medmcqa_arc.yaml)"
    )
    
    args = parser.parse_args()
    verify_token_alignment(args.config)
