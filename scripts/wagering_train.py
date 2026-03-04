#!/usr/bin/env python3
"""
Training script for multi-LLM wagering methods.

Does NOT call wandb.finish() to keep the run active for evaluation.

Usage: python wagering_train.py <config_file.yaml>
"""

import logging
import os
import sys
import yaml
from pathlib import Path
from typing import Optional

# Ensure the local src/ tree and wagering package are importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wagering.utils import (
    load_models_from_config,
    load_datasets_from_config,
    load_and_merge_configs,
    generate_checkpoint_dir,
    get_checkpoint_metadata,
)
from wagering.utils.multi_llm_ensemble import get_cached_logits_and_hidden_states_for_model
from lm_polygraph.utils.dataset import Dataset
from wagering.methods.factory import load_wagering_method
from wagering.training import WageringTrainer
from wagering.aggregation.factory import load_aggregation_function

log = logging.getLogger("wagering")


def load_api_keys_from_config():
    """Load API keys from .api_keys.yaml file if it exists."""
    api_keys_path = PROJECT_ROOT / ".api_keys.yaml"
    if not api_keys_path.exists():
        return {}
    
    with open(api_keys_path, "r") as f:
        config = yaml.safe_load(f)
        if not config:
            return {}
        
        filtered = {}
        for k, v in config.items():
            if v is None or v == "null" or v == "":
                continue
            if isinstance(v, str) and ("your-" in v.lower() and "-here" in v.lower()):
                continue
            filtered[k] = v
        return filtered


def main(config_path: Optional[str] = None):
    """Main training function."""
    if config_path is None:
        if len(sys.argv) > 1:
            config_path = sys.argv[1]
        else:
            raise ValueError(
                "Config file path required. "
                "Usage: python wagering_train.py <config_file.yaml>"
            )
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load config
    args = load_and_merge_configs(config_path)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
    
    # Suppress verbose library logging
    logging.getLogger("lm_polygraph").setLevel(logging.INFO)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    # Generate unique checkpoint directory
    base_checkpoint_dir = Path(args.get("checkpoint_base_dir", "/common/users/yl2310/MultiLLMs/checkpoints"))
    checkpoint_dir = generate_checkpoint_dir(
        base_dir=base_checkpoint_dir,
        models=args["models"],
        datasets=args["datasets"],
        wagering_method=args["wagering_method"],
        aggregation=args["aggregation"],
        create_hash=True,
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Checkpoint directory: {checkpoint_dir}")
    
    # Get metadata
    checkpoint_metadata = get_checkpoint_metadata(
        models=args["models"],
        datasets=args["datasets"],
        wagering_method=args["wagering_method"],
        aggregation=args["aggregation"],
    )
    
    # Initialize wandb
    wandb_logger = None
    if args.get("report_to_wandb", False):
        try:
            import wandb
            api_keys = load_api_keys_from_config()
            wandb_api_key = api_keys.get("wandb_api_key")
            if wandb_api_key:
                os.environ["WANDB_API_KEY"] = wandb_api_key
            
            wandb_config = args.copy()
            wandb_config.update(checkpoint_metadata)
            
            wandb.init(
                project=args.get("wandb_project", "multi-llm-wagering"),
                entity=args.get("wandb_entity", None),
                name=args.get("wandb_name", None),
                config=wandb_config,
                tags=[
                    f"wagering_{args['wagering_method']['name']}",
                    f"agg_{args['aggregation']['name']}",
                    f"models_{len(args['models'])}",
                    f"datasets_{len(args['datasets'])}",
                    "training",
                ],
            )
            wandb_logger = wandb
            log.info(f"Initialized wandb run: {wandb.run.id}")
        except ImportError as e:
            raise RuntimeError("wandb not available but report_to_wandb is enabled in config") from e
    
    # Load training datasets
    log.info("Loading training datasets...")
    train_datasets, dataset_names = load_datasets_from_config(
        args["datasets"],
        split="train",
    )
    log.info(f"Loaded {len(train_datasets)} training datasets: {dataset_names}")

    # Cache checks are performed per-dataset (matches trainer logic)

    # Load wagering method early so we can decide cache requirements
    wagering_config = args["wagering_method"]
    num_models = len(args["models"])
    wagering_method = load_wagering_method(
        wagering_config["name"],
        num_models=num_models,
        config=wagering_config.get("config", {}),
    )
    log.info(f"Loaded wagering method: {wagering_config['name']}")

    wagering_method_name = type(wagering_method).__name__
    needs_hidden_states = wagering_method_name not in ["EqualWagers", "ZeroOneWagers", "OneZeroWagers"]

    # Determine which models need to be loaded based on cache
    option_tokens = args.get("option_tokens", ["A", "B", "C", "D"])
    model_cfgs = args["models"]

    cache_miss_indices = []
    cached_model_names = []
    for idx, model_cfg in enumerate(model_cfgs):
        model_path = model_cfg["path"]
        cached_model_names.append(model_path.replace("/", "_"))

        model_cache_ok = True
        for dataset in train_datasets:
            cached_logits, cached_hidden_states, _ = get_cached_logits_and_hidden_states_for_model(
                model_path, dataset, option_tokens
            )
            if cached_logits is None or (needs_hidden_states and cached_hidden_states is None):
                model_cache_ok = False
                break

        if not model_cache_ok:
            cache_miss_indices.append(idx)

    models = []
    model_names = cached_model_names[:]
    if cache_miss_indices:
        log.info(f"Cache miss for {len(cache_miss_indices)}/{num_models} models. Loading missing models...")
        missing_cfgs = [model_cfgs[i] for i in cache_miss_indices]
        missing_models, missing_names = load_models_from_config(
            missing_cfgs,
            cache_kwargs={"cache_dir": args.get("cache_path", "./workdir/cache")} if args.get("cache_path") else {},
        )
        missing_name_map = {idx: name for idx, name in zip(cache_miss_indices, missing_names)}
        missing_iter = iter(missing_models)
        for i in range(num_models):
            if i in cache_miss_indices:
                models.append(next(missing_iter))
                model_names[i] = missing_name_map.get(i, model_names[i])
            else:
                models.append(model_cfgs[i]["path"])
    else:
        log.info("All models are cached. Skipping model loading.")
        models = [cfg["path"] for cfg in model_cfgs]

    log.info(f"Prepared {len(models)} models: {model_names}")
    
    # Load aggregation function
    aggregation_config = args["aggregation"]
    aggregation_function = load_aggregation_function(
        aggregation_config["name"],
        config=aggregation_config.get("config", {}),
    )
    log.info(f"Loaded aggregation function: {aggregation_config['name']}")
    
    # Check for resume checkpoint
    auto_resume = args.get("auto_resume", True)
    resume_checkpoint = args.get("resume_from_checkpoint", None)
    
    if resume_checkpoint:
        resume_path = Path(resume_checkpoint)
        if not resume_path.is_absolute():
            resume_path = checkpoint_dir / resume_checkpoint
        if resume_path.exists():
            log.info(f"Resuming from checkpoint: {resume_path}")
            resume_checkpoint = str(resume_path)
        else:
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    elif auto_resume:
        checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*_step_*.pt"))
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
            log.info(f"Auto-resuming from: {latest_checkpoint}")
            resume_checkpoint = str(latest_checkpoint)
    
    # Create trainer
    trainer = WageringTrainer(
        models=models,
        datasets=train_datasets,
        wagering_method=wagering_method,
        aggregation_function=aggregation_function,
        option_tokens=args.get("option_tokens", ["A", "B", "C", "D"]),
        checkpoint_dir=checkpoint_dir,
        wandb_logger=wandb_logger,
        save_every=args.get("save_every", 100),
        metadata=checkpoint_metadata,
        resume_from_checkpoint=resume_checkpoint,
        shuffle_data=args.get("shuffle_data", True),
        shuffle_seed=args.get("shuffle_seed", 42),
        early_stopping_patience=args.get("early_stopping_patience", 10),
        batch_size=args.get("training_batch_size", 100),
        validation_split_ratio=args.get("validation_split_ratio", 0.1),
    )
    
    # Train
    log.info("Starting training...")
    results = trainer.train(num_epochs=args.get("num_epochs", 100))
    
    log.info(f"Training complete! Final accuracy: {results['final_accuracy']:.4f}")
    
    # Save final checkpoint
    final_checkpoint_dir = Path(checkpoint_dir) / "final"
    trainer.save_final_checkpoint(str(final_checkpoint_dir))
    
    # Return results (wandb run stays active)
    results["checkpoint_path"] = str(checkpoint_dir)
    if wandb_logger and hasattr(wandb_logger, 'run') and wandb_logger.run is not None:
        results["wandb_run_id"] = wandb_logger.run.id
        results["wandb_run_name"] = wandb_logger.run.name
    
    return results


if __name__ == "__main__":
    main()
