#!/usr/bin/env python3
"""
Evaluation script for multi-LLM wagering methods.

Checks if wandb.run is active from training and continues it.

Usage: python wagering_eval.py <config_file.yaml>
"""

import logging
import os
import sys
import yaml
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Ensure the local src/ tree and wagering package are importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wagering.utils import load_models_from_config, load_datasets_from_config, load_and_merge_configs
from wagering.utils.multi_llm_ensemble import get_cached_logits_and_hidden_states_for_model
from wagering.methods.factory import load_wagering_method
from wagering.inference import WageringEvaluator
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


def main(config_path: Optional[str] = None, checkpoint_path: Optional[str] = None):
    """Main evaluation function."""
    if config_path is None:
        if len(sys.argv) > 1:
            config_path = sys.argv[1]
        else:
            raise ValueError(
                "Config file path required. "
                "Usage: python wagering_eval.py <config_file.yaml>"
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
    
    # Load API keys
    api_keys = load_api_keys_from_config()
    
    # Initialize wandb
    wandb_logger = None
    if args.get("report_to_wandb", False):
        try:
            import wandb
            
            wandb_api_key = api_keys.get("wandb_api_key")
            if wandb_api_key:
                os.environ["WANDB_API_KEY"] = wandb_api_key
            
            # Check if wandb is already active from training
            if wandb.run is not None:
                log.debug(f"Continuing wandb run from training: {wandb.run.id}")
                wandb_logger = wandb
            else:
                log.info("Creating new wandb run for evaluation")
                wandb.init(
                    project=args.get("wandb_project", "multi-llm-wagering"),
                    entity=args.get("wandb_entity", None),
                    name=args.get("wandb_name", None),
                    tags=["evaluation"],
                )
                wandb_logger = wandb
            
            # Add evaluation tag
            if wandb.run and "evaluation" not in (wandb.run.tags or []):
                wandb.run.tags = list(wandb.run.tags or []) + ["evaluation"]
                    
        except ImportError as e:
            raise RuntimeError("wandb not available but report_to_wandb is enabled in config") from e
    
    # Load checkpoint path
    if checkpoint_path is None:
        checkpoint_path = args.get("checkpoint_path")
    if checkpoint_path is None:
        log.error("Please provide a checkpoint path in config file")
        sys.exit(1)
    
    checkpoint_path = Path(checkpoint_path)
    
    # Load test datasets
    test_datasets = []
    if "test_datasets" in args:
        test_ds, test_names = load_datasets_from_config(
            args["test_datasets"],
            split="test",
        )
        test_datasets = [(ds, name) for ds, name in zip(test_ds, test_names)]

    # Load OOD dataset
    ood_dataset = None
    if "ood_dataset" in args and args["ood_dataset"]:
        ood_configs = [args["ood_dataset"]] if isinstance(args["ood_dataset"], dict) else args["ood_dataset"]
        ood_ds, ood_names = load_datasets_from_config(ood_configs, split="test")
        if ood_ds:
            ood_dataset = (ood_ds[0], ood_names[0])

    # Load wagering method (before models so we can decide cache requirements)
    wagering_config = args["wagering_method"]
    num_models = len(args["models"])
    wagering_method = load_wagering_method(
        wagering_config["name"],
        num_models=num_models,
        config=wagering_config.get("config", {}),
    )

    # Determine if hidden states are required
    wagering_method_name = type(wagering_method).__name__
    needs_hidden_states = wagering_method_name not in ["EqualWagers", "ZeroOneWagers", "OneZeroWagers"]

    # Check cache per model across all eval datasets
    option_tokens = args.get("option_tokens", ["A", "B", "C", "D"])
    model_cfgs = args["models"]

    eval_datasets = [ds for ds, _ in test_datasets]
    if ood_dataset is not None:
        eval_datasets.append(ood_dataset[0])

    cache_miss_indices = []
    cached_model_names = []
    cached_hidden_dims = {}
    for idx, model_cfg in enumerate(model_cfgs):
        model_path = model_cfg["path"]
        model_cached = True
        for ds in eval_datasets:
            cached_logits, cached_hidden_states, _ = get_cached_logits_and_hidden_states_for_model(
                model_path, ds, option_tokens
            )
            if cached_logits is None or (needs_hidden_states and cached_hidden_states is None):
                model_cached = False
                break
            if needs_hidden_states and cached_hidden_states is not None and idx not in cached_hidden_dims:
                try:
                    cached_hidden_dims[idx] = cached_hidden_states.shape[1]
                except Exception:
                    pass
        if not model_cached:
            cache_miss_indices.append(idx)
        cached_model_names.append(model_path.replace("/", "_"))

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
        log.debug("All models are cached for evaluation. Skipping model loading.")
        models = [cfg["path"] for cfg in model_cfgs]

    log.debug(f"Prepared {len(models)} models: {model_names}")

    # Log hidden dimensions for debugging projection key mismatches
    if needs_hidden_states:
        log.debug("Model hidden dimensions:")
        for i in range(num_models):
            if i in cached_hidden_dims:
                hidden_dim = cached_hidden_dims[i]
                log.debug(f"  Model {i} ({model_names[i]}): hidden_dim={hidden_dim}, expected proj key=proj_{i}")
            elif i in cache_miss_indices:
                log.debug(f"  Model {i} ({model_names[i]}): hidden_dim=unknown (cache miss, will compute during eval)")
    
    # Load checkpoint
    # Check if the method has trainable parameters
    requires_checkpoint = len(wagering_method.get_trainable_parameters()) > 0
    baseline_wagering_method = None
    
    if requires_checkpoint:
        log.info(f"\n{'='*80}")
        log.info("CHECKPOINT LOADING")
        log.info(f"{'='*80}")
        
        # Training returns the final checkpoint directory path directly
        # So we just need to load wagering_state.pt from that directory
        checkpoint_file = checkpoint_path / "final" / "wagering_state.pt"
        
        log.debug(f"Looking for checkpoint at: {checkpoint_file}")
        log.debug(f"Checkpoint exists: {checkpoint_file.exists()}")
        
        if not checkpoint_file.exists():
            log.error(f"✗ Checkpoint file not found: {checkpoint_file}")
            log.error(f"   Training should save the best checkpoint to this location.")
            log.error(f"   Received checkpoint_path: {checkpoint_path}")
            log.error(f"   Directory contents: {list(checkpoint_path.iterdir()) if checkpoint_path.exists() else 'directory does not exist'}")
            sys.exit(1)
        
        import torch

        def _nested_tensor_sum(obj) -> float:
            total = 0.0
            if isinstance(obj, dict):
                for v in obj.values():
                    total += _nested_tensor_sum(v)
            elif torch.is_tensor(obj):
                total += float(obj.detach().sum().cpu().item())
            return total

        def _subset_tensor_sum(obj, keys) -> float:
            total = 0.0
            if isinstance(obj, dict):
                for k in keys:
                    if k in obj:
                        total += _nested_tensor_sum(obj[k])
            return total
        
        # Get state dict before loading for comparison
        state_before = wagering_method.state_dict()
        log.debug(f"State dict keys before loading: {list(state_before.keys())}")
        if state_before:
            first_key = list(state_before.keys())[0]
            if hasattr(state_before[first_key], 'sum'):
                log.debug(f"  Sample parameter '{first_key}' sum before: {state_before[first_key].sum():.6f}")
        
        log.debug(f"Loading checkpoint from {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location="cpu")
        
        log.debug(f"Checkpoint type: {type(checkpoint)}")
        if isinstance(checkpoint, dict):
            log.debug(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        if isinstance(checkpoint, dict) and "wagering_method_state" in checkpoint:
            log.debug("Loading from checkpoint['wagering_method_state']")
            wagering_method.load_state_dict(checkpoint["wagering_method_state"])
            checkpoint_state = checkpoint["wagering_method_state"]
        else:
            log.debug("Loading checkpoint directly as state_dict")
            wagering_method.load_state_dict(checkpoint)
            checkpoint_state = checkpoint

        # Verify loaded state matches checkpoint (routers + projections)
        try:
            loaded_state = wagering_method.state_dict()
            ckpt_routers = checkpoint_state.get("routers_state_dict", {})
            ckpt_projs = checkpoint_state.get("model_projections_state_dict", {})
            loaded_routers = loaded_state.get("routers_state_dict", {})
            loaded_projs = loaded_state.get("model_projections_state_dict", {})

            ckpt_router_keys = set(ckpt_routers.keys())
            loaded_router_keys = set(loaded_routers.keys())
            ckpt_proj_keys = set(ckpt_projs.keys())
            loaded_proj_keys = set(loaded_projs.keys())

            missing_router_keys = ckpt_router_keys - loaded_router_keys
            extra_router_keys = loaded_router_keys - ckpt_router_keys
            missing_proj_keys = ckpt_proj_keys - loaded_proj_keys
            extra_proj_keys = loaded_proj_keys - ckpt_proj_keys

            if missing_router_keys or extra_router_keys or missing_proj_keys or extra_proj_keys:
                log.error("Checkpoint/model key mismatch detected:")
                if missing_router_keys:
                    log.error(f"  Missing router keys in model: {sorted(missing_router_keys)}")
                if extra_router_keys:
                    log.error(f"  Extra router keys in model: {sorted(extra_router_keys)}")
                if missing_proj_keys:
                    log.error(f"  Missing projection keys in model: {sorted(missing_proj_keys)}")
                    log.error(f"  This means the models loaded for evaluation have different hidden dimensions")
                    log.error(f"  than the models used during training.")
                    log.error(f"  Checkpoint expects: {sorted(ckpt_proj_keys)}")
                    log.error(f"  Current model has: {sorted(loaded_proj_keys)}")
                if extra_proj_keys:
                    log.error(f"  Extra projection keys in model: {sorted(extra_proj_keys)}")
                log.error("Refusing to evaluate with mismatched checkpoint/model state.")
                log.error("Solution: Ensure the same models are specified in the config for training and evaluation.")
                sys.exit(1)

            ckpt_sum = _subset_tensor_sum(ckpt_routers, ckpt_router_keys) + _subset_tensor_sum(ckpt_projs, ckpt_proj_keys)
            loaded_sum = _subset_tensor_sum(loaded_routers, ckpt_router_keys) + _subset_tensor_sum(loaded_projs, ckpt_proj_keys)
            log.debug(f"Checkpoint tensors sum (routers+projections): {ckpt_sum:.6f}")
            log.debug(f"Loaded tensors sum (routers+projections):    {loaded_sum:.6f}")
            if not torch.isclose(torch.tensor(ckpt_sum), torch.tensor(loaded_sum), rtol=1e-5, atol=1e-5):
                raise RuntimeError("Loaded state checksum does not match checkpoint checksum. This indicates corruption or incorrect checkpoint loading.")
            else:
                log.debug("✓ Loaded state checksum matches checkpoint")
        except Exception as e:
            raise Exception(f"Could not verify loaded state checksum: {e}")
        
        # Verify state changed
        state_after = wagering_method.state_dict()
        if state_before and state_after:
            first_key = list(state_after.keys())[0]
            if hasattr(state_after[first_key], 'sum'):
                log.info(f"  Sample parameter '{first_key}' sum after: {state_after[first_key].sum():.6f}")
                before_sum = state_before[first_key].sum()
                after_sum = state_after[first_key].sum()
                if torch.allclose(before_sum, after_sum):
                    raise Exception("⚠ WARNING: State dict appears unchanged after loading!")
                else:
                    log.info("✓ State dict successfully updated")
        
        log.debug(f"✓ Successfully loaded wagering method checkpoint")
        log.debug(f"{'='*80}\n")
    else:
        log.info("Wagering method has no trainable parameters - skipping checkpoint loading")

    # Load aggregation function
    aggregation_config = args["aggregation"]
    aggregation_function = load_aggregation_function(
        aggregation_config["name"],
        config=aggregation_config.get("config", {}),
    )
    
    # Set up evaluation checkpoint directory
    eval_checkpoint_dir = None
    if args.get("eval_checkpoint_dir"):
        eval_checkpoint_dir = Path(args["eval_checkpoint_dir"])
    elif checkpoint_path:
        eval_checkpoint_dir = checkpoint_path / "eval"
    else:
        eval_checkpoint_dir = Path("./eval_outputs")
    
    eval_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Evaluation outputs: {eval_checkpoint_dir}")
    
    # Prepare metadata
    metadata = {"model_names": model_names}
    training_datasets = args.get("training_datasets", [])
    if training_datasets:
        if isinstance(training_datasets, str):
            training_datasets = [training_datasets]
        elif isinstance(training_datasets, list) and training_datasets and isinstance(training_datasets[0], dict):
            training_datasets = [ds.get("name", ds.get("path", str(ds))) for ds in training_datasets]
        metadata["training_datasets"] = training_datasets
    
    seed = args.get("seed", args.get("shuffle_seed", None))
    
    # Get starting step from wandb if active
    wandb_starting_step = 0
    if wandb_logger and wandb.run is not None:
        try:
            wandb_starting_step = wandb.run.step
            log.info(f"Continuing from wandb step {wandb_starting_step}")
        except Exception as e:
            log.warning(f"Could not get wandb step: {e}")
    
    # Create evaluator
    evaluator = WageringEvaluator(
        models=models,
        wagering_method=wagering_method,
        aggregation_function=aggregation_function,
        option_tokens=args.get("option_tokens", ["A", "B", "C", "D"]),
        wandb_logger=wandb_logger,
        checkpoint_dir=eval_checkpoint_dir,
        metadata=metadata,
        training_checkpoint_path=str(checkpoint_path),
        seed=seed,
        wandb_starting_step=wandb_starting_step,
    )
    
    # Evaluate
    log.info("Starting evaluation...")
    log.info(f"  Test datasets: {len(test_datasets)}")
    if ood_dataset:
        log.info(f"  OOD dataset: {ood_dataset[1]}")
    
    results = evaluator.evaluate_multiple(
        test_datasets=test_datasets,
        ood_dataset=ood_dataset,
        resume=False,  # Always evaluate from scratch
    )

    # Print results
    log.info("Evaluation Results:")
    results_summary = {}
    aggregate_metrics = {
        "accuracy": [],
        "nll": [],
        "auc": [],
        "ece": [],
        "d_regret": [],
        "meta_acc": [],
        "meta_nll": [],
        "meta_auc": [],
    }
    
    for dataset_name, result in results.items():
        def get_metric_str(result, key):
            val = result.get(key, None)
            if val is None:
                return "N/A"
            try:
                val_scalar = float(val) if not isinstance(val, (list, np.ndarray)) else float(val[0])
                return f"{val_scalar:.4f}" if not np.isnan(val_scalar) else "N/A"
            except (ValueError, TypeError):
                return "N/A"
        
        def get_metric_float(result, key):
            val = result.get(key, None)
            if val is None:
                return None
            try:
                val_scalar = float(val) if not isinstance(val, (list, np.ndarray)) else float(val[0])
                return None if np.isnan(val_scalar) else val_scalar
            except (ValueError, TypeError):
                return None

        accuracy_val = get_metric_float(result, "accuracy")
        nll_val = get_metric_float(result, "nll")
        auc_val = get_metric_float(result, "auc")
        ece_val = get_metric_float(result, "ece")
        d_regret_val = get_metric_float(result, "d_regret")
        meta_acc_val = get_metric_float(result, "meta_acc")
        meta_nll_val = get_metric_float(result, "meta_nll")
        meta_auc_val = get_metric_float(result, "meta_auc")

        if accuracy_val is not None:
            aggregate_metrics["accuracy"].append(accuracy_val)
        if nll_val is not None:
            aggregate_metrics["nll"].append(nll_val)
        if auc_val is not None:
            aggregate_metrics["auc"].append(auc_val)
        if ece_val is not None:
            aggregate_metrics["ece"].append(ece_val)
        if d_regret_val is not None:
            aggregate_metrics["d_regret"].append(d_regret_val)
        if meta_acc_val is not None:
            aggregate_metrics["meta_acc"].append(meta_acc_val)
        if meta_nll_val is not None:
            aggregate_metrics["meta_nll"].append(meta_nll_val)
        if meta_auc_val is not None:
            aggregate_metrics["meta_auc"].append(meta_auc_val)

        accuracy_str = get_metric_str(result, "accuracy")
        nll_str = get_metric_str(result, "nll")
        auc_str = get_metric_str(result, "auc")
        ece_str = get_metric_str(result, "ece")
        d_regret_str = get_metric_str(result, "d_regret")
        meta_acc_str = get_metric_str(result, "meta_acc")
        meta_nll_str = get_metric_str(result, "meta_nll")
        meta_auc_str = get_metric_str(result, "meta_auc")
        
        log.info(
            f"{dataset_name}: Accuracy={accuracy_str}, "
            f"NLL={nll_str}, AUC={auc_str}, ECE={ece_str}, "
            f"DRegret={d_regret_str}, MetaAcc={meta_acc_str}, MetaNLL={meta_nll_str}, MetaAUC={meta_auc_str}"
        )
        
        results_summary[dataset_name] = f"Accuracy={accuracy_str}, AUC={auc_str}, ECE={ece_str}, NLL={nll_str}, DRegret={d_regret_str}, MetaAcc={meta_acc_str}, MetaAuc={meta_auc_str}, MetaNLL={meta_nll_str}"

    def aggregate_metric_str(values):
        if not values:
            return "N/A"
        mean_val = float(np.mean(values))
        return f"{mean_val:.4f}" if not np.isnan(mean_val) else "N/A"

    overall_accuracy = aggregate_metric_str(aggregate_metrics["accuracy"])
    overall_nll = aggregate_metric_str(aggregate_metrics["nll"])
    overall_auc = aggregate_metric_str(aggregate_metrics["auc"])
    overall_ece = aggregate_metric_str(aggregate_metrics["ece"])
    overall_d_regret = aggregate_metric_str(aggregate_metrics["d_regret"])
    overall_meta_acc = aggregate_metric_str(aggregate_metrics["meta_acc"])
    overall_meta_nll = aggregate_metric_str(aggregate_metrics["meta_nll"])
    overall_meta_auc = aggregate_metric_str(aggregate_metrics["meta_auc"])

    results_summary["overall"] = (
        f"Accuracy={overall_accuracy}, AUC={overall_auc}, ECE={overall_ece}, "
        f"NLL={overall_nll}, DRegret={overall_d_regret}, MetaAcc={overall_meta_acc}, "
        f"MetaAuc={overall_meta_auc}, MetaNLL={overall_meta_nll}"
    )
    
    # Log summary to wandb
    if wandb_logger and wandb.run is not None:
        summary_df = pd.DataFrame([
            {"Dataset": name, "Metrics": metrics} 
            for name, metrics in results_summary.items()
        ])
        # Use commit=True to ensure table is logged independently without step tracking
        wandb.log({"evaluation/results_summary": wandb.Table(dataframe=summary_df)}, commit=True)
    
    # Finish wandb
    if wandb_logger and wandb.run is not None:
        log.info(f"Finishing wandb run: {wandb.run.id}")
        wandb.finish()
    
    return results


if __name__ == "__main__":
    main()
