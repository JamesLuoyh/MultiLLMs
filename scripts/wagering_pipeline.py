#!/usr/bin/env python3
"""
End-to-end pipeline for multi-LLM wagering.

Keeps wandb run active between training and evaluation phases.

Usage: python wagering_pipeline.py <config_file.yaml>
"""

import logging
import os
import sys
import argparse
from pathlib import Path
from typing import Optional
import importlib.util

# Ensure the local src/ tree is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
SCRIPTS_PATH = PROJECT_ROOT / "scripts"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(SCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PATH))

# Import training and evaluation functions
def load_module_from_path(module_name, file_path):
    """Load a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

train_module = load_module_from_path("wagering_train", SCRIPTS_PATH / "wagering_train.py")
eval_module = load_module_from_path("wagering_eval", SCRIPTS_PATH / "wagering_eval.py")

train_main = train_module.main
eval_main = eval_module.main

log = logging.getLogger("lm_polygraph")


def run_pipeline(
    config_path: Optional[Path] = None,
    skip_training: bool = False,
    skip_evaluation: bool = False,
    checkpoint_path_override: Optional[str] = None,
):
    """Run end-to-end pipeline with unified wandb run."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(name)s - %(levelname)s - %(message)s'
    )
    
    # Suppress verbose library logging
    logging.getLogger("lm_polygraph").setLevel(logging.INFO)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    
    checkpoint_path = None
    
    # Training phase
    if not skip_training:
        log.info("\n" + "=" * 80)
        log.info("PHASE 1: TRAINING")
        log.info("=" * 80)
        
        try:
            train_results = train_main(config_path=str(config_path))
            checkpoint_path = train_results.get("checkpoint_path")
            
            import wandb
            if wandb.run is not None:
                log.debug(f"WandB run {wandb.run.id} active - continuing to evaluation")
                
        except Exception as e:
            log.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        log.info("Skipping training phase")
        checkpoint_path = checkpoint_path_override
    
    if checkpoint_path is None:
        log.error("No checkpoint path available for evaluation")
        sys.exit(1)
    
    # Evaluation phase
    if not skip_evaluation:
        log.info("\n" + "=" * 80)
        log.info("PHASE 2: EVALUATION")
        log.info("=" * 80)
        
        try:
            eval_results = eval_main(
                config_path=str(config_path),
                checkpoint_path=checkpoint_path
            )
            log.info("Evaluation complete")
            
        except Exception as e:
            log.error(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        log.info("Skipping evaluation phase")
    
    log.info("\n" + "=" * 80)
    log.info("PIPELINE COMPLETE")
    log.info("=" * 80)


def main():
    """Parse arguments and run pipeline."""
    parser = argparse.ArgumentParser(
        description="Run multi-LLM wagering pipeline"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to config file (YAML)"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training phase"
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation phase"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Override checkpoint path (use with --skip-training)"
    )
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        log.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    run_pipeline(
        config_path=config_path,
        skip_training=args.skip_training,
        skip_evaluation=args.skip_evaluation,
        checkpoint_path_override=args.checkpoint_path,
    )


if __name__ == "__main__":
    main()
