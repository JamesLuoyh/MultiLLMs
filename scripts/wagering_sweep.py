#!/usr/bin/env python3
"""
Script to run wagering pipeline with multiple random seeds and aggregate results.

This script runs the training and evaluation pipeline multiple times with different
random seeds, then aggregates the evaluation metrics across all runs.

Usage:
    python wagering_sweep.py <config_file.yaml> --seeds 42 43 44 45 46
    python wagering_sweep.py <config_file.yaml> --num-seeds 5 --seed-start 42
    python wagering_sweep.py <config_file.yaml> --seeds 42 43 44 --output-dir ./sweep_results
"""

import logging
import os
import sys
import argparse
import yaml
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any
import pandas as pd

# Ensure the local src/ tree is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
SCRIPTS_PATH = PROJECT_ROOT / "scripts"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(SCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PATH))

# Import pipeline and analytics
from wagering.training.analytics import WageringAnalytics
import importlib.util

def load_module_from_path(module_name, file_path):
    """Load a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

pipeline_module = load_module_from_path("wagering_pipeline", PROJECT_ROOT / "scripts" / "wagering_pipeline.py")
run_pipeline = pipeline_module.run_pipeline
main_pipeline = pipeline_module.main

log = logging.getLogger("lm_polygraph")


def update_config_with_seed(config_path: Path, seed: int, output_path: Path) -> Path:
    """
    Create a temporary config file with the specified seed.
    
    Args:
        config_path: Path to original config file
        seed: Random seed to use
        output_path: Path to save updated config
        
    Returns:
        Path to updated config file
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update shuffle_seed in config
    if 'shuffle_seed' not in config:
        config['shuffle_seed'] = seed
    else:
        config['shuffle_seed'] = seed
    
    # Also update seed if present
    if 'seed' in config:
        config['seed'] = seed
    
    # Save updated config
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return output_path


def find_analytics_files(checkpoint_base_dir: Path, pattern: str = "analytics*.csv") -> List[Path]:
    """
    Find all analytics CSV files in checkpoint directories.
    
    Args:
        checkpoint_base_dir: Base directory containing checkpoints
        pattern: Glob pattern to match analytics files
        
    Returns:
        List of paths to analytics files
    """
    analytics_files = []
    
    # Look for analytics files in checkpoint directories
    # Pattern: checkpoint_base_dir/*/analytics.csv (training)
    # Pattern: checkpoint_base_dir/*/eval_checkpoints/analytics_*.csv (evaluation)
    # Pattern: checkpoint_base_dir/*/eval_checkpoints/analytics_all.csv (combined evaluation)
    
    for checkpoint_dir in checkpoint_base_dir.iterdir():
        if not checkpoint_dir.is_dir():
            continue
        
        # Training analytics
        training_analytics = checkpoint_dir / "analytics.csv"
        if training_analytics.exists():
            analytics_files.append(training_analytics)
        
        # Evaluation analytics
        eval_checkpoint_dir = checkpoint_dir / "eval_checkpoints"
        if eval_checkpoint_dir.exists():
            for analytics_file in eval_checkpoint_dir.glob(pattern):
                analytics_files.append(analytics_file)
    
    return analytics_files


def run_sweep(
    config_path: Path,
    seeds: List[int],
    output_dir: Optional[Path] = None,
    skip_training: bool = False,
    skip_evaluation: bool = False,
    checkpoint_path_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the wagering pipeline with multiple seeds and aggregate results.
    
    Args:
        config_path: Path to config file
        seeds: List of random seeds to use
        output_dir: Directory to save aggregated results (default: config directory)
        skip_training: If True, skip training phase
        skip_evaluation: If True, skip evaluation phase
        checkpoint_path_override: Override checkpoint path for evaluation
        
    Returns:
        Dictionary with aggregated results and paths to analytics files
    """
    log.info("=" * 80)
    log.info("WAGERING SWEEP")
    log.info("=" * 80)
    log.info(f"Config: {config_path}")
    log.info(f"Seeds: {seeds}")
    log.info(f"Number of runs: {len(seeds)}")
    
    # Set up output directory
    if output_dir is None:
        output_dir = config_path.parent / "sweep_results"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load original config to get checkpoint_base_dir
    with open(config_path, 'r') as f:
        original_config = yaml.safe_load(f)
    
    checkpoint_base_dir = Path(original_config.get("checkpoint_base_dir", "/common/users/yl2310/MultiLLMs/checkpoints"))
    
    # Run pipeline for each seed
    all_checkpoint_dirs = []
    for i, seed in enumerate(seeds):
        log.info("\n" + "=" * 80)
        log.info(f"RUN {i+1}/{len(seeds)}: SEED {seed}")
        log.info("=" * 80)
        
        # Create temporary config with this seed
        temp_config_path = output_dir / f"config_seed_{seed}.yaml"
        update_config_with_seed(config_path, seed, temp_config_path)
        
        try:
            # Run pipeline directly
            run_pipeline(
                config_path=temp_config_path,
                skip_training=skip_training,
                skip_evaluation=skip_evaluation,
                checkpoint_path_override=checkpoint_path_override,
            )
            
            log.info(f"✓ Completed run {i+1}/{len(seeds)} with seed {seed}")
        except Exception as e:
            log.error(f"✗ Failed run {i+1}/{len(seeds)} with seed {seed}: {e}", exc_info=True)
            continue
    
    log.info("\n" + "=" * 80)
    log.info("COLLECTING ANALYTICS FILES")
    log.info("=" * 80)
    
    # Find all analytics files
    analytics_files = find_analytics_files(checkpoint_base_dir)
    
    log.info(f"Found {len(analytics_files)} analytics files:")
    for f in analytics_files:
        log.info(f"  - {f}")
    
    if not analytics_files:
        log.warning("No analytics files found. Cannot aggregate results.")
        return {
            "analytics_files": [],
            "aggregated_results": None,
            "output_dir": str(output_dir),
        }
    
    # Load all analytics DataFrames
    log.info("\n" + "=" * 80)
    log.info("LOADING AND AGGREGATING ANALYTICS")
    log.info("=" * 80)
    
    analytics_dfs = []
    for analytics_file in analytics_files:
        try:
            df = pd.read_csv(analytics_file)
            analytics_dfs.append(df)
            log.info(f"Loaded {analytics_file.name}: {len(df)} rows")
        except Exception as e:
            log.warning(f"Failed to load {analytics_file}: {e}")
    
    if not analytics_dfs:
        log.warning("No analytics DataFrames loaded. Cannot aggregate results.")
        return {
            "analytics_files": [str(f) for f in analytics_files],
            "aggregated_results": None,
            "output_dir": str(output_dir),
        }
    
    # Separate training and evaluation analytics
    training_dfs = []
    evaluation_dfs = []
    for df in analytics_dfs:
        if 'result_type' in df.columns:
            result_type = df['result_type'].iloc[0] if len(df) > 0 else ''
            if result_type == 'training':
                training_dfs.append(df)
            elif result_type == 'evaluation':
                evaluation_dfs.append(df)
        else:
            # If no result_type column, try to infer from other columns
            # Training has 'final_accuracy', evaluation has 'evaluation_dataset'
            if 'final_accuracy' in df.columns and 'evaluation_dataset' not in df.columns:
                training_dfs.append(df)
            elif 'evaluation_dataset' in df.columns:
                evaluation_dfs.append(df)
    
    log.info(f"Training analytics: {len(training_dfs)} files")
    log.info(f"Evaluation analytics: {len(evaluation_dfs)} files")
    
    # Aggregate results
    aggregated_results = {}
    
    if training_dfs:
        log.info("\nAggregating training results...")
        aggregated_training = WageringAnalytics.aggregate_results_by_settings(training_dfs)
        aggregated_results["training"] = aggregated_training
        
        # Save aggregated training results
        training_output_path = output_dir / "aggregated_training_analytics.csv"
        aggregated_training.to_csv(training_output_path, index=False)
        log.info(f"Saved aggregated training analytics to {training_output_path}")
    
    if evaluation_dfs:
        log.info("\nAggregating evaluation results...")
        aggregated_evaluation = WageringAnalytics.aggregate_results_by_settings(evaluation_dfs)
        aggregated_results["evaluation"] = aggregated_evaluation
        
        # Save aggregated evaluation results
        eval_output_path = output_dir / "aggregated_evaluation_analytics.csv"
        aggregated_evaluation.to_csv(eval_output_path, index=False)
        log.info(f"Saved aggregated evaluation analytics to {eval_output_path}")
    
    # Also save combined (all analytics together)
    if analytics_dfs:
        log.info("\nAggregating all results together...")
        combined_aggregated = WageringAnalytics.aggregate_results_by_settings(analytics_dfs)
        aggregated_results["combined"] = combined_aggregated
        
        combined_output_path = output_dir / "aggregated_all_analytics.csv"
        combined_aggregated.to_csv(combined_output_path, index=False)
        log.info(f"Saved combined aggregated analytics to {combined_output_path}")
    
    # Print summary
    log.info("\n" + "=" * 80)
    log.info("AGGREGATION SUMMARY")
    log.info("=" * 80)
    
    if "evaluation" in aggregated_results:
        eval_df = aggregated_results["evaluation"]
        log.info(f"\nEvaluation Results (aggregated over {len(seeds)} runs):")
        log.info(f"Number of unique settings: {len(eval_df)}")
        
        # Print key metrics
        if 'accuracy' in eval_df.columns:
            log.info("\nAccuracy (mean ± std):")
            for idx, row in eval_df.iterrows():
                eval_dataset = row.get('evaluation_dataset', 'unknown')
                acc_mean = row.get('accuracy', 'N/A')
                acc_std = row.get('accuracy_std', None)
                num_runs = row.get('num_runs', 'N/A')
                if acc_std is not None and not pd.isna(acc_std):
                    log.info(f"  {eval_dataset}: {acc_mean:.4f} ± {acc_std:.4f} (n={num_runs})")
                else:
                    log.info(f"  {eval_dataset}: {acc_mean:.4f} (n={num_runs})")
    
    log.info("\n" + "=" * 80)
    log.info("SWEEP COMPLETE")
    log.info("=" * 80)
    log.info(f"Results saved to: {output_dir}")
    
    return {
        "analytics_files": [str(f) for f in analytics_files],
        "aggregated_results": aggregated_results,
        "output_dir": str(output_dir),
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run wagering pipeline with multiple seeds and aggregate results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with specific seeds
  python wagering_sweep.py config.yaml --seeds 42 43 44 45 46
  
  # Run with 5 seeds starting from 42
  python wagering_sweep.py config.yaml --num-seeds 5 --seed-start 42
  
  # Run only evaluation (skip training)
  python wagering_sweep.py config.yaml --seeds 42 43 --skip-training
  
  # Custom output directory
  python wagering_sweep.py config.yaml --seeds 42 43 --output-dir ./my_sweep_results
        """
    )
    
    parser.add_argument(
        "config",
        type=str,
        help="Path to config file"
    )
    
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="List of random seeds to use"
    )
    
    parser.add_argument(
        "--num-seeds",
        type=int,
        help="Number of seeds to generate (used with --seed-start)"
    )
    
    parser.add_argument(
        "--seed-start",
        type=int,
        default=42,
        help="Starting seed value (default: 42)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save aggregated results (default: config directory / sweep_results)"
    )
    
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training phase (only run evaluation)"
    )
    
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation phase (only run training)"
    )
    
    parser.add_argument(
        "--checkpoint-path-override",
        type=str,
        help="Override checkpoint path for evaluation (useful when skipping training)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Determine seeds
    if args.seeds:
        seeds = args.seeds
    elif args.num_seeds:
        seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    else:
        log.error("Must provide either --seeds or --num-seeds")
        sys.exit(1)
    
    # Validate config file
    config_path = Path(args.config)
    if not config_path.exists():
        log.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    # Run sweep
    try:
        results = run_sweep(
            config_path=config_path,
            seeds=seeds,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            skip_training=args.skip_training,
            skip_evaluation=args.skip_evaluation,
            checkpoint_path_override=args.checkpoint_path_override,
        )
        
        log.info("\nSweep completed successfully!")
        log.info(f"Output directory: {results['output_dir']}")
        
    except Exception as e:
        log.error(f"Sweep failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

