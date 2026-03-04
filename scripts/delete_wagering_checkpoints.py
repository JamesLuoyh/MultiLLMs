#!/usr/bin/env python3
"""
Script to delete existing wagering checkpoints for fresh training runs.
"""

import sys
import shutil
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from lm_polygraph.utils.config_loader import load_and_merge_configs
from lm_polygraph.utils.checkpoint_utils import generate_checkpoint_dir


def delete_checkpoints(config_path: Path, method_name: str):
    """Delete checkpoints for a given config file."""
    print(f"\n{'='*60}")
    print(f"Checking {method_name} checkpoints...")
    print(f"{'='*60}")
    
    config = load_and_merge_configs(config_path)
    base_dir = Path(config.get('checkpoint_base_dir', '/common/users/yl2310/MultiLLMs/checkpoints'))
    checkpoint_dir = generate_checkpoint_dir(
        base_dir=base_dir,
        models=config['models'],
        datasets=config['datasets'],
        wagering_method=config['wagering_method'],
        aggregation=config['aggregation'],
        create_hash=True,
    )
    
    print(f"Expected checkpoint directory: {checkpoint_dir}")
    
    if checkpoint_dir.exists():
        print(f"  ✓ Found checkpoint directory")
        print(f"  Deleting {checkpoint_dir}...")
        try:
            shutil.rmtree(checkpoint_dir)
            print(f"  ✓ Successfully deleted!")
            return True
        except Exception as e:
            print(f"  ✗ Error deleting: {e}")
            return False
    else:
        print(f"  No checkpoint directory found (nothing to delete)")
        return False


def main():
    """Main function."""
    project_root = Path(__file__).resolve().parents[1]
    
    # Delete centralized wagers checkpoints
    centralized_config = project_root / "examples/configs/wagering_training/centralized_wagers_1000samples.yaml"
    deleted_centralized = delete_checkpoints(centralized_config, "Centralized Wagers")
    
    # Delete decentralized wagers checkpoints
    decentralized_config = project_root / "examples/configs/wagering_training/decentralized_wagers_1000samples.yaml"
    deleted_decentralized = delete_checkpoints(decentralized_config, "Decentralized Wagers")
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    if deleted_centralized or deleted_decentralized:
        print("✓ Checkpoints deleted successfully!")
        print("You can now run training from scratch.")
    else:
        print("No checkpoints found to delete.")
        print("Training will start from scratch automatically.")
    print()


if __name__ == "__main__":
    main()
