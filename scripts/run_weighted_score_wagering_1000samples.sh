#!/bin/bash
# Script to run weighted score wagering approach end-to-end with 1000 samples FROM SCRATCH
# Uses centralized_wagers with weighted_linear_pooling aggregation
# Config: centralized_wagers_1000samples.yaml
# GPUs: 0,1
#
# This script:
# 1. Clears existing checkpoints for this config
# 2. Clears cached logits/hidden states in wagering_cache
# 3. Runs training and evaluation from scratch

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Config file path
CONFIG_FILE="${PROJECT_ROOT}/examples/configs/wagering_training/weighted_score_wagers_1000samples.yaml"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Change to project root
cd "$PROJECT_ROOT"

# Activate virtual environment
source .venv/bin/activate

echo "=========================================="
echo "Clearing existing checkpoints and cache..."
echo "=========================================="

# Clear checkpoints and cache using Python script
python - <<EOF
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path("$PROJECT_ROOT")
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from lm_polygraph.utils.config_loader import load_and_merge_configs
from lm_polygraph.utils.checkpoint_utils import generate_checkpoint_dir
import shutil

# Load config
config_file = Path("$CONFIG_FILE")
config = load_and_merge_configs(config_file)

# 1. Delete checkpoint directory
base_dir = Path(config.get('checkpoint_base_dir', '/common/users/yl2310/MultiLLMs/checkpoints'))
checkpoint_dir = generate_checkpoint_dir(
    base_dir=base_dir,
    models=config['models'],
    datasets=config['datasets'],
    wagering_method=config['wagering_method'],
    aggregation=config['aggregation'],
    create_hash=True,
)

print(f"Checkpoint directory: {checkpoint_dir}")
if checkpoint_dir.exists():
    print(f"  Deleting checkpoint directory...")
    shutil.rmtree(checkpoint_dir)
    print(f"  ✓ Checkpoint directory deleted")
else:
    print(f"  No checkpoint directory found (nothing to delete)")

# 2. Clear wagering cache (logits and hidden states)
wagering_cache_dir = Path("/common/users/yl2310/MultiLLMs/wagering_cache")
if wagering_cache_dir.exists():
    print(f"\nWagering cache directory: {wagering_cache_dir}")
    cache_files = list(wagering_cache_dir.glob("*.npz"))
    if cache_files:
        print(f"  Found {len(cache_files)} cache files")
        print(f"  Deleting cached logits/hidden states...")
        for cache_file in cache_files:
            cache_file.unlink()
        print(f"  ✓ Cache cleared ({len(cache_files)} files deleted)")
    else:
        print(f"  No cache files found (nothing to delete)")
else:
    print(f"\nWagering cache directory does not exist (nothing to delete)")

print("\n✓ All caches and checkpoints cleared!")
EOF

echo ""
echo "=========================================="
echo "Running pipeline from scratch..."
echo "=========================================="

# Run pipeline with GPU 0,1
# CUDA_VISIBLE_DEVICES must be set right before the python command (not before source)
CUDA_VISIBLE_DEVICES=0,1 python scripts/wagering_pipeline.py "$CONFIG_FILE"

echo ""
echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
