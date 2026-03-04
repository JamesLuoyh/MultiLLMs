#!/bin/bash
# Script to run equal wagering evaluation with 1000 samples
# Uses the same configuration as centralized_wagers_1000samples.yaml but with equal_wagers method
# This script uses cached logits and hidden states if available, otherwise collects them

set -e  # Exit on error

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Config file path
CONFIG_FILE="examples/configs/wagering_training/equal_wagers_eval_1000samples.yaml"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at $CONFIG_FILE"
    exit 1
fi

# Wagering cache directory (where logits and hidden states are cached)
WAGERING_CACHE_DIR="/common/users/yl2310/MultiLLMs/wagering_cache"

echo "=========================================="
echo "Running Equal Wagering Evaluation"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "GPUs: 2,3"
echo ""
echo "This will:"
echo "  1. Use cached logits and hidden states if available"
echo "  2. Collect logits/hidden states from scratch if cache is not available"
echo "  3. Set resume_eval: false (start fresh evaluation)"
echo "=========================================="
echo ""

# Check if cache exists and inform user
if [ -d "$WAGERING_CACHE_DIR" ] && [ "$(find "$WAGERING_CACHE_DIR" -mindepth 1 -type f 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "Found cached logits/hidden states in: $WAGERING_CACHE_DIR"
    echo "Will use cached data if available for the models and datasets."
else
    echo "No cached logits/hidden states found. Will collect from scratch."
fi
echo ""

# Create a temporary config file with resume_eval: false in the same directory as the original config
# This ensures relative paths in the config (like models/...) resolve correctly
CONFIG_DIR="$(dirname "$CONFIG_FILE")"
TEMP_CONFIG=$(mktemp -p "$CONFIG_DIR" --suffix=.yaml)
trap "rm -f $TEMP_CONFIG" EXIT

# Use Python to modify the config file to set resume_eval: false
source .venv/bin/activate && python3 -c "
import yaml
import sys

config_path = sys.argv[1]
temp_config_path = sys.argv[2]

# Load config
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Set resume_eval to false
config['resume_eval'] = False

# Save to temp file
with open(temp_config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
" "$CONFIG_FILE" "$TEMP_CONFIG"

echo "Using modified config with resume_eval: false"
echo ""

# Run evaluation with GPU 2,3
# Note: CUDA_VISIBLE_DEVICES must be set in the same command as python
source .venv/bin/activate && CUDA_VISIBLE_DEVICES=2,3 .venv/bin/python scripts/wagering_eval.py "$TEMP_CONFIG"

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "=========================================="
