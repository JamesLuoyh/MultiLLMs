#!/bin/bash
# Commands to run wagering sweeps for centralized and decentralized approaches
# Each sweep uses 2 GPUs and runs with seeds 42 and 43
#
# NOTE: If you need to activate the virtual environment, use:
#   source .venv/bin/activate && CUDA_VISIBLE_DEVICES=0,1 python ...
# NOT: CUDA_VISIBLE_DEVICES=0,1 source .venv/bin/activate && python ...
# CUDA_VISIBLE_DEVICES must be set right before the python command to work correctly.

# ============================================================================
# SWEEP 1: CENTRALIZED WAGERS
# ============================================================================
# This command runs the centralized wagers sweep with seeds 42 and 43
# Using GPUs 0 and 1 (adjust CUDA_VISIBLE_DEVICES if you want different GPUs)

echo "======================================================================"
echo "Starting Centralized Wagers Sweep"
echo "======================================================================"
CUDA_VISIBLE_DEVICES=0,1 python scripts/wagering_sweep.py \
    examples/configs/wagering_training/centralized_wagers_1000samples.yaml \
    --seeds 42 43 \
    --output-dir examples/configs/wagering_training/sweep_results_centralized

echo ""
echo "======================================================================"
echo "Centralized Wagers Sweep Complete"
echo "======================================================================"
echo ""

# ============================================================================
# SWEEP 2: DECENTRALIZED WAGERS
# ============================================================================
# This command runs the decentralized wagers sweep with seeds 42 and 43
# Using GPUs 0 and 1 (adjust CUDA_VISIBLE_DEVICES if you want different GPUs)

echo "======================================================================"
echo "Starting Decentralized Wagers Sweep"
echo "======================================================================"
CUDA_VISIBLE_DEVICES=0,1 python scripts/wagering_sweep.py \
    examples/configs/wagering_training/decentralized_wagers_1000samples.yaml \
    --seeds 42 43 \
    --output-dir examples/configs/wagering_training/sweep_results_decentralized

echo ""
echo "======================================================================"
echo "Decentralized Wagers Sweep Complete"
echo "======================================================================"
echo ""
echo "All sweeps completed!"

