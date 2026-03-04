#!/bin/bash
# Script to clear cached logits and re-run with fixed token resolution

CACHE_DIR="./workdir/output/multi_llm_logits"

echo "Clearing cached logits..."
rm -f ${CACHE_DIR}/*.npz
echo "Cache cleared!"

echo ""
echo "Now re-run your command:"
echo "python3 scripts/run_multi_llm_ensemble.py \\"
echo "  --dataset-name mcq \\"
echo "  --split train \\"
echo "  --batch-size 8 \\"
echo "  --num-examples 100 \\"
echo "  --model-paths BioMistral/BioMistral-7B meta-llama/Llama-3.1-8B-Instruct \\"
echo "  --cache-dir ${CACHE_DIR}"


