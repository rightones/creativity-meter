#!/bin/bash
# Script to run the full analysis pipeline
# 1. Download full dataset
# 2. Extract images
# 3. Resume evaluation on all images

echo "Starting full analysis pipeline..."

# Ensure we are in the project root
cd /root/workspace/ccp

# 1. Download and Extract (No max_samples implies all)
echo "[1/2] Downloading and Validating Data..."
uv run scripts/download_wikiart_sd.py

if [ $? -ne 0 ]; then
    echo "Download script failed. Exiting."
    exit 1
fi

# 2. Run Evaluation
echo "[2/2] Running Evaluation (Optimized for ~8h)..."
OUTPUT_FILE="results/wikiart_sd_full_scores.csv"

# Using nohup logic here isn't needed if we run this script in background via agent tool
uv run experiments/evaluate_dataset.py --metadata_csv data/wikiart_sd/dataset_map.csv --output_file $OUTPUT_FILE --max_samples 3000 --num_steps 20

echo "Analysis pipeline finished."
