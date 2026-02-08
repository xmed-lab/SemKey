#!/bin/bash

# This script will run inference on SemKey stage 1
# (Aka, GLIM with 2 * classification + 2 * regression tasks)
# !!! You'll need this for stage 2 and e2e <- We'll generate 
#     stage 2 dataset here
# WARNING: If you choose to use different checkpoints, make sure
#          your system has sufficient RAM

# ============================================================
# MODE SELECTION
# Set USE_SINGLE_CHECKPOINT=true to use CKPT for all tasks
# and run prediction only once (faster, single model)
# ============================================================
USE_SINGLE_CHECKPOINT=true

# ============================================================
# DATASET SELECTION
# Set USE_ZUCO1_ONLY=true to use only ZuCo1 dataset
# and drop all ZuCo2 samples
# ============================================================
USE_ZUCO1_ONLY=false

# Set paths to your checkpoints
CKPT="./logs/...ckpt"
# (LEAVE the follows as it is if you have USE_SINGLE_CHECKPOINT = true)
SENTIMENT_CKPT="./logs/...ckpt"
TOPIC_CKPT="./logs/...ckpt" 
LENGTH_CKPT="./logs/...ckpt"
SURPRISAL_CKPT="./logs/...ckpt"

# Set data path
# WARNING: if you've used whitening, please specify that cached whitened dataframe !!!
DATA_PATH="data/zuco_preprocessed_dataframe/zuco_merged_with_variants.df"

# Set output path
OUTPUT_PATH="./data/zuco_preprocessed_dataframe/....df"

# Create output directory if it doesn't exist
mkdir -p $(dirname $OUTPUT_PATH)

# Set batch size
BATCH_SIZE=72

# Set device
DEVICE=0

# Set split (for stage 2 training, please use 'all')
SPLIT="all" # run predict on 'all', 'train', 'test' and 'val' splits

# Build the zuco1 only flag
ZUCO1_FLAG=""

if [ "$USE_ZUCO1_ONLY" = true ]; then
    ZUCO1_FLAG="--use_zuco1_only"
    echo "Note: Using ZuCo1 dataset only (ZuCo2 samples will be dropped)"
fi

# Run prediction based on mode
if [ "$USE_SINGLE_CHECKPOINT" = true ]; then
    echo "Running in SINGLE CHECKPOINT mode using CKPT for all tasks"
    python -m inference.predict_semkey_parallel_and_pack \
        --data_path "$DATA_PATH" \
        --single_checkpoint "$CKPT" \
        --output_path "$OUTPUT_PATH" \
        --device "$DEVICE" \
        --batch_size "$BATCH_SIZE" \
        --split "$SPLIT" \
        $ZUCO1_FLAG
else
    echo "Running in MULTI-CHECKPOINT mode with separate checkpoints per task"
    python -m inference.predict_semkey_parallel_and_pack \
        --data_path "$DATA_PATH" \
        --sentiment_checkpoint "$SENTIMENT_CKPT" \
        --topic_checkpoint "$TOPIC_CKPT" \
        --length_checkpoint "$LENGTH_CKPT" \
        --surprisal_checkpoint "$SURPRISAL_CKPT" \
        --output_path "$OUTPUT_PATH" \
        --device "$DEVICE" \
        --batch_size "$BATCH_SIZE" \
        --split "$SPLIT" \
        $ZUCO1_FLAG
fi