#!/bin/bash

# Training script for SEMKEY_PARALLEL model
# Multi-task: 2 classification (sentiment, topic) + 2 regression (length, surprisal)
#
# Output directory structure:
#   ./logs/<experiment_name>_<timestamp>/
#     ├── tensorboard/    # TensorBoard logs
#     ├── checkpoints/    # Model checkpoints
#     ├── training.log    # Console output log
#     └── training_error.log  # Error log

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================
USE_ZUCO1_ONLY=false
USE_CHANNEL_WEIGHTS=true

USE_SCALED_LR=true          # Scale LR by number of GPUs in multi-GPU training
USE_PER_GPU_BATCH_SIZE=true # Treat BATCH_SIZE as per-GPU (global = BATCH_SIZE × num_gpus)
USE_CLASS_WEIGHTS=false     # Enable per-class loss weighting based on inverse frequency

# Data
DATA_PATH="./data/zuco_preprocessed_dataframe/zuco_merged_with_variants.df"

# Signal preprocessing
SPECTRAL_WHITENING=true  # Apply spectral whitening to EEG signals
ROBUST_NORMALIZE=false    # Apply robust normalization to EEG signals

# Classification tasks
SENTIMENT_LABELS=("non_neutral" "neutral")
TOPIC_LABELS=("Biographies and Factual Knowledge" "Movie Reviews and Sentiment")

# Model Architecture
TEXT_MODEL="google/flan-t5-large"

# Model Cache Directory
# Set to empty string "" to use default Hugging Face cache (~/.cache/huggingface)
# Set to a persistent path on AFS to avoid re-downloading weights on restart
MODEL_CACHE_DIR="data/zuco_preprocessed_dataframe/hf_cache"
# MODEL_CACHE_DIR="" # Use default cache

HIDDEN_DIM=128
EMBED_DIM=1024
# Have to be set at 96
HIDDEN_EEG_LEN=96
N_IN_BLOCKS=6
N_OUT_BLOCKS=6
NUM_HEADS=8
ENCODER_DROPOUT=0.1
MLP_HIDDEN_DIMS="512 256 128"
MLP_DROPOUT=0.3

# Loss Weights (6 total)
CLIP_LOSS_WEIGHT=0.5
LM_LOSS_WEIGHT=0.5
COMMITMENT_LOSS_WEIGHT=0.7
SENTIMENT_LOSS_WEIGHT=0.3
TOPIC_LOSS_WEIGHT=0.3
LENGTH_LOSS_WEIGHT=0.9
SURPRISAL_LOSS_WEIGHT=0.3

# Training
# Per-GPU batch size (global batch size = BATCH_SIZE × number of GPUs)
# Example: 32 per GPU × 8 GPUs = 256 global batch size
BATCH_SIZE=72
VAL_BATCH_SIZE=24
MAX_EPOCHS=50

# Base learning rate for single GPU
# NOTE: When using multiple GPUs, LR will be scaled automatically in the training script
# Formula: effective_lr = base_lr × num_gpus
# Example: 1e-4 × 8 GPUs = 1.6e-3
LR=2e-4
MIN_LR=1e-5
WARMUP_EPOCHS=15
SEED=42
DATASPLIT_SEED="None" # Set to "None" to use default split, or use a single number (i.e. DATASPLIT_SEED=648)

# Hardware single-GPU settings
ACCELERATOR="gpu"
STRATEGY="auto"       
DEVICE=(0)            
PRECISION="bf16-mixed"
NUM_WORKERS=4

# Logging
LOG_DIR="./logs/semkey_parallel"
EXPERIMENT_NAME="exp"

# ============================================================================

echo "=========================================="
echo "Training Configuration:"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Sentiment Labels: ${SENTIMENT_LABELS[@]}"
echo "  Topic Labels: ${TOPIC_LABELS[@]}"
echo "  Data Path: $DATA_PATH"
echo "=========================================="
echo

# Initialize the base command in an array
CMD_ARGS=(
    -m train.train_parallel
    --data_path "$DATA_PATH"
    --sentiment_labels "${SENTIMENT_LABELS[@]}"
    --topic_labels "${TOPIC_LABELS[@]}"
    --text_model "$TEXT_MODEL"
    --hidden_dim $HIDDEN_DIM
    --embed_dim $EMBED_DIM
    --hidden_eeg_len $HIDDEN_EEG_LEN
    --n_in_blocks $N_IN_BLOCKS
    --n_out_blocks $N_OUT_BLOCKS
    --num_heads $NUM_HEADS
    --encoder_dropout $ENCODER_DROPOUT
    --mlp_hidden_dims $MLP_HIDDEN_DIMS
    --mlp_dropout $MLP_DROPOUT
    --clip_loss_weight $CLIP_LOSS_WEIGHT
    --lm_loss_weight $LM_LOSS_WEIGHT
    --commitment_loss_weight $COMMITMENT_LOSS_WEIGHT
    --sentiment_loss_weight $SENTIMENT_LOSS_WEIGHT
    --topic_loss_weight $TOPIC_LOSS_WEIGHT
    --length_loss_weight $LENGTH_LOSS_WEIGHT
    --surprisal_loss_weight $SURPRISAL_LOSS_WEIGHT
    --batch_size $BATCH_SIZE
    --val_batch_size $VAL_BATCH_SIZE
    --max_epochs $MAX_EPOCHS
    --lr $LR
    --min_lr $MIN_LR
    --warmup_epochs $WARMUP_EPOCHS
    --log_dir "$LOG_DIR"
    --experiment_name "$EXPERIMENT_NAME"
    --accelerator "$ACCELERATOR"
    --device "${DEVICE[@]}"
    --precision "$PRECISION"
    --seed $SEED
    --num_workers $NUM_WORKERS
)

# Add conditional Boolean flags
[[ "$USE_ZUCO1_ONLY" == true ]] && CMD_ARGS+=(--use_zuco1_only)
[[ "$USE_CHANNEL_WEIGHTS" == true ]] && CMD_ARGS+=(--use_channel_weights)
[[ "$USE_SCALED_LR" == true ]] && CMD_ARGS+=(--use_scaled_lr)
[[ "$USE_PER_GPU_BATCH_SIZE" == true ]] && CMD_ARGS+=(--use_per_gpu_batch_size)
[[ "$USE_CLASS_WEIGHTS" == true ]] && CMD_ARGS+=(--use_class_weights)
[[ "$SPECTRAL_WHITENING" == false ]] && CMD_ARGS+=(--no_spectral_whitening)
[[ "$ROBUST_NORMALIZE" == false ]] && CMD_ARGS+=(--no_robust_normalize)

# Add Path and Seed with "None"/Empty checks
[[ -n "$MODEL_CACHE_DIR" ]] && CMD_ARGS+=(--model_cache_dir "$MODEL_CACHE_DIR")

# Dataset split
if [[ -n "$DATASPLIT_SEED" && "$DATASPLIT_SEED" != "None" && "$DATASPLIT_SEED" =~ ^[0-9]+$ ]]; then
    CMD_ARGS+=(--datasplit_seed $DATASPLIT_SEED)
fi

# Execute the command
python "${CMD_ARGS[@]}"
