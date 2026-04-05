# !/bin/bash

# This script runs the evaluation for the SEMKEY model predicted CSV file.
# You can feed this script with the CSV file generated when you train the
# SEMKEY End-to-end (E2E) model.
# This script runs: ./evaluation/evaluate.py

# Set Hugging Face cache directory
export HF_HOME="./data/zuco_preprocessed_dataframe/hf_cache"
export TRANSFORMERS_CACHE="./data/zuco_preprocessed_dataframe/hf_cache"
# Set nltk data directory
NLTK_DATA="./data/zuco_preprocessed_dataframe/nltk_data"

# Path to the generated CSV file
CSV_FILE_PATH="..."

# Device (you can use CPU for this)
DEVICE="cuda:0"

# MTV CONFIG (Leave empty if you have every file in the default path)
# Path to the MTV file
MTV_FILE_PATH=""
# MTV KEYS (for BLEU)
MTV_KEYS=""

# Construct CMD
CMD="python -m evaluation.evaluate"

CMD+=" --csv-path $CSV_FILE_PATH"
CMD+=" --device $DEVICE"

if [ -n "$MTV_FILE_PATH" ]; then
    CMD+=" --variants-path $MTV_FILE_PATH"
fi

if [ -n "$MTV_KEYS" ]; then
    CMD+=" --variant-keys $MTV_KEYS"
fi

if [ -n "$NLTK_DATA" ]; then
    CMD+=" --nltk-data-path $NLTK_DATA"
fi

# Verbose and run
echo "Running evaluation with the following command:"
echo "$CMD"
eval "$CMD"