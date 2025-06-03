#!/bin/bash

# Activate your environment (if needed)
# conda activate mission_code

# Set the folder where your files are
DATA_FOLDER="/Users/chenyujie/Desktop/text_data/babylm_en"

# Python script path
SCRIPT_PATH="data/run_tokenize.py"

# Loop over all files in the folder
for file in "$DATA_FOLDER"/*
do
    echo "🚀 Processing file: $file"
    python "$SCRIPT_PATH" "$file"
done

echo "✅ All files processed!"