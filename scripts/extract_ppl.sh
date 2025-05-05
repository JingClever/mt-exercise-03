#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

base="$(dirname "$script_dir")"

# Activate the virtual environment
source "$base/venvs/torch3/bin/activate"

models_dir="$base/models"

# Create directory to store the visualization files
viz_dir="$base/visualization"
mkdir -p "$viz_dir"

# Log files for the models
baseline_log="$models_dir/deen_transformer_baseline/train.log"
prenorm_log="$models_dir/deen_transformer_pre/train.log"
postnorm_log="$models_dir/deen_transformer_post/train.log"

python "$script_dir/extract_ppl.py" \
    "$baseline_log" \
    "$prenorm_log" \
    "$postnorm_log" \
    --csv_output "$viz_dir/validation_ppl.csv" \
    --pdf_output "$viz_dir/validation_ppl.pdf"





