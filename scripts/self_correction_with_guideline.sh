#!/bin/bash

# Set variables based on arguments
max_threads=30
initial_pred_path="./results/dev_initial_pred_bird.json"
gold_df_path="./data/bird/dev.json"
trajectory_path="./results/MAGIC-trajectory/"
db_path="./data/bird/dev_databases"
model_key_name="gpt-4"

# Generate feedbacks
python3 -u -m magic.self_correction_with_guideline \
  --model_key_name "${model_key_name}" \
  --max_threads "${max_threads}" \
  --db_path "${db_path}" \
  --gold_df_path "${gold_df_path}" \
  --initial_pred_path "${initial_pred_path}" \
  --trajectory_path "${trajectory_path}" \
  --max_iterations "${max_iterations}"