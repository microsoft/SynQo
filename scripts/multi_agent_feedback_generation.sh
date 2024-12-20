#!/bin/bash

# Set variables based on arguments
max_threads=10
max_iterations=5
initial_pred_path="./results/train_initial_pred.json"
gold_df_path="./data/bird/train.json"
trajectory_path="./results/MAGIC-trajectory/"
db_path="./data/bird/train_databases"
model_key_name="gpt-4"

# Generate feedbacks
python3 -u -m magic.multi_agent_feedback_generation \
  --model_key_name "${model_key_name}" \
  --max_threads "${max_threads}" \
  --temperature "${temperature}" \
  --db_path "${db_path}" \
  --gold_df_path "${gold_df_path}" \
  --initial_pred_path "${initial_pred_path}" \
  --trajectory_path "${trajectory_path}" \
  --max_iterations "${max_iterations}"
