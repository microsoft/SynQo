#!/bin/bash

# Set variables based on arguments
initial_pred_path="./data/bird/train_initial_pred.json"
gold_df_path="./data/bird/train.json"
trajectory_path="./results/MAGIC-trajectory/*"
guideline_out_path="./results/MAGIC-Guideline/guideline_progress_per_batch.json"

# Generate guidelines
python3 -u -m magic.guideline_generation \
  --initial_pred_path "${initial_pred_path}" \
  --gold_df_path "${gold_df_path}" \
  --trajectory_path "${trajectory_path}" \
  --guideline_out_path "${guideline_out_path}"
