#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,4,5
export PYTHONPATH=/root/fu_wj/clone2github/Reward-Design-in-RL/Code:$PYTHONPATH


TRAIN_JSONL="Datasets/gsm8k_json/train.jsonl"
BASE_MODEL_DIR="Models/qwen_3_1_7b"
OUTPUT_DIR="Models/qwen_3_1_7b_rm"

accelerate launch \
  --num_processes 4 \
  --mixed_precision bf16 \
  Code/train_rm_gsm8k_json.py \
  --train_jsonl "${TRAIN_JSONL}" \
  --base_model_dir "${BASE_MODEL_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --per_device_batch_size 2 \
  --grad_accum 16 \
  --epochs 1 \
  --max_length 512 \
  --use_chat_template

