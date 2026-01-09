#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,4,5
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_INIT_TIMEOUT=600

BASE_DIR="/root/fu_wj/clone2github/Reward-Design-in-RL" # 项目根目录
export PYTHONPATH="${BASE_DIR}/Code:$PYTHONPATH"

MODEL_DIR="${BASE_DIR}/Models"
DATASETS_DIR="${BASE_DIR}/Datasets"

TRAIN_JSONL="${DATASETS_DIR}/gsm8k_json/train.jsonl"
EVAL_JSONL="${DATASETS_DIR}/gsm8k_json/test.jsonl"

ACTOR_DIR="${MODEL_DIR}/qwen_3_1_7b"
OUT_DIR="${MODEL_DIR}/qwen_3_1_7b_grpo_bf16_kl_0.001"

ACC_CONFIG="${BASE_DIR}/Code/accelerate/ds_zero2_bf16.yaml"

accelerate launch \
  --config_file "${ACC_CONFIG}" \
  "${BASE_DIR}/Code/grpo_rl_gsm8k.py" \
  --train_jsonl "${TRAIN_JSONL}" \
  --eval_jsonl  "${EVAL_JSONL}" \
  --actor_model_dir "${ACTOR_DIR}" \
  --output_dir "${OUT_DIR}" \
  --max_prompt_len 256 \
  --response_length 512 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 8 \
  --eval_strategy "epoch" \
  --eval_steps 1 \
  --per_device_eval_batch_size 16 \
  --num_train_epochs 1 \
  --num_generations 8 \
