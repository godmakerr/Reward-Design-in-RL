#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,4,5
export PYTHONPATH=/root/fu_wj/clone2github/Reward-Design-in-RL/Code:$PYTHONPATH
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_INIT_TIMEOUT=600

MODEL_DIR="/root/fu_wj/clone2github/Reward-Design-in-RL"
TRAIN_JSONL="Datasets/gsm8k_json/train.jsonl"
EVAL_JSONL="Datasets/gsm8k_json/test.jsonl"
ACTOR_DIR="${MODEL_DIR}/Models/qwen_3_1_7b"
RM_DIR="${MODEL_DIR}/Models/qwen_3_1_7b_rm"
OUT_DIR="${MODEL_DIR}/Models/qwen_3_1_7b_ppo_bf16_selfrm"

# accelerate launch \
#   --config_file Code/accelerate/multi_gpu_bf16.yaml \
#   Code/ppo_rlhf_gsm8k.py -- \
#   --train_jsonl "${TRAIN_JSONL}" \
#   --eval_jsonl "${EVAL_JSONL}" \
#   --actor_model_dir "${ACTOR_DIR}" \
#   --reward_model_dir "${RM_DIR}" \
#   --output_dir "${OUT_DIR}" \
#   --per_device_train_batch_size 1 \
#   --gradient_accumulation_steps 8 \
#   --per_device_eval_batch_size 1 \
#   --world_size 4 \
#   --max_prompt_len 128 \
#   --response_length 128 \
#   --local_rollout_forward_batch_size 4

accelerate launch \
  --config_file /root/fu_wj/clone2github/Reward-Design-in-RL/Code/accelerate/ds_zero3_bf16.yaml \
  Code/ppo_rlhf_gsm8k.py -- \
  --train_jsonl "${TRAIN_JSONL}" \
  --eval_jsonl "${EVAL_JSONL}" \
  --actor_model_dir "${ACTOR_DIR}" \
  --reward_model_dir "${RM_DIR}" \
  --output_dir "${OUT_DIR}" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --per_device_eval_batch_size 1 \
  --world_size 4 \
  --num_train_epochs 1 \
  --max_prompt_len 256 \
  --response_length 512 \
  --local_rollout_forward_batch_size 8

