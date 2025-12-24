#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,4,5
export PYTHONPATH=/root/fu_wj/clone2github/Reward-Design-in-RL/Code:$PYTHONPATH


MODEL_DIR="Models/qwen_3_1_7b_ppo_bf16_selfrm"
TEST_JSONL="Datasets/gsm8k_json/test.jsonl"
OUTPUT_PATH="Infer/gsm8k_predictions.jsonl"

accelerate launch \
  --num_processes 4 \
  --mixed_precision fp16 \
  Code/eval_gsm8k_json.py \
  --model_dir "${MODEL_DIR}" \
  --data_jsonl "${TEST_JSONL}" \
  --output_path "${OUTPUT_PATH}" \
  --batch_size 32 \
  --max_new_tokens 512 \
  --use_chat_template
