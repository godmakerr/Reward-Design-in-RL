#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
import wandb

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)

from trl.experimental.ppo import PPOConfig, PPOTrainer

from utils_gsm8k import set_seed, load_jsonl_as_dataset, save_policy_only


def tok_fn(ex):
        q = str(ex["question"])
        messages = [{"role": "user", "content": q}]
        prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
        out = tok(
            prompt,
            truncation=True,
            max_length=args.max_prompt_len,
        )
        return out

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_jsonl", type=str, required=True)
    ap.add_argument("--eval_jsonl", type=str, required=True)

    ap.add_argument("--actor_model_dir", type=str, required=True)
    ap.add_argument("--reward_model_dir", type=str, required=True)
    ap.add_argument("--value_model_dir", type=str, default=None)
    
    ap.add_argument("--proj_name", type=str, default="reward_design_in_rl")
    ap.add_argument("--run_name", type=str, default="qwen_ppo_rlhf")

    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--max_prompt_len", type=int, default=256)
    ap.add_argument("--response_length", type=int, default=256)
    ap.add_argument("--enable_thinking", action="store_true")

    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--world_size", type=int, default=4)
    ap.add_argument("--eval_strategy", type=str, default="steps")
    ap.add_argument("--eval_steps", type=int, default=10)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)

    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--num_ppo_epochs", type=int, default=1)

    ap.add_argument("--learning_rate", type=float, default=1e-6)
    ap.add_argument("--kl_coef", type=float, default=0.05)
    ap.add_argument("--cliprange", type=float, default=0.2)
    ap.add_argument("--vf_coef", type=float, default=1.0)
    ap.add_argument("--cliprange_value", type=float, default=0.2)

    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--missing_eos_penalty", type=float, default=None)

    ap.add_argument("--local_rollout_forward_batch_size", type=int, default=2)

    ap.add_argument("--eval_max_samples", type=int, default=1319)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--bf16", action="store_true", default=True)

    args = ap.parse_args()


    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    if not args.bf16:
        raise RuntimeError("你要求必须用 bf16：请保持 --bf16 (默认已开启)，不要关闭。")

    # ===== tokenizer =====
    tok = AutoTokenizer.from_pretrained(args.actor_model_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # ===== dataset: tokenize -> input_ids/attention_mask =====
    train_raw = load_jsonl_as_dataset(args.train_jsonl)
    eval_raw = load_jsonl_as_dataset(args.eval_jsonl)

    

    train_ds = train_raw.map(tok_fn, remove_columns=train_raw.column_names)
    eval_ds = eval_raw.map(tok_fn, remove_columns=eval_raw.column_names)

    if args.eval_max_samples is not None and len(eval_ds) > args.eval_max_samples:
        eval_ds = eval_ds.select(range(args.eval_max_samples))

    # ===== models: policy/value/reward =====
    dtype = torch.bfloat16

    policy_model = AutoModelForCausalLM.from_pretrained(
        args.actor_model_dir,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    policy_model.config.pad_token_id = tok.pad_token_id
    policy_model.config.eos_token_id = tok.eos_token_id
    policy_model.generation_config.eos_token_id = tok.eos_token_id
    policy_model.generation_config.pad_token_id = tok.pad_token_id

    value_dir = args.value_model_dir or args.actor_model_dir
    value_model = AutoModelForSequenceClassification.from_pretrained(
        value_dir,
        num_labels=1,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    value_model.config.pad_token_id = tok.pad_token_id
    value_model.config.eos_token_id = tok.eos_token_id
    # value_model.generation_config.eos_token_id = tok.eos_token_id
    # value_model.generation_config.pad_token_id = tok.pad_token_id

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_dir,
        num_labels=1,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    reward_model.config.pad_token_id = tok.pad_token_id
    reward_model.config.eos_token_id = tok.eos_token_id
    # reward_model.generation_config.eos_token_id = tok.eos_token_id
    # reward_model.generation_config.pad_token_id = tok.pad_token_id
    
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.actor_model_dir,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    ref_model.config.pad_token_id = tok.pad_token_id
    ref_model.config.eos_token_id = tok.eos_token_id
    ref_model.generation_config.eos_token_id = tok.eos_token_id
    ref_model.generation_config.pad_token_id = tok.pad_token_id

    ppo_args = PPOConfig(
        output_dir=args.output_dir,

        # batch sizes
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        world_size=args.world_size,

        # 强制 bf16
        bf16=True,
        fp16=False,
        
        # PPO hyperparams
        learning_rate=args.learning_rate,
        kl_coef=args.kl_coef,
        cliprange=args.cliprange,
        vf_coef=args.vf_coef,
        cliprange_value=args.cliprange_value,
        num_ppo_epochs=args.num_ppo_epochs,
        response_length=args.response_length,
        temperature=args.temperature,
        missing_eos_penalty=args.missing_eos_penalty,

        # rollout forward chunk
        local_rollout_forward_batch_size=args.local_rollout_forward_batch_size,

        num_train_epochs=args.num_train_epochs,

        save_strategy="no",
        # save_steps=2, 
        logging_steps=1,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        report_to=["wandb"],

        seed=args.seed,
        # deepspeed="/root/fu_wj/clone2github/Reward-Design-in-RL/Code/deepspeed/ds_zero3_bf16.json",
    )

    trainer = PPOTrainer(
        args=ppo_args,
        processing_class=tok,
        model=policy_model,
        ref_model=ref_model,
        reward_model=reward_model,
        train_dataset=train_ds,
        value_model=value_model,
        eval_dataset=eval_ds,
    )
    if trainer.accelerator.is_main_process:
        ds_cfg = trainer.accelerator.state.deepspeed_plugin.hf_ds_config
        print("=== zero stage ===", ds_cfg.config["zero_optimization"]["stage"])
        wandb.init(project=args.proj_name, name=args.run_name)


    # ===== train =====
    trainer.train()

    trainer.accelerator.wait_for_everyone()
    
    save_policy_only(trainer, tok, args.output_dir)


if __name__ == "__main__":
    main()
