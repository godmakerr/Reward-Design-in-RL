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


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl_as_dataset(jsonl_path: str):
    ds = load_dataset("json", data_files={"data": jsonl_path})["data"]
    return ds


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_jsonl", type=str, required=True)
    ap.add_argument("--eval_jsonl", type=str, required=True)

    ap.add_argument("--actor_model_dir", type=str, required=True)
    ap.add_argument("--reward_model_dir", type=str, required=True)
    ap.add_argument("--value_model_dir", type=str, default=None)

    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--max_prompt_len", type=int, default=256)
    ap.add_argument("--response_length", type=int, default=256)
    ap.add_argument("--enable_thinking", action="store_true")

    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--world_size", type=int, default=4)

    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--num_ppo_epochs", type=int, default=1)

    ap.add_argument("--learning_rate", type=float, default=1e-6)
    ap.add_argument("--kl_coef", type=float, default=0.05)
    ap.add_argument("--cliprange", type=float, default=0.2)
    ap.add_argument("--vf_coef", type=float, default=1.0)
    ap.add_argument("--cliprange_value", type=float, default=0.2)

    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--missing_eos_penalty", type=float, default=None)

    ap.add_argument("--local_rollout_forward_batch_size", type=int, default=2)  # 显存不够就调小到1

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
        logging_steps=10,
        eval_steps=70,
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
        print("=== ds_cfg type ===", type(ds_cfg))
        print("=== zero stage ===", ds_cfg.config["zero_optimization"]["stage"])
        wandb.init(project="reward_design_in_rl", name="qwen_ppo_rlhf")


    # ===== train =====
    trainer.train()

    trainer.accelerator.wait_for_everyone()
    
    def save_policy_only(trainer, tok, out_dir: str):
        acc = trainer.accelerator
        acc.wait_for_everyone()
        os.makedirs(out_dir, exist_ok=True)

        # 1) DeepSpeedEngine（一般 trainer.model 就是 DS engine）
        engine = getattr(trainer, "deepspeed", None) or trainer.model

        # 2) ⚠️ 所有 rank 都必须跑到这里（内部会 allgather）
        full_sd = acc.get_state_dict(engine)

        acc.wait_for_everyone()

        if acc.is_main_process:
            # 3) 只保留 policy.*，并 strip 前缀
            policy_sd = {k[len("policy."):]: v for k, v in full_sd.items() if k.startswith("policy.")}

            # 4) 拿到未包装 wrapper -> policy
            wrapper = acc.unwrap_model(trainer.model)  # PolicyAndValueWrapper
            policy = wrapper.policy

            # 5) save_pretrained + safe_serialization 会正确处理 tied weights
            policy.save_pretrained(out_dir, state_dict=policy_sd, safe_serialization=True)
            tok.save_pretrained(out_dir)

        # 6) 释放 + 再 barrier，避免别的 rank 还在用
        del full_sd
        acc.wait_for_everyone()


    # 训练结束后：
    save_policy_only(trainer, tok, args.output_dir)

    
    # if trainer.accelerator.is_main_process:
    #     unwrapped = trainer.accelerator.unwrap_model(policy_model)
    #     sd = trainer.accelerator.get_state_dict(policy_model)  # DS/ZeRO3 会做 consolidate
    #     unwrapped.save_pretrained(args.output_dir, state_dict=sd, safe_serialization=True)
    #     tok.save_pretrained(args.output_dir)
    
    # # 先拷贝正确的 config/tokenizer（从 actor 模型目录）
    # if trainer.accelerator.is_main_process:
    #     import shutil, glob
    #     src = args.actor_model_dir
    #     dst = args.output_dir
    #     for fn in [
    #         "config.json", "generation_config.json",
    #         "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    #         "vocab.json", "merges.txt"
    #     ]:
    #         p = os.path.join(src, fn)
    #         if os.path.exists(p):
    #             shutil.copy2(p, os.path.join(dst, fn))
    #     # 如果是 trust_remote_code 模型，很多时候还需要这些 python 文件
    #     for p in glob.glob(os.path.join(src, "*.py")):
    #         shutil.copy2(p, os.path.join(dst, os.path.basename(p)))

    # trainer.accelerator.wait_for_everyone()

    # engine = getattr(trainer, "deepspeed", None)

    # if trainer.accelerator.is_main_process:
    #     os.makedirs(args.output_dir, exist_ok=True)
    # trainer.accelerator.wait_for_everyone()
    # engine.save_16bit_model(args.output_dir)

    # if trainer.accelerator.is_main_process:
    #     tok.save_pretrained(args.output_dir)
    #     print(f"[OK] PPO policy saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
