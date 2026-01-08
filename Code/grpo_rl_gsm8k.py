#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
import wandb

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from trl.trainer import GRPOConfig, GRPOTrainer

from Code.utils_gsm8k import compute_binary_rewards, compute_format_rewards, compute_closeness_rewards, compute_format_and_closeness_rewards, set_seed, load_jsonl_as_dataset
def build_prompt(ex):
    """
    将 GSM8K 原始样本转换为 GRPO 支持的“对话格式 + gold 答案”。
    约定原始 jsonl 至少有字段:
      - question: str
      - answer: str  (官方 GSM8K 答案，末尾有 '#### <number>')
    """
    q = str(ex["question"])
    a = str(ex["answer"])
    return {
        # 对话格式：列表中每个元素是一个 message dict
        "prompt": [{"role": "user", "content": q}],
        # gold 答案文本，reward 函数会用来判对错
        "answer": a,
    }


def gsm8k_rlvr_reward(completions, answer, **kwargs):
    """
    RLVR reward 函数：
    - completions: List[List[{"role": "assistant", "content": str}]]
    - answer: List[str] (gold answer), Trainer 会自动重复到 completions 等长
    """
    completion_contents = []
    for completion in completions:
        if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[0], dict):
            completion_contents.append(completion[0].get("content", ""))
        else:
            completion_contents.append(str(completion))
    print("=== completion_contents[0] ===")
    print(completion_contents[0])
    print("=== answer[0] ===")
    print(answer[0])
    rewards = compute_binary_rewards(completion_contents, answer)
    print("=== rewards ===")
    print(rewards)
    return rewards

def make_gsm8k_reward_fn(reward_mode: str):
    mode2fn = {
        "binary": compute_binary_rewards,
        "format": compute_format_rewards,
        "closeness": compute_closeness_rewards,
        "format_and_closeness": compute_format_and_closeness_rewards,
    }
    if reward_mode not in mode2fn:
        raise ValueError(f"Unknown reward_mode: {reward_mode}")

    compute_fn = mode2fn[reward_mode]

    def reward_fn(completions, answer, **kwargs):
        completion_contents = []
        for completion in completions:
            if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[0], dict):
                completion_contents.append(completion[0].get("content", ""))
            else:
                completion_contents.append(str(completion))

        rewards = compute_fn(completion_contents, answer)

        # 可选：少打印一点，不然多进程下会爆日志
        if kwargs.get("global_step", 0) % 50 == 0:  # 有些版本不传 global_step，可删
            print(f"[reward_mode={reward_mode}] sample_reward={rewards[0]}")

        return rewards

    return reward_fn

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_jsonl", type=str, required=True)
    ap.add_argument("--eval_jsonl", type=str, required=True)

    ap.add_argument("--actor_model_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    
    ap.add_argument("--proj_name", type=str, default="reward_design_in_rl")
    ap.add_argument("--run_name", type=str, default="qwen_grpo_rl")

    ap.add_argument("--max_prompt_len", type=int, default=256)
    ap.add_argument("--response_length", type=int, default=256)
    ap.add_argument("--enable_thinking", action="store_true")

    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--eval_strategy", type=str, default="epochs")
    ap.add_argument("--eval_steps", type=int, default=10)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)

    ap.add_argument("--num_train_epochs", type=float, default=1.0)

    ap.add_argument("--learning_rate", type=float, default=1e-6)
    ap.add_argument("--kl_coef", type=float, default=0.0)
    ap.add_argument("--temperature", type=float, default=0.7)

    ap.add_argument("--eval_max_samples", type=int, default=1319)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--bf16", action="store_true", default=True)

    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--num_generations_eval", type=int, default=1)
    ap.add_argument("--reward_mode", type=str, default="binary", choices=["binary", "format", "closeness", "format_and_closeness"],)

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

    # ===== dataset: 保留 question + answer，构造对话格式 prompt =====
    train_raw = load_jsonl_as_dataset(args.train_jsonl)
    eval_raw = load_jsonl_as_dataset(args.eval_jsonl)

    train_ds = train_raw.map(build_prompt, remove_columns=train_raw.column_names)
    eval_ds = eval_raw.map(build_prompt, remove_columns=eval_raw.column_names)

    if args.eval_max_samples is not None and len(eval_ds) > args.eval_max_samples:
        eval_ds = eval_ds.select(range(args.eval_max_samples))

    # ===== policy model =====
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

    # ===== GRPO config (PPO-family RLVR) =====
    grpo_args = GRPOConfig(
        output_dir=args.output_dir,

        # batch sizes
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.per_device_eval_batch_size,

        # precision
        bf16=args.bf16,
        fp16=False,

        # optimizer / schedule
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,

        # generation-related
        max_prompt_length=args.max_prompt_len,
        max_completion_length=args.response_length,
        temperature=args.temperature,
        chat_template_kwargs={"enable_thinking": args.enable_thinking},

        # RL-related
        num_generations=args.num_generations,  # 每个 prompt 采样多少条 completion 做 GRPO
        num_generations_eval=args.num_generations_eval,
        beta=args.kl_coef,  # KL penalty，设为 0 就是纯 RLVR

        # logging / eval
        logging_steps=1,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy="no",
        report_to=["wandb"],

        # 需要保留 answer 字段给 reward_fn 用
        remove_unused_columns=False,

        seed=args.seed,
    )
    reward_fn = make_gsm8k_reward_fn(args.reward_mode)
    trainer = GRPOTrainer(
        model=policy_model,
        args=grpo_args,
        processing_class=tok,
        reward_funcs=reward_fn,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    if trainer.accelerator.is_main_process:
        # DeepSpeed Zero stage 信息（如有）
        if trainer.accelerator.state.deepspeed_plugin is not None:
            ds_cfg = trainer.accelerator.state.deepspeed_plugin.hf_ds_config
            print("=== ds_cfg type ===", type(ds_cfg))
            print("=== zero stage ===", ds_cfg.config["zero_optimization"]["stage"])
        wandb.init(project=args.proj_name, name=args.run_name)

    # ===== train =====
    trainer.train()
    trainer.accelerator.wait_for_everyone()

    # ===== save policy model =====
    if trainer.accelerator.is_main_process:
        trainer.save_model(args.output_dir)
        tok.save_pretrained(args.output_dir)
        print(f"[OK] GRPO RLVR policy saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
