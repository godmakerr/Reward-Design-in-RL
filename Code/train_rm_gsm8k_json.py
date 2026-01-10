# Code/train_rm_gsm8k_json.py
import argparse
import os
import random

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from trl.trainer.reward_config import RewardConfig
from trl.trainer.reward_trainer import RewardTrainer

from Code.utils_gsm8k import corrupt_final_answer, build_rm_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", type=str, required=True)
    ap.add_argument("--base_model_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--per_device_batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--use_chat_template", action="store_true")
    ap.add_argument("--enable_thinking", action="store_true")

    # 可选：datasets.map 多进程
    ap.add_argument("--dataset_num_proc", type=int, default=1)

    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) load dataset
    ds = load_dataset("json", data_files={"train": args.train_jsonl})["train"]

    # 2) tokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # 3) build preference pairs (chosen/rejected text)
    def map_fn(ex):
        q = str(ex["question"])
        chosen_ans = str(ex["answer"])
        rejected_ans = corrupt_final_answer(chosen_ans)

        chosen = build_rm_text(
            q, chosen_ans,
            tokenizer=tok,
            use_chat_template=args.use_chat_template,
            enable_thinking=args.enable_thinking,
        )
        rejected = build_rm_text(
            q, rejected_ans,
            tokenizer=tok,
            use_chat_template=args.use_chat_template,
            enable_thinking=args.enable_thinking,
        )
        return {"chosen": chosen, "rejected": rejected}

    pref = ds.map(
        map_fn,
        remove_columns=ds.column_names,
        num_proc=args.dataset_num_proc if args.dataset_num_proc and args.dataset_num_proc > 1 else None,
        desc="Building (chosen, rejected) pairs",
    )

    # 4) reward model (sequence classification head)
    rm = AutoModelForSequenceClassification.from_pretrained(
        args.base_model_dir,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    rm.config.pad_token_id = tok.pad_token_id

    # 5) TRL RewardConfig (NOT TrainingArguments)
    rargs = RewardConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        bf16=True,
        fp16=False,
        remove_unused_columns=False,
        report_to=[],
        seed=args.seed,

        # ✅ 关键：RewardTrainer 从 args.max_length 读取，并在内部 tokenize + filter 超长样本
        max_length=args.max_length,

        # 可选优化：padding 对齐，提升效率
        gradient_checkpointing=True,
        pad_to_multiple_of=8,
        dataset_num_proc=args.dataset_num_proc if args.dataset_num_proc and args.dataset_num_proc > 1 else None,
    )

    # 6) trainer
    trainer = RewardTrainer(
        model=rm,
        args=rargs,
        train_dataset=pref,
        processing_class=tok,
    )

    trainer.train()

    # 7) save
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"[OK] RM saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
