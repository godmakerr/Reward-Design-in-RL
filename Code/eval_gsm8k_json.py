import argparse
import json
import math
import os
from tqdm.auto import tqdm

import torch
from accelerate import Accelerator
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from Code.utils_gsm8k import extract_final_number

try:
    from trl import AutoModelForCausalLMWithValueHead
except Exception:
    AutoModelForCausalLMWithValueHead = None


def build_prompt(question: str, tokenizer, prompt_suffix: str, use_chat_template: bool, enable_thinking: bool) -> str:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": f"{question}{prompt_suffix}"}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    return f"{question}{prompt_suffix}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--data_jsonl", type=str, required=True)
    ap.add_argument("--output_path", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--use_chat_template", action="store_true")
    ap.add_argument("--enable_thinking", action="store_true")
    ap.add_argument("--prompt_suffix", type=str, default="")
    args = ap.parse_args()

    accelerator = Accelerator()

    dataset = load_dataset("json", data_files={"test": args.data_jsonl})["test"]
    total = len(dataset)
    shard_size = math.ceil(total / accelerator.num_processes)
    start = shard_size * accelerator.process_index
    end = min(start + shard_size, total)
    if start >= total:
        local_dataset = dataset.select([])
    else:
        local_dataset = dataset.select(range(start, end))

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if AutoModelForCausalLMWithValueHead is not None:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            args.model_dir,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

    model = accelerator.prepare(model)
    model.eval()
    gen_model = accelerator.unwrap_model(model)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    shard_path = f"{args.output_path}.rank{accelerator.process_index}"
    shard_file = open(shard_path, "w", encoding="utf-8")

    correct = 0
    processed = 0
    
    if accelerator.is_main_process:
        pbar = tqdm(
            total=len(local_dataset),
            desc="Evaluating",
            dynamic_ncols=True,
        )
    else:
        pbar = None

    for idx in range(0, len(local_dataset), args.batch_size):
        batch = local_dataset[idx : idx + args.batch_size]
        questions = [str(q) for q in batch["question"]]
        answers = [str(a) for a in batch["answer"]]

        prompts = [
            build_prompt(q, tokenizer, args.prompt_suffix, args.use_chat_template, args.enable_thinking)
            for q in questions
        ]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        encoded = {k: v.to(accelerator.device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = gen_model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        responses = []
        attention_mask = encoded.get("attention_mask")
        for i in range(outputs.shape[0]):
            prompt_len = int(attention_mask[i].sum().item()) if attention_mask is not None else encoded["input_ids"].shape[1]
            responses.append(tokenizer.decode(outputs[i][prompt_len:], skip_special_tokens=True))

        for question, gold, resp in zip(questions, answers, responses):
            pred_num = extract_final_number(resp)
            gold_num = extract_final_number(gold)
            is_right = pred_num is not None and gold_num is not None and abs(pred_num - gold_num) < 1e-2
            if is_right:
                correct += 1
            processed += 1

            record = {
                "question": question,
                "answer": gold,
                "prediction": resp,
                "pred_final": pred_num,
                "gold_final": gold_num,
                "correct": is_right,
            }
            shard_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        if pbar is not None:
            pbar.update(args.batch_size)
    if pbar is not None:
        pbar.close()

    shard_file.close()

    stats = torch.tensor([correct, processed], device=accelerator.device, dtype=torch.long)
    gathered = accelerator.gather_for_metrics(stats)
    if gathered.dim() == 1:
        print("gathered维度为1")
        gathered = gathered.view(-1, 2)
    total_correct = int(gathered[:, 0].sum().item())
    total_seen = int(gathered[:, 1].sum().item())

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        acc = total_correct / max(1, total_seen)
        print(f"[EVAL] accuracy={acc:.4f} ({total_correct}/{total_seen})")
        with open(args.output_path, "w", encoding="utf-8") as out_f:
            for rank in range(accelerator.num_processes):
                rank_path = f"{args.output_path}.rank{rank}"
                if not os.path.exists(rank_path):
                    continue
                with open(rank_path, "r", encoding="utf-8") as in_f:
                    for line in in_f:
                        out_f.write(line)
                os.remove(rank_path)


if __name__ == "__main__":
    main()
