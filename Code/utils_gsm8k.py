import re
import random
import json
import math
from typing import Optional, List, Any
import torch
from datasets import load_dataset

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl_as_dataset(jsonl_path: str):
    ds = load_dataset("json", data_files={"data": jsonl_path})["data"]
    return ds

def save_policy_only(trainer, tok, out_dir: str):
    acc = trainer.accelerator
    acc.wait_for_everyone()
    os.makedirs(out_dir, exist_ok=True)

    # 1) DeepSpeedEngine（一般 trainer.model 就是 DS engine）
    engine = getattr(trainer, "deepspeed", None) or trainer.model

    # 2) 所有 rank 都必须跑到这里（内部会 allgather）
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

FINAL_RE = re.compile(r"####\s*([-+]?\d[\d,]*\.?\d*)")  # 允许逗号和小数
NUM_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")            # 抓取任意数字（含逗号/小数）

def _to_float(num_str: str) -> Optional[float]:
    if num_str is None:
        return None
    s = num_str.strip()
    # 去掉常见符号
    s = s.replace(",", "")
    s = s.replace("$", "")
    s = s.replace("¥", "")
    s = s.replace("%", "")
    # 有些模型会输出类似 "18." 也能 float
    try:
        return float(s)
    except Exception:
        return None
    
def _strip_think_tail(text: Optional[str]) -> str:
    """
    If the model output contains <think>...</think>, only score the tail after </think>.
    This keeps your extraction stable when enable_thinking is on.
    """
    if text is None:
        return ""
    s = str(text)
    tag = "</think>"
    pos = s.rfind(tag)
    if pos != -1:
        return s[pos + len(tag):].strip()
    return s.strip()

def extract_final_number(text: str) -> Optional[float]:
    """
    Return final numeric answer as float.
    Priority:
      1) '#### <number>'
      2) last number in text
    """
    if text is None:
        return None

    m = FINAL_RE.search(text)
    if m:
        return _to_float(m.group(1))

    # fallback: take last number in the whole text
    nums = NUM_RE.findall(text)
    if not nums:
        return None
    return _to_float(nums[-1])


def build_rm_text(
    question: str,
    response: str,
    tokenizer=None,
    use_chat_template: bool = False,
    enable_thinking: bool = False,
) -> str:
    """
    For reward model scoring.
    - If use_chat_template=True and tokenizer supports it: format as chat.
    - Else: simple concat: question + "\n" + response
    """
    q = str(question)
    r = str(response)
    if use_chat_template and tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": q}, {"role": "assistant", "content": r}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
        )
    return q + "\n" + r


def corrupt_final_answer(answer_text: str, seed: Optional[int] = None) -> str:
    """
    Corrupt a GSM8K-style answer string to create a 'rejected' answer.
    Strategy:
      1) If has '#### <number>', replace that final number with a wrong nearby number.
      2) Else, find last number in text and replace it.
      3) If no number found, append a wrong final line.

    Keeps formatting similar to original to avoid trivial discrimination.
    """
    rng = random.Random(seed) if seed is not None else random

    if answer_text is None:
        return "#### 0"

    s = str(answer_text)

    # helper: sample a wrong number close-ish to gold
    def _wrong_number(gold: float) -> float:
        # Prefer integer-ish corruption when gold is near int
        is_int_like = abs(gold - round(gold)) < 1e-9
        base = int(round(gold)) if is_int_like else gold

        # choose corruption type
        # - small additive noise
        # - sign flip
        # - multiply by 10 or 2
        t = rng.random()
        if t < 0.55:
            delta = rng.choice([-5, -3, -2, -1, 1, 2, 3, 5, 7, 10])
            wrong = base + delta
        elif t < 0.75:
            wrong = -base if base != 0 else 1
        elif t < 0.90:
            wrong = base * rng.choice([2, 10]) if base != 0 else rng.choice([2, 10])
        else:
            wrong = base + rng.choice([11, 12, 13, 17, 19])

        # Avoid accidentally equal
        if wrong == base:
            wrong = base + 1

        return float(int(wrong)) if is_int_like else float(wrong)

    # 1) GSM8K canonical final line: #### <number>
    m = FINAL_RE.search(s)
    if m:
        gold = _to_float(m.group(1))
        if gold is None:
            # if parse fails, just replace the matched group with "0"
            return FINAL_RE.sub("#### 0", s, count=1)

        wrong = _wrong_number(gold)

        # keep integer formatting if possible
        rep = str(int(wrong)) if abs(wrong - round(wrong)) < 1e-9 else str(wrong)
        # Replace only the first occurrence of the final pattern
        return FINAL_RE.sub(f"#### {rep}", s, count=1)

    # 2) fallback: replace last number anywhere
    nums = NUM_RE.findall(s)
    if nums:
        gold = _to_float(nums[-1])
        if gold is None:
            # can't parse; blunt replace last matched substring with "0"
            last = nums[-1]
            idx = s.rfind(last)
            if idx >= 0:
                return s[:idx] + "0" + s[idx + len(last):]
            return s + "\n#### 0"

        wrong = _wrong_number(gold)
        rep = str(int(wrong)) if abs(wrong - round(wrong)) < 1e-9 else str(wrong)

        last = nums[-1]
        idx = s.rfind(last)
        if idx >= 0:
            return s[:idx] + rep + s[idx + len(last):]
        return s + f"\n#### {rep}"

    # 3) no number at all: append a wrong final answer
    return s.rstrip() + "\n#### 0"


def compute_binary_rewards(pred_responses: List[str], gold_answers: List[str]) -> List[float]:
    """
    +1 if final number matches gold; else -1.
    (Useful for sanity-check eval; PPO reward should come from RM in RLHF.)
    """
    out = []
    for pred, gold in zip(pred_responses, gold_answers):
        pred_tail = _strip_think_tail(pred)
        gold_tail = _strip_think_tail(gold)
        p = extract_final_number(pred_tail)
        g = extract_final_number(gold_tail)
        out.append(1.0 if (p is not None and g is not None and abs(p - g) < 1e-2) else -1.0)
    return out


def compute_format_rewards(pred_responses: List[str], gold_answers: List[str]) -> List[float]:
    """
    Only changes ONE condition vs. compute_binary_rewards:
      - Require strict GSM8K final format: must contain '#### <number>' in pred.
    Scoring:
      - if no '#### <number>' -> -1
      - else: correct (+1) / incorrect (-1) by numeric match
    """
    out: List[float] = []
    for pred, gold in zip(pred_responses, gold_answers):
        pred_tail = _strip_think_tail(pred)
        gold_tail = _strip_think_tail(gold)

        # Strict format requirement: must have #### <number>
        if FINAL_RE.search(pred_tail) is None:
            out.append(-1.0)
            continue

        p = extract_final_number(pred_tail)
        g = extract_final_number(gold_tail)
        out.append(1.0 if (p is not None and g is not None and abs(p - g) < 1e-2) else -1.0)

    return out


def compute_closeness_rewards(pred_responses: List[str], gold_answers: List[str]) -> List[float]:
    """
    Only changes ONE condition vs. compute_binary_rewards:
      - Replace hard +/-1 with a continuous reward based on numeric closeness.

    Reward (clipped to [-1, 1]):
      r = 2 * exp(-k * rel_err) - 1
      rel_err = |p - g| / max(1, |g|)
      - exact match -> 1
      - large error  -> approaches -1

    Does NOT require '#### <number>' (still uses your extract_final_number fallback).
    """
    out: List[float] = []
    k = 5.0  # decay strength; higher -> only very close answers get positive reward

    for pred, gold in zip(pred_responses, gold_answers):
        pred_tail = _strip_think_tail(pred)
        gold_tail = _strip_think_tail(gold)

        p = extract_final_number(pred_tail)
        g = extract_final_number(gold_tail)
        if p is None or g is None:
            out.append(-1.0)
            continue

        denom = max(1.0, abs(g))
        rel_err = abs(p - g) / denom
        r = 2.0 * math.exp(-k * rel_err) - 1.0

        # clip to [-1, 1]
        if r > 1.0:
            r = 1.0
        elif r < -1.0:
            r = -1.0

        out.append(float(r))

    return out

def compute_format_and_closeness_rewards(pred_responses: List[str], gold_answers: List[str]) -> List[float]:
    """
    Soft version of "format + closeness":
      - Base reward: continuous closeness based on numeric distance (same as before)
      - Format shaping: add a small bonus if '#### <number>' exists (and is parseable)
        instead of hard gating to -1 when missing.

    This makes rewards much less sparse early in training, while still pushing the model
    toward the GSM8K '#### <number>' convention.
    """
    out: List[float] = []
    k = 5.0

    # format shaping strength (tuneable)
    format_bonus = 0.15   # reward if format is present
    format_penalty = 0.05 # optional mild penalty if format is absent (set to 0.0 if you prefer)

    for pred, gold in zip(pred_responses, gold_answers):
        pred_tail = _strip_think_tail(pred)
        gold_tail = _strip_think_tail(gold)

        # Parse numbers (keep your existing extraction behavior)
        p = extract_final_number(pred_tail)
        g = extract_final_number(gold_tail)

        # If cannot parse predicted or gold number, give a strong negative (same spirit as before)
        if p is None or g is None:
            out.append(-1.0)
            continue

        # Base closeness reward in [-1, 1]
        denom = max(1.0, abs(g))
        rel_err = abs(p - g) / denom
        r = 2.0 * math.exp(-k * rel_err) - 1.0

        # Soft format shaping (NO hard -1 gating)
        has_hash = (FINAL_RE.search(pred_tail) is not None)
        if has_hash:
            r += format_bonus
        else:
            r -= format_penalty

        # Clip to [-1, 1]
        if r > 1.0:
            r = 1.0
        elif r < -1.0:
            r = -1.0

        out.append(float(r))

    return out





