import re
from typing import Optional, List, Any

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

import random
from typing import Optional

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
        p = extract_final_number(pred)
        g = extract_final_number(gold)
        out.append(1.0 if (p is not None and g is not None and abs(p - g) < 1e-2) else -1.0)
    return out
