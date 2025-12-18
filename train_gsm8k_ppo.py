import os
import warnings
import logging
from transformers import logging as hf_logging
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from tqdm import tqdm
import wandb
import re
import numpy as np

# --- 0. 环境设置 ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
hf_logging.set_verbosity_error()

# ===========================
# 1. 关键超参配置 (解决报错与稳定性)
# ===========================
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct" 
WANDB_PROJECT = "rl-gsm8k-final"

config = PPOConfig(
    model_name=MODEL_NAME,
    # [关键调整] 学习率调小，防止震荡
    learning_rate=5e-6, 
    
    # [关键调整] 核心公式：Batch Size (采样池) / (Mini Batch * Grad Acc) = 整数
    # 这里的 batch_size=64 意味着：每做 64 道题，才更新一次参数。这能极大降低噪声！
    batch_size=32,          
    mini_batch_size=2,      # 显存限制，保持 2
    gradient_accumulation_steps=4, # 2*4=8，64能被8整除，修复报错
    
    optimize_cuda_cache=True,
    target_kl=0.1,
    ppo_epochs=2,
    seed=42,
    log_with="wandb",
    tracker_project_name=WANDB_PROJECT,
)

# ===========================
# 2. 数据准备 (训练集 + 测试集)
# ===========================
# 加载全量训练集
train_dataset = load_dataset("gsm8k", "main", split="train[:1000]") 
# 加载测试集用于评估 Acc
eval_dataset = load_dataset("gsm8k", "main", split="test[:50]") # 为了速度，每次只测50个

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def build_dataset(example):
    ground_truth = example['answer'].split("####")[-1].strip()
    messages = [{"role": "user", "content": f"{example['question']}\nLet's think step by step."}]
    prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    prompt_str = tokenizer.decode(prompt_ids)
    return {
        "input_ids": prompt_ids, 
        "query": f"{prompt_str}|||GT|||{ground_truth}", # 拼接GT方便后续取用
        "ground_truth_clean": ground_truth # 保留纯净的 GT 用于评估
    }

train_dataset = train_dataset.map(build_dataset, remove_columns=["question", "answer"])
# eval 数据集不需要 map input_ids，因为我们在评估时是动态生成的
# 但为了统一格式，我们简单处理一下 query
eval_dataset = eval_dataset.map(lambda x: {"query": x["question"], "ground_truth": x["answer"].split("####")[-1].strip()})

def my_collate_fn(data):
    return {
        "input_ids": [d["input_ids"] for d in data],
        "query": [d["query"] for d in data],
    }

# ===========================
# 3. 模型与 LoRA
# ===========================
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    peft_config=lora_config,
    torch_dtype=torch.float16,
    device_map="auto", 
    trust_remote_code=True
)

# ===========================
# 4. 奖励函数 & 评估函数
# ===========================
def extract_number(text):
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    if numbers: return numbers[-1]
    return None

def compute_reward_score(response_text, ground_truth):
    # 奖励函数逻辑
    if "Question:" in response_text:
        response_text = response_text.split("Question:")[0]
    pred_num = extract_number(response_text)
    try:
        # 宽松匹配
        if pred_num and float(pred_num.replace(',', '')) == float(ground_truth.replace(',', '')):
            return 1.0
    except:
        pass
    return 0.0

# [新增] 专门用于评估准确率的函数
def evaluate_accuracy(ppo_trainer, eval_ds, generation_kwargs):
    print("\n>>> 开始评估测试集 Accuracy...")
    correct_count = 0
    total = len(eval_ds)
    
    # 构建 batch
    queries = [f"{q}\nLet's think step by step." for q in eval_ds["query"]]
    # 批量编码
    inputs = [tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=True, add_generation_prompt=True) for q in eval_ds["query"]]
    input_tensors = [torch.tensor(i).to(ppo_trainer.accelerator.device) for i in inputs]

    # 为了不爆显存，我们需要手动切分 eval batch
    eval_batch_size = 4
    for i in range(0, total, eval_batch_size):
        batch_input = input_tensors[i : i+eval_batch_size]
        batch_gt = eval_ds["ground_truth"][i : i+eval_batch_size]
        
        with torch.no_grad():
            # 生成
            batch_responses = ppo_trainer.generate(batch_input, return_prompt=False, **generation_kwargs)
            batch_responses_text = tokenizer.batch_decode(batch_responses, skip_special_tokens=True)
            
            for resp, gt in zip(batch_responses_text, batch_gt):
                score = compute_reward_score(resp, gt)
                if score == 1.0:
                    correct_count += 1
    
    acc = correct_count / total
    print(f">>> 评估结束: Accuracy = {acc:.2%} ({correct_count}/{total})")
    return acc

# ===========================
# 5. 训练循环
# ===========================
ppo_trainer = PPOTrainer(
    config, model, ref_model=None, tokenizer=tokenizer, dataset=train_dataset, data_collator=my_collate_fn
)

generation_kwargs = {
    "min_length": -1, "top_k": 0.0, "top_p": 1.0, "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 512,
}

print(f"开始训练 (Batch Size={config.batch_size}, Total Updates={len(ppo_trainer.dataloader)})...")

# 手动初始化 wandb 以确保 config 同步
if ppo_trainer.accelerator.is_main_process:
    if wandb.run is None:
        wandb.init(project=WANDB_PROJECT, config=config)

for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    # 1. 解析数据
    query_tensors = [torch.tensor(ids).to(ppo_trainer.accelerator.device) for ids in batch["input_ids"]]
    real_prompts, ground_truths = [], []
    for q_str in batch["query"]:
        if "|||GT|||" in q_str:
            p, g = q_str.split("|||GT|||")
            real_prompts.append(p)
            ground_truths.append(g)
        else:
            real_prompts.append(q_str)
            ground_truths.append("")

    # 2. 生成 (Rollout)
    response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)
    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
    
    # 3. 计算奖励
    rewards = [torch.tensor(compute_reward_score(r, gt), dtype=torch.float32) for r, gt in zip(batch["response"], ground_truths)]
    
    # 4. PPO 更新
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    
    # 5. 数据处理与日志
    batch_mean_reward = torch.stack(rewards).mean()
    
    # 辅助函数
    def to_float(x):
        if hasattr(x, "item"): return x.item()
        if isinstance(x, list): return sum(x) / len(x)
        return float(x)

    # 记录 Training Log
    logs = {
        "train/reward": batch_mean_reward.item(),
        "train/loss": to_float(stats['ppo/loss/total']),
        "train/kl": to_float(stats['objective/kl']),
    }
    
    # === [关键新增] 每隔 20 个 Step 跑一次测试集评估 ===
    # 为什么是 20？因为现在 batch_size=64，20步相当于看过了 1280 条数据
    if step > 0 and step % 20 == 0:
        acc = evaluate_accuracy(ppo_trainer, eval_dataset, generation_kwargs)
        logs["eval/accuracy"] = acc
        
        # 顺便保存一下中间 Checkpoint
        save_path = f"checkpoints/step_{step}"
        if not os.path.exists(save_path): os.makedirs(save_path)
        ppo_trainer.model.save_pretrained(save_path)

    # 上传日志
    if ppo_trainer.accelerator.is_main_process:
        wandb.log(logs)
    
    # 终端简略打印
    if step % 5 == 0:
        print(f"Step {step}: Reward={logs['train/reward']:.2f} | KL={logs['train/kl']:.4f}")

# 最终保存
final_dir = "gsm8k-ppo-final"
if not os.path.exists(final_dir): os.makedirs(final_dir)
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
print("训练完成！")