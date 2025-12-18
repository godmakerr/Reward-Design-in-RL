import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import re
import os

# ===========================
# 配置参数
# ===========================
BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
# ！！！这里换成你训练结束时保存权重的文件夹名！！！
# 根據上一轮代码，最后的保存路径应该是 "gsm8k-ppo-final" 
# 如果你那是中间停掉的，可能是 "checkpoints/step_xxx"
ADAPTER_PATH = "gsm8k-ppo-final" 

TEST_SIZE = 200  # 和基线测试保持一致，保证公平
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"正在加载基座模型: {BASE_MODEL_NAME} ...")

# 1. 加载基座模型
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# 2. 加载 RLHF 训练后的 LoRA 权重
print(f"正在加载 RLHF LoRA 权重: {ADAPTER_PATH} ...")
if not os.path.exists(ADAPTER_PATH):
    raise FileNotFoundError(f"找不到文件夹: {ADAPTER_PATH}，请检查你的训练脚本最后保存的路径名字！")

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval() # 开启评估模式
# model.merge_and_unload() # 可选：如果推理太慢，可以合并权重，但在 peft 0.7+ 通常不需要

# 3. 准备测试数据
print(f"正在加载测试集 (前 {TEST_SIZE} 条)...")
dataset = load_dataset("gsm8k", "main", split=f"test[:{TEST_SIZE}]")

# 4. 辅助函数 (保持和基线测试完全一致)
def extract_answer(text):
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    if numbers:
        return numbers[-1]
    return None

def is_correct(response, ground_truth):
    pred = extract_answer(response)
    gt = extract_answer(ground_truth)
    try:
        if pred and gt and float(pred.replace(',', '')) == float(gt.replace(',', '')):
            return True
    except:
        pass
    return False

# 5. 开始评估
print("="*50)
print("开始评估 RLHF 模型...")
correct_count = 0

for example in tqdm(dataset):
    question = example['question']
    ground_truth = example['answer'].split("####")[-1].strip()
    
    # 构造 Prompt
    messages = [
        {"role": "user", "content": f"{question}\nLet's think step by step."}
    ]
    text_input = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text_input, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512, # 长度给够
            do_sample=False,    # 评估时通常关掉采样，用贪婪搜索，结果更稳定
            pad_token_id=tokenizer.eos_token_id
        )
        
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    
    if is_correct(response, ground_truth):
        correct_count += 1

accuracy = correct_count / TEST_SIZE
print("\n" + "="*30)
print(f"RLHF 模型 (After PPO) 最终评估结果:")
print(f"文件夹: {ADAPTER_PATH}")
print(f"Accuracy: {accuracy:.2%} ({correct_count}/{TEST_SIZE})")
print("="*30)