import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re

# ===========================
# 配置
# ===========================
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
# 使用测试集的前 200 条作为评估基准 (与训练时保持一致或使用全量)
TEST_SIZE = 200 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"正在加载基座模型: {MODEL_NAME} ...")

# 1. 加载原始模型 (不带 Value Head，不带 Adapter)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.eval() # 开启评估模式

# 2. 准备数据
dataset = load_dataset("gsm8k", "main", split=f"test[:{TEST_SIZE}]")

# 3. 辅助函数
def extract_answer(text):
    # 提取最后一个数字
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

# 4. 开始评估
print(f"开始评估 {TEST_SIZE} 条数据 (Before RLHF)...")
correct_count = 0

for example in tqdm(dataset):
    question = example['question']
    ground_truth = example['answer'].split("####")[-1].strip()
    
    # 构造 Prompt (必须与训练时完全一致)
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
            max_new_tokens=512,
            do_sample=False, # 贪婪搜索，保证结果确定性
            pad_token_id=tokenizer.eos_token_id
        )
        
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    
    if is_correct(response, ground_truth):
        correct_count += 1

accuracy = correct_count / TEST_SIZE
print("\n" + "="*30)
print(f"基座模型 (Before RLHF) 评估结果:")
print(f"Accuracy: {accuracy:.2%} ({correct_count}/{TEST_SIZE})")
print("="*30)