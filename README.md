# RLHF on GSM8k with Qwen3-1.7B

本项目实现了基于 PPO 和 GRPO 算法对 Qwen-3-1.7B 模型在 GSM8k 数学推理任务上的强化学习微调，使用 trl 库。

## Papers will comming soon

## 📂 项目结构
```
.
├── Code/                核心训练与评估脚本
│   ├── README.md
│   ├── ppo_rlhf_gsm8k.py         PPO 训练代码
│   ├── train_rm_gsm8k_json.py    RM 训练代码
│   ├── grpo_rl_gsm8k.py          GRPO 训练代码         
│   ├── eval_gsm8k_json.py        GSM8k 评估代码
│   └── utils_gsm8k.py            工具函数（包括奖励计算、结果匹配等代码）
├── Datasets/            GSM8k 数据与处理后的 JSONL
│   ├── gsm8k/                    GSM8k 数据
│   ├── gsm8k_json/               JSONL 格式的 GSM8k 数据
│   │   ├── test.jsonl            测试集  
│   │   └── train.jsonl           训练集
│   └── README.md
├── Infer/               评估与推理输出
│   ├── gsm8k_predictions_*.jsonl 模型推理的输出
│   └── README.md
├── Models/              模型与检查点（请在 modelscope 平台上下载）
├── Scripts/             训练与评估脚本入口
│   ├── eval.sh                   评估脚本
│   ├── train_rm.sh               RM 训练脚本
│   ├── train_grpo_rl.sh          GRPO 训练脚本
│   ├── train_ppo_rlhf.sh         PPO 训练脚本  
│   └── README.md
├── trl/                 第三方 TRL 库
├── requirements.txt     环境依赖文件
└── README.md            项目说明
```

各目录的细节说明见对应的 `README.md`。

## 🚀 快速开始

### 1. 安装依赖
#### 1.1 安装 torch
我们使用的是 2.9.0 版本的 torch，CUDA 版本为 12.6。

```bash
pip install torch==2.9.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

#### 1.2 安装 trl（源码安装）
我们使用的是 0.27.0.dev0 最新开发版本的 trl，可通过源码安装。trl目录位于项目目录下，位置详见📂 项目结构。

```bash
git clone https://github.com/huggingface/trl.git
cd trl/
pip install -e .
```

#### 1.3 安装其他
```bash
pip install -r requirements.txt
```
### 2. 下载模型与配置环境
#### 2.1 下载模型
本项目基于 Qwen3-1.7B 模型进行训练。请从 ModelScope 下载模型（其中包含我们已经训练好的模型）：
```bash
# 创建 Models 文件夹
mkdir Models
# 方法1：使用 modelscope 命令行工具
modelscope download --model shireshire/prml_qwen3_1_7b --local_dir Models

# 方法2：手动下载后放置到 Models/ 目录下
# 从 https://modelscope.cn/models/shireshire/prml_qwen3_1_7b 手动下载并放入 Models/ 目录
```
如果想要自行下载，请将模型重命名为`qwen_3_1_7b`并放入 Models/ 目录。
#### 2.2 配置环境
- 登录 wandb 便于可视化训练流程：
```bash
wandb login
```

- 根据你的 GPU 配置修改脚本（位于 Scripts/ 目录下）中的环境变量：
```bash
# 在 Scripts/ 目录下的脚本中修改：
# 1. 设置可用的 GPU 设备（根据实际 GPU 数量调整）
export CUDA_VISIBLE_DEVICES=0,1,4,5  # 使用4个GPU

# 2. 更新项目路径（如果项目位置不同）
BASE_DIR="/root/fu_wj/clone2github/Reward-Design-in-RL"
export PYTHONPATH="${BASE_DIR}/Code:$PYTHONPATH"

# 3. 内存优化设置（可选，用于处理大模型）
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
#### 2.3 配置训练参数

主要训练参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--per_device_batch_size` | 2 (RM) / 16 (GRPO) | 每GPU批大小 |
| `--gradient_accumulation_steps` | 16 (RM) / 8 (GRPO) | 梯度累积步数 |
| `--num_train_epochs` | 1 | 训练轮数 |
| `--max_length` | 512 | 最大序列长度 |
| `--response_length` | 512 | 生成响应长度 |
| `--kl_coef` | 0.001 | KL散度系数（GRPO） |

根据你的 GPU 显存调整参数：

- 如果遇到 **OOM（显存不足）**，减小 `per_device_batch_size`。
- 如果想**加快训练**，增大 `gradient_accumulation_steps`。

我们的实验均在4张Nvidia GeForce RTX 3090 GPU上完成。

### 3. 训练与评估流程
#### 3.1 完整训练流程
```bash
# 1. PPO 训练流程
## 1.1 训练奖励模型 或 使用现有的奖励模型（如 Models/skywork_reward_qwen_3_1_7b）
bash Scripts/train_rm.sh

## 1.2 使用 PPO 进行 RLHF 训练
bash Scripts/train_ppo_rlhf.sh

# 2. GRPO 训练流程  
bash Scripts/train_grpo_rl.sh

# 3. 评估训练后的模型
bash Scripts/eval.sh
```

## 实验结果
### 1. 模型说明
- `qwen_3_1_7b/`: Baseline 模型。
- `skywork_reward_qwen_3_1_7b/`: skywork RM。
- `qwen_3_1_7b_rm/`: 自己训练的 RM。
- `qwen_3_1_7b_ppo_bf16/`: PPO 微调后的模型。
- `qwen_3_1_7b_ppo_bf16_selfrm/`: 使用自训练 RM 的 PPO 微调后的模型。
- `qwen_3_1_7b_grpo_bf16/`: GRPO 微调（KL散度系数为0，奖励为答案）后的模型。
- `qwen_3_1_7b_grpo_bf16_kl/`: GRPO 微调（KL散度系数为0.01，奖励为答案）后的模型。
- `qwen_3_1_7b_grpo_bf16_kl_0.001/`: GRPO 微调（KL散度系数为0.001，奖励为答案）后的模型。
- `qwen_3_1_7b_grpo_bf16_kl_0.001_format/`: GRPO 微调（KL散度系数为0.001，奖励为格式奖励）后的模型。
- `qwen_3_1_7b_grpo_bf16_kl_0.001_closeness/`: GRPO 微调（KL散度系数为0.001，奖励为与答案的接近程度）后的模型。
- `qwen_3_1_7b_grpo_bf16_kl_0.001_format_and_closeness/`: GRPO 微调（KL散度系数为0.001，奖励为与答案接近程度和格式奖励的加权）后的模型。

### 2. GSM8K 评测结果（分数从高到低）
| 模型名称 | 准确率 | 正确数/总数 |
|----------|--------|-------------|
| grpo | 82.03% | 1082/1319 |
| grpo_kl | 80.82% | 1066/1319 |
| grpo_kl_0.001_closeness | 80.29% | 1059/1319 |
| grpo_kl_0.001 | 80.06% | 1056/1319 |
| grpo_kl_0.001_format_and_closeness | 79.30% | 1046/1319 |
| ppo_self_rm | 78.17% | 1031/1319 |
| ppo_skywork_rm | 77.94% | 1028/1319 |
| baseline | 77.86% | 1027/1319 |
| grpo_kl_0.001_format | 76.72% | 1012/1319 |