# RLHF on GSM8k with Qwen3-1.7B

本项目实现了基于 PPO (Proximal Policy Optimization) 算法对 Qwen-3-1.7B 模型在 GSM8k 数学推理任务上的强化学习微调。

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

### 2. 运行训练脚本
```bash
bash Scripts/train_rm.sh
bash Scripts/train_ppo_rlhf.sh
```

### 3. 运行评估脚本
```bash
bash Scripts/eval.sh
```

评估结果会写入 `Infer/gsm8k_predictions.jsonl`。
