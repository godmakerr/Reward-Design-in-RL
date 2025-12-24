# RLHF on GSM8k with Qwen-1.5B

本项目实现了基于 PPO (Proximal Policy Optimization) 算法对 Qwen-2.5-1.5B-Instruct 小模型在 GSM8k 数学推理任务上的强化学习微调。

## 📂 项目结构
```
.
├── Code/                核心训练与评估脚本
│   ├── README.md
│   ├── ppo_rlhf_gsm8k.py
│   ├── train_rm_gsm8k_json.py
│   └── eval_gsm8k_json.py
├── Datasets/            GSM8k 数据与处理后的 JSONL
│   └── README.md
├── Infer/               评估与推理输出
│   └── README.md
├── Models/              模型与检查点（已忽略）
│   └── README.md
├── Scripts/             训练与评估脚本入口
│   └── README.md
├── trl/                 第三方 TRL 库
│   └── README.md
├── wandb/               实验日志（已忽略）
│   └── README.md
├── requirements.txt     环境依赖文件
└── README.md            项目说明
```

各目录的细节说明见对应的 `README.md`。

## 🚀 快速开始

### 1. 安装依赖
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
