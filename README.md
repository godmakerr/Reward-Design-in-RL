# RLHF on GSM8k with Qwen-1.5B

本项目实现了基于 PPO (Proximal Policy Optimization) 算法对 Qwen-2.5-1.5B-Instruct 小模型在 GSM8k 数学推理任务上的强化学习微调。

## 📂 项目结构
- `train_gsm8k_ppo.py`: PPO 强化学习训练主脚本（包含 Reward Function 定义）。
- `eval_base_model.py`: 原始模型基准测试脚本。
- `requirements.txt`: 环境依赖文件。

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
