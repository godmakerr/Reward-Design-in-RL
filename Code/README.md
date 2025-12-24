# Code

该目录包含 GSM8k 任务的训练与评估核心脚本，以及配套配置文件。

## 主要文件
- `train_rm_gsm8k_json.py`: 训练奖励模型（Reward Model）。
- `ppo_rlhf_gsm8k.py`: 使用 PPO 进行 RLHF 训练。
- `eval_gsm8k_json.py`: 在 GSM8k 上评估模型并输出预测。
- `utils_gsm8k.py`: 数据处理与评估工具函数。
- `convert_to_safetensors.py`: 模型权重格式转换辅助脚本。

## 子目录
- `accelerate/`: Accelerate 多卡训练配置。
- `deepspeed/`: DeepSpeed ZeRO 训练配置。
