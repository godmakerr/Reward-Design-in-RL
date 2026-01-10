#!/bin/bash
bash Scripts/train_grpo_rl_format_and_closeness.sh
sleep 10
bash Scripts/train_grpo_rl_format.sh
sleep 10
bash Scripts/train_grpo_rl.sh
