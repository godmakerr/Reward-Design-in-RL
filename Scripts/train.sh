#!/bin/bash

bash Scripts/train_grpo_rl_format_and_closeness.sh
sleep 60
bash Scripts/train_grpo_rl.sh
sleep 60
bash Scripts/train_grpo_rl_format.sh
sleep 60
bash Scripts/train_grpo_rl_closeness.sh