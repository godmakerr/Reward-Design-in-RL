import os, torch
from safetensors.torch import save_file

d = "/root/fu_wj/clone2github/Reward-Design-in-RL/Models/qwen_3_1_7b_copy"
bin_path = os.path.join(d, "pytorch_model.bin")
sd = torch.load(bin_path, map_location="cpu")
if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
    sd = sd["state_dict"]

# 抽 policy.* 并去掉前缀 -> 变成 model.* / lm_head.* 这种 Qwen 能识别的键
policy_sd = {}
for k,v in sd.items():
    if k.startswith("policy."):
        policy_sd[k[len("policy."):]] = v

assert any(k.startswith("model.") for k in policy_sd.keys()), "没抽到 policy 的主干权重，检查 bin 里键名！"
assert "lm_head.weight" in policy_sd or any(k.endswith("lm_head.weight") for k in policy_sd.keys()), "没抽到 lm_head，检查 bin 里键名！"

out = os.path.join(d, "model.safetensors")
save_file(policy_sd, out)
print("OK: wrote", out, "keys=", len(policy_sd))

# 非常重要：别留 shard index，否则 transformers 会以为你有分片
idx = os.path.join(d, "model.safetensors.index.json")
if os.path.exists(idx):
    os.remove(idx)
    print("removed", idx)