# ppo_custom.py
import torch
import torch.nn as nn
import numpy as np

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes=(64, 64)):
        super().__init__()
        layers = []
        last_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.Tanh())
            last_dim = h
        # 末尾输出层，对应 SB3 的 action_net
        layers.append(nn.Linear(last_dim, action_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PPOAgent:
    def __init__(self, obs_dim, action_dim, policy_path, device="cpu"):
        self.device = device
        self.net = PolicyNet(obs_dim, action_dim).to(device)

        # 1. 载入原始的完整 state_dict
        raw = torch.load(policy_path, map_location=device)

        # 2. 筛选出跟 “policy_net” 和 “action_net” 相关的键，并重命名到你的 model.* 上
        policy_sd = {}
        for k, v in raw.items():
            # SB3 里两层隐藏层的键以 "mlp_extractor.policy_net.{idx}.weight/bias" 开头
            if k.startswith("mlp_extractor.policy_net"):
                # 把 "mlp_extractor.policy_net.0.weight" → "model.0.weight"
                new_key = "model" + k[len("mlp_extractor.policy_net"):]
                policy_sd[new_key] = v
            # SB3 最后一层 action_net
            elif k == "action_net.weight":
                policy_sd["model.4.weight"] = v
            elif k == "action_net.bias":
                policy_sd["model.4.bias"] = v

        # 3. 加载到你的网络里
        self.net.load_state_dict(policy_sd)
        self.net.eval()

    def choose_action(self, obs: np.ndarray) -> int:
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.net(x)
        return int(torch.argmax(logits, dim=-1).item())
