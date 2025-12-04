import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from omegaconf import OmegaConf

current_dir = os.path.dirname(os.path.abspath(__file__))
core_path = os.path.join(current_dir, "core")
if core_path not in sys.path: sys.path.insert(0, core_path)
if current_dir not in sys.path: sys.path.append(current_dir)

from custom_envs.joint_1dof_env import Joint1DofEnv
import torch.nn as nn

# === 必须与 train_sim.py 一致 ===
class CausalTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim=64, num_layers=2, num_heads=4, history_len=10):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, history_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4,
            batch_first=True, dropout=0.0, activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        B, T, D = x.shape
        emb = self.embedding(x) + self.pos_embed[:, :T, :]
        mask = torch.triu(torch.ones(T, T) * float('-inf'), diagonal=1).to(x.device)
        feat = self.transformer(emb, mask=mask)
        return self.head(feat[:, -1, :])

def get_features(state):
    dim = state.shape[-1]
    half = dim // 2
    q = state[..., :half]
    qd = state[..., half:]
    return torch.cat([torch.sin(q), torch.cos(q), qd], dim=-1)

def check():
    device = "cuda"
    cfg = OmegaConf.load("conf/joint_1dof.yaml")
    cfg.task.num_envs = 1 
    env = Joint1DofEnv(cfg.task, device=device)
    
    if not os.path.exists("nerd_stats.pt"): return
    stats = torch.load("nerd_stats.pt")
    for k, v in stats.items(): stats[k] = v.to(device)

    # 维度计算
    obs = env.reset()
    state_dim = obs.shape[-1] 
    feat_dim = (state_dim // 2) * 3 
    input_dim = feat_dim + 1
    output_dim = state_dim 
    
    # 简单的模型参数
    model = CausalTransformer(
        input_dim=input_dim, output_dim=output_dim, 
        embed_dim=64, num_layers=2, # 回退
        history_len=cfg.task.train.history_len
    ).to(device)
    
    model.load_state_dict(torch.load("nerd_sim_weights.pth"))
    model.eval()

    # GT 生成
    n_steps = 300
    h = cfg.task.train.history_len
    t = np.linspace(0, 6*np.pi, n_steps)
    # 动作幅度小一点，与训练一致
    raw_action = 2.0 * np.sin(t * np.linspace(0.5, 2.0, n_steps)) 
    
    actions = torch.tensor(raw_action, dtype=torch.float32, device=device).unsqueeze(1)
    actions_batch = actions.unsqueeze(1)

    gt_states = []
    curr = env.reset()
    for i in range(n_steps):
        next_s = env.step(actions_batch[i])
        gt_states.append(next_s[0]) 
    gt_states = torch.stack(gt_states)

    # NeRD 预测
    pred_states = []
    with torch.no_grad():
        for i in range(h, n_steps):
            s_raw_win = gt_states[i-h : i]
            a_raw_win = actions[i-h : i]
            
            s_feat = get_features(s_raw_win)
            s_norm = (s_feat - stats['s_mean']) / stats['s_std']
            a_norm = (a_raw_win - stats['a_mean']) / stats['a_std']
            
            model_in = torch.cat([s_norm, a_norm], dim=-1).unsqueeze(0)
            
            pred_delta_norm = model(model_in)[0]
            pred_delta_raw = pred_delta_norm * stats['d_std'] + stats['d_mean']
            
            # NeRD 预测的是 Delta，加到上一帧 GT 上
            pred_next = gt_states[i-1] + pred_delta_raw
            pred_states.append(pred_next.cpu().numpy())

    gt_numpy = gt_states.cpu().numpy()
    pred_numpy = np.array(pred_states)
    
    time = np.arange(h, n_steps)
    plt.figure(figsize=(10, 8))
    
    # Pos (Dim 0)
    plt.subplot(2, 1, 1)
    plt.plot(time, gt_numpy[h:, 0], 'k', label='GT', lw=2, alpha=0.6)
    plt.plot(time, pred_numpy[:, 0], 'r--', label='NeRD', lw=1.5)
    plt.title("Motor Position")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Vel (Dim 2)
    vel_idx = state_dim // 2
    plt.subplot(2, 1, 2)
    plt.plot(time, gt_numpy[h:, vel_idx], 'k', label='GT', alpha=0.6)
    plt.plot(time, pred_numpy[:, vel_idx], 'r--', label='NeRD', lw=1.5)
    plt.title("Motor Velocity")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("sim_check_result.png")
    print("✅ 画图完成")

if __name__ == "__main__":
    check()