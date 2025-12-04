import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import sys
import random

current_dir = os.path.dirname(os.path.abspath(__file__))
core_path = os.path.join(current_dir, "core")
if core_path not in sys.path: sys.path.insert(0, core_path)
if current_dir not in sys.path: sys.path.append(current_dir)

try:
    from custom_envs.joint_1dof_env import Joint1DofEnv
except ImportError:
    sys.exit(1)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==============================================================================
# Model (保持简单稳健)
# ==============================================================================
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

# ==============================================================================
# Structured Dataset (修复瞬移 Bug)
# ==============================================================================
class StructuredSimDataset(Dataset):
    def __init__(self, states, actions, targets, history_len=10):
        # 接收数据形状: [N, T, D] (Env, Time, Dim)
        # 这样我们可以安全地在 Time 轴上切片，而不跨越 Env
        self.states = states
        self.actions = actions
        self.targets = targets
        self.h = history_len
        
        self.num_envs = states.shape[0]
        self.num_steps = states.shape[1]
        
        # 每个环境能提供的样本数 = Time - History
        self.samples_per_env = self.num_steps - self.h
        self.length = self.num_envs * self.samples_per_env

    def __len__(self):
        return max(0, self.length)

    def __getitem__(self, idx):
        # 将线性索引 idx 映射回 (env_idx, time_idx)
        env_idx = idx // self.samples_per_env
        time_idx = idx % self.samples_per_env
        
        # 构造窗口 [time_idx : time_idx + h]
        # 注意：inputs 的最后一帧是 time_idx + h - 1
        s_win = self.states[env_idx, time_idx : time_idx+self.h]
        a_win = self.actions[env_idx, time_idx : time_idx+self.h]
        
        inputs = torch.cat([s_win, a_win], dim=-1)
        
        # Target 对应 inputs 的最后一帧的下一帧预测
        target = self.targets[env_idx, time_idx + self.h - 1]
        
        return inputs, target

class SimpleTrainer:
    def __init__(self, cfg, env, device):
        self.cfg = cfg
        self.env = env
        self.device = device
        self.model = None
        self.stats = None

    def collect_and_process_data(self):
        n_steps = 400
        print(f"[Sim] Collecting {n_steps} steps (512 Envs)...")
        
        states_list = []
        actions_list = []
        
        current_state = self.env.reset()
        for _ in range(10): self.env.step(torch.zeros(self.env.num_envs, 1, device=self.device))
        
        for i in range(n_steps):
            action = (torch.rand(self.env.num_envs, 1, device=self.device) * 4.0) - 2.0
            next_state = self.env.step(action)
            
            states_list.append(current_state) # [N, D]
            actions_list.append(action)       # [N, 1]
            current_state = next_state

        # Stack -> [T, N, D]
        raw_states = torch.stack(states_list, dim=0)
        raw_actions = torch.stack(actions_list, dim=0)
        
        # === 1. 计算统计量 (基于原始数据) ===
        print("[Sim] Computing Statistics...")
        # 特征工程临时计算
        dim = raw_states.shape[-1]
        half = dim // 2
        q = raw_states[..., :half]
        qd = raw_states[..., half:]
        feat_states = torch.cat([torch.sin(q), torch.cos(q), qd], dim=-1)
        
        # 全局均值/方差
        s_mean = feat_states.mean(dim=(0, 1))
        s_std = feat_states.std(dim=(0, 1)) + 1e-5
        a_mean = raw_actions.mean(dim=(0, 1))
        a_std = raw_actions.std(dim=(0, 1)) + 1e-5
        
        # 计算 Delta (严格沿时间轴)
        raw_deltas = raw_states[1:] - raw_states[:-1] # [T-1, N, D]
        d_mean = raw_deltas.mean(dim=(0, 1))
        d_std = raw_deltas.std(dim=(0, 1)) + 1e-5
        
        print(f"   >>> Delta Std: {d_std.cpu().numpy()}")
        
        self.stats = {
            's_mean': s_mean, 's_std': s_std,
            'a_mean': a_mean, 'a_std': a_std,
            'd_mean': d_mean, 'd_std': d_std
        }
        
        # === 2. 归一化数据 (在构建 Dataset 前完成) ===
        # 切片对齐: Inputs 用 [0 ~ T-1], Targets 用 Deltas
        # 丢弃最后一步 state，因为它没有 delta target
        valid_states = raw_states[:-1] 
        valid_actions = raw_actions[:-1]
        
        # 归一化 States
        q = valid_states[..., :half]
        qd = valid_states[..., half:]
        feat = torch.cat([torch.sin(q), torch.cos(q), qd], dim=-1)
        norm_states = (feat - s_mean) / s_std
        
        # 归一化 Actions
        norm_actions = (valid_actions - a_mean) / a_std
        
        # 归一化 Targets
        norm_targets = (raw_deltas - d_mean) / d_std
        
        # === 3. 转换维度: [T, N, D] -> [N, T, D] ===
        # 这样 Dataset 可以简单地按 Env 索引
        final_states = norm_states.permute(1, 0, 2).contiguous() # [N, T-1, 6]
        final_actions = norm_actions.permute(1, 0, 2).contiguous() # [N, T-1, 1]
        final_targets = norm_targets.permute(1, 0, 2).contiguous() # [N, T-1, 4]
        
        return final_states, final_actions, final_targets

    def train(self):
        # 获取结构化数据 [N, T, D]
        states, actions, targets = self.collect_and_process_data()
        
        dataset = StructuredSimDataset(states, actions, targets, history_len=self.cfg.train.history_len)
        torch.save(self.stats, "nerd_stats.pt")
        
        # Batch Size 可以很大，因为数据已经在 GPU 上了
        loader = DataLoader(dataset, batch_size=4096, shuffle=True)
        
        # 动态获取维度
        # states last dim is 6 (sin, cos, vel)
        # actions last dim is 1
        input_dim = states.shape[-1] + actions.shape[-1]
        output_dim = targets.shape[-1]
        
        print(f"[Model] In: {input_dim}, Out: {output_dim}")
        
        self.model = CausalTransformer(
            input_dim=input_dim, 
            output_dim=output_dim,
            embed_dim=64, num_layers=2, num_heads=4,
            history_len=self.cfg.train.history_len
        ).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        epochs = 200
        print(f"[Train] Starting {epochs} epochs...")
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            steps = 0
            for x, y in loader:
                # x, y 已经在 GPU 上 (如果 dataset tensors 在 GPU)
                # 如果不在，这里 to(device)
                if x.device != self.device: x = x.to(self.device)
                if y.device != self.device: y = y.to(self.device)
                
                optimizer.zero_grad()
                pred = self.model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                steps += 1
            
            if (epoch+1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/steps:.6f}")

    def save(self, path):
        torch.save(self.model.state_dict(), path)

@hydra.main(config_path="conf", config_name="joint_1dof", version_base="1.2")
def main(cfg: DictConfig):
    set_seed(42)
    device = "cuda"
    print(">>> ⚡ SETTING NUM_ENVS TO 512 (Structured Data) ⚡ <<<")
    cfg.task.num_envs = 512
    env = Joint1DofEnv(cfg.task, device=device)
    trainer = SimpleTrainer(cfg.task, env, device)
    trainer.train()
    trainer.save("nerd_sim_weights.pth")

if __name__ == "__main__":
    main()