import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys

# 路径 hack
current_dir = os.path.dirname(os.path.abspath(__file__))
core_path = os.path.join(current_dir, "core")
if core_path not in sys.path: sys.path.insert(0, core_path)

# ==============================================================================
# 复用模型定义 (必须与 train_sim.py 这里的 2层小模型 一致)
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
# Real Dataset (适配 1-DOF -> 2-DOF 映射)
# ==============================================================================
class RealDataset(Dataset):
    def __init__(self, npz_path, stats_path, history_len=10):
        # 1. 加载实物数据
        data = np.load(npz_path)
        # 确保数据是 Float32
        q_real = torch.from_numpy(data['q']).float()
        qd_real = torch.from_numpy(data['qd']).float()
        self.tau = torch.from_numpy(data['tau']).float()
        
        # 2. 加载 Sim 统计量
        if not os.path.exists(stats_path):
            raise FileNotFoundError("找不到 nerd_stats.pt！")
        self.stats = torch.load(stats_path, map_location='cpu')
        
        # 3. 特征工程 + 维度填充
        # Sim Feat: [sin0, sin1, cos0, cos1, vel0, vel1]
        # Real Data: Only 0 (Motor)
        # 策略: 假设 sin1=sin0, cos1=cos0, vel1=vel0 (初始猜测)
        
        sin_q = torch.sin(q_real)
        cos_q = torch.cos(q_real)
        
        self.feat_states = torch.cat([
            sin_q, sin_q,   # sin
            cos_q, cos_q,   # cos
            qd_real, qd_real # vel
        ], dim=-1) # [N, 6]
        
        # 4. 归一化 (使用 Sim 的标准)
        self.norm_states = (self.feat_states - self.stats['s_mean']) / self.stats['s_std']
        self.norm_actions = (self.tau - self.stats['a_mean']) / self.stats['a_std']
        
        # 5. 准备 Target
        # 我们只能计算 Motor 的 Delta
        # Sim Target Delta: [dq0, dq1, dv0, dv1]
        raw_dq = q_real[1:] - q_real[:-1]
        raw_dv = qd_real[1:] - qd_real[:-1]
        
        # 归一化 Delta
        d_mean = self.stats['d_mean']
        d_std = self.stats['d_std']
        
        # 构造全零 Target，稍后只训练有数据的维度
        self.norm_targets = torch.zeros(len(raw_dq), 4)
        
        # 填充 Motor Pos Delta (Index 0)
        self.norm_targets[:, 0] = (raw_dq.squeeze() - d_mean[0]) / d_std[0]
        # 填充 Motor Vel Delta (Index 2)
        self.norm_targets[:, 2] = (raw_dv.squeeze() - d_mean[2]) / d_std[2]
        
        self.h = history_len
        self.length = len(self.norm_states) - self.h - 1

    def __len__(self):
        return max(0, self.length)

    def __getitem__(self, idx):
        # Input
        s_win = self.norm_states[idx : idx+self.h]
        a_win = self.norm_actions[idx : idx+self.h]
        inputs = torch.cat([s_win, a_win], dim=-1) # [h, 7]
        
        # Target
        target = self.norm_targets[idx + self.h - 1] # [4]
        
        # Mask: [1, 0, 1, 0] -> 只监督 Motor Pos 和 Motor Vel
        mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
        
        return inputs, target, mask

def finetune():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== Sim-to-Real Fine-tuning ===")
    
    # 1. 维度配置
    # In: 6(feat) + 1(act) = 7
    # Out: 4(delta)
    model = CausalTransformer(
        input_dim=7, output_dim=4, 
        embed_dim=64, num_layers=2, num_heads=4,
        history_len=10
    ).to(device)
    
    # 2. 加载权重
    if os.path.exists("nerd_sim_weights.pth"):
        model.load_state_dict(torch.load("nerd_sim_weights.pth"))
        print("✅ Sim 权重加载成功")
    else:
        print("❌ 权重丢失")
        return

    # 3. 准备数据
    if not os.path.exists("data/real_data.npz"):
        print("⚠️ 生成测试用假实物数据...")
        os.makedirs("data", exist_ok=True)
        N = 2000
        t = np.linspace(0, 10, N)
        # 加一点 Sim 不具备的特性 (比如摩擦力更大导致幅度变小)
        q = 0.8 * np.sin(t) 
        qd = 0.8 * np.cos(t)
        tau = np.sin(t)
        np.savez("data/real_data.npz", q=q[:,None], qd=qd[:,None], tau=tau[:,None])
        
    dataset = RealDataset("data/real_data.npz", "nerd_stats.pt")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 4. 微调 (Low LR)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    criterion = nn.MSELoss(reduction='none') 
    
    model.train()
    print("开始微调...")
    for epoch in range(20):
        total_loss = 0
        steps = 0
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            
            optimizer.zero_grad()
            pred = model(x)
            
            # Masked Loss
            loss_all = criterion(pred, y)
            loss = (loss_all * mask).sum() / mask.sum()
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1
            
        print(f"Finetune Epoch {epoch+1} | Loss: {total_loss/steps:.6f}")

    torch.save(model.state_dict(), "nerd_real_final.pth")
    print("🎉 All Done! 模型已保存。")

if __name__ == "__main__":
    finetune()