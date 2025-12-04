import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
core_path = os.path.join(current_dir, "core")
if core_path not in sys.path: sys.path.insert(0, core_path)

import torch.nn as nn

# ==============================================================================
# 模型定义 (必须完全一致)
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

def check_real():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 加载数据
    if not os.path.exists("data/real_data.npz"):
        print("❌ 没有数据文件")
        return
    
    data = np.load("data/real_data.npz")
    q = data['q']
    qd = data['qd']
    tau = data['tau']
    
    # 转 Tensor
    q_gpu = torch.from_numpy(q).float().to(device)
    qd_gpu = torch.from_numpy(qd).float().to(device)
    tau_gpu = torch.from_numpy(tau).float().to(device)
    
    # 2. 加载统计量
    stats = torch.load("nerd_stats.pt", map_location=device)
    
    # 3. 加载微调后的模型
    model = CausalTransformer(
        input_dim=7, output_dim=4, 
        embed_dim=64, num_layers=2, 
        history_len=10
    ).to(device)
    
    if os.path.exists("nerd_real_final.pth"):
        model.load_state_dict(torch.load("nerd_real_final.pth"))
        print("✅ 加载微调模型: nerd_real_final.pth")
    else:
        print("❌ 未找到微调模型")
        return
        
    model.eval()
    
    # 4. 滚动预测 (Open Loop Prediction)
    # 给定历史 -> 预测下一步 -> 对比真实下一步
    pred_q = []
    pred_qd = []
    
    h = 10
    total_steps = min(500, len(q)) # 只看前500步
    
    print(f"正在验证前 {total_steps} 步数据...")
    
    with torch.no_grad():
        for i in range(h, total_steps):
            # A. 准备输入窗口
            q_win = q_gpu[i-h : i]
            qd_win = qd_gpu[i-h : i]
            tau_win = tau_gpu[i-h : i]
            
            # B. 特征工程 (Sin/Cos) & 维度填充
            # 假设 Load = Motor (Copy Heuristic)
            sin_q = torch.sin(q_win)
            cos_q = torch.cos(q_win)
            
            feat = torch.cat([sin_q, sin_q, cos_q, cos_q, qd_win, qd_win], dim=-1)
            
            # C. 归一化
            s_norm = (feat - stats['s_mean']) / stats['s_std']
            a_norm = (tau_win - stats['a_mean']) / stats['a_std']
            
            # [1, h, 7]
            model_in = torch.cat([s_norm, a_norm], dim=-1).unsqueeze(0)
            
            # D. 预测 Normalized Delta
            pred_delta_norm = model(model_in)[0] # [4]
            
            # E. 反归一化
            pred_delta = pred_delta_norm * stats['d_std'] + stats['d_mean']
            
            # F. 恢复绝对值 (Next = Curr + Delta)
            # 我们只关心 Motor (Index 0, 2)
            # 这里的 curr 是 i-1 时刻
            next_q_val = q_gpu[i-1] + pred_delta[0]
            next_qd_val = qd_gpu[i-1] + pred_delta[2]
            
            pred_q.append(next_q_val.cpu().item())
            pred_qd.append(next_qd_val.cpu().item())

    # 5. 绘图
    time = np.arange(h, total_steps)
    
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(time, q[h:total_steps], 'k-', label='Real Data', lw=2, alpha=0.6)
    plt.plot(time, pred_q, 'r--', label='NeRD Finetuned', lw=1.5)
    plt.title("Real Motor Position (Sim-to-Real Check)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(time, qd[h:total_steps], 'k-', label='Real Data', alpha=0.6)
    plt.plot(time, pred_qd, 'r--', label='NeRD Finetuned', lw=1.5)
    plt.title("Real Motor Velocity")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("real_check_result.png")
    print("✅ 验证完成！图片已保存至 real_check_result.png")

if __name__ == "__main__":
    check_real()