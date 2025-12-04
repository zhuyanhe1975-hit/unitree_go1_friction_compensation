import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
core_path = os.path.join(current_dir, "core")
if core_path not in sys.path: sys.path.insert(0, core_path)

# ==============================================================================
# 模型定义 (必须与 train_sim.py / finetune_real.py 完全一致)
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
        # 返回 [Batch, Output_Dim] (2D Tensor)
        return self.head(feat[:, -1, :])

def run_mpc():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Running Differentiable MPC on {device} ===")
    
    # 1. 加载模型
    model_path = "nerd_sim_weights.pth" 
    if os.path.exists("nerd_real_final.pth"):
        print("💡 发现微调后的模型，使用实物模型！")
        model_path = "nerd_real_final.pth"
    else:
        print("⚠️ 未找到微调模型，使用 Sim 模型")
        
    if not os.path.exists("nerd_stats.pt"):
        print("❌ 缺少 nerd_stats.pt，无法进行归一化")
        return

    # 加载统计量
    stats = torch.load("nerd_stats.pt", map_location=device)
    
    # 初始化模型 (确保参数与训练时一致: layers=2, embed=64)
    model = CausalTransformer(
        input_dim=7, output_dim=4, embed_dim=64, num_layers=2, history_len=10
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path))
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
        
    model.eval()
    
    # 冻结参数
    for param in model.parameters():
        param.requires_grad = False

    # 2. 定义控制任务
    target_pos = 1.5 
    horizon = 50 
    h = 10 
    
    # 初始状态 (静止)
    # [1, h, 1]
    curr_q = torch.zeros(1, h, 1, device=device)
    curr_qd = torch.zeros(1, h, 1, device=device)
    curr_action = torch.zeros(1, h, 1, device=device)
    
    # 3. 待优化的控制序列
    # 初始化为一个小的随机值或者0
    future_actions = torch.zeros(1, horizon, 1, device=device, requires_grad=True)
    
    # 学习率可以大一点，因为我们是在优化 Input，不是 Weights
    optimizer = optim.Adam([future_actions], lr=0.5)
    
    print(f"🎯 目标: 移动到 {target_pos} rad")
    
    # MPC 优化循环
    for step in range(300):
        optimizer.zero_grad()
        loss = 0
        
        # 模拟未来 (Rollout)
        sim_q = curr_q.clone()
        sim_qd = curr_qd.clone()
        sim_action = curr_action.clone()
        
        predicted_traj = []
        
        for t in range(horizon):
            # A. 构造输入特征 [1, h, 6]
            feat = torch.cat([
                torch.sin(sim_q), torch.cos(sim_q), 
                torch.sin(sim_q), torch.cos(sim_q), # Copy Heuristic for Load
                sim_qd, sim_qd
            ], dim=-1)
            
            # B. 归一化
            s_norm = (feat - stats['s_mean']) / stats['s_std']
            
            # 动作窗口处理
            act_t = future_actions[:, t:t+1, :] # [1, 1, 1]
            
            # 拼接动作历史: 取 [old_action[1:], new_action]
            next_action_window = torch.cat([sim_action[:, 1:], act_t], dim=1)
            a_norm = (next_action_window - stats['a_mean']) / stats['a_std']
            
            # C. 模型预测
            model_in = torch.cat([s_norm, a_norm], dim=-1) # [1, h, 7]
            
            # 【关键修复】模型返回 [1, 4] (2D)，我们需要升维成 [1, 1, 4] (3D)
            # 这样才能和后面的切片运算匹配
            pred_delta_norm = model(model_in).unsqueeze(1) 
            
            # D. 反归一化
            pred_delta = pred_delta_norm * stats['d_std'] + stats['d_mean']
            
            # E. 积分 (Next State = Current Last + Delta)
            # Delta 0: Motor Pos, Delta 2: Motor Vel
            next_q = sim_q[:, -1:, :] + pred_delta[:, :, 0:1]
            next_qd = sim_qd[:, -1:, :] + pred_delta[:, :, 2:3]
            
            # F. 更新滑动窗口
            sim_q = torch.cat([sim_q[:, 1:], next_q], dim=1)
            sim_qd = torch.cat([sim_qd[:, 1:], next_qd], dim=1)
            sim_action = next_action_window
            
            # 记录轨迹
            predicted_traj.append(next_q)
            
            # --- Loss 计算 ---
            # 1. 位置误差 (权重最大)
            loss += (next_q - target_pos) ** 2 * 20.0
            
            # 2. 速度惩罚 (希望终点静止)
            if t > horizon - 10:
                loss += (next_qd) ** 2 * 2.0
            
            # 3. 动作能量惩罚 (省电/平滑)
            loss += (act_t) ** 2 * 0.001
            
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            print(f"Iter {step}: Loss = {loss.item():.4f}")

    # 4. 绘图
    actions_np = future_actions.detach().cpu().numpy().flatten()
    traj_np = torch.cat(predicted_traj, dim=1).detach().cpu().numpy().flatten()
    
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(traj_np, 'r-', label='MPC Plan', lw=2)
    plt.axhline(y=target_pos, color='g', ls='--', label='Target')
    plt.title("Planned Trajectory")
    plt.ylabel("Position (rad)")
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(actions_np, 'b-', label='Torque', lw=2)
    plt.title("Optimized Control Sequence")
    plt.ylabel("Torque (Nm)")
    plt.xlabel("Time Step (0.01s)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("mpc_result.png")
    print("✅ MPC 规划完成！查看 mpc_result.png")

if __name__ == "__main__":
    run_mpc()