import torch
import numpy as np
import mujoco
import mujoco.viewer
import time
import os
import sys
import argparse
from collections import deque

# 路径 Hack
current_dir = os.path.dirname(os.path.abspath(__file__))
core_path = os.path.join(current_dir, "core")
if core_path not in sys.path: sys.path.insert(0, core_path)

import torch.nn as nn

# ==============================================================================
# 模型定义 (必须与 train_sim.py 中回退后的 2层小模型一致)
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

def get_args():
    parser = argparse.ArgumentParser(description="NeRD 1-DOF Real-time Visualization")
    parser.add_argument("--mode", type=str, choices=["neural", "sim"], default="neural", help="Mode: 'neural' (AI prediction) or 'sim' (Physics Engine)")
    parser.add_argument("--xml", type=str, default="assets/joint_1dof.xml", help="Path to MuJoCo XML")
    return parser.parse_args()

def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== NeRD Visualization ({args.mode.upper()} Mode) ===")
    print(f"Device: {device}")

    # 1. 准备 MuJoCo (用于显示)
    if not os.path.exists(args.xml):
        print(f"❌ XML not found: {args.xml}")
        return
    
    mj_model = mujoco.MjModel.from_xml_path(args.xml)
    mj_data = mujoco.MjData(mj_model)

    # 2. 初始化资源
    model = None
    stats = None
    env = None
    history_queue = deque(maxlen=10)
    
    # 初始状态 [q0, q1, qd0, qd1]
    current_state_neural = torch.zeros(4, device=device)
    
    if args.mode == "neural":
        # 加载统计量
        if not os.path.exists("nerd_stats.pt"):
            print("❌ nerd_stats.pt not found.")
            return
        stats = torch.load("nerd_stats.pt", map_location=device)

        # 加载模型
        # Input=7 (6 feat + 1 action), Output=4 (delta)
        model = CausalTransformer(input_dim=7, output_dim=4, embed_dim=64, num_layers=2, history_len=10).to(device)
        
        weight_path = "nerd_real_final.pth"
        if not os.path.exists(weight_path):
            print(f"⚠️ {weight_path} not found, trying sim weights...")
            weight_path = "nerd_sim_weights.pth"
            
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path, map_location=device))
            model.eval()
            print(f"✅ Neural Model Loaded: {weight_path}")
        else:
            print("❌ No model weights found!")
            return
            
        # 填充历史 Buffer
        for _ in range(10): 
            history_queue.append(torch.zeros(4, device=device))

    elif args.mode == "sim":
        # 初始化 Warp 环境
        try:
            from custom_envs.joint_1dof_env import Joint1DofEnv
            from omegaconf import OmegaConf
            cfg = OmegaConf.load("conf/joint_1dof.yaml")
            cfg.task.num_envs = 1 
            # 必须用 GPU
            env = Joint1DofEnv(cfg.task, device=device)
            env.reset()
            print("✅ Warp Physics Engine Initialized.")
        except ImportError as e:
            print(f"❌ Warp Init Error: {e}")
            return

    # 3. 启动 Viewer
    print("\n>>> 正在启动窗口... (按 ESC 退出) <<<\n")
    
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        step = 0
        while viewer.is_running():
            step_start = time.time()

            # --- 生成控制信号 ---
            # 随时间变化的正弦波力矩
            t = step * 0.01
            freq = 1.0 + 0.5 * np.sin(t * 0.2)
            # --- 修改前：开环力矩 (能量无限注入 -> 发散) ---
            # raw_tau = 3.0 * np.sin(t * freq)
            # action_val = torch.tensor([raw_tau], dtype=torch.float32, device=device)

            # --- 修改后：PD 控制 (模拟伺服电机 -> 稳定) ---
            # 1. 设定目标位置 (让它在 +/- 1.5 rad 之间摆动)
            target_pos = 1.5 * np.sin(t * freq)
            target_vel = 1.5 * freq * np.cos(t * freq)
            
            # 2. 获取当前状态 (Feedback)
            if args.mode == "neural":
                curr_q = current_state_neural[0]
                curr_qd = current_state_neural[2]
            elif args.mode == "sim":
                # 从 Warp state 获取 (为了简单，这里我们假设上一帧的输出作为近似)
                # 在 Sim 模式下，最好是在 env.step 内部做 PID，或者这里简单估算
                # 暂时用开环力矩测 Sim 物理稳定性，或者手动获取 sim state
                # 这里为了演示 Sim-to-Real 的效果，我们主要关注 neural 分支
                curr_q = 0.0 # 占位
                curr_qd = 0.0
            
            # 3. 计算 PD 力矩
            # Kp: 刚度, Kd: 阻尼
            kp = 10.0
            kd = 0.5 
            
            if args.mode == "neural":
                # PD 公式: tau = Kp*(target - q) + Kd*(target_v - qd)
                pd_tau = kp * (target_pos - curr_q) + kd * (target_vel - curr_qd)
                # 限制幅度
                pd_tau = torch.clamp(pd_tau, -5.0, 5.0)
                action_val = pd_tau.view(1) # [1]
            else:
                # Sim 模式继续用开环力矩测试物理极限，或者你也加上 PD
                # 如果 XML 阻尼改小了，开环力矩还是会发散，建议 Sim 也加 PD
                # 但为了不修改 env 代码，我们这里还是发正弦波，但幅度小一点
                action_val = torch.tensor([1.0 * np.sin(t * freq)], dtype=torch.float32, device=device)

            if args.mode == "neural":
                # --- A. 神经网络推理 ---
                
                # 准备历史 Tensor: [1, 10, 4]
                hist_states = torch.stack(list(history_queue)).unsqueeze(0)
                
                # 准备动作 Tensor: [1, 10, 1] (简单重复当前动作)
                action_seq = action_val.view(1, 1, 1).repeat(1, 10, 1)
                
                # 特征工程: [q, qd] -> [sin, cos, qd]
                q_hist = hist_states[..., :2]
                qd_hist = hist_states[..., 2:]
                feat = torch.cat([torch.sin(q_hist), torch.cos(q_hist), qd_hist], dim=-1)
                
                # 归一化
                s_norm = (feat - stats['s_mean']) / stats['s_std']
                a_norm = (action_seq - stats['a_mean']) / stats['a_std']
                
                # 拼接输入
                model_in = torch.cat([s_norm, a_norm], dim=-1)
                
                # 预测 Delta
                with torch.no_grad():
                    pred_delta_norm = model(model_in)[0] # [4]
                
                # 反归一化
                pred_delta = pred_delta_norm * stats['d_std'] + stats['d_mean']
                
                # 更新状态
                current_state_neural = current_state_neural + pred_delta
                history_queue.append(current_state_neural)
                
                # 同步到 MuJoCo 显示
                s_cpu = current_state_neural.cpu().numpy()
                mj_data.qpos[0] = s_cpu[0]
                mj_data.qpos[1] = s_cpu[1]
                mj_data.qvel[0] = s_cpu[2]
                mj_data.qvel[1] = s_cpu[3]

            elif args.mode == "sim":
                # --- B. 物理引擎步进 ---
                # Warp 需要 [N, 1] 的输入
                action_tensor = action_val.unsqueeze(0) 
                
                # 执行物理步进
                state_warp = env.step(action_tensor) # [1, 4]
                
                # 同步到 MuJoCo
                s_cpu = state_warp[0].cpu().numpy()
                mj_data.qpos[0] = s_cpu[0]
                mj_data.qpos[1] = s_cpu[1]
                mj_data.qvel[0] = s_cpu[2]
                mj_data.qvel[1] = s_cpu[3]

            # 刷新渲染
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()

            # 帧率控制 (保持实时感)
            time_until_next = mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)
            
            step += 1

    print("Done.")

if __name__ == "__main__":
    main()