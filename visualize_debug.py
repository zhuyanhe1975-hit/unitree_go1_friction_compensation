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
# 模型定义 (2层小模型)
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
    parser = argparse.ArgumentParser(description="NeRD 1-DOF Debug Visualization")
    parser.add_argument("--mode", type=str, choices=["neural", "sim"], default="neural")
    parser.add_argument("--xml", type=str, default="assets/joint_1dof.xml")
    return parser.parse_args()

def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== DEBUG VISUALIZATION: {args.mode.upper()} MODE ===\n")

    # 1. MuJoCo Viewer Setup
    if not os.path.exists(args.xml):
        print(f"❌ XML not found: {args.xml}")
        return
    mj_model = mujoco.MjModel.from_xml_path(args.xml)
    mj_data = mujoco.MjData(mj_model)

    # 2. Init Resources
    model = None
    stats = None
    env = None
    history_queue = deque(maxlen=10)
    current_state_neural = torch.zeros(4, device=device) # [q0, q1, qd0, qd1]
    
    if args.mode == "neural":
        if not os.path.exists("nerd_stats.pt"):
            print("❌ nerd_stats.pt not found.")
            return
        stats = torch.load("nerd_stats.pt", map_location=device)
        print(f"✅ Stats Loaded. Mean Q0: {stats['s_mean'][0]:.4f}, Std Q0: {stats['s_std'][0]:.4f}")

        # Try loading real fine-tuned weights first, then sim
        weight_path = "nerd_real_final.pth"
        if not os.path.exists(weight_path):
            weight_path = "nerd_sim_weights.pth"
            
        if os.path.exists(weight_path):
            model = CausalTransformer(input_dim=7, output_dim=4, embed_dim=64, num_layers=2, history_len=10).to(device)
            model.load_state_dict(torch.load(weight_path, map_location=device))
            model.eval()
            print(f"✅ Model Loaded: {weight_path}")
        else:
            print("❌ No weights found.")
            return
            
        for _ in range(10): history_queue.append(torch.zeros(4, device=device))

    elif args.mode == "sim":
        try:
            from custom_envs.joint_1dof_env import Joint1DofEnv
            from omegaconf import OmegaConf
            cfg = OmegaConf.load("conf/joint_1dof.yaml")
            cfg.task.num_envs = 1 
            env = Joint1DofEnv(cfg.task, device=device)
            env.reset()
            print("✅ Warp Physics Engine Ready.")
        except ImportError as e:
            print(f"❌ Warp Error: {e}")
            return

    # 3. Main Loop
    print("\n>>> 开始运行... 请观察下方输出数据 <<<")
    print(f"{'Step':<6} | {'Tgt Pos':<8} | {'Cur Pos':<8} | {'Error':<8} | {'Torque':<8} | {'Pred Delta (rad)':<20}")
    print("-" * 80)

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        step = 0
        while viewer.is_running():
            step_start = time.time()

            # --- 控制逻辑: PD 追踪正弦波 ---
            t = step * 0.01
            # 目标: 幅度 1.5 rad, 频率随时间略变
            freq = 1.0
            target_pos = 1.5 * np.sin(t * freq)
            target_vel = 1.5 * freq * np.cos(t * freq)
            
            # 获取当前状态 (Feedback)
            if args.mode == "neural":
                curr_q = current_state_neural[0].item()
                curr_v = current_state_neural[2].item()
            else:
                # Sim 模式下暂无法直接获得上一帧 Warp 状态(除非修改 Env 返回)，
                # 这里为了简单，假设 Open Loop 或者从 MuJoCo 读 (近似)
                curr_q = mj_data.qpos[0]
                curr_v = mj_data.qvel[0]

            # PD Controller
            kp = 20.0
            kd = 1.0
            error_p = target_pos - curr_q
            error_d = target_vel - curr_v
            
            raw_tau = kp * error_p + kd * error_d
            # 限幅 +/- 5.0 Nm
            raw_tau = np.clip(raw_tau, -5.0, 5.0)
            
            action_val = torch.tensor([raw_tau], dtype=torch.float32, device=device)

            # --- 调试数据容器 ---
            debug_delta_info = "N/A"

            if args.mode == "neural":
                # Neural Inference
                hist_states = torch.stack(list(history_queue)).unsqueeze(0)
                action_seq = action_val.view(1, 1, 1).repeat(1, 10, 1)
                
                # Feat: [Sin, Cos, Vel]
                q_hist = hist_states[..., :2]
                qd_hist = hist_states[..., 2:]
                feat = torch.cat([torch.sin(q_hist), torch.cos(q_hist), qd_hist], dim=-1)
                
                # Norm
                s_norm = (feat - stats['s_mean']) / stats['s_std']
                a_norm = (action_seq - stats['a_mean']) / stats['a_std']
                
                model_in = torch.cat([s_norm, a_norm], dim=-1)
                
                with torch.no_grad():
                    pred_delta_norm = model(model_in)[0]
                
                pred_delta = pred_delta_norm * stats['d_std'] + stats['d_mean']
                
                # Update State
                current_state_neural = current_state_neural + pred_delta
                history_queue.append(current_state_neural)
                
                # 更新调试信息 (只看电机 Pos 和 Vel 的 Delta)
                d_pos = pred_delta[0].item()
                d_vel = pred_delta[2].item()
                debug_delta_info = f"dP={d_pos:.5f}, dV={d_vel:.3f}"

                # Sync MuJoCo
                s_cpu = current_state_neural.cpu().numpy()
                mj_data.qpos[0] = s_cpu[0]
                mj_data.qpos[1] = s_cpu[1]
                mj_data.qvel[0] = s_cpu[2]
                mj_data.qvel[1] = s_cpu[3]

            elif args.mode == "sim":
                # Warp Physics
                action_tensor = action_val.unsqueeze(0)
                state_warp = env.step(action_tensor)
                s_cpu = state_warp[0].cpu().numpy()
                
                mj_data.qpos[0] = s_cpu[0]
                mj_data.qpos[1] = s_cpu[1]
                mj_data.qvel[0] = s_cpu[2]
                mj_data.qvel[1] = s_cpu[3]
                
                debug_delta_info = "Physics Engine"

            # --- 渲染与打印 ---
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()

            # 每 10 帧打印一次调试信息
            if step % 10 == 0:
                print(f"{step:<6d} | {target_pos:<8.3f} | {curr_q:<8.3f} | {error_p:<8.3f} | {raw_tau:<8.3f} | {debug_delta_info}")

            # 帧率控制
            time_until_next = mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)
            
            step += 1

    print("Done.")

if __name__ == "__main__":
    main()