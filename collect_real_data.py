import numpy as np
import time
import os
import signal
import sys

# ==============================================================================
# 1. 配置参数
# ==============================================================================
CONFIG = {
    "duration": 60.0,       # 采集时长 (秒)
    "dt": 0.01,             # 控制周期 0.01s (100Hz) -> 必须与 Sim 训练一致!
    "max_torque": 3.0,      # 最大力矩限制 (Nm) - 保护电机
    "chirp_freq_start": 0.1,# 起始频率 (Hz)
    "chirp_freq_end": 5.0,  # 终止频率 (Hz)
    "chirp_amplitude": 2.0, # 信号幅度 (Nm) - 不要给太大，防止飞车
    "save_path": "data/real_data.npz"
}

# ==============================================================================
# 2. 硬件接口类 (请根据你的电机 SDK 修改此处!)
# ==============================================================================
class RealMotorInterface:
    def __init__(self):
        print("[Hardware] Initializing motor connection...")
        # -----------------------------------------------------------
        # TODO: 在这里初始化你的电机驱动
        # 例如: self.motor = odrive.find_any()
        #       self.motor.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
        # -----------------------------------------------------------
        pass

    def get_state(self):
        """
        读取电机当前状态
        返回: (position_rad, velocity_rad_s)
        """
        # -----------------------------------------------------------
        # TODO: 读取你的传感器数据
        # pos = self.motor.encoder.pos_estimate
        # vel = self.motor.encoder.vel_estimate
        # return pos, vel
        # -----------------------------------------------------------
        
        # --- 模拟数据 (如果你没有连接硬件，用这个测试代码逻辑) ---
        # 实际使用时请删除下面这几行
        t = time.time()
        sim_q = np.sin(t) 
        sim_qd = np.cos(t)
        return sim_q, sim_qd
        # -----------------------------------------------------------

    def set_torque(self, torque_nm):
        """
        发送力矩指令
        """
        # 安全截断
        torque_nm = np.clip(torque_nm, -CONFIG["max_torque"], CONFIG["max_torque"])
        
        # -----------------------------------------------------------
        # TODO: 发送指令给电机
        # self.motor.controller.input_torque = torque_nm
        # -----------------------------------------------------------
        # print(f"Cmd Torque: {torque_nm:.2f}") # Debug用，正式采集建议注释掉以免拖慢循环
        pass

    def close(self):
        """
        安全停机
        """
        print("[Hardware] Shutting down motor...")
        # -----------------------------------------------------------
        # TODO: 发送 0 力矩，或者进入 Idle 模式
        # self.motor.controller.input_torque = 0
        # self.motor.requested_state = AXIS_STATE_IDLE
        # -----------------------------------------------------------
        pass

# ==============================================================================
# 3. 信号生成器 (Chirp Signal)
# ==============================================================================
def get_chirp_signal(t, total_time):
    """
    生成线性 Chirp 信号: f(t) = f0 + (f1-f0) * t / T
    Phase phi(t) = 2*pi * (f0 * t + (k/2) * t^2)
    """
    f0 = CONFIG["chirp_freq_start"]
    f1 = CONFIG["chirp_freq_end"]
    
    # 瞬间频率 k
    k = (f1 - f0) / total_time
    
    # 相位
    phase = 2 * np.pi * (f0 * t + 0.5 * k * t**2)
    
    # 信号
    val = CONFIG["chirp_amplitude"] * np.sin(phase)
    return val

# ==============================================================================
# 4. 主循环
# ==============================================================================
def main():
    # 准备目录
    os.makedirs(os.path.dirname(CONFIG["save_path"]), exist_ok=True)
    
    motor = RealMotorInterface()
    
    # 数据容器
    logs = {
        'q': [],
        'qd': [],
        'tau': [],
        'time': []
    }
    
    print(f"=== 开始采集数据 ({CONFIG['duration']}s) ===")
    print("按 Ctrl+C 可以提前结束并保存数据。")
    print("3秒后开始...")
    time.sleep(3)
    
    start_time = time.perf_counter()
    next_step_time = start_time
    
    try:
        while True:
            now = time.perf_counter()
            t = now - start_time
            
            if t > CONFIG["duration"]:
                break
                
            # 1. 计算控制信号 (Chirp)
            cmd_torque = get_chirp_signal(t, CONFIG["duration"])
            
            # 2. 发送指令
            motor.set_torque(cmd_torque)
            
            # 3. 读取反馈
            q, qd = motor.get_state()
            
            # 4. 记录数据
            logs['q'].append(q)
            logs['qd'].append(qd)
            logs['tau'].append(cmd_torque) # 记录命令力矩作为输入
            logs['time'].append(t)
            
            # 5. 精确控频 (Spin/Sleep wait)
            next_step_time += CONFIG["dt"]
            sleep_time = next_step_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # 打印进度 (每秒一次)
            if int(t * 100) % 100 == 0:
                print(f"Time: {t:.1f}/{CONFIG['duration']}s | Q: {q:.3f} | Tau: {cmd_torque:.3f}")

    except KeyboardInterrupt:
        print("\n[User] 采集被中断！")
    
    except Exception as e:
        print(f"\n[Error] 发生错误: {e}")
        
    finally:
        # 安全操作：必须先停机
        motor.close()
        
        # 保存数据
        print("正在保存数据...")
        
        # 转换为 numpy 数组并调整形状为 [N, 1]
        q_np = np.array(logs['q']).reshape(-1, 1)
        qd_np = np.array(logs['qd']).reshape(-1, 1)
        tau_np = np.array(logs['tau']).reshape(-1, 1)
        
        # 检查数据长度
        if len(q_np) > 0:
            np.savez(CONFIG["save_path"], q=q_np, qd=qd_np, tau=tau_np)
            print(f"✅ 数据已保存至: {CONFIG['save_path']}")
            print(f"   数据形状: {q_np.shape}")
            
            # 简单的可视化检查
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                plt.subplot(3,1,1)
                plt.plot(logs['time'], q_np, label='Pos')
                plt.legend(); plt.grid()
                plt.subplot(3,1,2)
                plt.plot(logs['time'], qd_np, label='Vel')
                plt.legend(); plt.grid()
                plt.subplot(3,1,3)
                plt.plot(logs['time'], tau_np, label='Torque (Chirp)', color='r')
                plt.legend(); plt.grid()
                plt.tight_layout()
                plt.savefig("real_data_preview.png")
                print("   预览图已保存: real_data_preview.png")
            except ImportError:
                pass
        else:
            print("❌ 没有采集到数据")

if __name__ == "__main__":
    main()