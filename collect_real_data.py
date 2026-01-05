import numpy as np
import time
import os
import signal
import sys

from project_config import ensure_dir, get, load_config

# ==============================================================================
# 2. 硬件接口类 (Unitree GO-M8010-6)
# ==============================================================================
class RealMotorInterface:
    def __init__(self, cfg: dict, max_torque_nm: float):
        print("[Hardware] Initializing Unitree actuator SDK...")
        sdk_lib = get(cfg, "real.unitree_sdk_lib", required=False)
        if sdk_lib:
            sys.path.insert(0, sdk_lib)
        else:
            # Best-effort fallback: known local SDK path.
            local_sdk = "/home/yhzhu/Industrial Robot/unitree_actuator_sdk/lib"
            if os.path.isdir(local_sdk):
                sys.path.insert(0, local_sdk)

        try:
            import unitree_actuator_sdk as u  # type: ignore
        except Exception as e:
            raise ImportError(
                "Cannot import unitree_actuator_sdk. Set env UNITREE_ACTUATOR_SDK_LIB to the folder containing "
                "`unitree_actuator_sdk*.so` (e.g. /home/yhzhu/Industrial Robot/unitree_actuator_sdk/lib), "
                "or set `real.unitree_sdk_lib` in config.json. "
                f"Original error: {e}"
            ) from e

        self.u = u
        port = str(get(cfg, "real.serial_port"))
        try:
            self.serial = u.SerialPort(port)
        except RuntimeError as e:
            msg = str(e)
            if "Permission denied" in msg or "IO Exception (13)" in msg:
                raise RuntimeError(
                    f"Failed to open serial port {port} (permission denied).\n"
                    f"- Quick test: run with sudo: `sudo {sys.executable} collect_real_data.py`\n"
                    f"- Proper fix: add user to dialout then re-login:\n"
                    f"    `sudo usermod -a -G dialout $USER`\n"
                    f"  then logout/login (or `newgrp dialout`) and retry.\n"
                    f"- You can also set `real.serial_port` to a stable symlink like:\n"
                    f"    `/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTALVWMV-if00-port0`\n"
                    f"Original error: {msg}"
                ) from e
            raise
        self.cmd = u.MotorCmd()
        self.data = u.MotorData()

        self.data.motorType = u.MotorType.GO_M8010_6
        self.cmd.motorType = u.MotorType.GO_M8010_6
        self.cmd.mode = u.queryMotorMode(u.MotorType.GO_M8010_6, u.MotorMode.FOC)
        self.cmd.id = int(get(cfg, "real.motor_id"))

        # Torque-centric control: keep position/velocity loops off unless explicitly configured.
        self.cmd.q = 0.0
        self.cmd.dq = 0.0
        self.cmd.kp = float(get(cfg, "real.kp", required=False, default=0.0))
        self.cmd.kd = float(get(cfg, "real.kd", required=False, default=0.0))
        self.cmd.tau = 0.0
        if abs(self.cmd.kp) > 0.0 or abs(self.cmd.kd) > 0.0:
            print(
                f"[Warning] real.kp={self.cmd.kp:.6g}, real.kd={self.cmd.kd:.6g} (non-zero). "
                "In this mode, feedback torque data.tau will include position/velocity loop components and "
                "may differ significantly from cmd.tau. For pure torque chirp identification, set both to 0."
            )

        self.max_torque_nm = float(max_torque_nm)

        # Prime one read so downstream code gets a valid state immediately.
        self.serial.sendRecv(self.cmd, self.data)

    def get_state(self):
        """
        读取电机当前状态
        注意：按你的实际情况，data.q/data.dq 是电机侧（磁编码器单圈绝对角度/角速度）。
        返回: (q_m_rad, qd_m_rad_s, tau_feedback)
        """
        return float(self.data.q), float(self.data.dq), float(self.data.tau)

    def set_torque(self, torque_nm):
        """
        发送力矩指令 (units per Unitree SDK convention; logged as tau_cmd)
        """
        # 安全截断
        torque_nm = float(np.clip(float(torque_nm), -self.max_torque_nm, self.max_torque_nm))
        self.cmd.tau = float(torque_nm)
        self.serial.sendRecv(self.cmd, self.data)

    def close(self):
        """
        安全停机
        """
        print("[Hardware] Shutting down (send 0 torque)...")
        try:
            self.cmd.tau = 0.0
            self.serial.sendRecv(self.cmd, self.data)
        except Exception:
            pass

# ==============================================================================
# 3. 信号生成器 (Chirp Signal)
# ==============================================================================
def get_chirp_signal(t: float, total_time: float, cfg: dict) -> float:
    """
    生成线性 Chirp 信号: f(t) = f0 + (f1-f0) * t / T
    Phase phi(t) = 2*pi * (f0 * t + (k/2) * t^2)
    """
    chirp = get(cfg, "real.chirp")
    f0 = float(chirp["f0_hz"])
    f1 = float(chirp["f1_hz"])
    
    # 瞬间频率 k
    k = (f1 - f0) / total_time
    
    # 相位
    phase = 2 * np.pi * (f0 * t + 0.5 * k * t**2)
    
    # 信号
    amp = float(chirp["amplitude_nm"])
    return float(amp * np.sin(phase))

def _smooth_sign(x: float, eps: float = 1e-3) -> float:
    return float(x / (abs(x) + eps))


# ==============================================================================
# 4. 主循环
# ==============================================================================
def main():
    cfg = load_config()
    dt = float(get(cfg, "real.dt"))
    duration = float(get(cfg, "real.duration"))
    max_torque = float(get(cfg, "real.max_torque"))
    gear_ratio = float(get(cfg, "motor.gear_ratio"))
    unwrap_motor_angle = bool(get(cfg, "real.unwrap_motor_angle", required=False, default=True))
    tau_slew = float(get(cfg, "real.tau_slew_nm_s", required=False, default=50.0))
    tau_static = float(get(cfg, "friction.tau_static_out_nm", required=False, default=0.0))
    tau_static_enable_th = float(get(cfg, "real.tau_static_enable_threshold_nm", required=False, default=0.0))
    temp_limit_c = float(get(cfg, "real.temp_limit_c", required=False, default=80.0))
    abort_on_merror = bool(get(cfg, "real.abort_on_merror", required=False, default=True))

    save_path = str(get(cfg, "paths.real_log"))
    ensure_dir(os.path.dirname(save_path) or ".")

    motor = RealMotorInterface(cfg, max_torque_nm=max_torque)

    # 数据容器
    logs = {
        # Motor-side signals (from encoder)
        "q_m_raw": [],
        "q_m": [],
        "qd_m": [],

        # Unified kinematics (we treat ALL signals as motor-side for consistency with Unitree SDK):
        # - q_out/qd_out are kept for downstream pipeline compatibility, but represent motor-side q/dq.
        "q_out": [],
        "qd_out": [],

        # Torque (motor-side; consistent with cmd.tau and data.tau conventions)
        "tau_ref": [],
        "tau_cmd": [],
        "tau_out_raw": [],  # raw feedback data.tau from SDK
        "tau_out": [],  # effective torque channel (tau_out_raw + kd * qd_m)

        'temp': [],
        'merror': [],
        'time': [],
    }
    
    print(f"=== 开始采集数据 ({duration}s) ===")
    print("按 Ctrl+C 可以提前结束并保存数据。")
    print("3秒后开始...")
    time.sleep(3)
    
    start_time = time.perf_counter()
    next_step_time = start_time
    q_m_cont = None
    q_m_prev = None
    tau_cmd_prev = 0.0
    
    try:
        while True:
            now = time.perf_counter()
            t = now - start_time
            
            if t > duration:
                break
                
            # 1. 计算控制信号 (Chirp)
            tau_ref = get_chirp_signal(t, duration, cfg)

            # 2. 静摩擦前馈（方向由 tau_ref 决定；小于阈值时不加，避免零附近抖动）
            tau_ff = 0.0
            if abs(tau_ref) >= tau_static_enable_th and tau_static > 0.0:
                tau_ff = tau_static * _smooth_sign(tau_ref)

            tau_des = float(tau_ref + tau_ff)

            # 3. 限幅 + 斜率限制（安全）
            tau_des = float(np.clip(tau_des, -max_torque, max_torque))
            max_step = max(0.0, tau_slew) * dt
            tau_cmd = float(np.clip(tau_des, tau_cmd_prev - max_step, tau_cmd_prev + max_step))
            tau_cmd_prev = tau_cmd
            
            # 4. 发送指令
            motor.set_torque(tau_cmd)
            
            # 5. 读取反馈
            q_m_raw, qd_m, tau_out_raw = motor.get_state()

            # 6. 安全监控：温度/错误码
            if abort_on_merror and int(motor.data.merror) != 0:
                raise RuntimeError(f"motor merror={int(motor.data.merror)}")
            if float(motor.data.temp) >= temp_limit_c:
                raise RuntimeError(f"motor temp={float(motor.data.temp):.1f}C >= limit={temp_limit_c:.1f}C")

            # 7. 单圈角度展开（避免 2π 跳变破坏微分/学习）
            if q_m_prev is None:
                q_m_cont = q_m_raw
            else:
                dq_wrap = q_m_raw - q_m_prev
                if unwrap_motor_angle:
                    if dq_wrap > np.pi:
                        dq_wrap -= 2.0 * np.pi
                    elif dq_wrap < -np.pi:
                        dq_wrap += 2.0 * np.pi
                q_m_cont = float(q_m_cont + dq_wrap)
            q_m_prev = q_m_raw

            # Unify everything on motor side: no gear ratio conversion.
            q_out = float(q_m_cont)
            qd_out = float(qd_m)

            # The actuator's internal velocity loop (kd) can contribute an additional torque-like term.
            # For analysis, we log an "effective" torque channel that adds kd * qd_m (dq_cmd is 0 here).
            tau_out_eff = float(tau_out_raw + motor.cmd.kd * float(qd_m))
            
            # 8. 记录数据
            logs["q_m_raw"].append(q_m_raw)
            logs["q_m"].append(q_m_cont)
            logs["qd_m"].append(qd_m)
            logs["q_out"].append(q_out)
            logs["qd_out"].append(qd_out)
            logs["tau_ref"].append(tau_ref)
            logs["tau_cmd"].append(tau_cmd)
            logs["tau_out_raw"].append(tau_out_raw)
            logs["tau_out"].append(tau_out_eff)
            logs['temp'].append(float(motor.data.temp))
            logs['merror'].append(float(motor.data.merror))
            logs['time'].append(t)
            
            # 5. 精确控频 (Spin/Sleep wait)
            next_step_time += dt
            sleep_time = next_step_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # 打印进度 (每秒一次)
            if int(t) != int((t - dt) if t - dt >= 0 else 0):
                print(f"Time: {t:.1f}/{duration:.1f}s | q_out: {q_out:+.3f} | tau_cmd: {tau_cmd:+.3f} | tau_out: {tau_out:+.3f}")

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
        # Save as 1D time series [T] for easier downstream use.
        q_m_raw_np = np.asarray(logs["q_m_raw"], dtype=np.float64)
        q_m_np = np.asarray(logs["q_m"], dtype=np.float64)
        qd_m_np = np.asarray(logs["qd_m"], dtype=np.float64)
        q_out_np = np.asarray(logs["q_out"], dtype=np.float64)
        qd_out_np = np.asarray(logs["qd_out"], dtype=np.float64)
        tau_ref_np = np.asarray(logs["tau_ref"], dtype=np.float64)
        tau_cmd_np = np.asarray(logs["tau_cmd"], dtype=np.float64)
        tau_out_raw_np = np.asarray(logs["tau_out_raw"], dtype=np.float64)
        tau_out_np = np.asarray(logs["tau_out"], dtype=np.float64)
        temp_np = np.asarray(logs["temp"], dtype=np.float64)
        merror_np = np.asarray(logs["merror"], dtype=np.float64)
        
        # 检查数据长度
        if len(q_m_np) > 0:
            np.savez(
                save_path,
                # Motor-side
                q_m_raw=q_m_raw_np,
                q_m=q_m_np,
                qd_m=qd_m_np,
                # Derived output-side
                q_out=q_out_np,
                qd_out=qd_out_np,
                # Torque
                tau_ref=tau_ref_np,
                tau_cmd=tau_cmd_np,
                tau_out_raw=tau_out_raw_np,
                tau_out=tau_out_np,
                temp=temp_np,
                merror=merror_np,
                t=np.array(logs["time"], dtype=np.float64),
                meta=np.array(
                    [
                        dt,
                        duration,
                        max_torque,
                        gear_ratio,
                        float(unwrap_motor_angle),
                        tau_slew,
                        tau_static,
                        tau_static_enable_th,
                        temp_limit_c,
                        float(abort_on_merror),
                        float(motor.cmd.kp),
                        float(motor.cmd.kd),
                    ],
                    dtype=np.float64,
                ),
            )
            print(f"✅ 数据已保存至: {save_path}")
            print(f"   数据形状: {q_m_np.shape}")

            # 简单的可视化检查
            try:
                import matplotlib.pyplot as plt
                preview_path = os.path.join(os.path.dirname(save_path) or ".", "real_data_preview_motor.png")
                plt.figure(figsize=(10, 6))
                plt.subplot(3,1,1)
                plt.plot(logs['time'], q_m_np, label='q_m (motor, unwrapped)')
                plt.legend(); plt.grid()
                plt.subplot(3,1,2)
                plt.plot(logs['time'], qd_m_np, label='qd_m (motor)')
                plt.legend(); plt.grid()
                plt.subplot(3,1,3)
                plt.plot(logs['time'], tau_cmd_np, label='tau_cmd', color='r')
                plt.plot(logs['time'], tau_out_raw_np, label='tau_out_raw (data.tau)', color='k', alpha=0.35)
                plt.plot(logs['time'], tau_out_np, label=f'tau_out_eff = tau_out_raw + kd*qd_m (kd={motor.cmd.kd:g})', color='k', alpha=0.9)
                plt.legend(); plt.grid()
                plt.tight_layout()
                plt.savefig(preview_path)
                print(f"   预览图已保存: {preview_path}")

                # Best-effort keep legacy top-level path for convenience.
                legacy = "real_data_preview.png"
                try:
                    plt.savefig(legacy)
                    print(f"   预览图已保存: {legacy}")
                except Exception as e:
                    print(f"   [warn] cannot overwrite {legacy}: {e}")
            except ImportError:
                pass

            # If running under sudo, chown artifacts back to the invoking user to avoid root-owned files.
            try:
                sudo_uid = os.environ.get("SUDO_UID")
                sudo_gid = os.environ.get("SUDO_GID")
                if sudo_uid is not None and sudo_gid is not None:
                    uid = int(sudo_uid)
                    gid = int(sudo_gid)
                    for p in [save_path, preview_path]:
                        try:
                            os.chown(p, uid, gid)
                        except Exception:
                            pass
            except Exception:
                pass
        else:
            print("❌ 没有采集到数据")

if __name__ == "__main__":
    main()
