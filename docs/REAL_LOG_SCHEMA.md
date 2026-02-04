# 实机日志字段说明（real_log / ff_demo）

本文档用于集中说明本项目中常见的实机日志（`.npz`）字段含义，方便后续训练/评估/复现实验。

> 说明：
> - `.npz` 中每个字段通常是一维数组 `[T]`，表示按时间采样的序列。
> - 不同脚本生成的日志字段可能不同；以下按“最常用”与“demo 专用”两类整理。

---

## A. `runs/real_log.npz`（用于训练 torque-delta / friction 等离线模型）

典型来源：
- `scripts/collect_real_data.py`（实机采集）
- 或由外部 CSV 转换得到（如 `data/*.csv -> runs/real_log.npz`）

常用字段：
- `t`：时间戳（秒），`[T]`
- `q_out`：关节角（rad），`[T]`
- `qd_out`：关节角速度（rad/s），`[T]`
- `tau_out`：反馈/估计的实际力矩（Nm），`[T]`
- `tau_cmd`：下发力矩指令（Nm），`[T]`（位置闭环模式下可能为 0 或不存在）
- `temp`：温度（C），`[T]`（可选）
- `merror`：电机错误码（int/float），`[T]`（可选）
- `q_ref` / `qd_ref`：位置闭环参考（rad / rad/s），`[T]`（coverage 采集 CSV 转换后会带）
- `kp` / `kd`：位置环增益（float），`[T]`（coverage 采集 CSV 转换后会带，通常为常数）
- `stage_id`：采集阶段编号（int），`[T]`（用于切段，避免跨 stage 拼接样本）

补充字段（某些采集脚本会带）：
- `q_m/q_m_raw`、`qd_m`：电机侧角度/速度（rad/rad/s），`[T]`
- `tau_out_raw`：SDK 原始 `data.tau` 通道，`[T]`

来源追踪（强烈建议保留）：
- `parent_csv`：本 `real_log.npz` 由哪个 CSV 转换而来（字符串），形如 `[".../coverage_capture_*.csv"]`
- `parent_csv_sha256`：CSV 的 SHA256（字符串），用于确保“没搞错来源”
- `created_at`：转换时间（字符串）
- `git_head`：转换时仓库 git commit（字符串，可能为空）

---

## B. `runs/ff_demo_*_<stamp>.npz`（用于对比有无前馈补偿的在线 demo 日志）

典型来源：
- `scripts/demo_ff_sine.py`

字段（核心）：
- `t`：时间戳（秒），`[T]`
- `q`：实际角度（rad），`[T]`
- `qd`：实际角速度（rad/s），`[T]`
- `q_ref`：参考角度（rad），`[T]`
- `qd_ref`：参考角速度（rad/s），`[T]`
- `e_q`：位置误差 `q_ref - q`（rad），`[T]`
- `e_qd`：速度误差 `qd_ref - qd`（rad/s），`[T]`
- `tau_out`：反馈/估计的实际力矩（Nm），`[T]`
- `tau_ff`：前馈力矩（Nm），`[T]`（baseline 为 0）
- `tau_cmd_ff`：实际下发的前馈力矩（Nm），`[T]`（与 `tau_ff` 等价，保留为“语义明确”字段）
- `kp`、`kd`：本次试验采用的 PD 增益（逐步记录），`[T]`

字段（实时性/调试）：
- `loop_dt`：控制循环实际运行间隔（秒），`[T]`
- `temp`：温度（C），`[T]`
- `merror`：错误码，`[T]`

字段（由 q/t 推导，便于分析）：
- `qd_from_q`：用 `q/t` 差分得到的速度（rad/s），`[T]`
- `qdd_from_q`：用 `qd_from_q/t` 差分得到的加速度（rad/s^2），`[T]`

---

## 备注：字段命名与“时刻 k”

在 torque-delta 训练中，我们通常使用历史窗口 `[k-H .. k-1]` 作为输入，预测下一步的增量：
- 目标：`delta_tau_out[k] = tau_out[k] - tau_out[k-1]`

因此部署时“当前测量”对应训练样本里的 `k-1`，模型预测的是下一步 `k` 的增量。
