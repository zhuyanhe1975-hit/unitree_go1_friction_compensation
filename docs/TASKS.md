# 任务清单：Torque-Delta 前馈补偿提升路线图

范围：保持当前已验证成功的做法（**直接用实机 `runs/real_log.npz` 训练**），并逐步提升 torque-delta 前馈补偿的跟踪性能与鲁棒性。

关键入口：
- 离线：`inverse_torque/prepare.py`、`inverse_torque/train.py`、`inverse_torque/eval_low_speed_reversal.py`
- 在线 demo：`scripts/demo_ff_sine.py`（`--ff_type torque_delta`）

---

## Phase 0 — 基线冻结（现在）
- [x] 用一个小表格记录当前“已知好用”的 demo 默认参数（amp/freq/kp/kd/limit/scale/update_div）。
- [x] 定义我们每次都要输出的成功指标：
- [x] `rmse_e_q`、`maxabs_e_q`、`meanabs_tau_out`、`meanabs_tau_ff`、`loop_dt_median`、`loop_dt_p90`
- [x] 定义 1 条标准轨迹（低速）+ 1 条压力轨迹（较高速）用于回归测试。

交付物：
- 在本文档中固定一条“黄金命令”（golden run）。
- 给出回归门槛：例如“新模型不能让 baseline 指标恶化超过 X%”。

### Phase 0 产物（已完成）

已知好用参数（来自 2026-02-04 的实机 demo 结果，`ff_type=torque_delta`）：

| 场景 | 轨迹 (amp,freq,dur,dt) | kp/kd | FF(limit,scale,div) | baseline (rmse/max) | ff (rmse/max) | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 标准（低频低速，偏软反馈） | 0.2, 0.1, 20, 0.01 | 1 / 0.01 | 0.15, 1, 1 | 0.033862 / 0.063797 | 0.009577 / 0.034417 | `runs/ff_demo_report_20260204_155343.md` |
| 低速（较高频，偏软反馈） | 0.2, 0.2, 20, 0.01 | 1 / 0.01 | 0.15, 1, 1 | 0.035657 / 0.062428 | 0.011010 / 0.038489 | `runs/ff_demo_report_20260204_154740.md` |
| 低速（偏硬反馈） | 0.2, 0.2, 20, 0.01 | 5 / 0.05 | 0.15, 1, 1 | 0.009029 / 0.016698 | 0.004603 / 0.011644 | `runs/ff_demo_report_20260204_154347.md` |
| 压力（更高频） | 0.2, 0.5, 20, 0.01 | 1 / 0.01 | 0.15, 1, 1 | 0.038112 / (n/a) | 0.016497 / (n/a) | `runs/ff_demo_report_20260204_155218.md` |

黄金命令（标准轨迹，建议作为回归基线）：

```bash
# 位置闭环 + torque-delta 前馈（标准轨迹）
PYTHONPATH=. python3 scripts/demo_ff_sine.py \
  --mode both \
  --ff_type torque_delta \
  --amp 0.2 --freq 0.1 --duration 20 \
  --kp 1 --kd 0.01 \
  --tau_ff_limit 0.15 --tau_ff_scale 1 \
  --ff_update_div 1
```

压力轨迹（用于验证更高频/更高速度时是否会退化）：

```bash
# 位置闭环 + torque-delta 前馈（压力轨迹）
PYTHONPATH=. python3 scripts/demo_ff_sine.py \
  --mode both \
  --ff_type torque_delta \
  --amp 0.2 --freq 0.5 --duration 20 \
  --kp 1 --kd 0.01 \
  --tau_ff_limit 0.15 --tau_ff_scale 1 \
  --ff_update_div 1
```

回归门槛（建议，后续可根据数据再收紧）：
- 标准轨迹：`ff.rmse_e_q` 不得劣化超过 10%，且 `loop_dt_p90 <= 0.012s`
- 压力轨迹：`ff.rmse_e_q` 不得劣化超过 15%，且 `loop_dt_p90 <= 0.012s`

---

## Phase 1 — 评估工具链（让进展可量化）
- [ ] 增加一个离线评估脚本：回放 `ff_demo_*.npz`，对多次实验输出一张汇总表。
- [ ] 增加一个“分段指标”报告，专注低速换向段：
- [ ] 用 `qd` 自动检测换向区间，只在这些区间统计误差指标。
- [ ] 每一批实验产出一个 `runs/summary_*.md`（不强制截图）。

验收标准：
- 一条命令可以生成 markdown 汇总，对比多组实验结果。

---

## Phase 2 — 更好的实机日志（补齐“控制器上下文”）
目标：让模型“看见”位置环到底想让系统做什么。

- [ ] 扩展 `scripts/demo_ff_sine.py` 的日志字段，至少包含：
- [ ] `kp`、`kd`（逐步记录）
- [ ] `tau_cmd_ff`（实际下发的前馈力矩）
- [ ] `qd_from_q`（由 q 和 dt 差分得到）
- [ ] 可选：`qdd_from_q`（轻度滤波后差分）
- [ ] 日志中已有 `q_ref`、`qd_ref`，并额外计算/存储：
- [ ] `e_q = q_ref - q`
- [ ] `e_qd = qd_ref - qd`
- [ ] 把 `runs/real_log.npz` 的字段/含义集中写在一个文档里。

验收标准：
- 新的 `real_log` 包含足够信息，可训练：
- [ ] 纯状态驱动模型（dynamics-ish），或
- [ ] 控制器感知模型（基于 ref/error）。

---

## Phase 3 — Torque-Delta 模型 v2（特征升级）
目标：在“部署的真实工况”（位置闭环）下提升可预测性。

- [ ] 新建一份数据集版本（不破坏现有 v1）：
- [ ] 输出：`runs/torque_delta_dataset_v2.npz`
- [ ] 每步输入特征（建议）：
- [ ] `sin(q)`、`cos(q)`、`qd`
- [ ] `q_ref`、`qd_ref`（或 `e_q`、`e_qd`）
- [ ] `tau_out`（只用到 k-1 的历史，避免泄漏）
- [ ] 可选：`temp`
- [ ] 目标保持一致：`delta_tau_out[k]`
- [ ] 更新 `scripts/demo_ff_sine.py`：支持选择使用 v1 还是 v2 的模型/数据集。

验收标准：
- v2 在标准低速轨迹上优于 v1，且不会引入明显的实时性问题。

---

## Phase 4 — 训练改进（降低增量学习的噪声）
- [ ] 增加“先平滑再差分”的可选项：
- [ ] 对 `tau_out` 先低通，再计算 `delta_tau_out`
- [ ] 增加鲁棒损失函数可选项：
- [ ] Huber loss（或 clipped MSE），降低对力矩量化尖峰的敏感性
- [ ] 对“摩擦关键区域”做样本加权：
- [ ] 当 `|qd| < v_th` 时权重更高
- [ ] 当 `qd` 发生符号变化附近权重更高
- [ ] 基于时间序列验证集做 early-stopping（避免过拟合）。

验收标准：
- 验证集指标提升，且多次训练的方差更小、更稳定。

---

## Phase 5 — 在线集成（更安全 + 更有效）
- [ ] 在 demo 中增加 `torque_delta_mode` 选项：
- [ ] `tau_hat`：`tau_ff = tau_out_prev + delta_pred`（当前方案）
- [ ] `delta_only`：`tau_ff = delta_pred`（更像“补偿器”）
- [ ] 增加“换向附近 gating”选项：
- [ ] 仅在 `|qd| < v_gate` 或换向附近窗口内应用前馈
- [ ] 增加一个简单的“置信度/跳变限制器”：
- [ ] 若模型输出跳变过快，则对 delta 的限幅/限斜率更严格（比 `tau_ff_slew` 更激进）

验收标准：
- `delta_only` 与 `gated` 模式不会导致跟踪不稳定，同时在换向误差上仍有可见收益。

---

## Phase 6 — 泛化验证矩阵（证明不是偶然）
- [ ] 在实机上跑一个网格（在安全范围内尽量多）：
- [ ] `amp ∈ {0.1, 0.2, 0.3}`
- [ ] `freq ∈ {0.1, 0.2, 0.5}`
- [ ] `kp/kd` 两套：偏软与偏硬
- [ ] 用表格汇总：baseline vs ff 的改进幅度。

验收标准：
- 在低速/低频区域多数工况提升明显；在其它区域不出现灾难性退化。

---

## Phase 7 — 打包 / 发布（可选）
- [ ] 增加 torque-delta 前馈的 quickstart（最少步骤）。
- [ ] 把实验配置版本化并纳入仓库：
- [ ] `configs/ff_demo_soft.json`
- [ ] `configs/ff_demo_stiff.json`
- [ ] 增加一个引导脚本：采集 → 训练 → 运行 demo 一键串联。

验收标准：
- 其他人从零开始按 5–8 条命令能复现实机里程碑效果。
