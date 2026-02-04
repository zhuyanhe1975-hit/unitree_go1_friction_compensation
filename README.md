# nerd_1dof_goM8010_6

单关节真实动力学建模（Unitree GO-M8010-6），目标是学习/辨识一个可用于预测的关节动力学模型（不包含补偿器相关内容）。

## 目录结构（精简版）

- `config.json`：全局唯一参数入口（仿真/训练/实机采集/路径）。
- `scripts/`：所有 CLI 脚本（仿真采集、real 采集、训练/finetune、评估、对比等）。
- `pipeline/`：数据准备、模型定义、训练调度等可复用模块。
- `assets/`：单关节 MJCF（包含摩擦/回差等结构）。
- `custom_envs/joint_1dof_env.py`：Warp 单关节环境（用于 sim rollout）。

原工程的完整功能可参考：`/home/yhzhu/myWorks/nerd_1dof`。

## 快速开始

### 完整链条（推荐）

1) generation（仿真生成原始 log）：
- `python3 scripts/generate.py`

2) prepare（构建监督数据集 + 统计量）：
- `python3 scripts/prepare.py --mode sim`

3) train（训练仿真动力学模型）：
- `python3 scripts/train.py`

4) finetune（采集真实数据 + 构建数据集 + 微调）：
- 采集：`python3 scripts/collect_real_data.py`（输出到 `paths.real_log`）
- prepare：`python3 scripts/prepare.py --mode real`（使用 `paths.stats_npz` 做归一化）
- finetune：`python3 scripts/finetune.py`

5) eval（评估/画图）：
- `python3 scripts/eval.py`

## Milestone: Torque-Delta Feedforward Compensation (2026-02-04)

在位置闭环模式下（给定 `q_ref/dq_ref/kp/kd`），我们验证了一个可重复的阶段性成果：
使用 **torque-delta 一步预测**构造前馈力矩，可显著降低正弦轨迹跟踪误差，尤其在低速换向/低频场景更明显。

- 说明文档：`docs/MILESTONE_20260204_torque_delta_ff.md`
- 关键 demo：`scripts/demo_ff_sine.py`（baseline vs feedforward 对比）
- torque-delta 管线：`inverse_torque/`（prepare/train/eval）

快速跑硬件 demo（示例）：

```bash
PYTHONPATH=. python3 scripts/demo_ff_sine.py \
  --mode both \
  --ff_type torque_delta \
  --amp 0.2 --freq 0.1 --duration 20 \
  --kp 1 --kd 0.01 \
  --tau_ff_limit 0.15 --tau_ff_scale 1 \
  --ff_update_div 1
```

## 实机采集前：Unitree SDK Python 模块

如果你在 `python3 scripts/collect_real_data.py` 遇到：
`ModuleNotFoundError: No module named 'unitree_actuator_sdk'`
通常是因为 SDK 自带的 `.so` 与当前 Python 版本不匹配（例如 SDK 里是 `cpython-38`，你在用 `py310`）。

用当前 Python 重新编译一次：
- `python3 scripts/build_unitree_sdk.py --sdk_src "/home/yhzhu/Industrial Robot/unitree_actuator_sdk"`

编译输出会在 `runs/unitree_sdk_build/lib/`，把它写到 `config.json`：
- `real.unitree_sdk_lib = "runs/unitree_sdk_build/lib"`

也可以直接设置环境变量（更通用）：
- `UNITREE_ACTUATOR_SDK_LIB=/path/to/unitree_actuator_sdk/lib`

## 实机采集注意事项（关键）

- 如果你使用覆盖性采集工具（`coverage_capture.py`）生成了 `coverage_capture_*.csv`，建议用本仓库脚本转换为 `runs/real_log.npz`，并写入来源追踪字段（`parent_csv/_sha256` 等），避免训练时搞错 CSV 来源：

```bash
PYTHONPATH=. python3 scripts/convert_coverage_csv_to_real_log.py \
  --csv data/coverage_capture_YYYYMMDD_HHMMSS.csv \
  --out runs/real_log.npz \
  --kp 2.0 --kd 0.02 \
  --tau_cmd 0.0 \
  --force
```

检查当前 `real_log.npz` 的来源：

```bash
PYTHONPATH=. python3 scripts/inspect_real_log_source.py --npz runs/real_log.npz
```

- 采集 chirp 力矩时，请确保 `config.json` 里 `real.kp=0` 且 `real.kd=0`，否则反馈的 `tau_out` 会混入内部位置/速度环的力矩分量，导致 `tau_cmd` 与 `tau_out` 形状差异很大。
- `config.json` 的 `data.real.action_key` 决定 real 数据集训练/评估用哪一个力矩作为输入：
  - `"tau_cmd"`：下发的力矩指令
  - `"tau_out"`：SDK 反馈的力矩（kp/kd=0 时更接近“真实作用力矩”）
- 如果后续确认 SDK 的 `data.tau` 是电机侧力矩，可把 `data.real.tau_out_scale_to_out=true`，在进入模型前按 `(gear_ratio * efficiency)` 缩放到输出端等效力矩。
