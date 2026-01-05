from __future__ import annotations

import argparse
import os

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

import torch

from custom_envs.joint_1dof_env import Joint1DofEnv
from pipeline.config import load_cfg
from pipeline.features import state_to_features
from pipeline.model import CausalTransformer
from pipeline.train import _resolve_device
from project_config import ensure_dir, get


def _chirp_scalar(t: float, T: float, f0_hz: float, f1_hz: float, amp: float) -> float:
    k = (float(f1_hz) - float(f0_hz)) / max(1e-6, float(T))
    phase = 2.0 * np.pi * (float(f0_hz) * t + 0.5 * k * (t * t))
    return float(amp * np.sin(phase))


def _smooth_sign(x: float, eps: float = 1e-3) -> float:
    return float(x / (abs(x) + eps))


def _gen_tau_cmd_series(cfg: dict, steps: int, dt: float, source: str) -> np.ndarray:
    if source == "real":
        chirp = get(cfg, "real.chirp")
        f0 = float(chirp["f0_hz"])
        f1 = float(chirp["f1_hz"])
        amp = float(chirp["amplitude_nm"])
        # Important: chirp sweep rate depends on the *planned* duration used during real collection.
        # Even if we only plot the first N steps, we must keep the same total duration to match the real signal.
        T_total = float(get(cfg, "real.duration"))
        tau_limit = float(get(cfg, "real.max_torque"))
        tau_slew = float(get(cfg, "real.tau_slew_nm_s", required=False, default=0.0))
        tau_static = float(get(cfg, "friction.tau_static_out_nm", required=False, default=0.0))
        tau_static_th = float(get(cfg, "real.tau_static_enable_threshold_nm", required=False, default=0.0))
    else:
        chirp = get(cfg, "data.sim.chirp", required=False, default={})
        f0 = float(chirp.get("f0_hz", 0.1))
        f1 = float(chirp.get("f1_hz", 5.0))
        amp = float(chirp.get("amplitude_nm", 0.45))
        T_total = float(get(cfg, "data.sim.steps")) * float(get(cfg, "sim.frame_dt"))
        tau_limit = float(max(abs(float(get(cfg, "data.sim.torque_low"))), abs(float(get(cfg, "data.sim.torque_high")))))
        tau_slew = float(get(cfg, "data.sim.tau_slew_nm_s", required=False, default=0.0))
        tau_static = 0.0
        tau_static_th = 0.0

    tau_cmd = np.zeros((steps,), dtype=np.float64)
    prev = 0.0
    for k in range(steps):
        t = float(k) * float(dt)
        tau_ref = _chirp_scalar(t, T_total, f0, f1, amp)
        tau_ff = 0.0
        if abs(tau_ref) >= tau_static_th and tau_static > 0.0:
            tau_ff = tau_static * _smooth_sign(tau_ref)
        des = float(np.clip(tau_ref + tau_ff, -tau_limit, tau_limit))
        if tau_slew > 0:
            max_step = float(tau_slew) * float(dt)
            des = float(np.clip(des, prev - max_step, prev + max_step))
        tau_cmd[k] = des
        prev = des
    return tau_cmd


def _load_tau_series_from_real_log(cfg: dict, key: str, steps: int, scale_to_out: bool) -> np.ndarray:
    log_path = str(get(cfg, "paths.real_log"))
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"missing real log: {log_path}")
    ds = dict(np.load(log_path, allow_pickle=True))
    if key not in ds:
        raise KeyError(f"missing '{key}' in {log_path}")
    tau = np.asarray(ds[key], dtype=np.float64)
    if tau.ndim == 2 and tau.shape[1] == 1:
        tau = tau[:, 0]
    tau = tau.reshape(-1)
    if len(tau) < steps:
        raise ValueError(f"real log shorter than requested steps: {len(tau)} < {steps}")
    tau = tau[:steps].copy()
    if scale_to_out:
        N = float(get(cfg, "motor.gear_ratio"))
        eta = float(get(cfg, "real.efficiency", required=False, default=1.0))
        tau = tau * (N * eta)
    return tau


def _load_tau_out_eff_from_real_log(cfg: dict, steps: int, kd: float | None) -> np.ndarray:
    log_path = str(get(cfg, "paths.real_log"))
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"missing real log: {log_path}")
    ds = dict(np.load(log_path, allow_pickle=True))
    if "qd_m" not in ds:
        raise KeyError(f"tau_out_eff requires 'qd_m' in {log_path}")
    qd_m = np.asarray(ds["qd_m"], dtype=np.float64).reshape(-1)
    if "tau_out_raw" in ds:
        tau_raw = np.asarray(ds["tau_out_raw"], dtype=np.float64).reshape(-1)
    elif "tau_out" in ds:
        tau_raw = np.asarray(ds["tau_out"], dtype=np.float64).reshape(-1)
    else:
        raise KeyError(f"tau_out_eff requires 'tau_out_raw' or 'tau_out' in {log_path}")
    if len(qd_m) < steps or len(tau_raw) < steps:
        raise ValueError(f"real log shorter than requested steps: {min(len(qd_m), len(tau_raw))} < {steps}")
    if kd is None:
        kd = float(get(cfg, "real.kd", required=False, default=0.0))
    return tau_raw[:steps].copy() + float(kd) * qd_m[:steps].copy()


def _load_real_state_series(cfg: dict, steps: int) -> tuple[np.ndarray, np.ndarray]:
    log_path = str(get(cfg, "paths.real_log"))
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"missing real log: {log_path}")
    ds = dict(np.load(log_path, allow_pickle=True))
    q_key = "q_m" if "q_m" in ds else "q_out"
    qd_key = "qd_m" if "qd_m" in ds else "qd_out"
    q = np.asarray(ds[q_key], dtype=np.float64).reshape(-1)
    qd = np.asarray(ds[qd_key], dtype=np.float64).reshape(-1)
    if len(q) < steps or len(qd) < steps:
        raise ValueError(f"real log shorter than requested steps: {min(len(q), len(qd))} < {steps}")
    return q[:steps].copy(), qd[:steps].copy()


def _load_sim_model(cfg: dict, device: str) -> tuple[CausalTransformer, dict]:
    model_path = str(get(cfg, "paths.sim_model"))
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"missing sim model: {model_path} (run train.py first)")
    ckpt = torch.load(model_path, map_location=torch.device(device))
    model = CausalTransformer(
        input_dim=int(ckpt["input_dim"]),
        output_dim=int(ckpt["output_dim"]),
        embed_dim=int(get(cfg, "model.embed_dim")),
        num_layers=int(get(cfg, "model.num_layers")),
        num_heads=int(get(cfg, "model.num_heads")),
        history_len=int(get(cfg, "model.history_len")),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    stats_path = str(get(cfg, "paths.stats_npz"))
    st = dict(np.load(stats_path))
    stats = {k: st[k].astype(np.float32) for k in ["s_mean", "s_std", "a_mean", "a_std", "d_mean", "d_std"]}
    return model, stats


@torch.no_grad()
def _rollout_model(
    model: CausalTransformer,
    stats: dict,
    q0: float,
    qd0: float,
    tau_cmd: np.ndarray,
    history_len: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    s_mean = stats["s_mean"]
    s_std = stats["s_std"]
    a_mean = stats["a_mean"]
    a_std = stats["a_std"]
    d_mean = stats["d_mean"]
    d_std = stats["d_std"]

    # Initialize history buffers in *raw* units, then normalize per step.
    q_hist = [float(q0)] * history_len
    qd_hist = [float(qd0)] * history_len
    a_hist = [float(tau_cmd[0])] * history_len

    q_pred = np.zeros((len(tau_cmd),), dtype=np.float64)
    qd_pred = np.zeros((len(tau_cmd),), dtype=np.float64)
    q_pred[0] = float(q0)
    qd_pred[0] = float(qd0)

    for k in range(1, len(tau_cmd)):
        # Update action history with the current command (what you would apply for the next step).
        a_hist = a_hist[1:] + [float(tau_cmd[k])]

        feat = state_to_features(np.array(q_hist, dtype=np.float64), np.array(qd_hist, dtype=np.float64)).astype(np.float32)
        act = np.array(a_hist, dtype=np.float32)[:, None]

        feat_n = (feat - s_mean) / s_std
        act_n = (act - a_mean) / a_std
        x = np.concatenate([feat_n, act_n], axis=-1)[None, :, :]  # [1, H, 4]

        pred_norm = model(torch.from_numpy(x).to(device)).cpu().numpy().reshape(-1)
        delta = pred_norm * d_std + d_mean

        q_next = q_hist[-1] + float(delta[0])
        qd_next = qd_hist[-1] + float(delta[1])

        q_hist = q_hist[1:] + [q_next]
        qd_hist = qd_hist[1:] + [qd_next]
        q_pred[k] = q_next
        qd_pred[k] = qd_next

    return q_pred, qd_pred


@torch.no_grad()
def _predict_deltas_teacher_forced(
    model: CausalTransformer,
    stats: dict,
    q_gt: np.ndarray,
    qd_gt: np.ndarray,
    tau_cmd: np.ndarray,
    history_len: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One-step delta prediction using *ground-truth* history windows (teacher forcing).
    Returns (delta_gt, delta_pred), each shape [T-1, 2] for [Δq, Δqd].
    """
    s_mean = stats["s_mean"]
    s_std = stats["s_std"]
    a_mean = stats["a_mean"]
    a_std = stats["a_std"]
    d_mean = stats["d_mean"]
    d_std = stats["d_std"]

    T = len(tau_cmd)
    if not (len(q_gt) == len(qd_gt) == T):
        raise ValueError("q_gt/qd_gt/tau_cmd must have same length")
    if T < history_len + 2:
        raise ValueError("sequence too short for history")

    delta_gt = np.zeros((T - 1, 2), dtype=np.float64)
    delta_gt[:, 0] = q_gt[1:] - q_gt[:-1]
    delta_gt[:, 1] = qd_gt[1:] - qd_gt[:-1]

    delta_pred = np.zeros_like(delta_gt)
    for k in range(history_len - 1, T - 1):
        q_hist = q_gt[k - (history_len - 1) : k + 1]
        qd_hist = qd_gt[k - (history_len - 1) : k + 1]
        a_hist = tau_cmd[k - (history_len - 1) : k + 1]

        feat = state_to_features(q_hist, qd_hist).astype(np.float32)
        act = a_hist.astype(np.float32)[:, None]
        feat_n = (feat - s_mean) / s_std
        act_n = (act - a_mean) / a_std
        x = np.concatenate([feat_n, act_n], axis=-1)[None, :, :]  # [1, H, 4]

        pred_norm = model(torch.from_numpy(x).to(device)).cpu().numpy().reshape(-1)
        delta = pred_norm * d_std + d_mean
        delta_pred[k, 0] = float(delta[0])
        delta_pred[k, 1] = float(delta[1])

    # Remove leading region where we didn't have enough history.
    return delta_gt[history_len - 1 :], delta_pred[history_len - 1 :]


@torch.no_grad()
def _predict_deltas_teacher_forced_physical(
    model: CausalTransformer,
    stats_in: dict,
    d_mean: np.ndarray,
    d_std: np.ndarray,
    q_gt: np.ndarray,
    qd_gt: np.ndarray,
    tau_cmd: np.ndarray,
    history_len: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One-step delta prediction using ground-truth history, returning deltas in physical units.

    - Inputs are normalized using stats_in (s/a).
    - Outputs are de-normalized using (d_mean, d_std).
    """
    s_mean = stats_in["s_mean"]
    s_std = stats_in["s_std"]
    a_mean = stats_in["a_mean"]
    a_std = stats_in["a_std"]

    T = len(tau_cmd)
    if not (len(q_gt) == len(qd_gt) == T):
        raise ValueError("q_gt/qd_gt/tau_cmd must have same length")
    if T < history_len + 2:
        raise ValueError("sequence too short for history")

    delta_gt = np.zeros((T - 1, 2), dtype=np.float64)
    delta_gt[:, 0] = q_gt[1:] - q_gt[:-1]
    delta_gt[:, 1] = qd_gt[1:] - qd_gt[:-1]

    delta_pred = np.zeros_like(delta_gt)
    for k in range(history_len - 1, T - 1):
        q_hist = q_gt[k - (history_len - 1) : k + 1]
        qd_hist = qd_gt[k - (history_len - 1) : k + 1]
        a_hist = tau_cmd[k - (history_len - 1) : k + 1]

        feat = state_to_features(q_hist, qd_hist).astype(np.float32)
        act = a_hist.astype(np.float32)[:, None]
        feat_n = (feat - s_mean) / s_std
        act_n = (act - a_mean) / a_std
        x = np.concatenate([feat_n, act_n], axis=-1)[None, :, :]  # [1, H, 4]

        pred_norm = model(torch.from_numpy(x).to(device)).cpu().numpy().reshape(-1)
        delta = pred_norm * d_std + d_mean
        delta_pred[k, 0] = float(delta[0])
        delta_pred[k, 1] = float(delta[1])

    return delta_gt[history_len - 1 :], delta_pred[history_len - 1 :]


@torch.no_grad()
def _rollout_gt(
    cfg: dict,
    tau_cmd: np.ndarray,
    device: str,
    init_q: float | None = None,
    init_qd: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    env_cfg = {
        "num_envs": 1,
        "frame_dt": float(get(cfg, "sim.frame_dt")),
        "sim_substeps": int(get(cfg, "sim.sim_substeps")),
        "asset_mjcf": str(get(cfg, "paths.asset_mjcf")),
        "mjcf_override": dict(get(cfg, "sim.mjcf_override")),
        "runs_dir": str(get(cfg, "paths.runs_dir")),
    }
    env = Joint1DofEnv(env_cfg, device=device)
    st = env.reset()  # [1, 2]
    if init_q is not None or init_qd is not None:
        import warp as wp
        import numpy as np

        q0 = float(init_q) if init_q is not None else float(st[0, 0].cpu().item())
        qd0 = float(init_qd) if init_qd is not None else float(st[0, 1].cpu().item())
        with wp.ScopedDevice(device):
            env.state.joint_q.assign(wp.from_numpy(np.array([q0], dtype=np.float32), device=device, dtype=wp.float32))
            env.state.joint_qd.assign(wp.from_numpy(np.array([qd0], dtype=np.float32), device=device, dtype=wp.float32))
        st = env.step(torch.zeros((1, 1), dtype=torch.float32, device=device))
    q = np.zeros((len(tau_cmd),), dtype=np.float64)
    qd = np.zeros((len(tau_cmd),), dtype=np.float64)
    q[0] = float(st[0, 0].cpu().item())
    qd[0] = float(st[0, 1].cpu().item())

    for k in range(1, len(tau_cmd)):
        a = torch.tensor([[float(tau_cmd[k])]], dtype=torch.float32, device=device)
        st = env.step(a)
        q[k] = float(st[0, 0].cpu().item())
        qd[k] = float(st[0, 1].cpu().item())
    return q, qd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", choices=["real", "sim"], default="real", help="which chirp settings to use")
    ap.add_argument("--steps", type=int, default=None, help="override rollout steps (default: real.duration/dt or data.sim.steps)")
    ap.add_argument("--no_gt", action="store_true", help="skip Warp ground-truth rollout (faster)")
    ap.add_argument("--delta", action="store_true", help="also save one-step delta plot (teacher forcing)")
    ap.add_argument("--compare_real_scratch", action="store_true", help="compare gt deltas vs real_model_scratch 1-step prediction")
    ap.add_argument(
        "--tau_input",
        choices=["chirp_cmd", "log_tau_cmd", "log_tau_out", "log_tau_out_eff", "log_tau_out_scaled"],
        default="chirp_cmd",
        help="torque input to use for the GT rollout and delta comparisons",
    )
    ap.add_argument("--tau_out_eff_kd", type=float, default=None, help="override kd for log_tau_out_eff")
    ap.add_argument("--plot_real", action="store_true", help="also plot real q/qd (from log) against GT response")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = _resolve_device(cfg)

    dt = float(get(cfg, "sim.frame_dt"))
    if args.steps is not None:
        steps = int(args.steps)
    else:
        if args.source == "real":
            steps = int(get(cfg, "eval.plot_steps", required=False, default=int(round(float(get(cfg, "real.duration")) / float(get(cfg, "real.dt"))))))
            dt = float(get(cfg, "real.dt"))
        else:
            steps = int(get(cfg, "data.sim.steps"))
            dt = float(get(cfg, "sim.frame_dt"))

    if args.tau_input == "chirp_cmd":
        tau_cmd = _gen_tau_cmd_series(cfg, steps=steps, dt=dt, source=args.source)
    elif args.tau_input == "log_tau_cmd":
        tau_cmd = _load_tau_series_from_real_log(cfg, key="tau_cmd", steps=steps, scale_to_out=False)
    elif args.tau_input == "log_tau_out":
        tau_cmd = _load_tau_series_from_real_log(cfg, key="tau_out", steps=steps, scale_to_out=False)
    elif args.tau_input == "log_tau_out_eff":
        tau_cmd = _load_tau_out_eff_from_real_log(cfg, steps=steps, kd=args.tau_out_eff_kd)
    else:  # log_tau_out_scaled
        tau_cmd = _load_tau_series_from_real_log(cfg, key="tau_out", steps=steps, scale_to_out=True)

    model, stats = _load_sim_model(cfg, device=device)
    H = int(get(cfg, "model.history_len"))

    # Initial state: use gt initial state when available, otherwise zeros.
    if args.no_gt:
        q0, qd0 = 0.0, 0.0
        q_gt = qd_gt = None
    else:
        init_q = init_qd = None
        if args.plot_real and args.source == "real":
            try:
                q_real0, qd_real0 = _load_real_state_series(cfg, steps=steps)
                init_q, init_qd = float(q_real0[0]), float(qd_real0[0])
            except Exception:
                init_q = init_qd = None
        q_gt, qd_gt = _rollout_gt(cfg, tau_cmd=tau_cmd, device=device, init_q=init_q, init_qd=init_qd)
        q0, qd0 = float(q_gt[0]), float(qd_gt[0])

    q_pred, qd_pred = _rollout_model(model, stats, q0=q0, qd0=qd0, tau_cmd=tau_cmd, history_len=H, device=device)

    out_dir = str(get(cfg, "paths.runs_dir"))
    ensure_dir(out_dir)

    if plt is None:
        print("matplotlib not available; skipping plot")
        return

    t = np.arange(steps, dtype=np.float64) * float(dt)

    q_real = qd_real = None
    if args.plot_real and args.source == "real":
        try:
            q_real, qd_real = _load_real_state_series(cfg, steps=steps)
        except Exception as e:
            print(f"[warn] plot_real disabled: {e}")
            q_real = qd_real = None

    plt.figure(figsize=(12, 9))

    plt.subplot(3, 1, 1)
    plt.plot(t, tau_cmd, label="tau_cmd", color="k", lw=1.2)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(3, 1, 2)
    if q_gt is not None:
        plt.plot(t, q_gt, label="q_out (gt)", color="k", alpha=0.6)
    if q_real is not None:
        plt.plot(t, q_real, label="q_out (real log)", color="g", alpha=0.5)
    plt.plot(t, q_pred, label="q_out (sim_model)", color="r", lw=1.2)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(3, 1, 3)
    if qd_gt is not None:
        plt.plot(t, qd_gt, label="qd_out (gt)", color="k", alpha=0.6)
    if qd_real is not None:
        plt.plot(t, qd_real, label="qd_out (real log)", color="g", alpha=0.5)
    plt.plot(t, qd_pred, label="qd_out (sim_model)", color="r", lw=1.2)
    plt.grid(True, alpha=0.3)
    plt.legend()

    out = os.path.join(out_dir, f"rollout_sim_model_chirp_{args.source}_{args.tau_input}.png")
    plt.tight_layout()
    plt.savefig(out)
    print(f"saved: {out}")

    if args.delta and q_gt is not None and qd_gt is not None:
        delta_gt, delta_pred = _predict_deltas_teacher_forced(
            model,
            stats=stats,
            q_gt=q_gt,
            qd_gt=qd_gt,
            tau_cmd=tau_cmd,
            history_len=H,
            device=device,
        )
        tt = t[H:steps]  # align with delta arrays length
        n = min(len(tt), len(delta_gt))
        tt = tt[:n]
        delta_gt = delta_gt[:n]
        delta_pred = delta_pred[:n]

        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(tt, delta_gt[:, 0], label="delta_q (gt)", alpha=0.7)
        plt.plot(tt, delta_pred[:, 0], label="delta_q (pred, 1-step)", alpha=0.7)
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(tt, delta_gt[:, 1], label="delta_qd (gt)", alpha=0.7)
        plt.plot(tt, delta_pred[:, 1], label="delta_qd (pred, 1-step)", alpha=0.7)
        plt.grid(True, alpha=0.3)
        plt.legend()

        out = os.path.join(out_dir, f"rollout_sim_model_chirp_{args.source}_{args.tau_input}_delta.png")
        plt.tight_layout()
        plt.savefig(out)
        print(f"saved: {out}")

    if args.compare_real_scratch and q_gt is not None and qd_gt is not None:
        real_scratch_path = str(get(cfg, "paths.real_model_scratch"))
        real_dataset_path = str(get(cfg, "paths.real_dataset"))
        if not os.path.exists(real_scratch_path):
            raise FileNotFoundError(f"missing real_model_scratch: {real_scratch_path} (run train.py --mode real)")
        if not os.path.exists(real_dataset_path):
            raise FileNotFoundError(f"missing real_dataset: {real_dataset_path} (run prepare.py --mode real)")

        ds_real = dict(np.load(real_dataset_path, allow_pickle=True))
        d_mean = ds_real["d_mean"].astype(np.float32)
        d_std = ds_real["d_std"].astype(np.float32)
        stats_real = {
            "s_mean": ds_real["s_mean"].astype(np.float32),
            "s_std": ds_real["s_std"].astype(np.float32),
            "a_mean": ds_real["a_mean"].astype(np.float32),
            "a_std": ds_real["a_std"].astype(np.float32),
            "d_mean": d_mean,
            "d_std": d_std,
        }

        ckpt = torch.load(real_scratch_path, map_location=torch.device(device))
        model_real = CausalTransformer(
            input_dim=int(ckpt["input_dim"]),
            output_dim=int(ckpt["output_dim"]),
            embed_dim=int(get(cfg, "model.embed_dim")),
            num_layers=int(get(cfg, "model.num_layers")),
            num_heads=int(get(cfg, "model.num_heads")),
            history_len=int(get(cfg, "model.history_len")),
        ).to(device)
        model_real.load_state_dict(ckpt["model"])
        model_real.eval()

        delta_gt, delta_pred = _predict_deltas_teacher_forced_physical(
            model_real,
            stats_in=stats_real,
            d_mean=d_mean,
            d_std=d_std,
            q_gt=q_gt,
            qd_gt=qd_gt,
            tau_cmd=tau_cmd,
            history_len=H,
            device=device,
        )

        tt = t[H:steps]
        n = min(len(tt), len(delta_gt))
        tt = tt[:n]
        delta_gt = delta_gt[:n]
        delta_pred = delta_pred[:n]

        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(tt, delta_gt[:, 0], label="delta_q (gt)", alpha=0.7)
        plt.plot(tt, delta_pred[:, 0], label="delta_q (real_scratch, 1-step)", alpha=0.7)
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(tt, delta_gt[:, 1], label="delta_qd (gt)", alpha=0.7)
        plt.plot(tt, delta_pred[:, 1], label="delta_qd (real_scratch, 1-step)", alpha=0.7)
        plt.grid(True, alpha=0.3)
        plt.legend()

        out = os.path.join(out_dir, f"compare_gt_vs_real_scratch_delta_{args.source}_{args.tau_input}.png")
        plt.tight_layout()
        plt.savefig(out)
        print(f"saved: {out}")


if __name__ == "__main__":
    main()
