from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import numpy as np

from project_config import ensure_dir, get
from .features import state_to_features


def _one_pole_lpf(x: np.ndarray, dt: float, cutoff_hz: float) -> np.ndarray:
    """
    Simple 1st-order low-pass filter for time series.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if cutoff_hz <= 0 or len(x) == 0:
        return x
    rc = 1.0 / (2.0 * np.pi * float(cutoff_hz))
    alpha = float(dt) / (rc + float(dt))
    y = np.empty_like(x)
    y[0] = x[0]
    for k in range(1, len(x)):
        y[k] = y[k - 1] + alpha * (x[k] - y[k - 1])
    return y


def _compute_stats(feat: np.ndarray, act: np.ndarray, delta: np.ndarray) -> Dict[str, np.ndarray]:
    s_mean = feat.mean(axis=(0, 1)).astype(np.float32)
    s_std = (feat.std(axis=(0, 1)) + 1e-6).astype(np.float32)
    a_mean = act.mean(axis=(0, 1)).astype(np.float32)
    a_std = (act.std(axis=(0, 1)) + 1e-6).astype(np.float32)
    d_mean = delta.mean(axis=(0, 1)).astype(np.float32)
    d_std = (delta.std(axis=(0, 1)) + 1e-6).astype(np.float32)
    return {"s_mean": s_mean, "s_std": s_std, "a_mean": a_mean, "a_std": a_std, "d_mean": d_mean, "d_std": d_std}


def prepare_dataset(
    cfg: Dict[str, Any],
    raw_npz: str,
    out_npz: str,
    stats_npz: str | None = None,
    real_stats_npz: str | None = None,
) -> None:
    """
    Converts a raw log into a supervised dataset for sequence-to-delta prediction.

    Raw format (sim or real):
      - t: [T]
      - q_out: [T] or [K, T]
      - qd_out: [T] or [K, T]
      - tau_cmd: [T] or [K, T]   (commanded torque)
      - tau_out: [T] or [K, T]   (measured torque, optional; real logs)

    Action selection:
      - sim logs always use tau_cmd
      - real logs use cfg `data.real.action_key` ("tau_cmd" or "tau_out")
        If `data.real.tau_out_scale_to_out=true`, tau_out is scaled by (gear_ratio * efficiency).

    Output:
      - x: [N, H, Din] normalized
      - y: [N, Dout] normalized (delta of [q_out, qd_out])
      - stats: mean/std arrays
    """
    ds = dict(np.load(raw_npz, allow_pickle=True))
    q = ds["q_out"]
    qd = ds["qd_out"]
    is_real = ("q_m" in ds) or ("q_m_raw" in ds)
    if is_real:
        action_key = str(get(cfg, "data.real.action_key", required=False, default="tau_cmd"))
        if action_key not in ds:
            raise KeyError(f"real log missing action_key='{action_key}' (available keys: {sorted(ds.keys())})")
        tau = ds[action_key]
        if action_key == "tau_out" and bool(get(cfg, "data.real.tau_out_scale_to_out", required=False, default=False)):
            N = float(get(cfg, "motor.gear_ratio"))
            eta = float(get(cfg, "real.efficiency", required=False, default=1.0))
            tau = np.asarray(tau, dtype=np.float64) * (N * eta)
    else:
        tau = ds["tau_cmd"]
    t = ds.get("t", None)

    q = np.asarray(q, dtype=np.float64)
    qd = np.asarray(qd, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)

    # Accept common log shapes:
    # - [T] (preferred for single-trajectory logs)
    # - [1, T] or [K, T] (multi-trajectory logs)
    # - [T, 1] (column-vector logs; squeeze to [T])
    if q.ndim == 2 and q.shape[1] == 1:
        q = q[:, 0]
    if qd.ndim == 2 and qd.shape[1] == 1:
        qd = qd[:, 0]
    if tau.ndim == 2 and tau.shape[1] == 1:
        tau = tau[:, 0]

    # Normalize shapes to [K, T]
    if q.ndim == 1:
        q = q[None, :]
        qd = qd[None, :]
        tau = tau[None, :]
    if not (q.shape == qd.shape == tau.shape):
        raise ValueError(f"shape mismatch: q={q.shape}, qd={qd.shape}, tau={tau.shape}")

    H = int(get(cfg, "model.history_len"))
    K, T = q.shape
    if T < H + 2:
        raise ValueError("trajectory too short for history_len")

    # Optional filtering for real logs: motor-provided velocity can be noisy/quantized,
    # which blows up delta_qd and hurts supervised training.
    if "q_m" in ds or "q_m_raw" in ds:
        if t is not None:
            tt = np.asarray(t, dtype=np.float64).reshape(-1)
            if len(tt) >= 2:
                dt = float(np.median(np.diff(tt)))
            else:
                dt = 0.0
        else:
            dt = 0.0
        if dt > 0:
            qd_source = str(get(cfg, "data.real.qd_source", required=False, default="from_log"))
            if qd_source == "from_q":
                # Derive velocity from encoder position for internal consistency.
                qd_from_q = np.zeros_like(q, dtype=np.float64)
                qd_from_q[:, 1:] = (q[:, 1:] - q[:, :-1]) / dt
                qd = qd_from_q

            qd_lpf_hz = float(get(cfg, "data.real.qd_lpf_hz", required=False, default=0.0))
            if qd_lpf_hz > 0:
                qd_f = np.zeros_like(qd, dtype=np.float64)
                for traj in range(K):
                    qd_f[traj] = _one_pole_lpf(qd[traj], dt=dt, cutoff_hz=qd_lpf_hz)
                qd = qd_f

    # Feature tensors [K, T, ...]
    feat = state_to_features(q, qd).astype(np.float32)
    act = tau[..., None].astype(np.float32)
    delta = np.stack([q[:, 1:] - q[:, :-1], qd[:, 1:] - qd[:, :-1]], axis=-1).astype(np.float32)  # [K, T-1, 2]

    # Stats
    # - For sim: compute and persist full stats in stats_npz.
    # - For real: by default reuse sim's s/a stats, but compute d stats from real.
    sim_stats = None
    if stats_npz is not None and os.path.exists(stats_npz):
        st = dict(np.load(stats_npz))
        sim_stats = {k: st[k].astype(np.float32) for k in ["s_mean", "s_std", "a_mean", "a_std", "d_mean", "d_std"]}

    if sim_stats is None:
        stats = _compute_stats(feat, act, delta)
        if stats_npz is not None:
            ensure_dir(os.path.dirname(stats_npz) or ".")
            np.savez(stats_npz, **stats)
    else:
        stats = dict(sim_stats)
        if ("q_m" in ds or "q_m_raw" in ds) and str(get(cfg, "data.real.d_stats", required=False, default="sim")) == "real":
            # Replace only delta stats with real ones.
            d_mean = delta.reshape(-1, 2).mean(axis=0).astype(np.float32)
            d_std = (delta.reshape(-1, 2).std(axis=0) + 1e-6).astype(np.float32)
            stats["d_mean"] = d_mean
            stats["d_std"] = d_std
            if real_stats_npz is not None:
                ensure_dir(os.path.dirname(real_stats_npz) or ".")
                np.savez(real_stats_npz, d_mean=d_mean, d_std=d_std)

    # Build windows: inputs at [k-H+1..k], target = delta at k (i.e., x_k -> x_{k+1})
    xs = []
    ys = []
    for traj in range(K):
        for k in range(H - 1, T - 1):
            s_win = feat[traj, k - (H - 1) : k + 1]  # [H, 3]
            a_win = act[traj, k - (H - 1) : k + 1]  # [H, 1]
            x = np.concatenate([s_win, a_win], axis=-1)  # [H, 4]
            y = delta[traj, k]  # [2]
            xs.append(x)
            ys.append(y)

    x = np.stack(xs, axis=0).astype(np.float32)
    y = np.stack(ys, axis=0).astype(np.float32)

    # Normalize
    s_mean, s_std = stats["s_mean"], stats["s_std"]
    a_mean, a_std = stats["a_mean"], stats["a_std"]
    d_mean, d_std = stats["d_mean"], stats["d_std"]
    x[..., :3] = (x[..., :3] - s_mean) / s_std
    x[..., 3:] = (x[..., 3:] - a_mean) / a_std
    y = (y - d_mean) / d_std

    ensure_dir(os.path.dirname(out_npz) or ".")
    np.savez(
        out_npz,
        x=x,
        y=y,
        s_mean=s_mean,
        s_std=s_std,
        a_mean=a_mean,
        a_std=a_std,
        d_mean=d_mean,
        d_std=d_std,
        stats_path=np.array([stats_npz or ""], dtype=object),
        real_stats_path=np.array([real_stats_npz or ""], dtype=object),
    )


def load_prepared(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    ds = dict(np.load(npz_path, allow_pickle=True))
    return ds["x"].astype(np.float32), ds["y"].astype(np.float32)
