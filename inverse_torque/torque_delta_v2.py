from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np

from pipeline.features import state_error_to_features
from inverse_torque.torque_delta import _apply_by_stage, _lpf_1pole, _lpf_1pole_zero_phase
from project_config import ensure_dir, get


def _compute_stats(x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    x_mean = x.mean(axis=(0, 1)).astype(np.float32)
    x_std = (x.std(axis=(0, 1)) + 1e-6).astype(np.float32)
    y_mean = y.mean(axis=0).astype(np.float32)
    y_std = (y.std(axis=0) + 1e-6).astype(np.float32)
    return {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}


def _get_any(ds: dict[str, Any], keys: tuple[str, ...]) -> np.ndarray | None:
    for k in keys:
        if k in ds:
            return np.asarray(ds[k], dtype=np.float64).reshape(-1)
    return None


def prepare_torque_delta_dataset_v2(
    cfg: Dict[str, Any],
    raw_npz: str,
    out_npz: str,
    stats_npz: str | None = None,
    *,
    tau_lpf_hz: float | None = None,
    qd_lpf_hz: float | None = None,
    zero_phase: bool = True,
) -> None:
    """
    Torque-delta dataset v2: add controller context for position-loop conditions.

      input (history ending at k-1):
        [sin(q), cos(q), qd, e_q, e_qd, (temp?), tau_out]

      target:
        delta_tau_out[k] = tau_out[k] - tau_out[k-1]

    Requirements on raw log:
    - Must contain at least: q/qd/tau_out and either (q_ref, qd_ref) or (e_q, e_qd).
    - Accepts both naming conventions:
      - real log: q_out/qd_out
      - ff demo log: q/qd
    """
    ds = dict(np.load(raw_npz, allow_pickle=True))

    q = _get_any(ds, ("q_out", "q"))
    qd = _get_any(ds, ("qd_out", "qd"))
    tau_out = _get_any(ds, ("tau_out",))
    temp = _get_any(ds, ("temp",))
    t = _get_any(ds, ("t", "time"))

    if q is None or qd is None or tau_out is None:
        raise KeyError("raw log missing required keys: q(_out), qd(_out), tau_out")

    q_ref = _get_any(ds, ("q_ref",))
    qd_ref = _get_any(ds, ("qd_ref",))
    e_q = _get_any(ds, ("e_q",))
    e_qd = _get_any(ds, ("e_qd",))

    if q_ref is None and e_q is not None:
        q_ref = q + e_q
    if qd_ref is None and e_qd is not None:
        qd_ref = qd + e_qd

    if q_ref is None or qd_ref is None:
        raise KeyError("raw log missing controller context: need (q_ref, qd_ref) or (e_q, e_qd)")

    stage_id = _get_any(ds, ("stage_id",))
    if stage_id is None or stage_id.size != q.size:
        stage_id = np.zeros_like(q, dtype=np.int64)
    stage_id = stage_id.astype(np.int64).reshape(-1)

    if not (q.shape == qd.shape == tau_out.shape == q_ref.shape == qd_ref.shape):
        raise ValueError(
            f"shape mismatch: q={q.shape} qd={qd.shape} q_ref={q_ref.shape} qd_ref={qd_ref.shape} tau_out={tau_out.shape}"
        )

    H = int(get(cfg, "model.history_len"))
    T = int(q.shape[0])
    if T < H + 1:
        raise ValueError("trajectory too short for history_len")

    # Optional smoothing (training-time only). Apply within stages to avoid boundary artifacts.
    if t is not None and t.size >= 3:
        dt = float(np.median(np.diff(t)))
    else:
        dt = float(get(cfg, "real.dt", required=False, default=0.01))

    def _make_filter(fc: float):
        if zero_phase:
            return lambda z: _lpf_1pole_zero_phase(z, fc_hz=fc, dt=dt)
        return lambda z: _lpf_1pole(z, fc_hz=fc, dt=dt)

    if qd_lpf_hz is not None and float(qd_lpf_hz) > 0.0:
        qd = _apply_by_stage(qd, stage_id, _make_filter(float(qd_lpf_hz)))
    if tau_lpf_hz is not None and float(tau_lpf_hz) > 0.0:
        tau_out = _apply_by_stage(tau_out, stage_id, _make_filter(float(tau_lpf_hz)))

    feat_state = state_error_to_features(q, qd, q_ref, qd_ref).astype(np.float32)  # [T, 5]
    tau_col = tau_out.astype(np.float32).reshape(-1, 1)
    if temp is not None and temp.size == T:
        temp_col = temp.astype(np.float32).reshape(-1, 1)
        feat_full = np.concatenate([feat_state, temp_col, tau_col], axis=-1)  # [T, 7]
    else:
        feat_full = np.concatenate([feat_state, tau_col], axis=-1)  # [T, 6]

    # delta at k uses k-1, so define delta[0]=0 for convenience.
    delta_tau = np.zeros((T, 1), dtype=np.float32)
    delta_tau[1:, 0] = (tau_out[1:] - tau_out[:-1]).astype(np.float32)

    xs = []
    ys = []
    for k in range(H, T):
        # Avoid mixing samples across capture stages.
        if stage_id[k - H] != stage_id[k]:
            continue
        if not np.all(stage_id[k - H : k + 1] == stage_id[k]):
            continue
        xs.append(feat_full[k - H : k])  # [H, Din]
        ys.append(delta_tau[k])

    x = np.stack(xs, axis=0).astype(np.float32)  # [N, H, Din]
    y = np.stack(ys, axis=0).astype(np.float32)  # [N, 1]

    val_ratio = float(get(cfg, "data.prepare.val_ratio", required=False, default=0.1))
    val_ratio = float(np.clip(val_ratio, 0.0, 0.5))
    n = int(x.shape[0])
    n_train = int(round(n * (1.0 - val_ratio)))
    n_train = max(1, min(n - 1, n_train)) if n >= 2 else n
    train_idx = np.arange(0, n_train, dtype=np.int64)
    val_idx = np.arange(n_train, n, dtype=np.int64)

    stats = _compute_stats(x, y)
    if stats_npz is not None:
        ensure_dir(os.path.dirname(stats_npz) or ".")
        np.savez(stats_npz, **stats)

    x_n = (x - stats["x_mean"]) / stats["x_std"]
    y_n = (y - stats["y_mean"]) / stats["y_std"]

    ensure_dir(os.path.dirname(out_npz) or ".")
    np.savez(
        out_npz,
        x=x_n,
        y=y_n,
        x_mean=stats["x_mean"],
        x_std=stats["x_std"],
        y_mean=stats["y_mean"],
        y_std=stats["y_std"],
        train_idx=train_idx,
        val_idx=val_idx,
        stats_path=np.array([stats_npz or ""], dtype=object),
        parent_raw=np.array([raw_npz], dtype=object),
        feature_version=np.array(["v2_state_error_tau"], dtype=object),
    )
