from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from pipeline.features import state_to_features
from pipeline.model import CausalTransformer
from pipeline.train import _resolve_device
from project_config import ensure_dir, get


class TorqueDeltaDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def _compute_stats(x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    x_mean = x.mean(axis=(0, 1)).astype(np.float32)
    x_std = (x.std(axis=(0, 1)) + 1e-6).astype(np.float32)
    y_mean = y.mean(axis=0).astype(np.float32)
    y_std = (y.std(axis=0) + 1e-6).astype(np.float32)
    return {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}


def _lpf_1pole(x: np.ndarray, fc_hz: float, dt: float) -> np.ndarray:
    """
    1st-order IIR low-pass filter (causal):
      y[n] = a*y[n-1] + (1-a)*x[n]
    where a = exp(-2*pi*fc*dt).
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return x.copy()
    if not (fc_hz > 0.0) or not (dt > 0.0):
        return x.copy()
    a = float(np.exp(-2.0 * np.pi * float(fc_hz) * float(dt)))
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, x.size):
        y[i] = a * y[i - 1] + (1.0 - a) * x[i]
    return y


def _lpf_1pole_zero_phase(x: np.ndarray, fc_hz: float, dt: float) -> np.ndarray:
    """Forward-backward (zero-phase) variant of _lpf_1pole."""
    y = _lpf_1pole(x, fc_hz=fc_hz, dt=dt)
    y = _lpf_1pole(y[::-1], fc_hz=fc_hz, dt=dt)[::-1]
    return y


def _apply_by_stage(x: np.ndarray, stage_id: np.ndarray, fn) -> np.ndarray:
    """Apply a function within each contiguous stage segment only."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    stage_id = np.asarray(stage_id, dtype=np.int64).reshape(-1)
    if x.size == 0 or stage_id.size != x.size:
        return x.copy()
    out = x.copy()
    s = 0
    for i in range(1, x.size + 1):
        if i == x.size or stage_id[i] != stage_id[s]:
            out[s:i] = fn(x[s:i])
            s = i
    return out


def prepare_torque_delta_dataset(
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
    Torque-delta prediction dataset (inverse-dynamics flavored, but autoregressive on measured torque):

      input: history of [sin(q), cos(q), qd, (temp?), tau_out] up to time (k-1)
      target: delta_tau_out[k] = tau_out[k] - tau_out[k-1]

    Important: we must NOT include tau_out[k] in the input window (otherwise the task leaks labels).

    This matches the use-case: \"on top of last actual torque, predict the increment for current torque\":
      tau_out_pred[k] = tau_out[k-1] + delta_tau_out_pred[k]
    """
    ds = dict(np.load(raw_npz, allow_pickle=True))
    q = np.asarray(ds.get("q_out", ds.get("q", [])), dtype=np.float64).reshape(-1)
    qd = np.asarray(ds.get("qd_out", ds.get("qd", [])), dtype=np.float64).reshape(-1)
    tau_out = np.asarray(ds.get("tau_out", []), dtype=np.float64).reshape(-1)
    temp = np.asarray(ds.get("temp", []), dtype=np.float64).reshape(-1)
    stage_id = np.asarray(ds.get("stage_id", np.zeros_like(q, dtype=np.int64)), dtype=np.int64).reshape(-1)
    t = np.asarray(ds.get("t", ds.get("time", [])), dtype=np.float64).reshape(-1)

    if q.size == 0 or qd.size == 0:
        raise KeyError("raw log missing q_out/qd_out (or q/qd) for torque-delta dataset")
    if tau_out.size == 0:
        raise KeyError("raw log missing tau_out for torque-delta dataset")
    if not (q.shape == qd.shape == tau_out.shape):
        raise ValueError(f"q/qd/tau_out shape mismatch: q={q.shape} qd={qd.shape} tau_out={tau_out.shape}")
    if stage_id.size != q.size:
        stage_id = np.zeros_like(q, dtype=np.int64)

    H = int(get(cfg, "model.history_len"))
    T = int(q.shape[0])
    if T < H + 1:
        raise ValueError("trajectory too short for history_len")

    # Optional smoothing (training-time only). Prefer zero-phase offline to remove noise,
    # but NEVER use it online. Always apply within each stage to avoid boundary artifacts.
    if t.size >= 3:
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

    feat_state = state_to_features(q, qd).astype(np.float32)  # [T, 3]
    tau_col = tau_out.astype(np.float32).reshape(-1, 1)
    if temp.size == T:
        temp_col = temp.astype(np.float32).reshape(-1, 1)
        feat_full = np.concatenate([feat_state, temp_col, tau_col], axis=-1)  # [T, 5]
    else:
        feat_full = np.concatenate([feat_state, tau_col], axis=-1)  # [T, 4]

    # delta at k uses k-1, so define delta[0]=0 for convenience.
    delta_tau = np.zeros((T, 1), dtype=np.float32)
    delta_tau[1:, 0] = (tau_out[1:] - tau_out[:-1]).astype(np.float32)

    xs = []
    ys = []
    # Predict delta_tau at time k, using history ending at k-1 (length H).
    # Window indices: [k-H, ..., k-1] (inclusive) -> length H
    for k in range(H, T):
        # Do not stitch across stage boundaries (e.g., return-to-zero / step changes).
        if stage_id[k - H] != stage_id[k]:
            continue
        if not np.all(stage_id[k - H : k + 1] == stage_id[k]):
            continue
        x_win = feat_full[k - H : k]  # [H, Din], last row corresponds to (k-1)
        y_k = delta_tau[k]  # [1], delta from (k-1)->k
        xs.append(x_win)
        ys.append(y_k)

    x = np.stack(xs, axis=0).astype(np.float32)  # [N, H, Din]
    y = np.stack(ys, axis=0).astype(np.float32)  # [N, 1]

    # Time-series split (avoid random shuffling leakage): use the last val_ratio as validation.
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

    # normalize
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
    )


def train_torque_delta_model(
    cfg: Dict[str, Any],
    dataset_npz: str,
    out_weights: str,
    batch_size: int | None = None,
    epochs: int | None = None,
    lr: float | None = None,
) -> None:
    ds = dict(np.load(dataset_npz, allow_pickle=True))
    x = ds["x"].astype(np.float32)
    y = ds["y"].astype(np.float32)

    device = _resolve_device(cfg)
    model = CausalTransformer(
        input_dim=int(x.shape[-1]),
        output_dim=int(y.shape[-1]),
        embed_dim=int(get(cfg, "model.embed_dim")),
        num_layers=int(get(cfg, "model.num_layers")),
        num_heads=int(get(cfg, "model.num_heads")),
        history_len=int(get(cfg, "model.history_len")),
    ).to(device)

    bs = int(batch_size or get(cfg, "train.batch_size"))
    ep = int(epochs or get(cfg, "train.epochs"))
    learning_rate = float(lr or get(cfg, "train.lr"))

    loader = DataLoader(TorqueDeltaDataset(x, y), batch_size=bs, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    for e in range(ep):
        model.train()
        total = 0.0
        n = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * int(xb.shape[0])
            n += int(xb.shape[0])
        if (e + 1) % max(1, ep // 5) == 0 or e == 0:
            print(f"[torque_delta] epoch {e+1}/{ep} mse={total/max(1,n):.6f}")

    # Persist model + arch params for later eval without config coupling.
    torch.save(
        {
            "model": model.state_dict(),
            "input_dim": x.shape[-1],
            "output_dim": y.shape[-1],
            "embed_dim": int(get(cfg, "model.embed_dim")),
            "num_layers": int(get(cfg, "model.num_layers")),
            "num_heads": int(get(cfg, "model.num_heads")),
            "history_len": int(get(cfg, "model.history_len")),
        },
        out_weights,
    )
