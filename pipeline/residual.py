from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from project_config import get, ensure_dir
from .model import CausalTransformer
from .train import _resolve_device


class _ResidualDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def _load_base_model(cfg: Dict[str, Any], path: str, device: str) -> CausalTransformer:
    ckpt = torch.load(path, map_location=torch.device(device))
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
    return model


@torch.no_grad()
def prepare_residual_dataset(
    cfg: Dict[str, Any],
    dataset_npz: str,
    base_model_path: str,
    out_npz: str,
    batch_size: int = 8192,
) -> None:
    """
    Build residual targets: y_res = y_true - y_base_pred (all in normalized units).
    Input x stays the same as the prepared dataset.
    """
    ds = dict(np.load(dataset_npz, allow_pickle=True))
    x = ds["x"].astype(np.float32)
    y = ds["y"].astype(np.float32)

    device = _resolve_device(cfg)
    base_model = _load_base_model(cfg, base_model_path, device=device)

    loader = DataLoader(_ResidualDataset(x, y), batch_size=batch_size, shuffle=False)
    preds = []
    for xb, _ in loader:
        xb = xb.to(device)
        pred = base_model(xb).cpu().numpy().astype(np.float32)
        preds.append(pred)
    y_base = np.concatenate(preds, axis=0)
    y_res = y - y_base

    ensure_dir(os.path.dirname(out_npz) or ".")
    payload = {"x": x, "y": y_res}
    # Carry over stats for convenience.
    for k in ["s_mean", "s_std", "a_mean", "a_std", "d_mean", "d_std"]:
        if k in ds:
            payload[k] = ds[k]
    payload["base_model_path"] = np.array([base_model_path], dtype=object)
    payload["parent_dataset"] = np.array([dataset_npz], dtype=object)
    np.savez(out_npz, **payload)


def train_residual_model(
    cfg: Dict[str, Any],
    dataset_npz: str,
    base_model_path: str,
    out_weights: str,
    batch_size: int | None = None,
    epochs: int | None = None,
    lr: float | None = None,
) -> None:
    ds = dict(np.load(dataset_npz, allow_pickle=True))
    x = ds["x"].astype(np.float32)
    y = ds["y"].astype(np.float32)

    device = _resolve_device(cfg)
    base_ckpt = torch.load(base_model_path, map_location=torch.device(device))
    model = CausalTransformer(
        input_dim=int(base_ckpt["input_dim"]),
        output_dim=int(base_ckpt["output_dim"]),
        embed_dim=int(get(cfg, "model.embed_dim")),
        num_layers=int(get(cfg, "model.num_layers")),
        num_heads=int(get(cfg, "model.num_heads")),
        history_len=int(get(cfg, "model.history_len")),
    ).to(device)
    model.load_state_dict(base_ckpt["model"])

    bs = int(batch_size or get(cfg, "train.batch_size"))
    ep = int(epochs or get(cfg, "train.epochs"))
    learning_rate = float(lr or get(cfg, "train.lr"))

    loader = DataLoader(_ResidualDataset(x, y), batch_size=bs, shuffle=True)
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
            print(f"[residual] epoch {e+1}/{ep} mse={total/max(1,n):.6f}")

    torch.save(
        {
            "model": model.state_dict(),
            "input_dim": int(base_ckpt["input_dim"]),
            "output_dim": int(base_ckpt["output_dim"]),
            "base_model_path": base_model_path,
        },
        out_weights,
    )
