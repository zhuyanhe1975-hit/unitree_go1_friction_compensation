from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from project_config import get
from .model import CausalTransformer


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def _resolve_device(cfg: Dict[str, Any]) -> str:
    dev = str(get(cfg, "sim.device"))
    if dev.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] sim.device is cuda but CUDA not available; falling back to cpu")
        return "cpu"
    return dev


def train_model(cfg: Dict[str, Any], dataset_npz: str, out_weights: str) -> None:
    ds = dict(np.load(dataset_npz, allow_pickle=True))
    x = ds["x"].astype(np.float32)
    y = ds["y"].astype(np.float32)

    # Split
    n = x.shape[0]
    val_ratio = float(get(cfg, "data.prepare.val_ratio"))
    n_val = max(1, int(n * val_ratio))
    idx = np.arange(n)
    np.random.default_rng(int(get(cfg, "project.seed"))).shuffle(idx)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    x_tr, y_tr = x[tr_idx], y[tr_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    device = _resolve_device(cfg)
    model = CausalTransformer(
        input_dim=int(x.shape[-1]),
        output_dim=int(y.shape[-1]),
        embed_dim=int(get(cfg, "model.embed_dim")),
        num_layers=int(get(cfg, "model.num_layers")),
        num_heads=int(get(cfg, "model.num_heads")),
        history_len=int(get(cfg, "model.history_len")),
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=float(get(cfg, "train.lr")))
    loss_fn = nn.MSELoss()

    bs = int(get(cfg, "train.batch_size"))
    tr_loader = DataLoader(WindowDataset(x_tr, y_tr), batch_size=bs, shuffle=True)
    val_loader = DataLoader(WindowDataset(x_val, y_val), batch_size=bs, shuffle=False)

    epochs = int(get(cfg, "train.epochs"))
    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        tr_n = 0
        for xb, yb in tr_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item()) * int(xb.shape[0])
            tr_n += int(xb.shape[0])

        model.eval()
        va_loss = 0.0
        va_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                va_loss += float(loss.item()) * int(xb.shape[0])
                va_n += int(xb.shape[0])

        if (ep + 1) % max(1, epochs // 10) == 0 or ep == 0:
            print(f"epoch {ep+1}/{epochs} train_mse={tr_loss/max(1,tr_n):.6f} val_mse={va_loss/max(1,va_n):.6f}")

    torch.save({"model": model.state_dict(), "input_dim": x.shape[-1], "output_dim": y.shape[-1]}, out_weights)


def finetune_model(cfg: Dict[str, Any], dataset_npz: str, base_weights: str, out_weights: str) -> None:
    ds = dict(np.load(dataset_npz, allow_pickle=True))
    x = ds["x"].astype(np.float32)
    y = ds["y"].astype(np.float32)

    device = _resolve_device(cfg)
    ckpt = torch.load(base_weights, map_location=torch.device(device))
    model = CausalTransformer(
        input_dim=int(ckpt["input_dim"]),
        output_dim=int(ckpt["output_dim"]),
        embed_dim=int(get(cfg, "model.embed_dim")),
        num_layers=int(get(cfg, "model.num_layers")),
        num_heads=int(get(cfg, "model.num_heads")),
        history_len=int(get(cfg, "model.history_len")),
    ).to(device)
    model.load_state_dict(ckpt["model"])

    opt = optim.Adam(model.parameters(), lr=float(get(cfg, "train.finetune_lr")))
    loss_fn = nn.MSELoss()
    bs = int(get(cfg, "train.batch_size"))
    loader = DataLoader(WindowDataset(x, y), batch_size=bs, shuffle=True)

    epochs = int(get(cfg, "train.finetune_epochs"))
    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        tr_n = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item()) * int(xb.shape[0])
            tr_n += int(xb.shape[0])
        if (ep + 1) % max(1, epochs // 5) == 0 or ep == 0:
            print(f"finetune epoch {ep+1}/{epochs} mse={tr_loss/max(1,tr_n):.6f}")

    torch.save({"model": model.state_dict(), "input_dim": ckpt["input_dim"], "output_dim": ckpt["output_dim"]}, out_weights)
