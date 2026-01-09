from __future__ import annotations

import argparse
import os

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

import torch

from pipeline.config import load_cfg
from pipeline.model import CausalTransformer
from pipeline.train import _resolve_device
from project_config import ensure_dir, get
from pipeline.features import state_to_features


def _load_model(cfg, weights_path: str, device: str) -> CausalTransformer:
    ckpt = torch.load(weights_path, map_location=torch.device(device))
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


def _stats_from_prepared(ds: dict) -> dict:
    return {
        "s_mean": ds["s_mean"].astype(np.float32),
        "s_std": ds["s_std"].astype(np.float32),
        "a_mean": ds["a_mean"].astype(np.float32),
        "a_std": ds["a_std"].astype(np.float32),
        "d_mean": ds["d_mean"].astype(np.float32),
        "d_std": ds["d_std"].astype(np.float32),
    }


def _real_action_from_log(cfg: dict, lg: dict) -> np.ndarray:
    action_key = str(get(cfg, "data.real.action_key", required=False, default="tau_cmd"))
    if action_key not in lg:
        raise KeyError(f"real_log missing action_key='{action_key}' (available keys: {sorted(lg.keys())})")
    tau = np.asarray(lg[action_key], dtype=np.float64).reshape(-1)
    if action_key == "tau_out" and bool(get(cfg, "data.real.tau_out_scale_to_out", required=False, default=False)):
        N = float(get(cfg, "motor.gear_ratio"))
        eta = float(get(cfg, "real.efficiency", required=False, default=1.0))
        tau = tau * (N * eta)
    return tau


@torch.no_grad()
def _mse(model: CausalTransformer, x: np.ndarray, y: np.ndarray, device: str) -> float:
    xb = torch.from_numpy(x).float().to(device)
    yb = torch.from_numpy(y).float().to(device)
    pred = model(xb)
    return float(torch.mean((pred - yb) ** 2).item())

@torch.no_grad()
def _mse_physical(model: CausalTransformer, x: np.ndarray, y_norm: np.ndarray, d_mean: np.ndarray, d_std: np.ndarray, device: str) -> float:
    xb = torch.from_numpy(x).float().to(device)
    pred_norm = model(xb).cpu().numpy()
    y_pred = pred_norm * d_std + d_mean
    y_true = y_norm * d_std + d_mean
    return float(np.mean((y_pred - y_true) ** 2))


@torch.no_grad()
def _rollout_open_loop(
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

    q_hist = [float(q0)] * history_len
    qd_hist = [float(qd0)] * history_len
    a_hist = [float(tau_cmd[0])] * history_len

    q_pred = np.zeros((len(tau_cmd),), dtype=np.float64)
    qd_pred = np.zeros((len(tau_cmd),), dtype=np.float64)
    q_pred[0] = float(q0)
    qd_pred[0] = float(qd0)

    for k in range(1, len(tau_cmd)):
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

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", choices=["sim", "real", "real_scratch", "all"], default="all")
    ap.add_argument(
        "--residual_model",
        default=None,
        help="optional residual model path (default: <model_path> with _residual suffix if exists)",
    )
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    ensure_dir(get(cfg, "paths.runs_dir"))
    device = _resolve_device(cfg)

    def eval_one(model_path: str, dataset_path: str, label: str) -> tuple[float, float]:
        ds = dict(np.load(dataset_path, allow_pickle=True))
        x = ds["x"].astype(np.float32)
        y = ds["y"].astype(np.float32)
        if "d_mean" in ds and "d_std" in ds:
            d_mean = ds["d_mean"].astype(np.float32)
            d_std = ds["d_std"].astype(np.float32)
        else:
            st = dict(np.load(get(cfg, "paths.stats_npz")))
            d_mean = st["d_mean"].astype(np.float32)
            d_std = st["d_std"].astype(np.float32)
        model = _load_model(cfg, model_path, device=device)
        mse_norm = _mse(model, x, y, device=device)
        mse_phys = _mse_physical(model, x, y, d_mean, d_std, device=device)
        print(f"{label} mse_norm={mse_norm:.6f} mse_phys={mse_phys:.6f}")
        return mse_norm, mse_phys

    sim_dataset_path = str(get(cfg, "paths.sim_dataset"))
    sim_raw_log_path = str(get(cfg, "paths.sim_raw_log", required=False, default=""))
    real_dataset_path = str(get(cfg, "paths.real_dataset"))
    sim_model_path = str(get(cfg, "paths.sim_model"))
    real_model_path = str(get(cfg, "paths.real_model"))
    real_scratch_path = str(get(cfg, "paths.real_model_scratch"))
    residual_cli = args.residual_model

    if args.model in ("sim", "all"):
        eval_one(sim_model_path, sim_dataset_path, "sim_model on sim_dataset")

    real_ds_path = str(get(cfg, "paths.real_dataset"))
    real_model_path = str(get(cfg, "paths.real_model"))
    if os.path.exists(real_ds_path):
        if args.model in ("real", "all") and os.path.exists(real_model_path):
            eval_one(real_model_path, real_dataset_path, "real_model (finetune) on real_dataset")
        if args.model in ("real_scratch", "all") and os.path.exists(real_scratch_path):
            eval_one(real_scratch_path, real_dataset_path, "real_model_scratch on real_dataset")
        if args.model == "all":
            # Useful diagnostic: how far the sim model is from real distribution.
            if os.path.exists(sim_model_path):
                eval_one(sim_model_path, real_dataset_path, "sim_model on real_dataset")

    if plt is None:
        return

    def _resolve_residual_path(base_path: str) -> str | None:
        if residual_cli:
            return residual_cli
        if not base_path or "." not in base_path:
            return None
        stem, ext = base_path.rsplit(".", 1)
        cand = f"{stem}_residual.{ext}"
        return cand if os.path.exists(cand) else None

    # Plot a quick one-step prediction trace for sim (delta_q)
    plot_steps = int(get(cfg, "eval.plot_steps", required=False, default=2000))
    if args.model in ("sim", "all") and os.path.exists(sim_dataset_path) and os.path.exists(sim_model_path):
        sim_ds = dict(np.load(sim_dataset_path, allow_pickle=True))
        x_sim = sim_ds["x"].astype(np.float32)
        y_sim = sim_ds["y"].astype(np.float32)
        n_plot = min(1000, x_sim.shape[0])
        model_sim = _load_model(cfg, sim_model_path, device=device)
        with torch.no_grad():
            pred = model_sim(torch.from_numpy(x_sim[:n_plot]).float().to(device)).cpu().numpy()
        plt.figure(figsize=(10, 4))
        plt.plot(y_sim[:n_plot, 0], label="delta_q (gt)", alpha=0.7)
        plt.plot(pred[:n_plot, 0], label="delta_q (pred)", alpha=0.7)
        plt.grid(True, alpha=0.3)
        plt.legend()
        out = os.path.join(get(cfg, "paths.runs_dir"), "eval_sim_delta_q.png")
        ensure_dir(os.path.dirname(out) or ".")
        plt.tight_layout()
        plt.savefig(out)
        print(f"saved: {out}")

        # Plot open-loop rollout for sim log (absolute q/qd)
        if os.path.exists(sim_raw_log_path):
            lg_sim = dict(np.load(sim_raw_log_path, allow_pickle=True))
            t_sim = np.asarray(lg_sim.get("t", []), dtype=np.float64)
            q_sim = np.asarray(lg_sim.get("q_out", []), dtype=np.float64)
            qd_sim = np.asarray(lg_sim.get("qd_out", []), dtype=np.float64)
            tau_sim = np.asarray(lg_sim.get("tau_cmd", []), dtype=np.float64)

            # If multi-trajectory, take the first trajectory for plotting.
            if q_sim.ndim > 1:
                q_sim = q_sim[0]
            if qd_sim.ndim > 1:
                qd_sim = qd_sim[0]
            if tau_sim.ndim > 1:
                tau_sim = tau_sim[0]
            q_sim = q_sim.reshape(-1)
            qd_sim = qd_sim.reshape(-1)
            tau_sim = tau_sim.reshape(-1)
            t_sim = t_sim.reshape(-1)

            if len(q_sim) > 0 and len(tau_sim) > 0:
                if len(t_sim) >= 2:
                    dt_sim = float(np.median(np.diff(t_sim)))
                else:
                    dt_sim = float(get(cfg, "sim.frame_dt"))
                    t_sim = np.arange(len(q_sim), dtype=np.float64) * dt_sim

                n = min(plot_steps, len(q_sim), len(tau_sim))
                stats_sim = _stats_from_prepared(sim_ds)
                q_pred_sim, qd_pred_sim = _rollout_open_loop(
                    model_sim,
                    stats=stats_sim,
                    q0=float(q_sim[0]),
                    qd0=float(qd_sim[0]) if len(qd_sim) > 0 else 0.0,
                    tau_cmd=tau_sim[:n],
                    history_len=int(get(cfg, "model.history_len")),
                    device=device,
                )

                tt = t_sim[:n]
                plt.figure(figsize=(12, 7))
                plt.subplot(2, 1, 1)
                plt.plot(tt, q_sim[:n], label="q_out (gt)", color="k", alpha=0.6)
                plt.plot(tt, q_pred_sim, label="q_out (pred rollout)", color="r", lw=1.2)
                plt.grid(True, alpha=0.3)
                plt.legend()

                if len(qd_sim) > 0:
                    plt.subplot(2, 1, 2)
                    plt.plot(tt, qd_sim[:n], label="qd_out (gt)", color="k", alpha=0.6)
                    plt.plot(tt, qd_pred_sim, label="qd_out (pred rollout)", color="r", lw=1.2)
                    plt.grid(True, alpha=0.3)
                    plt.legend()

                out_sim_roll = os.path.join(get(cfg, "paths.runs_dir"), "eval_sim_rollout.png")
                ensure_dir(os.path.dirname(out_sim_roll) or ".")
                plt.tight_layout()
                plt.savefig(out_sim_roll)
                print(f"saved: {out_sim_roll}")

    def _plot_real_variant(model_path: str, dataset_path: str, out_prefix: str) -> None:
        if not os.path.exists(model_path):
            print(f"[skip] model not found for {out_prefix}: {model_path}")
            return
        if not os.path.exists(dataset_path):
            print(f"[skip] dataset not found for {out_prefix}: {dataset_path}")
            return
        residual_path = _resolve_residual_path(model_path)
        if residual_path and not os.path.exists(residual_path):
            residual_path = None

        real_ds = dict(np.load(dataset_path, allow_pickle=True))
        x_real = real_ds["x"].astype(np.float32)
        y_real_norm = real_ds["y"].astype(np.float32)
        d_mean = real_ds["d_mean"].astype(np.float32)
        d_std = real_ds["d_std"].astype(np.float32)
        n_plot = min(plot_steps, x_real.shape[0])

        model_real = _load_model(cfg, model_path, device=device)
        model_res = _load_model(cfg, residual_path, device=device) if residual_path else None

        def _pred_total(xb: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                base = model_real(torch.from_numpy(xb).float().to(device))
                if model_res is not None:
                    res = model_res(torch.from_numpy(xb).float().to(device))
                    return (base + res).detach().cpu().numpy()
                return base.detach().cpu().numpy()

        with torch.no_grad():
            pred_norm = _pred_total(x_real[:n_plot])
        y_pred = pred_norm * d_std + d_mean
        y_true = y_real_norm[:n_plot] * d_std + d_mean

        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(y_true[:, 0], label="delta_q (gt)", alpha=0.7)
        plt.plot(y_pred[:, 0], label="delta_q (pred)", alpha=0.7)
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(y_true[:, 1], label="delta_qd (gt)", alpha=0.7)
        plt.plot(y_pred[:, 1], label="delta_qd (pred)", alpha=0.7)
        plt.grid(True, alpha=0.3)
        plt.legend()

        suffix = "_residual" if model_res is not None else ""
        out = os.path.join(get(cfg, "paths.runs_dir"), f"eval_{out_prefix}_delta{suffix}.png")
        ensure_dir(os.path.dirname(out) or ".")
        plt.tight_layout()
        plt.savefig(out)
        print(f"saved: {out}")

        # Also plot open-loop rollout on real_log for intuition (absolute q_out/qd_out).
        real_log_path = str(get(cfg, "paths.real_log"))
        if os.path.exists(real_log_path):
            lg = dict(np.load(real_log_path, allow_pickle=True))
            t_log = np.asarray(lg["t"], dtype=np.float64).reshape(-1)
            q_log = np.asarray(lg["q_out"], dtype=np.float64).reshape(-1)
            tau_log = _real_action_from_log(cfg, lg)

            if len(t_log) >= 2:
                dt_log = float(np.median(np.diff(t_log)))
            else:
                dt_log = float(get(cfg, "real.dt"))

            n = min(plot_steps, len(q_log))
            qd_log = np.zeros_like(q_log)
            if len(q_log) >= 2:
                qd_log[1:] = (q_log[1:] - q_log[:-1]) / dt_log

            stats = _stats_from_prepared(real_ds)

            # rollout with combined model
            def _rollout_combined(
                model_base: CausalTransformer, model_residual: CausalTransformer | None, stats: dict
            ) -> tuple[np.ndarray, np.ndarray]:
                q_hist = [float(q_log[0])] * int(get(cfg, "model.history_len"))
                qd_hist = [float(qd_log[0])] * int(get(cfg, "model.history_len"))
                a_hist = [float(tau_log[0])] * int(get(cfg, "model.history_len"))

                q_pred = np.zeros((n,), dtype=np.float64)
                qd_pred = np.zeros((n,), dtype=np.float64)
                q_pred[0] = float(q_log[0])
                qd_pred[0] = float(qd_log[0])

                s_mean = stats["s_mean"]
                s_std = stats["s_std"]
                a_mean = stats["a_mean"]
                a_std = stats["a_std"]
                d_mean = stats["d_mean"]
                d_std = stats["d_std"]

                for k in range(1, n):
                    a_hist = a_hist[1:] + [float(tau_log[k])]
                    feat = state_to_features(np.array(q_hist, dtype=np.float64), np.array(qd_hist, dtype=np.float64)).astype(
                        np.float32
                    )
                    act = np.array(a_hist, dtype=np.float32)[:, None]
                    feat_n = (feat - s_mean) / s_std
                    act_n = (act - a_mean) / a_std
                    x = np.concatenate([feat_n, act_n], axis=-1)[None, :, :]
                    xb = torch.from_numpy(x).float().to(device)
                    with torch.no_grad():
                        pred_norm = model_base(xb)
                        if model_residual is not None:
                            pred_norm = pred_norm + model_residual(xb)
                    delta = pred_norm.detach().cpu().numpy().reshape(-1) * d_std + d_mean
                    q_next = q_hist[-1] + float(delta[0])
                    qd_next = qd_hist[-1] + float(delta[1])
                    q_hist = q_hist[1:] + [q_next]
                    qd_hist = qd_hist[1:] + [qd_next]
                    q_pred[k] = q_next
                    qd_pred[k] = qd_next
                return q_pred, qd_pred

            q_pred_roll, qd_pred_roll = _rollout_combined(model_real, model_res, stats=stats)

            tt = t_log[:n]
            plt.figure(figsize=(12, 7))
            plt.subplot(2, 1, 1)
            plt.plot(tt, q_log[:n], label="q_out (gt)", color="k", alpha=0.6)
            plt.plot(tt, q_pred_roll, label="q_out (pred rollout)", color="r", lw=1.2)
            plt.grid(True, alpha=0.3)
            plt.legend()

            plt.subplot(2, 1, 2)
            plt.plot(tt, qd_log[:n], label="qd_out (gt, from diff)", color="k", alpha=0.6)
            plt.plot(tt, qd_pred_roll, label="qd_out (pred rollout)", color="r", lw=1.2)
            plt.grid(True, alpha=0.3)
            plt.legend()

            out_roll = os.path.join(get(cfg, "paths.runs_dir"), f"eval_{out_prefix}_rollout{suffix}.png")
            ensure_dir(os.path.dirname(out_roll) or ".")
            plt.tight_layout()
            plt.savefig(out_roll)
            print(f"saved: {out_roll}")

    if args.model in ("real", "all"):
        _plot_real_variant(real_model_path, real_dataset_path, out_prefix="real")
    if args.model in ("real_scratch", "all"):
        _plot_real_variant(real_scratch_path, real_dataset_path, out_prefix="real_scratch")


if __name__ == "__main__":
    main()
