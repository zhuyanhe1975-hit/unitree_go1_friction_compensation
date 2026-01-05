from __future__ import annotations

import argparse

import numpy as np

from pipeline.config import load_cfg
from project_config import get


def _stats(x: np.ndarray) -> str:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return f"shape={x.shape} min={x.min():+.6f} max={x.max():+.6f} mean={x.mean():+.6f} std={x.std():+.6f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--mode", choices=["sim", "real"], required=True)
    ap.add_argument("--preview", action="store_true", help="regenerate real_data_preview.png from the saved log (real only)")
    ap.add_argument("--preview_kd", type=float, default=None, help="override kd for tau_out_eff=tau_out_raw+kd*qd_m (motor-side)")
    ap.add_argument("--preview_out", default=None, help="output path for preview image (default: runs/real_data_preview_motor.png)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    path = str(get(cfg, "paths.sim_raw_log" if args.mode == "sim" else "paths.real_log"))
    ds = dict(np.load(path, allow_pickle=True))
    print(f"log: {path}")
    print("keys:", sorted(ds.keys()))

    t = np.asarray(ds.get("t", []), dtype=np.float64).reshape(-1)
    if len(t) >= 2:
        dt = float(np.median(np.diff(t)))
        print(f"dt≈{dt:.9f}s, T={len(t)}")

    for k in ["q_out", "qd_out", "tau_cmd", "tau_out", "q_m", "qd_m"]:
        if k in ds:
            print(f"{k}: {_stats(ds[k])}")

    if "q_out" in ds and "qd_out" in ds and len(t) >= 2:
        q = np.asarray(ds["q_out"], dtype=np.float64).reshape(-1)
        qd = np.asarray(ds["qd_out"], dtype=np.float64).reshape(-1)
        qd_from_q = np.zeros_like(q)
        qd_from_q[1:] = (q[1:] - q[:-1]) / dt
        err = qd - qd_from_q
        corr = float(np.corrcoef(qd[10:], qd_from_q[10:])[0, 1]) if len(q) > 20 else float("nan")
        print(f"qd consistency: rms(err)={float(np.sqrt(np.mean(err**2))):.6f}, corr={corr:.6f}")

    if args.mode == "real" and args.preview:
        try:
            import matplotlib.pyplot as plt
            import os

            t = np.asarray(ds.get("t", []), dtype=np.float64).reshape(-1)
            if len(t) == 0:
                raise ValueError("missing 't' in log")

            # Motor-side view: prefer q_m/qd_m if present, else fall back.
            q_m = np.asarray(ds.get("q_m", ds.get("q_out", [])), dtype=np.float64).reshape(-1)
            qd_m = np.asarray(ds.get("qd_m", ds.get("qd_out", [])), dtype=np.float64).reshape(-1)
            tau_cmd = np.asarray(ds.get("tau_cmd", []), dtype=np.float64).reshape(-1)
            tau_out_raw = np.asarray(ds.get("tau_out_raw", ds.get("tau_out", [])), dtype=np.float64).reshape(-1)

            if not (len(q_m) == len(qd_m) == len(tau_cmd) == len(tau_out_raw) == len(t)):
                raise ValueError("preview: length mismatch among t/q_m/qd_m/tau_cmd/tau_out_raw")

            kd = args.preview_kd
            if kd is None:
                meta = np.asarray(ds.get("meta", []), dtype=np.float64).reshape(-1)
                kd = float(meta[-1]) if len(meta) >= 12 else float(get(cfg, "real.kd", required=False, default=0.0))

            tau_eff = tau_out_raw + float(kd) * qd_m

            plt.figure(figsize=(10, 6))
            plt.subplot(3, 1, 1)
            plt.plot(t, q_m, label="q_m (motor)")
            plt.legend()
            plt.grid(True)
            plt.subplot(3, 1, 2)
            plt.plot(t, qd_m, label="qd_m (motor)")
            plt.legend()
            plt.grid(True)
            plt.subplot(3, 1, 3)
            plt.plot(t, tau_cmd, label="tau_cmd", color="r")
            plt.plot(t, tau_out_raw, label="tau_out_raw (data.tau)", color="k", alpha=0.35)
            plt.plot(t, tau_eff, label=f"tau_out_eff=tau_out_raw+kd*qd_m (kd={kd:g})", color="k", alpha=0.9)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            out = args.preview_out
            if out is None:
                out = os.path.join(str(get(cfg, "paths.runs_dir")), "real_data_preview_motor.png")
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            plt.savefig(out)
            print(f"saved: {out}")

            # Best-effort also update the legacy top-level path if writable.
            legacy = "real_data_preview.png"
            try:
                plt.savefig(legacy)
                print(f"saved: {legacy}")
            except Exception as e:
                print(f"[warn] cannot overwrite {legacy}: {e}")
        except Exception as e:
            print(f"[warn] preview failed: {e}")


if __name__ == "__main__":
    main()
