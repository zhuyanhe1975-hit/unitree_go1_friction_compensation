from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Tuple

import numpy as np


def _one_pole_lpf(x: np.ndarray, dt: float, cutoff_hz: float) -> np.ndarray:
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


def _smooth_sign(x: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / (np.abs(x) + float(eps))


def _diff(x: np.ndarray, dt: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.zeros_like(x)
    if len(x) >= 2:
        y[1:] = (x[1:] - x[:-1]) / float(dt)
    return y


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config root must be an object")
    return cfg


def _save_config(path: str, cfg: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _get(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _fit_u_as_tau(
    t: np.ndarray,
    q: np.ndarray,
    u: np.ndarray,
    qd_source: str,
    qd_lpf_hz: float,
    qdd_lpf_hz: float,
    min_speed: float,
    trim_frac: float,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    if not (len(t) == len(q) == len(u)):
        raise ValueError("t, q, u must have same length")
    if len(t) < 10:
        raise ValueError("log too short")

    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        raise ValueError("invalid dt")

    if qd_source == "from_q":
        qd = _diff(q, dt=dt)
    else:
        raise ValueError("qd_source must be 'from_q' for this script")

    if qd_lpf_hz > 0:
        qd = _one_pole_lpf(qd, dt=dt, cutoff_hz=qd_lpf_hz)
    qdd = _diff(qd, dt=dt)
    if qdd_lpf_hz > 0:
        qdd = _one_pole_lpf(qdd, dt=dt, cutoff_hz=qdd_lpf_hz)

    # Drop a few samples to avoid filter transient + diff edge.
    pad = max(5, int(round(0.1 / dt)))
    sl = slice(pad, len(t))
    t2, q2, qd2, qdd2, u2 = t[sl], q[sl], qd[sl], qdd[sl], u[sl]

    # Use samples away from stiction (identified from speed).
    mask = np.abs(qd2) >= float(min_speed)
    if int(mask.sum()) < 200:
        raise ValueError("not enough samples after speed filtering; reduce --min_speed or collect longer log")

    qd_m = qd2[mask]
    qdd_m = qdd2[mask]
    u_m = u2[mask]
    sgn = _smooth_sign(qd_m)

    # Model (output-side):
    #   u ≈ J*qdd + b*qd + tau_c*sgn + c0
    X = np.stack([qdd_m, qd_m, sgn, np.ones_like(qd_m)], axis=1)  # [N, 4]

    # Robust trimming: drop largest residuals from an initial fit.
    w0, *_ = np.linalg.lstsq(X, u_m, rcond=None)
    resid0 = u_m - (X @ w0)
    if 0.0 < trim_frac < 0.49:
        thr = np.quantile(np.abs(resid0), 1.0 - float(trim_frac))
        keep = np.abs(resid0) <= thr
        X2 = X[keep]
        u2m = u_m[keep]
    else:
        X2, u2m = X, u_m

    w, *_ = np.linalg.lstsq(X2, u2m, rcond=None)
    u_hat = X @ w
    resid = u_m - u_hat
    r2 = 1.0 - float(np.sum(resid**2) / np.sum((u_m - u_m.mean()) ** 2))

    out = {
        "dt": dt,
        "J": float(w[0]),
        "b": float(w[1]),
        "tau_coulomb": float(w[2]),
        "tau_bias": float(w[3]),
        "fit_r2": float(r2),
        "used_samples": int(mask.sum()),
        "total_samples": int(len(t)),
    }
    series = {"t": t2, "q": q2, "qd": qd2, "qdd": qdd2, "u": u2, "u_hat": u_hat}
    return out, series


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--log", default=None, help="override log npz (default: config.paths.real_log)")
    ap.add_argument("--apply", action="store_true", help="write estimated params into config.json sim.mjcf_override")
    ap.add_argument("--q_key", default=None, help="position key to use (default: q_m if present else q_out)")
    ap.add_argument(
        "--u_key",
        default=None,
        help="torque key to use: tau_cmd | tau_out | tau_out_raw | tau_out_eff (default: tau_out_eff if possible else tau_cmd)",
    )
    ap.add_argument("--kd", type=float, default=None, help="override kd used in tau_out_eff=tau_out_raw+kd*qd_m (default: from config.real.kd)")

    ap.add_argument("--qd_lpf_hz", type=float, default=30.0)
    ap.add_argument("--qdd_lpf_hz", type=float, default=30.0)
    ap.add_argument("--min_speed", type=float, default=0.3)
    ap.add_argument("--trim_frac", type=float, default=0.02, help="drop this fraction of largest residuals")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    log_path = args.log or str(_get(cfg, "paths.real_log"))
    if not log_path or not os.path.exists(log_path):
        raise SystemExit(f"missing real log: {log_path}")

    ds = dict(np.load(log_path, allow_pickle=True))
    if "t" not in ds:
        raise SystemExit(f"missing 't' in {log_path}")
    t = np.asarray(ds["t"], dtype=np.float64).reshape(-1)

    q_key = args.q_key
    if q_key is None:
        q_key = "q_m" if "q_m" in ds else "q_out"
    if q_key not in ds:
        raise SystemExit(f"missing '{q_key}' in {log_path}")
    q = np.asarray(ds[q_key], dtype=np.float64)
    if q.ndim == 2 and q.shape[1] == 1:
        q = q[:, 0]

    u_key = args.u_key
    if u_key is None:
        if ("qd_m" in ds) and (("tau_out_raw" in ds) or ("tau_out" in ds)):
            u_key = "tau_out_eff"
        else:
            u_key = "tau_cmd"
    if u_key == "tau_out_eff":
        if "qd_m" not in ds:
            raise SystemExit(f"tau_out_eff requires 'qd_m' in {log_path}")
        qd_m = np.asarray(ds["qd_m"], dtype=np.float64).reshape(-1)
        tau_raw_key = "tau_out_raw" if "tau_out_raw" in ds else "tau_out"
        if tau_raw_key not in ds:
            raise SystemExit(f"tau_out_eff requires '{tau_raw_key}' in {log_path}")
        tau_raw = np.asarray(ds[tau_raw_key], dtype=np.float64).reshape(-1)
        kd = float(args.kd) if args.kd is not None else float(_get(cfg, "real.kd", default=0.0))
        u = tau_raw + kd * qd_m
    else:
        if u_key not in ds:
            raise SystemExit(f"missing '{u_key}' in {log_path}")
        u = np.asarray(ds[u_key], dtype=np.float64)
        if u.ndim == 2 and u.shape[1] == 1:
            u = u[:, 0]

    params, _ = _fit_u_as_tau(
        t=t,
        q=q,
        u=u,
        qd_source="from_q",
        qd_lpf_hz=float(args.qd_lpf_hz),
        qdd_lpf_hz=float(args.qdd_lpf_hz),
        min_speed=float(args.min_speed),
        trim_frac=float(args.trim_frac),
    )

    print(f"=== Identified joint parameters (q='{q_key}', u='{u_key}') ===")
    print(f"dt={params['dt']:.6f}s, used_samples={params['used_samples']}/{params['total_samples']}, fit_r2={params['fit_r2']:.4f}")
    print(f"J_eq ≈ {params['J']:.6e} kg*m^2")
    print(f"b_viscous ≈ {params['b']:.6e} N*m/(rad/s)")
    print(f"tau_coulomb ≈ {params['tau_coulomb']:.6e} N*m")
    print(f"tau_bias ≈ {params['tau_bias']:.6e} N*m")

    if args.apply:
        mj = cfg.setdefault("sim", {}).setdefault("mjcf_override", {})
        mj["armature"] = float(params["J"])
        mj["damping"] = float(max(0.0, params["b"]))
        # MuJoCo frictionloss is unsigned.
        mj["frictionloss"] = float(abs(params["tau_coulomb"]))
        _save_config(args.config, cfg)
        print(f"applied to {args.config}: sim.mjcf_override.armature/damping/frictionloss")


if __name__ == "__main__":
    main()
