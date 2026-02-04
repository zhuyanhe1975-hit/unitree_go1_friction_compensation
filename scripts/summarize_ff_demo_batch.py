from __future__ import annotations

import argparse
import glob
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Trial:
    report_path: str
    baseline_npz: str
    ff_npz: str
    meta: dict[str, Any]


def _rmse(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean(x**2))) if x.size else float("nan")


def _maxabs(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return float(np.max(np.abs(x))) if x.size else float("nan")


def _meanabs(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return float(np.mean(np.abs(x))) if x.size else float("nan")


def _q(x: np.ndarray, p: float) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return float(np.quantile(x, p)) if x.size else float("nan")


def _parse_report_md(path: str) -> Trial | None:
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")

    def m1(pat: str) -> str | None:
        m = re.search(pat, txt)
        return m.group(1).strip() if m else None

    baseline_log = m1(r"## baseline\s+[\s\S]*?- log:\s+([^\s]+)")
    ff_log = m1(r"## ff\s+[\s\S]*?- log:\s+([^\s]+)")
    if not baseline_log or not ff_log:
        return None

    meta: dict[str, Any] = {}
    # Best-effort parse common header lines
    top = re.search(r"amp\(rad\)=([0-9.]+), freq\(Hz\)=([0-9.]+), duration\(s\)=([0-9.]+), dt\(s\)=([0-9.]+)", txt)
    if top:
        meta["amp"] = float(top.group(1))
        meta["freq"] = float(top.group(2))
        meta["duration"] = float(top.group(3))
        meta["dt_cfg"] = float(top.group(4))

    gains = re.search(r"kp=([0-9.]+), kd=([0-9.]+), tau_ff_limit=([0-9.]+), tau_ff_slew=([0-9.]+)", txt)
    if gains:
        meta["kp"] = float(gains.group(1))
        meta["kd"] = float(gains.group(2))
        meta["tau_ff_limit"] = float(gains.group(3))
        meta["tau_ff_slew"] = float(gains.group(4))

    scale = m1(r"tau_ff_scale=([0-9.]+)")
    if scale is not None:
        meta["tau_ff_scale"] = float(scale)

    div = m1(r"ff_update_div=([0-9]+)")
    if div is not None:
        meta["ff_update_div"] = int(div)

    ff_type = m1(r"ff_type=([A-Za-z0-9_]+)")
    if ff_type is not None:
        meta["ff_type"] = ff_type

    return Trial(
        report_path=path,
        baseline_npz=os.path.join(os.path.dirname(path), baseline_log),
        ff_npz=os.path.join(os.path.dirname(path), ff_log),
        meta=meta,
    )


def _load_npz(path: str) -> dict[str, np.ndarray]:
    return {k: v for k, v in dict(np.load(path, allow_pickle=True)).items()}


def _zero_crossings(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size < 3:
        return np.array([], dtype=np.int64)
    zc = np.where(x[:-1] * x[1:] <= 0)[0] + 1
    zc = zc[(zc > 1) & (zc < len(x) - 2)]
    return zc.astype(np.int64)


def _pick_reversal_signal(ds: dict[str, np.ndarray], choice: str) -> tuple[str | None, np.ndarray]:
    """Pick a velocity-like signal for alignment (or seeds if qd_ref missing)."""
    if choice == "qd_from_q":
        return ("qd_from_q", np.asarray(ds.get("qd_from_q", []), dtype=np.float64).reshape(-1))
    if choice == "qd":
        return ("qd", np.asarray(ds.get("qd", []), dtype=np.float64).reshape(-1))
    if choice == "qd_ref":
        return ("qd_ref", np.asarray(ds.get("qd_ref", []), dtype=np.float64).reshape(-1))
    # auto: prefer qd_from_q -> qd -> qd_ref
    for k in ("qd_from_q", "qd", "qd_ref"):
        x = np.asarray(ds.get(k, []), dtype=np.float64).reshape(-1)
        if x.size:
            return (k, x)
    return (None, np.array([], dtype=np.float64))


def _find_reversals(
    ds: dict[str, np.ndarray],
    *,
    dt: float,
    reversal_signal: str,
    align_s: float,
) -> tuple[np.ndarray, dict[str, str]]:
    """
    Find reversal centers (velocity ~= 0 moments).

    Strategy:
    - Prefer using `qd_ref` as stable seeds when available.
    - Optionally align each seed to the nearest |qd_x| minimum in a local neighborhood,
      where qd_x is chosen by `reversal_signal` (auto: qd_from_q -> qd -> qd_ref).
    """
    meta: dict[str, str] = {}

    qd_ref = np.asarray(ds.get("qd_ref", []), dtype=np.float64).reshape(-1)
    seed = _zero_crossings(qd_ref) if qd_ref.size else np.array([], dtype=np.int64)
    if seed.size:
        meta["seed"] = "qd_ref"
    else:
        meta["seed"] = "fallback"

    k_align, qd_align = _pick_reversal_signal(ds, reversal_signal)
    if k_align is None or qd_align.size == 0:
        meta["align"] = "none"
        return (seed, meta)
    meta["align"] = k_align

    if seed.size == 0:
        # No qd_ref: use chosen signal directly for zero crossings.
        return (_zero_crossings(qd_align), meta)

    if align_s <= 0:
        return (seed, meta)

    rad = int(max(1, round(float(align_s) / max(1e-9, float(dt)))))
    centers: list[int] = []
    for c in seed:
        s = max(0, int(c) - rad)
        t = min(qd_align.size, int(c) + rad + 1)
        if t - s < 5:
            continue
        local = np.abs(qd_align[s:t])
        centers.append(int(s + int(np.argmin(local))))
    if not centers:
        return (seed, meta)
    return (np.asarray(centers, dtype=np.int64), meta)


def _window_metrics(ds: dict[str, np.ndarray], centers: np.ndarray, half_window: int) -> dict[str, float]:
    e = np.asarray(ds.get("e_q", []), dtype=np.float64).reshape(-1)
    tau_out = np.asarray(ds.get("tau_out", []), dtype=np.float64).reshape(-1)
    tau_ff = np.asarray(ds.get("tau_ff", []), dtype=np.float64).reshape(-1)

    if e.size == 0 or centers.size == 0:
        return {
            "rmse_rev": float("nan"),
            "maxabs_rev": float("nan"),
            "meanabs_tau_out_rev": float("nan"),
            "meanabs_tau_ff_rev": float("nan"),
            "n_rev": 0.0,
        }

    vals_e = []
    vals_tau_out = []
    vals_tau_ff = []
    maxes = []
    for c in centers:
        s = max(0, int(c) - half_window)
        t = min(len(e), int(c) + half_window + 1)
        if t - s < 5:
            continue
        w_e = e[s:t]
        vals_e.append(w_e)
        vals_tau_out.append(tau_out[s:t] if tau_out.size == e.size else np.array([], dtype=np.float64))
        vals_tau_ff.append(tau_ff[s:t] if tau_ff.size == e.size else np.array([], dtype=np.float64))
        maxes.append(_maxabs(w_e))

    if not vals_e:
        return {
            "rmse_rev": float("nan"),
            "maxabs_rev": float("nan"),
            "meanabs_tau_out_rev": float("nan"),
            "meanabs_tau_ff_rev": float("nan"),
            "n_rev": 0.0,
        }

    e_all = np.concatenate(vals_e)
    tau_out_all = np.concatenate([x for x in vals_tau_out if x.size])
    tau_ff_all = np.concatenate([x for x in vals_tau_ff if x.size])
    return {
        "rmse_rev": _rmse(e_all),
        "maxabs_rev": float(np.max(maxes)) if maxes else float("nan"),
        "meanabs_tau_out_rev": _meanabs(tau_out_all) if tau_out_all.size else float("nan"),
        "meanabs_tau_ff_rev": _meanabs(tau_ff_all) if tau_ff_all.size else float("nan"),
        "n_rev": float(len(maxes)),
    }


def _trial_summary(tr: Trial, window_s: float, *, reversal_signal: str, align_s: float) -> dict[str, Any] | None:
    if not (os.path.exists(tr.baseline_npz) and os.path.exists(tr.ff_npz)):
        return None

    b = _load_npz(tr.baseline_npz)
    f = _load_npz(tr.ff_npz)

    # Prefer baseline dt as reference for window sizing
    tb = np.asarray(b.get("t", []), dtype=np.float64).reshape(-1)
    if tb.size >= 2:
        dt = float(np.median(np.diff(tb)))
    else:
        dt = float(tr.meta.get("dt_cfg", 0.01))

    half_window = int(max(1, round(float(window_s) / dt)))

    centers, rev_meta = _find_reversals(b, dt=dt, reversal_signal=reversal_signal, align_s=align_s)
    centers = centers[(centers > half_window) & (centers < len(tb) - half_window - 1)]

    b_e = np.asarray(b.get("e_q", []), dtype=np.float64).reshape(-1)
    f_e = np.asarray(f.get("e_q", []), dtype=np.float64).reshape(-1)
    b_tau_out = np.asarray(b.get("tau_out", []), dtype=np.float64).reshape(-1)
    f_tau_out = np.asarray(f.get("tau_out", []), dtype=np.float64).reshape(-1)
    f_tau_ff = np.asarray(f.get("tau_ff", []), dtype=np.float64).reshape(-1)
    b_loop = np.asarray(b.get("loop_dt", []), dtype=np.float64).reshape(-1)
    f_loop = np.asarray(f.get("loop_dt", []), dtype=np.float64).reshape(-1)

    b_rev = _window_metrics(b, centers, half_window=half_window)
    f_rev = _window_metrics(f, centers, half_window=half_window)

    out: dict[str, Any] = {
        "report": os.path.basename(tr.report_path),
        "baseline_npz": os.path.basename(tr.baseline_npz),
        "ff_npz": os.path.basename(tr.ff_npz),
        "n": int(b_e.size),
        "dt_med": dt,
        "window_s": float(window_s),
        "n_rev": int(b_rev["n_rev"]),
        "rev_seed": str(rev_meta.get("seed", "n/a")),
        "rev_align": str(rev_meta.get("align", "n/a")),
        # Global metrics
        "b_rmse": _rmse(b_e),
        "f_rmse": _rmse(f_e),
        "b_max": _maxabs(b_e),
        "f_max": _maxabs(f_e),
        "b_tauout": _meanabs(b_tau_out),
        "f_tauout": _meanabs(f_tau_out),
        "f_tauff": _meanabs(f_tau_ff),
        # Timing metrics
        "b_loop_p90": _q(b_loop, 0.90),
        "f_loop_p90": _q(f_loop, 0.90),
        # Reversal-window metrics
        "b_rmse_rev": float(b_rev["rmse_rev"]),
        "f_rmse_rev": float(f_rev["rmse_rev"]),
        "b_max_rev": float(b_rev["maxabs_rev"]),
        "f_max_rev": float(f_rev["maxabs_rev"]),
        "b_tauout_rev": float(b_rev["meanabs_tau_out_rev"]),
        "f_tauout_rev": float(f_rev["meanabs_tau_out_rev"]),
        "f_tauff_rev": float(f_rev["meanabs_tau_ff_rev"]),
    }
    out.update(tr.meta)
    return out


def _fmt(x: float, digits: int = 6) -> str:
    if x != x:  # nan
        return "n/a"
    return f"{x:.{digits}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="runs/ff_demo_report_*.md", help="glob for demo report markdown files")
    ap.add_argument("--out", default=None, help="output markdown path (default: runs/summary_ff_demo_<stamp>.md)")
    ap.add_argument("--window_s", type=float, default=0.5, help="half-window around each reversal (seconds)")
    ap.add_argument(
        "--reversal_signal",
        default="auto",
        choices=["auto", "qd_from_q", "qd", "qd_ref"],
        help="signal used to align reversal centers (default: auto; prefer qd_from_q -> qd -> qd_ref)",
    )
    ap.add_argument(
        "--align_s",
        type=float,
        default=0.25,
        help="if qd_ref exists, align each qd_ref zero-crossing to nearest |qd_x| minimum within +/- align_s seconds (0 disables)",
    )
    ap.add_argument("--only_ff_type", default="torque_delta", help="only include reports with ff_type=... (default: torque_delta)")
    args = ap.parse_args()

    reports = sorted(glob.glob(args.glob))
    trials: list[Trial] = []
    for rp in reports:
        tr = _parse_report_md(rp)
        if tr is None:
            continue
        if args.only_ff_type and tr.meta.get("ff_type", "") != args.only_ff_type:
            continue
        trials.append(tr)

    rows: list[dict[str, Any]] = []
    for tr in trials:
        s = _trial_summary(tr, window_s=float(args.window_s), reversal_signal=str(args.reversal_signal), align_s=float(args.align_s))
        if s is not None:
            rows.append(s)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = args.out or f"runs/summary_ff_demo_{stamp}.md"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    lines: list[str] = []
    lines.append("# FF Demo 批量汇总")
    lines.append("")
    lines.append(f"- 输入 glob: `{args.glob}`")
    lines.append(f"- 过滤 ff_type: `{args.only_ff_type}`")
    lines.append(f"- 换向窗口半宽: {args.window_s:g}s")
    lines.append(f"- 换向检测: seed=`qd_ref`(若存在) + align=`{args.reversal_signal}`(±{args.align_s:g}s)")
    lines.append("")
    lines.append(f"共收集到 {len(rows)} 组可解析实验。")
    lines.append("")

    if not rows:
        Path(out_path).write_text("\n".join(lines), encoding="utf-8")
        print("saved:", out_path)
        return

    # Sort by global rmse improvement ratio
    def ratio(r: dict[str, Any]) -> float:
        b = float(r.get("b_rmse", float("nan")))
        f = float(r.get("f_rmse", float("nan")))
        if not (b > 0):
            return 9e9
        return f / b

    rows.sort(key=ratio)

    lines.append("## 总览表（按 RMSE 改善排序）")
    lines.append("")
    lines.append(
        "| report | amp | freq | kp | kd | limit | scale | div | rev(seed/align) | rmse(b→ff) | max(b→ff) | rmse_rev(b→ff) | loop_p90(b/ff) |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")

    for r in rows:
        lines.append(
            "| {report} | {amp:g} | {freq:g} | {kp:g} | {kd:g} | {lim:g} | {scale:g} | {div} | {rev} | {b_rmse}->{f_rmse} | {b_max}->{f_max} | {b_rmse_rev}->{f_rmse_rev} | {b_loop}/{f_loop} |".format(
                report=r.get("report", ""),
                amp=float(r.get("amp", float("nan"))),
                freq=float(r.get("freq", float("nan"))),
                kp=float(r.get("kp", float("nan"))),
                kd=float(r.get("kd", float("nan"))),
                lim=float(r.get("tau_ff_limit", float("nan"))),
                scale=float(r.get("tau_ff_scale", float("nan"))),
                div=int(r.get("ff_update_div", 0)),
                rev=f"{r.get('rev_seed','n/a')}/{r.get('rev_align','n/a')}",
                b_rmse=_fmt(float(r.get("b_rmse", float("nan"))), 6),
                f_rmse=_fmt(float(r.get("f_rmse", float("nan"))), 6),
                b_max=_fmt(float(r.get("b_max", float("nan"))), 6),
                f_max=_fmt(float(r.get("f_max", float("nan"))), 6),
                b_rmse_rev=_fmt(float(r.get("b_rmse_rev", float("nan"))), 6),
                f_rmse_rev=_fmt(float(r.get("f_rmse_rev", float("nan"))), 6),
                b_loop=_fmt(float(r.get("b_loop_p90", float("nan"))), 6),
                f_loop=_fmt(float(r.get("f_loop_p90", float("nan"))), 6),
            )
        )

    lines.append("")
    lines.append("## 指标说明")
    lines.append("")
    lines.append("- `rmse/max`：全程 `e_q` 的 RMSE / 最大绝对值")
    lines.append(
        "- `rmse_rev`：只在换向窗口内统计的 `e_q` RMSE（默认以 `qd_ref` 过零作为换向 seed，并在邻域内对齐到 `qd/qd_from_q` 的近零点）"
    )
    lines.append("- `loop_p90`：主循环执行间隔的 90 分位（用于确认实时性可比）")
    lines.append("")

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    print("saved:", out_path)


if __name__ == "__main__":
    main()
