from __future__ import annotations

import argparse

from inverse_torque.torque_delta import prepare_torque_delta_dataset
from pipeline.config import load_cfg
from project_config import get


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--raw", default=None, help="raw log npz (default: paths.real_log)")
    ap.add_argument("--out", default=None, help="output torque dataset npz")
    ap.add_argument("--stats", default=None, help="output stats npz (optional)")
    ap.add_argument("--tau_lpf_hz", type=float, default=0.0, help="optional tau_out low-pass cutoff (Hz); 0 disables")
    ap.add_argument("--qd_lpf_hz", type=float, default=0.0, help="optional qd low-pass cutoff (Hz); 0 disables")
    ap.add_argument("--causal", action="store_true", help="use causal (non-zero-phase) filtering if LPF enabled")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    raw = args.raw or str(get(cfg, "paths.real_log"))
    out = args.out or "runs/torque_delta_dataset.npz"
    stats = args.stats or "runs/torque_delta_stats.npz"

    # Best-effort: show provenance of the raw log so we don't mix up sources.
    try:
        import numpy as np

        ds = dict(np.load(raw, allow_pickle=True))
        if "parent_csv" in ds and getattr(ds["parent_csv"], "dtype", None) == object and ds["parent_csv"].size:
            print(f"[raw] parent_csv={ds['parent_csv'].reshape(-1)[0]}")
        if "parent_csv_sha256" in ds and getattr(ds["parent_csv_sha256"], "dtype", None) == object and ds["parent_csv_sha256"].size:
            print(f"[raw] parent_csv_sha256={str(ds['parent_csv_sha256'].reshape(-1)[0])[:16]}...")
    except Exception:
        pass

    prepare_torque_delta_dataset(
        cfg,
        raw_npz=raw,
        out_npz=out,
        stats_npz=stats,
        tau_lpf_hz=(float(args.tau_lpf_hz) if float(args.tau_lpf_hz) > 0.0 else None),
        qd_lpf_hz=(float(args.qd_lpf_hz) if float(args.qd_lpf_hz) > 0.0 else None),
        zero_phase=(not bool(args.causal)),
    )
    print(f"saved torque-delta dataset: {out}")
    if stats:
        print(f"saved torque-delta stats: {stats}")


if __name__ == "__main__":
    main()
