from __future__ import annotations

import argparse

from inverse_torque.torque_delta_v2 import prepare_torque_delta_dataset_v2
from pipeline.config import load_cfg
from project_config import get


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--raw", default=None, help="raw log npz (default: paths.real_log)")
    ap.add_argument("--out", default=None, help="output torque-delta v2 dataset npz")
    ap.add_argument("--stats", default=None, help="output stats npz (optional)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    raw = args.raw or str(get(cfg, "paths.real_log"))
    out = args.out or "runs/torque_delta_dataset_v2.npz"
    stats = args.stats or "runs/torque_delta_stats_v2.npz"

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

    prepare_torque_delta_dataset_v2(cfg, raw_npz=raw, out_npz=out, stats_npz=stats)
    print(f"saved torque-delta v2 dataset: {out}")
    if stats:
        print(f"saved torque-delta v2 stats: {stats}")


if __name__ == "__main__":
    main()
