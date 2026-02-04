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

    prepare_torque_delta_dataset_v2(cfg, raw_npz=raw, out_npz=out, stats_npz=stats)
    print(f"saved torque-delta v2 dataset: {out}")
    if stats:
        print(f"saved torque-delta v2 stats: {stats}")


if __name__ == "__main__":
    main()

