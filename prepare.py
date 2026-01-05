from __future__ import annotations

import argparse

from pipeline.config import load_cfg
from pipeline.prepare import prepare_dataset
from project_config import get


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--mode", choices=["sim", "real"], required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    if args.mode == "sim":
        raw = str(get(cfg, "paths.sim_raw_log"))
        out = str(get(cfg, "paths.sim_dataset"))
        stats = str(get(cfg, "paths.stats_npz"))
        prepare_dataset(cfg, raw_npz=raw, out_npz=out, stats_npz=stats)
        print(f"saved: {out}")
        print(f"saved stats: {stats}")
    else:
        raw = str(get(cfg, "paths.real_log"))
        out = str(get(cfg, "paths.real_dataset"))
        stats = str(get(cfg, "paths.stats_npz"))  # sim stats for feature/action normalization
        real_stats = str(get(cfg, "paths.real_stats_npz"))  # real delta stats (optional)
        prepare_dataset(cfg, raw_npz=raw, out_npz=out, stats_npz=stats, real_stats_npz=real_stats)
        print(f"saved: {out}")
        print(f"saved real stats: {real_stats}")


if __name__ == "__main__":
    main()
