from __future__ import annotations

import argparse

from pipeline.config import load_cfg
from pipeline.sim_generate import generate_sim_log
from project_config import get


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    out_npz = str(get(cfg, "paths.sim_raw_log"))
    generate_sim_log(cfg, out_npz=out_npz)
    print(f"saved: {out_npz}")


if __name__ == "__main__":
    main()

