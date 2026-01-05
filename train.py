from __future__ import annotations

import argparse
import os

from pipeline.config import load_cfg
from pipeline.train import train_model
from project_config import ensure_dir, get


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--mode", choices=["sim", "real"], default="sim")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    ensure_dir(get(cfg, "paths.runs_dir"))
    if args.mode == "sim":
        ds = str(get(cfg, "paths.sim_dataset"))
        out = str(get(cfg, "paths.sim_model"))
    else:
        ds = str(get(cfg, "paths.real_dataset"))
        out = str(get(cfg, "paths.real_model_scratch"))
    ensure_dir(os.path.dirname(out) or ".")
    train_model(cfg, dataset_npz=ds, out_weights=out)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
