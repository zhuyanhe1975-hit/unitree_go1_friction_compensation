from __future__ import annotations

import argparse
import os

from pipeline.config import load_cfg
from pipeline.train import finetune_model
from project_config import ensure_dir, get


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    ensure_dir(get(cfg, "paths.runs_dir"))

    ds = str(get(cfg, "paths.real_dataset"))
    base = str(get(cfg, "paths.sim_model"))
    out = str(get(cfg, "paths.real_model"))
    ensure_dir(os.path.dirname(out) or ".")
    finetune_model(cfg, dataset_npz=ds, base_weights=base, out_weights=out)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()

