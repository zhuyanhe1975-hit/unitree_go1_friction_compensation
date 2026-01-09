from __future__ import annotations

import argparse
import os

from pipeline.config import load_cfg
from pipeline.residual import prepare_residual_dataset, train_residual_model
from project_config import ensure_dir, get


def _default_paths(cfg: dict, mode: str) -> tuple[str, str, str]:
    if mode == "sim":
        base_model = str(get(cfg, "paths.sim_model"))
        dataset = str(get(cfg, "paths.sim_dataset"))
    elif mode == "real":
        base_model = str(get(cfg, "paths.real_model"))
        dataset = str(get(cfg, "paths.real_dataset"))
    else:
        base_model = str(get(cfg, "paths.real_model_scratch"))
        dataset = str(get(cfg, "paths.real_dataset"))

    def _suffix(path: str, suffix: str) -> str:
        if path.endswith(".npz") or path.endswith(".pt"):
            stem = path.rsplit(".", 1)[0]
            ext = path.rsplit(".", 1)[1]
            return f"{stem}{suffix}.{ext}"
        return f"{path}{suffix}"

    residual_dataset = _suffix(dataset, "_residual")
    out_model = _suffix(base_model, "_residual")
    return base_model, residual_dataset, out_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--mode", choices=["sim", "real", "real_scratch"], default="real")
    ap.add_argument("--base_model", default=None, help="optional override of base model path")
    ap.add_argument("--dataset", default=None, help="optional override of prepared dataset (.npz)")
    ap.add_argument("--residual_dataset", default=None, help="output path for residual dataset")
    ap.add_argument("--out_model", default=None, help="output path for residual model weights")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    ensure_dir(get(cfg, "paths.runs_dir"))

    base_model_default, residual_ds_default, out_model_default = _default_paths(cfg, args.mode)
    base_model = args.base_model or base_model_default
    dataset = args.dataset or (str(get(cfg, "paths.sim_dataset")) if args.mode == "sim" else str(get(cfg, "paths.real_dataset")))
    residual_dataset = args.residual_dataset or residual_ds_default
    out_model = args.out_model or out_model_default

    ensure_dir(os.path.dirname(residual_dataset) or ".")
    ensure_dir(os.path.dirname(out_model) or ".")

    prepare_residual_dataset(cfg, dataset_npz=dataset, base_model_path=base_model, out_npz=residual_dataset)
    print(f"saved residual dataset: {residual_dataset}")

    train_residual_model(cfg, dataset_npz=residual_dataset, base_model_path=base_model, out_weights=out_model)
    print(f"saved residual model: {out_model}")


if __name__ == "__main__":
    main()
