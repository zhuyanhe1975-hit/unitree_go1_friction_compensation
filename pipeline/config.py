from __future__ import annotations

from typing import Any, Dict

from project_config import get, load_config


def load_cfg(path: str | None = None) -> Dict[str, Any]:
    """
    Loads repo-level config.json (optionally overridden by path) and performs a few
    basic sanity checks for the pipeline.
    """
    cfg = load_config(path)
    # Touch commonly used keys so we fail early with a readable message.
    get(cfg, "paths.runs_dir")
    get(cfg, "paths.asset_mjcf")
    get(cfg, "sim.device")
    get(cfg, "sim.num_envs")
    get(cfg, "model.history_len")
    get(cfg, "train.lr")
    return cfg

