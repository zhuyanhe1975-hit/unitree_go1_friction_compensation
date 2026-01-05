from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


class ConfigError(RuntimeError):
    pass


def load_config(path: str | None = None) -> Dict[str, Any]:
    """
    Loads the single project config JSON.
    By default, reads `config.json` at repo root (CWD).
    """
    cfg_path = path or "config.json"
    if not os.path.exists(cfg_path):
        raise ConfigError(f"missing config file: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ConfigError("config root must be an object")
    return cfg


def get(cfg: Dict[str, Any], dotted: str, default: Any = None, required: bool = True) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            if required:
                raise ConfigError(f"missing config key: {dotted}")
            return default
        cur = cur[part]
    return cur


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

