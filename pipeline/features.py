from __future__ import annotations

import numpy as np


def state_to_features(q: np.ndarray, qd: np.ndarray) -> np.ndarray:
    """
    Feature map for a single joint:
      [sin(q), cos(q), qd]
    Accepts (...,) arrays and returns (..., 3).
    """
    q = np.asarray(q, dtype=np.float64)
    qd = np.asarray(qd, dtype=np.float64)
    return np.stack([np.sin(q), np.cos(q), qd], axis=-1)

