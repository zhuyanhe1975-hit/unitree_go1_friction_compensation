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


def state_error_to_features(q: np.ndarray, qd: np.ndarray, q_ref: np.ndarray, qd_ref: np.ndarray) -> np.ndarray:
    """
    Feature map for a single joint with controller context:
      [sin(q), cos(q), qd, e_q, e_qd]
    where:
      e_q  = q_ref - q
      e_qd = qd_ref - qd

    Accepts (...,) arrays and returns (..., 5).
    """
    q = np.asarray(q, dtype=np.float64)
    qd = np.asarray(qd, dtype=np.float64)
    q_ref = np.asarray(q_ref, dtype=np.float64)
    qd_ref = np.asarray(qd_ref, dtype=np.float64)
    e_q = q_ref - q
    e_qd = qd_ref - qd
    return np.stack([np.sin(q), np.cos(q), qd, e_q, e_qd], axis=-1)
