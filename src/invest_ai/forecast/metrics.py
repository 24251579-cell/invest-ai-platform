"""Quality metrics (formulas 2.16-2.20) and reference baselines."""

from __future__ import annotations

import numpy as np


def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - pred)))


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def directional_accuracy(actual: np.ndarray, pred: np.ndarray, prev: np.ndarray) -> float:
    """DA, %: sign(q_t - p_{t-1}) == sign(p_t - p_{t-1})."""
    return float(100 * np.mean(np.sign(pred - prev) == np.sign(actual - prev)))


def evaluate(actual: np.ndarray, pred: np.ndarray, prev: np.ndarray) -> dict:
    """Model metrics plus naive baseline q_t = p_{t-1} on the same test sample."""
    naive_rmse = rmse(actual, prev)
    up = np.sign(actual - prev)
    prev_dir = np.concatenate([[np.nan], up[:-1]])
    return {
        "n_test": int(len(actual)),
        "mae": mae(actual, pred),
        "rmse": rmse(actual, pred),
        "da": directional_accuracy(actual, pred, prev),
        "rmse_rel": rmse(actual, pred) / naive_rmse if naive_rmse > 0 else float("nan"),
        "naive_mae": mae(actual, prev),
        "naive_rmse": naive_rmse,
        "da_same_as_yesterday": float(100 * np.mean(up[1:] == prev_dir[1:])),
        "da_always_up": float(100 * np.mean(up > 0)),
    }
