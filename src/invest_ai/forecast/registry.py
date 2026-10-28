"""Model Registry: models/lstm_{ticker}_f{F}/ with weights, config, scaler and metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from invest_ai import config
from invest_ai.forecast.model import LSTMRegressor


def model_dir(ticker: str, n_features: int) -> Path:
    return config.MODELS_DIR / f"lstm_{ticker.lower()}_f{n_features}"


def save(path: Path, model: LSTMRegressor, cfg: dict, scaler: MinMaxScaler, metrics: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    (path / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), "utf-8")
    (path / "scaler.json").write_text(json.dumps({
        "data_min": scaler.data_min_.tolist(), "data_max": scaler.data_max_.tolist(),
    }, indent=2), "utf-8")
    (path / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), "utf-8")


def load(path: Path) -> tuple[LSTMRegressor, dict, MinMaxScaler]:
    if not (path / "model.pt").exists():
        raise FileNotFoundError(f"Model is not trained: {path}")
    cfg = json.loads((path / "config.json").read_text("utf-8"))
    model = LSTMRegressor(len(cfg["features"]), cfg["hidden_size"], cfg["num_layers"], cfg["dropout"])
    model.load_state_dict(torch.load(path / "model.pt", weights_only=True))
    model.eval()
    s = json.loads((path / "scaler.json").read_text("utf-8"))
    scaler = MinMaxScaler().fit(np.array([s["data_min"], s["data_max"]]))
    return model, cfg, scaler
