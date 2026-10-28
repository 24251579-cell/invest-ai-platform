"""Training with Adam + MSE, gradient clipping and early stopping on validation MSE."""

from __future__ import annotations

import copy
import logging
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from invest_ai import config
from invest_ai.etl.moex_loader import load_candles
from invest_ai.forecast import registry
from invest_ai.forecast.features import build_frame, feature_columns, make_dataset
from invest_ai.forecast.metrics import evaluate
from invest_ai.forecast.model import LSTMRegressor

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def load_daily_sentiment(ticker: str) -> pd.DataFrame | None:
    if not config.DAILY_SENTIMENT_CSV.exists():
        return None
    daily = pd.read_csv(config.DAILY_SENTIMENT_CSV)
    return daily[daily["ticker"] == ticker]


def _mse(model: nn.Module, X: np.ndarray, y: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        return float(nn.functional.mse_loss(model(torch.from_numpy(X)), torch.from_numpy(y)))


def train(ticker: str, use_sentiment: bool) -> dict:
    p = config.settings()["lstm"]
    set_seed(p["seed"])
    torch.set_num_threads(config.torch_threads())

    columns = feature_columns(use_sentiment)
    daily = load_daily_sentiment(ticker) if use_sentiment else None
    frame = build_frame(load_candles(ticker), daily, use_sentiment)
    ds = make_dataset(frame, columns, p["window"], p["split"])
    X_train, y_train = ds.part("train")
    X_val, y_val = ds.part("val")
    X_test, _ = ds.part("test")

    model = LSTMRegressor(len(columns), p["hidden_size"], p["num_layers"], p["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=p["lr"])
    loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                        batch_size=p["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(p["seed"]))

    best_val, best_state, best_epoch, bad_epochs = float("inf"), None, 0, 0
    history = []
    for epoch in range(1, p["max_epochs"] + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), p["grad_clip"])
            optimizer.step()
        train_mse, val_mse = _mse(model, X_train, y_train), _mse(model, X_val, y_val)
        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})
        if val_mse < best_val:
            best_val, best_state, best_epoch, bad_epochs = val_mse, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad_epochs += 1
            if bad_epochs >= p["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    log.info("%s F=%d: best epoch %d, val MSE %.6f", ticker, len(columns), best_epoch, best_val)

    with torch.no_grad():
        pred_test = ds.inverse_close(model(torch.from_numpy(X_test)).numpy())
    s = ds.splits["test"]
    metrics = evaluate(ds.close_next[s], pred_test, ds.close_prev[s])
    metrics.update({"best_epoch": best_epoch, "val_mse": best_val, "epochs_run": len(history)})

    n_features = len(columns)
    dates = ds.target_dates
    cfg = {
        "model_version": f"lstm_{ticker.lower()}_f{n_features}_{datetime.now():%Y%m%d%H%M}",
        "ticker": ticker,
        "use_sentiment": use_sentiment,
        "features": columns,
        **{k: p[k] for k in ["window", "hidden_size", "num_layers", "dropout", "lr", "batch_size",
                             "max_epochs", "patience", "grad_clip", "seed", "split"]},
        "sentiment_fill": p["sentiment_fill"] if use_sentiment else None,
        "news_days_with_sentiment": int((frame.get("sentiment", pd.Series(dtype=float)) != 0).sum()),
        "train_period": [str(dates[ds.splits["train"]][0])[:10], str(dates[ds.splits["train"]][-1])[:10]],
        "val_period": [str(dates[ds.splits["val"]][0])[:10], str(dates[ds.splits["val"]][-1])[:10]],
        "test_period": [str(dates[s][0])[:10], str(dates[s][-1])[:10]],
        "torch_version": torch.__version__,
        "trained_at": datetime.now(config.tz()).isoformat(),
    }
    path = registry.model_dir(ticker, n_features)
    registry.save(path, model, cfg, ds.scaler, metrics)
    pd.DataFrame(history).to_csv(path / "history.csv", index=False)
    pd.DataFrame({"target_date": pd.to_datetime(dates[s]).date, "close_prev": ds.close_prev[s],
                  "actual": ds.close_next[s], "pred": pred_test}).to_csv(path / "test_predictions.csv", index=False)
    return {"config": cfg, "metrics": metrics, "path": str(path)}
