"""Forecast for t+1 from the last N days and upsert into forecasts."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import torch

from invest_ai import config
from invest_ai.etl.moex_loader import load_candles
from invest_ai.forecast import registry
from invest_ai.forecast.features import build_frame
from invest_ai.forecast.train import load_daily_sentiment

FORECAST_COLUMNS = ["ticker", "forecast_date", "target_date", "pred_close", "pred_direction",
                    "model_version", "use_sentiment", "created_at"]


def next_trading_day(day: date) -> date:
    """Next weekday after `day` (the exchange calendar is not modelled in the MVP)."""
    day += timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day


def predict(ticker: str, use_sentiment: bool) -> dict:
    n_features = 4 if use_sentiment else 3
    model, cfg, scaler = registry.load(registry.model_dir(ticker, n_features))
    daily = load_daily_sentiment(ticker) if use_sentiment else None
    frame = build_frame(load_candles(ticker), daily, use_sentiment)
    window = frame[cfg["features"]].tail(cfg["window"]).to_numpy(dtype=np.float64)
    if len(window) < cfg["window"]:
        raise ValueError(f"{ticker}: need {cfg['window']} days, have {len(window)}")

    x = torch.from_numpy(scaler.transform(window).astype(np.float32))[None]
    with torch.no_grad():
        q_norm = float(model(x)[0])
    i = cfg["features"].index("close")
    pred_close = q_norm * (scaler.data_max_[i] - scaler.data_min_[i]) + scaler.data_min_[i]

    last = frame.iloc[-1]
    forecast_date = last["trade_date"].date()
    return {
        "ticker": ticker,
        "forecast_date": forecast_date,
        "target_date": next_trading_day(forecast_date),
        "pred_close": round(float(pred_close), 6),
        "pred_direction": int(pred_close > last["close"]),
        "model_version": cfg["model_version"],
        "use_sentiment": use_sentiment,
        "created_at": datetime.now(config.tz()).isoformat(),
    }


def upsert_forecasts(rows: list[dict]) -> pd.DataFrame:
    """Unique by (ticker, target_date, model_version, use_sentiment)."""
    new = pd.DataFrame(rows, columns=FORECAST_COLUMNS).astype({"forecast_date": str, "target_date": str})
    if config.FORECASTS_CSV.exists():
        new = pd.concat([pd.read_csv(config.FORECASTS_CSV), new], ignore_index=True)
    key = ["ticker", "target_date", "model_version", "use_sentiment"]
    new = new.drop_duplicates(key, keep="last").sort_values(["ticker", "target_date"])
    config.FORECASTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(config.FORECASTS_CSV, index=False)
    return new
