"""Feature builder: x_t = (p_t, v_t, r_t[, S_k,t]), windows of N days, chronological split."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from invest_ai import config

PRICE_FEATURES = ["close", "volume", "ret"]
SENTIMENT_FEATURE = "sentiment"


def build_frame(candles: pd.DataFrame, daily_sentiment: pd.DataFrame | None,
                use_sentiment: bool) -> pd.DataFrame:
    """Daily frame with features; r_t by formula (2.4); missing S -> sentiment_fill."""
    df = candles[["trade_date", "close", "volume"]].copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").reset_index(drop=True)
    df["ret"] = df["close"].pct_change()
    df = df.dropna(subset=["ret"]).reset_index(drop=True)
    if use_sentiment:
        fill = config.settings()["lstm"]["sentiment_fill"]
        if daily_sentiment is None or daily_sentiment.empty:
            df[SENTIMENT_FEATURE] = fill
        else:
            s = daily_sentiment[["trade_date", "sentiment"]].copy()
            s["trade_date"] = pd.to_datetime(s["trade_date"])
            df = df.merge(s, on="trade_date", how="left")
            df[SENTIMENT_FEATURE] = df[SENTIMENT_FEATURE].fillna(fill)
    return df


def feature_columns(use_sentiment: bool) -> list[str]:
    return PRICE_FEATURES + ([SENTIMENT_FEATURE] if use_sentiment else [])


@dataclass
class Dataset:
    X: np.ndarray            # (M, N, F) normalized windows
    y: np.ndarray            # (M,) normalized close at t+1
    close_prev: np.ndarray   # (M,) p_t in rubles (last day of the window)
    close_next: np.ndarray   # (M,) p_{t+1} in rubles (target)
    target_dates: np.ndarray
    splits: dict[str, slice]
    scaler: MinMaxScaler
    columns: list[str]

    def part(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        s = self.splits[name]
        return self.X[s], self.y[s]

    def inverse_close(self, y_norm: np.ndarray) -> np.ndarray:
        """Back to rubles using the close column of the scaler (formula 2.7 inverted)."""
        i = self.columns.index("close")
        return y_norm * (self.scaler.data_max_[i] - self.scaler.data_min_[i]) + self.scaler.data_min_[i]


def split_bounds(m: int, ratios: list[float]) -> dict[str, slice]:
    n_train = int(m * ratios[0])
    n_val = int(m * ratios[1])
    return {"train": slice(0, n_train),
            "val": slice(n_train, n_train + n_val),
            "test": slice(n_train + n_val, m)}


def make_dataset(frame: pd.DataFrame, columns: list[str], window: int,
                 ratios: list[float]) -> Dataset:
    """Windows X_t = (x_{t-N+1}..x_t) -> p_{t+1}. Scaler is fit on the train part only."""
    values = frame[columns].to_numpy(dtype=np.float64)
    close = frame["close"].to_numpy(dtype=np.float64)
    m = len(frame) - window
    if m <= 10:
        raise ValueError(f"Not enough data: {len(frame)} days for window {window}")
    splits = split_bounds(m, ratios)

    # Sample j uses rows j..j+N-1 as input and row j+N as target.
    last_train_row = splits["train"].stop - 1 + window
    scaler = MinMaxScaler().fit(values[:last_train_row + 1])
    scaled = scaler.transform(values)

    idx = np.arange(m)[:, None] + np.arange(window)[None, :]
    X = scaled[idx].astype(np.float32)
    y = scaled[np.arange(m) + window, columns.index("close")].astype(np.float32)
    return Dataset(X=X, y=y,
                   close_prev=close[np.arange(m) + window - 1],
                   close_next=close[np.arange(m) + window],
                   target_dates=frame["trade_date"].to_numpy()[np.arange(m) + window],
                   splits=splits, scaler=scaler, columns=columns)
