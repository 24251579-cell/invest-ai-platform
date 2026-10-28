"""Daily candles from MOEX ISS (TQBR, interval=24) with `start` pagination."""

from __future__ import annotations

import logging
import time
from datetime import date

import pandas as pd
import requests

from invest_ai import config

log = logging.getLogger(__name__)

CANDLE_COLUMNS = ["ticker", "trade_date", "open", "high", "low", "close", "volume", "value"]


def fetch_candles(ticker: str, date_from: date, date_till: date, retries: int = 3) -> pd.DataFrame:
    cfg = config.settings()["moex"]
    url = (f"{cfg['base_url']}/engines/stock/markets/shares/boards/{config.board()}"
           f"/securities/{ticker}/candles.json")
    rows: list[list] = []
    columns: list[str] = []
    start = 0
    while True:
        params = {"from": date_from.isoformat(), "till": date_till.isoformat(),
                  "interval": cfg["interval"], "start": start, "iss.meta": "off"}
        for attempt in range(1, retries + 1):
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                break
            except requests.RequestException as exc:
                if attempt == retries:
                    raise
                log.warning("ISS %s attempt %d failed: %s", ticker, attempt, exc)
                time.sleep(2 * attempt)
        block = resp.json()["candles"]
        columns = block["columns"]
        if not block["data"]:
            break
        rows.extend(block["data"])
        start += len(block["data"])

    if not rows:
        return pd.DataFrame(columns=CANDLE_COLUMNS)
    df = pd.DataFrame(rows, columns=columns)
    df["ticker"] = ticker
    df["trade_date"] = pd.to_datetime(df["begin"]).dt.date
    return df[CANDLE_COLUMNS]


def validate_candles(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (valid, rejected) per the ETL rules of СРО2 section 2.2."""
    df = df.copy()
    for col in ["open", "high", "low", "close", "value"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    reasons = pd.Series("", index=df.index)
    required = df[["ticker", "trade_date", "close"]].isna().any(axis=1)
    reasons[required] += "missing_required;"
    prices = df[["open", "high", "low", "close"]]
    reasons[(prices <= 0).any(axis=1) | prices.isna().any(axis=1)] += "non_positive_price;"
    reasons[df["high"] < df[["open", "close"]].max(axis=1)] += "high_lt_open_close;"
    reasons[df["low"] > df[["open", "close"]].min(axis=1)] += "low_gt_open_close;"
    reasons[(df["volume"] < 0) | (df["value"] < 0)] += "negative_volume;"
    reasons[pd.to_datetime(df["trade_date"]).dt.date > date.today()] += "future_date;"

    bad = reasons != ""
    rejected = df[bad].assign(reason=reasons[bad])
    valid = df[~bad].sort_values("trade_date").reset_index(drop=True)

    threshold = config.settings()["moex"]["anomaly_close_change"]
    jumps = valid["close"].pct_change().abs() > threshold
    for _, row in valid[jumps].iterrows():
        log.warning("Anomaly %s %s: close change > %.0f%%", row.ticker, row.trade_date, threshold * 100)
    return valid, rejected


def upsert_candles_csv(ticker: str, new: pd.DataFrame) -> pd.DataFrame:
    """Upsert by (ticker, trade_date): the latest load wins."""
    path = config.CANDLES_DIR / f"{ticker}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path, parse_dates=["trade_date"])
        old["trade_date"] = old["trade_date"].dt.date
        new = pd.concat([old, new]).drop_duplicates(["ticker", "trade_date"], keep="last")
    new = new.sort_values("trade_date").reset_index(drop=True)
    new.to_csv(path, index=False)
    return new


def load_candles(ticker: str) -> pd.DataFrame:
    path = config.CANDLES_DIR / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No candles for {ticker}: run scripts/load_candles.py first")
    df = pd.read_csv(path, parse_dates=["trade_date"])
    return df.sort_values("trade_date").reset_index(drop=True)
