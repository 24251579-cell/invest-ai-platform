"""Step 6: daily sentiment index S_{k,d} (formulas 2.2-2.3)."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from invest_ai import config

MARKET = "MARKET"


def _next_weekday(day: date) -> date:
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day


def assign_trading_day(published_at: pd.Series, trading_days: list[date]) -> pd.Series:
    """Formula (2.2): d(i) = min{d in D | t_i <= tau_d}.

    News after the cutoff or on non-trading days go to the next trading day.
    Beyond the last known trading day, the next weekday is used.
    """
    days = np.array(sorted(trading_days), dtype="datetime64[D]")
    tz = config.tz()
    result = []
    for ts in pd.to_datetime(published_at, utc=True):
        local = ts.tz_convert(tz).to_pydatetime()
        candidate = local.date()
        cutoff = datetime.combine(candidate, config.cutoff_for(candidate), tzinfo=tz)
        if local > cutoff:
            candidate += timedelta(days=1)
        pos = np.searchsorted(days, np.datetime64(candidate, "D"))
        result.append(days[pos].astype(date) if pos < len(days) else _next_weekday(candidate))
    return pd.Series(result, index=published_at.index)


def daily_index(news: pd.DataFrame, scores: pd.DataFrame, trading_days: list[date]) -> pd.DataFrame:
    """Formula (2.3): S_{k,d} = mean(s_i) over news of instrument k assigned to day d.

    Market-wide news (ticker = NULL) form a separate series with ticker = MARKET.
    Returns columns: ticker, trade_date, sentiment, n_news.
    """
    merged = scores.merge(news[["id", "published_at", "ticker"]], left_on="news_id", right_on="id")
    if merged.empty:
        return pd.DataFrame(columns=["ticker", "trade_date", "sentiment", "n_news"])
    merged["ticker"] = merged["ticker"].fillna(MARKET)
    merged["trade_date"] = assign_trading_day(merged["published_at"], trading_days)
    return (merged.groupby(["ticker", "trade_date"])["score"]
            .agg(sentiment="mean", n_news="size")
            .reset_index()
            .sort_values(["ticker", "trade_date"], ignore_index=True))
