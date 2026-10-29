"""PostgreSQL access: engine, health check, CSV → DB sync."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterator

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool

from invest_ai import config

log = logging.getLogger(__name__)


def _records(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    """Convert dataframe rows to plain Python values (NaN → None)."""
    out: list[dict] = []
    for row in df[cols].itertuples(index=False, name=None):
        item = {}
        for col, val in zip(cols, row):
            if val is None or (isinstance(val, float) and pd.isna(val)):
                item[col] = None
            elif hasattr(val, "item"):
                item[col] = val.item()
            elif hasattr(val, "isoformat") and not isinstance(val, str):
                item[col] = val.isoformat()
            else:
                item[col] = val
        out.append(item)
    return out


def database_url() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgresql+psycopg://invest:invest@localhost:5432/invest",
    )


def engine(url: str | None = None) -> Engine:
    return create_engine(url or database_url(), pool_pre_ping=True)


@contextmanager
def connect(url: str | None = None) -> Iterator:
    eng = engine(url)
    with eng.begin() as conn:
        yield conn


def ping(url: str | None = None, timeout: int = 3) -> dict:
    # Without connect_timeout an unreachable host blocks for minutes (OS TCP timeout).
    eng = create_engine(url or database_url(), poolclass=NullPool, connect_args={"connect_timeout": timeout})
    try:
        with eng.begin() as conn:
            version = conn.execute(text("SELECT version()")).scalar()
            counts = {
                table: int(conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar())
                for table in ("candles", "news", "sentiment_scores", "forecasts", "daily_sentiment")
            }
        return {"ok": True, "version": version, "counts": counts, "url_host": database_url().split("@")[-1]}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "url_host": database_url().split("@")[-1]}


def _upsert_candles(conn, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    rows = df.copy()
    if "loaded_at" not in rows.columns:
        rows["loaded_at"] = pd.Timestamp.now(tz="Europe/Moscow")
    sql = text("""
        INSERT INTO candles (ticker, trade_date, open, high, low, close, volume, value, loaded_at)
        VALUES (:ticker, :trade_date, :open, :high, :low, :close, :volume, :value, :loaded_at)
        ON CONFLICT (ticker, trade_date) DO UPDATE SET
            open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
            close = EXCLUDED.close, volume = EXCLUDED.volume, value = EXCLUDED.value,
            loaded_at = EXCLUDED.loaded_at
    """)
    cols = ["ticker", "trade_date", "open", "high", "low", "close", "volume", "value", "loaded_at"]
    conn.execute(sql, _records(rows, cols))
    return len(rows)


def _upsert_news(conn, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    rows = df.copy()
    rows["ticker"] = rows["ticker"].where(rows["ticker"].notna(), None)
    rows["body"] = rows["body"].where(rows["body"].notna(), None)
    rows["lang"] = rows["lang"].where(rows["lang"].notna(), None)
    sql = text("""
        INSERT INTO news (id, source, url, title, body, published_at, lang, ticker, content_hash, loaded_at)
        VALUES (:id, :source, :url, :title, :body, :published_at, :lang, :ticker, :content_hash, :loaded_at)
        ON CONFLICT (url) DO UPDATE SET
            title = EXCLUDED.title, body = EXCLUDED.body, published_at = EXCLUDED.published_at,
            lang = EXCLUDED.lang, ticker = EXCLUDED.ticker, content_hash = EXCLUDED.content_hash,
            loaded_at = EXCLUDED.loaded_at
    """)
    cols = ["id", "source", "url", "title", "body", "published_at", "lang", "ticker", "content_hash", "loaded_at"]
    conn.execute(sql, _records(rows, cols))
    conn.execute(text("SELECT setval(pg_get_serial_sequence('news', 'id'), COALESCE((SELECT MAX(id) FROM news), 1))"))
    return len(rows)


def _upsert_scores(conn, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    sql = text("""
        INSERT INTO sentiment_scores (id, news_id, model_name, label, p_pos, p_neg, p_neu, score, scored_at)
        VALUES (:id, :news_id, :model_name, :label, :p_pos, :p_neg, :p_neu, :score, :scored_at)
        ON CONFLICT (news_id, model_name) DO UPDATE SET
            label = EXCLUDED.label, p_pos = EXCLUDED.p_pos, p_neg = EXCLUDED.p_neg,
            p_neu = EXCLUDED.p_neu, score = EXCLUDED.score, scored_at = EXCLUDED.scored_at
    """)
    cols = ["id", "news_id", "model_name", "label", "p_pos", "p_neg", "p_neu", "score", "scored_at"]
    conn.execute(sql, _records(df, cols))
    conn.execute(text(
        "SELECT setval(pg_get_serial_sequence('sentiment_scores', 'id'), "
        "COALESCE((SELECT MAX(id) FROM sentiment_scores), 1))"
    ))
    return len(df)


def _upsert_forecasts(conn, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    rows = df.copy()
    rows["use_sentiment"] = rows["use_sentiment"].map(
        lambda v: str(v).lower() in {"1", "true", "t", "yes"} if not isinstance(v, bool) else v
    )
    sql = text("""
        INSERT INTO forecasts
            (ticker, forecast_date, target_date, pred_close, pred_direction, model_version, use_sentiment, created_at)
        VALUES
            (:ticker, :forecast_date, :target_date, :pred_close, :pred_direction, :model_version, :use_sentiment, :created_at)
        ON CONFLICT (ticker, target_date, model_version, use_sentiment) DO UPDATE SET
            forecast_date = EXCLUDED.forecast_date,
            pred_close = EXCLUDED.pred_close,
            pred_direction = EXCLUDED.pred_direction,
            created_at = EXCLUDED.created_at
    """)
    cols = ["ticker", "forecast_date", "target_date", "pred_close", "pred_direction",
            "model_version", "use_sentiment", "created_at"]
    conn.execute(sql, _records(rows, cols))
    return len(rows)


def _upsert_daily(conn, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    sql = text("""
        INSERT INTO daily_sentiment (ticker, trade_date, sentiment, n_news)
        VALUES (:ticker, :trade_date, :sentiment, :n_news)
        ON CONFLICT (ticker, trade_date) DO UPDATE SET
            sentiment = EXCLUDED.sentiment, n_news = EXCLUDED.n_news
    """)
    conn.execute(sql, _records(df, ["ticker", "trade_date", "sentiment", "n_news"]))
    return len(df)


def sync_from_csv(url: str | None = None) -> dict:
    """Load local CSV artifacts into PostgreSQL (idempotent upserts)."""
    stats = {"candles": 0, "news": 0, "sentiment_scores": 0, "forecasts": 0, "daily_sentiment": 0}
    with connect(url) as conn:
        if config.CANDLES_DIR.exists():
            frames = []
            for path in sorted(config.CANDLES_DIR.glob("*.csv")):
                frames.append(pd.read_csv(path, parse_dates=["trade_date"]))
            if frames:
                candles = pd.concat(frames, ignore_index=True)
                candles["trade_date"] = candles["trade_date"].dt.date
                stats["candles"] = _upsert_candles(conn, candles)

        if config.NEWS_CSV.exists():
            news = pd.read_csv(config.NEWS_CSV)
            stats["news"] = _upsert_news(conn, news)

        if config.SENTIMENT_CSV.exists():
            scores = pd.read_csv(config.SENTIMENT_CSV)
            stats["sentiment_scores"] = _upsert_scores(conn, scores)

        if config.FORECASTS_CSV.exists():
            forecasts = pd.read_csv(config.FORECASTS_CSV)
            stats["forecasts"] = _upsert_forecasts(conn, forecasts)

        if config.DAILY_SENTIMENT_CSV.exists():
            daily = pd.read_csv(config.DAILY_SENTIMENT_CSV, parse_dates=["trade_date"])
            daily["trade_date"] = daily["trade_date"].dt.date
            stats["daily_sentiment"] = _upsert_daily(conn, daily)

    log.info("DB sync done: %s", stats)
    return stats


def auto_sync(label: str = "") -> dict | None:
    """Best-effort CSV → Postgres after a pipeline step.

    Enabled by default (AUTO_SYNC_DB=1). Skips quietly if Postgres is down
    or AUTO_SYNC_DB is 0/false/no — so the lab still works without Docker DB.
    """
    flag = os.environ.get("AUTO_SYNC_DB", "1").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return None
    status = ping()
    if not status["ok"]:
        msg = f"auto-sync skipped{f' ({label})' if label else ''}: {status.get('error')}"
        log.warning(msg)
        print(msg)
        return None
    try:
        stats = sync_from_csv()
    except Exception as exc:  # noqa: BLE001
        msg = f"auto-sync failed{f' ({label})' if label else ''}: {exc}"
        log.warning(msg)
        print(msg)
        return None
    print(f"auto-sync OK{f' ({label})' if label else ''}: {stats}")
    return stats
