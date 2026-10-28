"""Project paths and YAML configuration."""

from __future__ import annotations

import os
from datetime import date, time
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

CANDLES_DIR = RAW_DIR / "candles"
NEWS_CSV = RAW_DIR / "news.csv"
SENTIMENT_CSV = PROCESSED_DIR / "sentiment_scores.csv"
DAILY_SENTIMENT_CSV = PROCESSED_DIR / "daily_sentiment.csv"
FORECASTS_CSV = PROCESSED_DIR / "forecasts.csv"


def _load_yaml(name: str) -> dict:
    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache
def settings() -> dict:
    return _load_yaml("settings.yaml")


@lru_cache
def tickers() -> dict:
    return _load_yaml("tickers.yaml")["tickers"]


def board() -> str:
    return _load_yaml("tickers.yaml")["board"]


@lru_cache
def rss_sources() -> list[dict]:
    return _load_yaml("rss_sources.yaml")["sources"]


def tz() -> ZoneInfo:
    return ZoneInfo(settings()["timezone"])


def cutoff_for(day: date) -> time:
    """Trading-day cutoff time in effect on `day` (formula 2.2)."""
    result = None
    for entry in settings()["cutoff_schedule"]:
        if date.fromisoformat(entry["since"]) <= day:
            result = time.fromisoformat(entry["time"])
    if result is None:
        raise ValueError(f"No cutoff configured for {day}")
    return result


def torch_threads() -> int:
    return int(os.environ.get("TORCH_NUM_THREADS", os.cpu_count() or 1))
