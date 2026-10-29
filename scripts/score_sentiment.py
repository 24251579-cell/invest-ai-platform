"""Score unscored news (FinBERT / RuBERT) and rebuild the daily sentiment index.

Usage: python scripts/score_sentiment.py [--limit 50]
Outputs: data/processed/sentiment_scores.csv, data/processed/daily_sentiment.csv
"""

import argparse
import logging

import pandas as pd

from invest_ai import config
from invest_ai.db import auto_sync
from invest_ai.etl.news_collector import read_news
from invest_ai.sentiment.aggregate import daily_index
from invest_ai.sentiment.scoring import read_scores, score_pending


def trading_days() -> list:
    days: set = set()
    for path in config.CANDLES_DIR.glob("*.csv"):
        days |= set(pd.read_csv(path, parse_dates=["trade_date"])["trade_date"].dt.date)
    return sorted(days)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="max news per language")
    parser.add_argument("--no-sync", action="store_true", help="skip PostgreSQL auto-sync")
    args = parser.parse_args()

    new = score_pending(limit=args.limit)
    print(f"Scored {len(new)} news")
    if len(new):
        print(new["label"].value_counts().to_dict())

    days = trading_days()
    if not days:
        print("No candles yet: daily index needs trading days (run load_candles.py)")
    else:
        daily = daily_index(read_news(), read_scores(), days)
        daily.to_csv(config.DAILY_SENTIMENT_CSV, index=False)
        print(f"Daily index: {len(daily)} rows -> {config.DAILY_SENTIMENT_CSV}")
        print(daily.groupby("ticker")["n_news"].sum().to_dict())
    if not args.no_sync:
        auto_sync("sentiment")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    main()
