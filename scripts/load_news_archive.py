"""Backfill news.csv from the Interfax day archive for the period of the candles.

Usage: python scripts/load_news_archive.py [--start 2023-10-01] [--end 2026-10-28] [--delay 1.0]
Interrupted runs continue from data/raw/archive_state.json.
"""

import argparse
import logging
from datetime import date

from invest_ai import config
from invest_ai.db import auto_sync
from invest_ai.etl.news_archive import collect_range
from invest_ai.etl.news_collector import read_news


def main() -> None:
    today = date.today()
    years = config.settings()["moex"]["history_years"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat,
                        default=today.replace(year=today.year - years))
    parser.add_argument("--end", type=date.fromisoformat, default=today)
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    parser.add_argument("--no-sync", action="store_true", help="skip PostgreSQL auto-sync")
    args = parser.parse_args()

    inserted = collect_range(args.start, args.end, args.delay)
    news = read_news()
    print(f"Inserted {inserted} news, total {len(news)}")
    print("By ticker:", news["ticker"].fillna("MARKET").value_counts().to_dict())
    if not args.no_sync:
        auto_sync("news")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
