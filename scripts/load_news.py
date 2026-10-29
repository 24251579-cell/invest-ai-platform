"""Collect news from the RSS feeds in config/rss_sources.yaml into data/raw/news.csv.

Usage: python scripts/load_news.py
"""

import argparse
import logging

from invest_ai.db import auto_sync
from invest_ai.etl.news_collector import collect_all, read_news


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-sync", action="store_true", help="skip PostgreSQL auto-sync")
    args = parser.parse_args()

    inserted = collect_all()
    news = read_news()
    print(f"Inserted {inserted} news, total {len(news)}")
    print("By ticker:", news["ticker"].fillna("MARKET").value_counts().to_dict())
    if not args.no_sync:
        auto_sync("news")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
