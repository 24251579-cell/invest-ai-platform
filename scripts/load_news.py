"""Collect news from the RSS feeds in config/rss_sources.yaml into data/raw/news.csv.

Usage: python scripts/load_news.py
"""

import logging

from invest_ai.etl.news_collector import collect_all, read_news


def main() -> None:
    inserted = collect_all()
    news = read_news()
    print(f"Inserted {inserted} news, total {len(news)}")
    print("By ticker:", news["ticker"].fillna("MARKET").value_counts().to_dict())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
