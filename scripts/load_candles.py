"""Load daily candles from MOEX ISS into data/raw/candles/{TICKER}.csv.

Usage: python scripts/load_candles.py [--tickers SBER GAZP] [--years 3]
"""

import argparse
import logging
from datetime import date, timedelta

from invest_ai import config
from invest_ai.db import auto_sync
from invest_ai.etl.moex_loader import fetch_candles, upsert_candles_csv, validate_candles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=list(config.tickers()))
    parser.add_argument("--years", type=int, default=config.settings()["moex"]["history_years"])
    parser.add_argument("--no-sync", action="store_true", help="skip PostgreSQL auto-sync")
    args = parser.parse_args()

    till = date.today()
    since = till - timedelta(days=365 * args.years)
    for ticker in args.tickers:
        raw = fetch_candles(ticker, since, till)
        valid, rejected = validate_candles(raw)
        if len(rejected):
            logging.warning("%s: rejected %d rows: %s", ticker, len(rejected),
                            rejected["reason"].value_counts().to_dict())
        saved = upsert_candles_csv(ticker, valid)
        print(f"{ticker}: {len(valid)} candles loaded, {len(saved)} stored "
              f"({saved['trade_date'].min()} .. {saved['trade_date'].max()})")
    if not args.no_sync:
        auto_sync("candles")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
