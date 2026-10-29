"""Forecast close for t+1 with trained models and write data/processed/forecasts.csv.

Usage: python scripts/predict.py --ticker SBER [--features 3 4]
"""

import argparse

from invest_ai.db import auto_sync
from invest_ai.forecast.predict import predict, upsert_forecasts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SBER")
    parser.add_argument("--features", nargs="+", type=int, choices=[3, 4], default=[3, 4])
    parser.add_argument("--no-sync", action="store_true", help="skip PostgreSQL auto-sync")
    args = parser.parse_args()

    rows = []
    for f in args.features:
        try:
            rows.append(predict(args.ticker, use_sentiment=(f == 4)))
        except FileNotFoundError as exc:
            print(f"F={f}: {exc}")
    for r in rows:
        arrow = "up" if r["pred_direction"] else "down/flat"
        print(f"{r['ticker']} F={4 if r['use_sentiment'] else 3}: {r['forecast_date']} -> "
              f"{r['target_date']}: {r['pred_close']:.2f} ({arrow}), {r['model_version']}")
    if rows:
        upsert_forecasts(rows)
    if not args.no_sync:
        auto_sync("predict")


if __name__ == "__main__":
    main()
