"""Train LSTM F=3 (prices) and/or F=4 (prices + sentiment), compare with the naive forecast.

Usage: python scripts/train_lstm.py --ticker SBER [--features 3 4]
Outputs: models/lstm_{ticker}_f{F}/, reports/hypothesis_{ticker}.json
"""

import argparse
import json
import logging

from invest_ai import config
from invest_ai.forecast.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SBER")
    parser.add_argument("--features", nargs="+", type=int, choices=[3, 4], default=[3, 4])
    args = parser.parse_args()

    results = {f: train(args.ticker, use_sentiment=(f == 4)) for f in args.features}

    m0 = next(iter(results.values()))["metrics"]
    print(f"\n{args.ticker}: test {m0['n_test']} days")
    print(f"{'model':<8}{'MAE':>10}{'RMSE':>10}{'DA, %':>9}{'RMSE_rel':>10}")
    print(f"{'naive':<8}{m0['naive_mae']:>10.3f}{m0['naive_rmse']:>10.3f}{'-':>9}{1:>10.3f}")
    for f, r in results.items():
        m = r["metrics"]
        print(f"{'F=' + str(f):<8}{m['mae']:>10.3f}{m['rmse']:>10.3f}{m['da']:>9.1f}{m['rmse_rel']:>10.3f}")
    print(f"DA baselines: same as yesterday {m0['da_same_as_yesterday']:.1f}%, "
          f"always up {m0['da_always_up']:.1f}%")

    if 3 in results and 4 in results:
        m3, m4 = results[3]["metrics"], results[4]["metrics"]
        improved = m4["rmse"] < m3["rmse"] and m4["da"] > m3["da"]
        verdict = {
            "ticker": args.ticker,
            "improved_by_sentiment": improved,
            "rmse_f3": m3["rmse"], "rmse_f4": m4["rmse"],
            "da_f3": m3["da"], "da_f4": m4["da"],
            "rmse_rel_f3": m3["rmse_rel"], "rmse_rel_f4": m4["rmse_rel"],
            "news_days_with_sentiment": results[4]["config"]["news_days_with_sentiment"],
            "test_period": results[4]["config"]["test_period"],
        }
        config.REPORTS_DIR.mkdir(exist_ok=True)
        out = config.REPORTS_DIR / f"hypothesis_{args.ticker.lower()}.json"
        out.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), "utf-8")
        print(f"Sentiment improved (RMSE down and DA up): {improved} -> {out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
