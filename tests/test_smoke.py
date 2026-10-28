from datetime import date

import numpy as np
import pandas as pd

from invest_ai.etl.moex_loader import validate_candles
from invest_ai.etl.news_collector import append_news, content_hash
from invest_ai.etl.ticker_matcher import match_ticker
from invest_ai.forecast.features import build_frame, make_dataset
from invest_ai.forecast.metrics import evaluate
from invest_ai.sentiment.aggregate import MARKET, assign_trading_day, daily_index
from invest_ai.sentiment.preprocess import clean_text, detect_lang


def test_ohlc_validation_rejects_inconsistent_rows():
    df = pd.DataFrame({
        "ticker": ["SBER"] * 3,
        "trade_date": [date(2026, 1, 12), date(2026, 1, 13), date(2026, 1, 14)],
        "open": [100, 100, -1], "high": [110, 90, 110], "low": [95, 95, 95],
        "close": [105, 100, 100], "volume": [10, 10, 10], "value": [1e3, 1e3, 1e3],
    })
    valid, rejected = validate_candles(df)
    assert len(valid) == 1
    assert set(rejected["trade_date"]) == {date(2026, 1, 13), date(2026, 1, 14)}


def test_news_dedup_by_url_and_hash(tmp_path, monkeypatch):
    from invest_ai import config
    monkeypatch.setattr(config, "NEWS_CSV", tmp_path / "news.csv")
    ts = pd.Timestamp("2026-10-27 12:00", tz="Europe/Moscow").to_pydatetime()
    item = {"source": "t", "url": "u1", "title": "Сбербанк вырос", "body": None,
            "published_at": ts.isoformat(), "lang": None, "ticker": "SBER",
            "content_hash": content_hash("Сбербанк вырос", ts), "loaded_at": ts.isoformat()}
    assert append_news([item]) == 1
    assert append_news([item]) == 0
    assert append_news([{**item, "url": "u2"}]) == 0  # reprint: same hash


def test_ticker_matcher_single_issuer_only():
    assert match_ticker("Сбербанк отчитался о прибыли") == "SBER"
    assert match_ticker("Сбербанк и Газпром заключили соглашение") is None
    assert match_ticker("Сбережения граждан выросли") is None


def test_lang_and_cleaning():
    assert detect_lang("Прибыль Сбербанка выросла") == "ru"
    assert detect_lang("Sberbank profit rose") == "en"
    assert detect_lang("12345") is None
    assert clean_text("<p>Прибыль  выросла на 5 %</p> https://x.ru") == "Прибыль выросла на 5 %"


def test_trading_day_assignment_respects_cutoff():
    days = [date(2026, 10, 23), date(2026, 10, 26), date(2026, 10, 27)]  # Fri, Mon, Tue
    published = pd.Series([
        "2026-10-26T18:59:00+03:00",  # before 19:00 -> same day
        "2026-10-26T19:30:00+03:00",  # evening session -> next trading day
        "2026-10-24T12:00:00+03:00",  # Saturday -> Monday
        "2026-10-27T20:00:00+03:00",  # after the last known day -> next weekday
    ])
    assert list(assign_trading_day(published, days)) == [
        date(2026, 10, 26), date(2026, 10, 27), date(2026, 10, 26), date(2026, 10, 28)]


def test_daily_index_mean_and_market_series():
    news = pd.DataFrame({"id": [1, 2, 3], "ticker": ["SBER", "SBER", None],
                         "published_at": ["2026-10-26T10:00:00+03:00"] * 3})
    scores = pd.DataFrame({"news_id": [1, 2, 3], "score": [0.8, -0.2, 0.5]})
    daily = daily_index(news, scores, [date(2026, 10, 26)])
    sber = daily[daily["ticker"] == "SBER"].iloc[0]
    assert np.isclose(sber["sentiment"], 0.3) and sber["n_news"] == 2
    assert (daily["ticker"] == MARKET).sum() == 1


def test_split_has_no_lookahead():
    n = 200
    candles = pd.DataFrame({"trade_date": pd.bdate_range("2025-01-01", periods=n),
                            "close": np.linspace(100, 300, n), "volume": np.full(n, 1000)})
    frame = build_frame(candles, None, use_sentiment=False)
    ds = make_dataset(frame, ["close", "volume", "ret"], window=30, ratios=[0.7, 0.15, 0.15])
    tr, va, te = ds.splits["train"], ds.splits["val"], ds.splits["test"]
    assert tr.stop == va.start and va.stop == te.start and te.stop == len(ds.y)
    assert ds.target_dates[tr][-1] < ds.target_dates[va][0] < ds.target_dates[te][0]
    # The scaler sees prices only up to the last train target.
    assert ds.scaler.data_max_[0] == ds.close_next[tr][-1]
    assert ds.y[te].max() > 1.0


def test_metrics_naive_baseline_rmse_rel_is_one():
    actual = np.array([101.0, 99.0, 102.0])
    prev = np.array([100.0, 101.0, 99.0])
    m = evaluate(actual, prev, prev)
    assert m["rmse_rel"] == 1.0
