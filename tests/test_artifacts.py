"""Regression checks for saved model artifacts (run without torch): metrics.json vs test_predictions.csv."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from invest_ai.forecast.metrics import evaluate

ROOT = Path(__file__).resolve().parents[1]
MODELS = [ROOT / "models" / "lstm_sber_f3", ROOT / "models" / "lstm_sber_f4"]
PERIOD_DAYS = {"1w": 7, "1m": 30, "3m": 91, "6m": 182, "1y": 365}  # same as web/app.py


def _load(model_dir: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(model_dir / "test_predictions.csv", parse_dates=["target_date"])
    return df, json.loads((model_dir / "metrics.json").read_text("utf-8"))


def _metrics(df: pd.DataFrame) -> dict:
    return evaluate(*(df[c].to_numpy(float) for c in ("actual", "pred", "close_prev")))


@pytest.mark.parametrize("model_dir", MODELS, ids=lambda p: p.name)
def test_saved_metrics_match_predictions(model_dir):
    df, saved = _load(model_dir)
    fresh = _metrics(df)
    assert fresh["n_test"] == saved["n_test"] == len(df)
    for key in ("mae", "rmse", "da", "rmse_rel", "naive_mae", "naive_rmse", "da_same_as_yesterday", "da_always_up"):
        assert np.isclose(fresh[key], saved[key], rtol=1e-9), key


@pytest.mark.parametrize("model_dir", MODELS, ids=lambda p: p.name)
def test_predictions_are_clean_and_chronological(model_dir):
    df, _ = _load(model_dir)
    assert df["target_date"].is_monotonic_increasing and df["target_date"].is_unique
    assert not df[["actual", "pred", "close_prev"]].isna().any().any()
    assert (df[["actual", "pred", "close_prev"]] > 0).all().all()
    # close_prev of day t is the actual close of day t-1: no gaps inside the test period
    assert np.allclose(df["close_prev"].to_numpy()[1:], df["actual"].to_numpy()[:-1])


def test_both_models_share_test_period_and_naive_baseline():
    (d3, m3), (d4, m4) = _load(MODELS[0]), _load(MODELS[1])
    assert list(d3["target_date"]) == list(d4["target_date"])
    assert np.allclose(d3["actual"], d4["actual"])
    assert m3["naive_rmse"] == m4["naive_rmse"] and m3["naive_mae"] == m4["naive_mae"]


def test_hypothesis_report_matches_metrics():
    report = json.loads((ROOT / "reports" / "hypothesis_sber.json").read_text("utf-8"))
    (_, m3), (_, m4) = _load(MODELS[0]), _load(MODELS[1])
    improved = m4["rmse"] < m3["rmse"] and m4["da"] > m3["da"]
    assert report["improved_by_sentiment"] == improved


def test_period_filter_full_test_for_1y_and_subset_for_3m():
    """Quality screen: '1y' covers the whole test sample, '3m' only its last part."""
    df, saved = _load(MODELS[1])
    last = df["target_date"].max()
    sizes = {p: int((df["target_date"] > last - pd.Timedelta(days=d)).sum()) for p, d in PERIOD_DAYS.items()}
    assert sizes["1y"] == saved["n_test"] == len(df)
    assert 0 < sizes["3m"] < sizes["1y"]
    print("days per period:", sizes)
