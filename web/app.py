"""Local web UI for invest-ai-platform pipeline and results."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pandas.tseries.offsets import BDay

from invest_ai import config
from invest_ai.forecast.metrics import evaluate

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable
SENTIMENT_METRICS_JSON = config.REPORTS_DIR / "sentiment_metrics.json"

PERIOD_DAYS = {"1w": 7, "1m": 30, "3m": 91, "6m": 182, "1y": 365}
TONES = ("positive", "neutral", "negative")
DB_PING_TTL = 15.0

app = FastAPI(title="invest-ai", version="0.2.0")
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))

_job_lock = threading.Lock()
_job: dict = {"running": False, "step": None, "log": "", "started_at": None, "finished_at": None, "ok": None}


def _run_script(script: str, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PYTHON, str(ROOT / "scripts" / script), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _append_log(text: str) -> None:
    _job["log"] = (_job["log"] + text)[-50_000:]


def _start_job(step: str, script: str, args: list[str]) -> None:
    with _job_lock:
        if _job["running"]:
            raise HTTPException(409, "Уже выполняется другая задача")
        _job.update({
            "running": True, "step": step, "log": "", "ok": None,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": None,
        })

    def worker() -> None:
        try:
            _append_log(f"$ python scripts/{script} {' '.join(args)}\n")
            proc = _run_script(script, args)
            _append_log(proc.stdout or "")
            if proc.stderr:
                _append_log(proc.stderr)
            _job["ok"] = proc.returncode == 0
            if proc.returncode != 0:
                _append_log(f"\nexit code: {proc.returncode}\n")
        except Exception as exc:  # noqa: BLE001
            _job["ok"] = False
            _append_log(f"\nERROR: {exc}\n")
        finally:
            _job["finished_at"] = datetime.now().isoformat(timespec="seconds")
            _job["running"] = False

    threading.Thread(target=worker, daemon=True).start()


# --- data access (CSV artifacts, cached by file mtime) -----------------------

_cache: dict[str, tuple] = {}


def _cached(key: str, paths: tuple[Path, ...], loader: Callable):
    stamp = tuple(p.stat().st_mtime if p.exists() else None for p in paths)
    hit = _cache.get(key)
    if hit is None or hit[0] != stamp:
        hit = (stamp, loader())
        _cache[key] = hit
    return hit[1]


def _days(period: str) -> int:
    if period not in PERIOD_DAYS:
        raise HTTPException(400, f"Unknown period: {period}")
    return PERIOD_DAYS[period]


def _check_ticker(ticker: str) -> str:
    ticker = ticker.upper()
    if ticker not in config.tickers():
        raise HTTPException(404, f"Unknown ticker: {ticker}")
    return ticker


def _num(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else value


def _candles(ticker: str) -> pd.DataFrame | None:
    path = config.CANDLES_DIR / f"{ticker}.csv"

    def load() -> pd.DataFrame | None:
        if not path.exists():
            return None
        df = pd.read_csv(path, parse_dates=["trade_date"])
        return df.sort_values("trade_date", ignore_index=True)

    return _cached(f"candles:{ticker}", (path,), load)


def _news() -> pd.DataFrame:
    """News joined with their sentiment label/score, sorted by publication time (MSK)."""

    def load() -> pd.DataFrame:
        cols = ["id", "source", "url", "title", "published_at", "lang", "ticker"]
        if not config.NEWS_CSV.exists():
            return pd.DataFrame(columns=[*cols, "label", "score"])
        news = pd.read_csv(config.NEWS_CSV, usecols=cols)
        news["published_at"] = pd.to_datetime(news["published_at"], utc=True).dt.tz_convert(config.tz())
        if config.SENTIMENT_CSV.exists():
            scores = pd.read_csv(config.SENTIMENT_CSV, usecols=["news_id", "label", "score"])
            scores = scores.drop_duplicates("news_id", keep="last")
            news = news.merge(scores, left_on="id", right_on="news_id", how="left").drop(columns="news_id")
        else:
            news["label"], news["score"] = None, float("nan")
        return news.sort_values("published_at", ignore_index=True)

    return _cached("news", (config.NEWS_CSV, config.SENTIMENT_CSV), load)


def _recent_news(days: int) -> pd.DataFrame:
    """News of the last `days` days, counted back from the latest collected news."""
    news = _news()
    if news.empty:
        return news
    return news[news["published_at"] > news["published_at"].max() - pd.Timedelta(days=days)]


def _text(value) -> str | None:
    return value if isinstance(value, str) else None


def _news_items(df: pd.DataFrame) -> list[dict]:
    return [{
        "id": int(r.id),
        "title": _text(r.title) or "",
        "source": _text(r.source),
        "url": _text(r.url),
        "published_at": r.published_at.isoformat(),
        "ticker": _text(r.ticker),
        "lang": _text(r.lang),
        "tone": _text(r.label) or "none",
        "score": _num(r.score),
    } for r in df.itertuples()]


def _forecasts() -> pd.DataFrame:
    def load() -> pd.DataFrame:
        if not config.FORECASTS_CSV.exists():
            return pd.DataFrame(columns=["ticker", "forecast_date", "target_date", "pred_close",
                                         "pred_direction", "model_version", "use_sentiment"])
        df = pd.read_csv(config.FORECASTS_CSV)
        df["use_sentiment"] = df["use_sentiment"].map(lambda v: str(v).lower() in {"1", "true", "t", "yes"})
        return df

    return _cached("forecasts", (config.FORECASTS_CSV,), load)


def _latest_forecast(ticker: str, features: int | None = None) -> dict | None:
    """Latest t+1 forecast; without `features` F=4 wins over F=3 on the same date."""
    df = _forecasts()
    df = df[df["ticker"] == ticker]
    if features is not None:
        df = df[df["use_sentiment"] == (features == 4)]
    if df.empty:
        return None
    r = df.sort_values(["forecast_date", "use_sentiment"]).iloc[-1]
    return {
        "forecast_date": r.forecast_date,
        "target_date": r.target_date,
        "pred_close": float(r.pred_close),
        "pred_direction": int(r.pred_direction),
        "features": 4 if r.use_sentiment else 3,
        "model_version": r.model_version,
    }


def _model_dir(ticker: str, features: int) -> Path:
    return config.MODELS_DIR / f"lstm_{ticker.lower()}_f{features}"


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text("utf-8-sig")) if path.exists() else None


def _price_change(candles: pd.DataFrame, days: int) -> tuple[pd.DataFrame, float | None]:
    """Candles of the period and close-to-close change (%) against the close before it."""
    start = candles["trade_date"].max() - pd.Timedelta(days=days)
    window = candles[candles["trade_date"] > start]
    before = candles[candles["trade_date"] <= start]
    base = before["close"].iloc[-1] if not before.empty else window["close"].iloc[0]
    return window, float(100 * (window["close"].iloc[-1] / base - 1))


def _data_freshness() -> dict:
    files = [*config.CANDLES_DIR.glob("*.csv"), config.NEWS_CSV, config.SENTIMENT_CSV, config.FORECASTS_CSV]
    mtimes = [p.stat().st_mtime for p in files if p.exists()]
    updated = datetime.fromtimestamp(max(mtimes), tz=config.tz()) if mtimes else None
    last_dates = [c["trade_date"].max() for t in config.tickers()
                  if (c := _candles(t)) is not None and not c.empty]
    last = max(last_dates) if last_dates else None
    expected = pd.Timestamp.now(tz=config.tz()).normalize().tz_localize(None) - BDay(1)
    return {
        "updated_at": updated.isoformat(timespec="minutes") if updated else None,
        "last_market_date": last.strftime("%Y-%m-%d") if last is not None else None,
        "stale": last is None or last < expected,
    }


# --- pages & API ---------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    # A list keeps the tickers.yaml order (tojson would sort dict keys).
    names = [[t, meta["name"]] for t, meta in config.tickers().items()]
    version = int(max(p.stat().st_mtime for p in (WEB_DIR / "static").iterdir()))
    return templates.TemplateResponse(request, "index.html", {"tickers": names, "version": version})


_db_status: dict = {"checked": 0.0, "value": None, "refreshing": False}


def _refresh_db_status() -> None:
    from invest_ai.db import ping
    try:
        _db_status.update(value=ping(), checked=time.monotonic())
    finally:
        _db_status["refreshing"] = False


def _db_ping(recheck: bool) -> dict:
    """Postgres health; an unreachable DB costs seconds, so expired values refresh in background."""
    if recheck or _db_status["value"] is None:
        _refresh_db_status()
    elif time.monotonic() - _db_status["checked"] > DB_PING_TTL and not _db_status["refreshing"]:
        _db_status["refreshing"] = True
        threading.Thread(target=_refresh_db_status, daemon=True).start()
    return _db_status["value"]


@app.get("/api/status")
def status(recheck: bool = False) -> dict:
    return {
        "job": {k: _job[k] for k in ("running", "step", "started_at", "finished_at", "ok")},
        "has_candles": any(config.CANDLES_DIR.glob("*.csv")) if config.CANDLES_DIR.exists() else False,
        "has_news": config.NEWS_CSV.exists(),
        "has_sentiment": config.SENTIMENT_CSV.exists(),
        "has_forecasts": config.FORECASTS_CSV.exists(),
        "models": sorted(p.name for p in config.MODELS_DIR.glob("lstm_*") if (p / "model.pt").exists()),
        "db": _db_ping(recheck),
        **_data_freshness(),
    }


@app.get("/api/job/log")
def job_log() -> dict:
    return {"running": _job["running"], "step": _job["step"], "ok": _job["ok"], "log": _job["log"]}


@app.post("/api/run/pipeline")
def run_pipeline(ticker: str = "SBER", limit: int = 50) -> JSONResponse:
    with _job_lock:
        if _job["running"]:
            raise HTTPException(409, "Уже выполняется другая задача")
        _job.update({
            "running": True, "step": "pipeline", "log": "", "ok": None,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": None,
        })

    # Each data step auto-syncs CSV → Postgres (AUTO_SYNC_DB=1).
    steps = [
        ("candles", "load_candles.py", ["--tickers", ticker]),
        ("news", "load_news.py", []),
        ("sentiment", "score_sentiment.py", ["--limit", str(limit)]),
        ("train", "train_lstm.py", ["--ticker", ticker]),
        ("predict", "predict.py", ["--ticker", ticker]),
    ]

    def worker() -> None:
        ok = True
        try:
            for name, script, args in steps:
                _job["step"] = name
                _append_log(f"\n=== {name} ===\n$ python scripts/{script} {' '.join(args)}\n")
                proc = _run_script(script, args)
                _append_log(proc.stdout or "")
                if proc.stderr:
                    _append_log(proc.stderr)
                if proc.returncode != 0:
                    ok = False
                    _append_log(f"\nFAILED at {name}, exit {proc.returncode}\n")
                    break
            _job["ok"] = ok
        except Exception as exc:  # noqa: BLE001
            _job["ok"] = False
            _append_log(f"\nERROR: {exc}\n")
        finally:
            _job["finished_at"] = datetime.now().isoformat(timespec="seconds")
            _job["running"] = False

    threading.Thread(target=worker, daemon=True).start()
    return JSONResponse({"started": "pipeline", "ticker": ticker})


@app.post("/api/run/{step}")
def run_step(step: str, ticker: str = "SBER", limit: int = 50) -> JSONResponse:
    mapping = {
        "candles": ("load_candles.py", ["--tickers", ticker]),
        "news": ("load_news.py", []),
        "sentiment": ("score_sentiment.py", ["--limit", str(limit)]),
        "train": ("train_lstm.py", ["--ticker", ticker]),
        "predict": ("predict.py", ["--ticker", ticker]),
        "sync_db": ("sync_db.py", []),
    }
    if step not in mapping:
        raise HTTPException(404, f"Unknown step: {step}")
    script, args = mapping[step]
    _start_job(step, script, args)
    return JSONResponse({"started": step, "ticker": ticker})


@app.get("/api/overview")
def overview(period: str = "3m") -> dict:
    """Screen 1 (dashboard): instruments table, summary cards, latest news."""
    days = _days(period)
    recent = _recent_news(days)
    rows = []
    for ticker, meta in config.tickers().items():
        candles = _candles(ticker)
        price = change = None
        if candles is not None and not candles.empty:
            window, change = _price_change(candles, days)
            price = float(window["close"].iloc[-1])
        scores = recent.loc[recent["ticker"] == ticker, "score"].dropna()
        rows.append({
            "ticker": ticker,
            "name": meta["name"],
            "last_price": price,
            "change_pct": change,
            "tone": _num(scores.mean()) if len(scores) else None,
            "n_news": int(len(scores)),
            "forecast": _latest_forecast(ticker),
        })
    with_forecast = [r for r in rows if r["forecast"]]
    company_news = _news()
    company_news = company_news[company_news["ticker"].notna()]
    tracked_scores = recent.loc[recent["ticker"].notna(), "score"].dropna()
    return {
        "period": period,
        "tickers": rows,
        "summary": {
            "tracked": len(rows),
            "avg_tone": _num(tracked_scores.mean()) if len(tracked_scores) else None,
            "n_forecast": len(with_forecast),
            "n_up": sum(r["forecast"]["pred_direction"] == 1 for r in with_forecast),
        },
        "latest_news": _news_items(company_news.tail(5).iloc[::-1]),
    }


@app.get("/api/ticker/{ticker}")
def ticker_view(ticker: str, period: str = "3m", features: int = 4) -> dict:
    """Screen 2: candles + volume, t+1 forecast, ticker news."""
    ticker, days = _check_ticker(ticker), _days(period)
    candles = _candles(ticker)
    out: dict = {"ticker": ticker, "name": config.tickers()[ticker]["name"], "candles": None,
                 "last_price": None, "change_pct": None}
    if candles is not None and not candles.empty:
        window, out["change_pct"] = _price_change(candles, days)
        out["last_price"] = float(window["close"].iloc[-1])
        out["candles"] = {
            "dates": window["trade_date"].dt.strftime("%Y-%m-%d").tolist(),
            **{col: window[col].astype(float).tolist() for col in ("open", "high", "low", "close", "volume")},
        }

    out["trained"] = (_model_dir(ticker, features) / "metrics.json").exists()
    forecast = _latest_forecast(ticker, features) if out["trained"] else None
    if forecast and candles is not None:
        base = candles.loc[candles["trade_date"] <= pd.Timestamp(forecast["forecast_date"]), "close"]
        if not base.empty:
            forecast["change_pct"] = float(100 * (forecast["pred_close"] / base.iloc[-1] - 1))
    out["forecast"] = forecast

    recent = _recent_news(days)
    scores = recent.loc[recent["ticker"] == ticker, "score"].dropna()
    out["tone"] = _num(scores.mean()) if len(scores) else None
    news = _news()
    out["news"] = _news_items(news[news["ticker"] == ticker].tail(5).iloc[::-1])
    return out


@app.get("/api/news")
def news_feed(ticker: str = "ALL", period: str = "1m", tones: str = ",".join(TONES),
              offset: int = 0, limit: int = 40) -> dict:
    """Screen 3: tone distribution, daily mean tone, news feed (newest first)."""
    items = _recent_news(_days(period))
    if ticker.upper() != "ALL":
        items = items[items["ticker"] == _check_ticker(ticker)]
    wanted = [t for t in tones.split(",") if t in TONES]
    items = items[items["label"].isna() | items["label"].isin(wanted)]
    scored = items.dropna(subset=["score"])
    daily = scored.groupby(scored["published_at"].dt.strftime("%Y-%m-%d"))["score"].mean()
    page = items.iloc[::-1].iloc[offset:offset + limit]
    return {
        "total": int(len(items)),
        "counts": {t: int((items["label"] == t).sum()) for t in TONES},
        "daily": {"dates": daily.index.tolist(), "tone": [float(v) for v in daily]},
        "items": _news_items(page),
    }


@app.get("/api/quality/{ticker}")
def quality(ticker: str, features: int = 4, period: str = "3m") -> dict:
    """Screen 4: LSTM test metrics over the evaluation period + sentiment model quality."""
    ticker, days = _check_ticker(ticker), _days(period)
    model_dir = _model_dir(ticker, features)
    out: dict = {
        "ticker": ticker, "features": features,
        "trained": (model_dir / "metrics.json").exists(),
        "hypothesis": _read_json(config.REPORTS_DIR / f"hypothesis_{ticker.lower()}.json"),
        "sentiment": _read_json(SENTIMENT_METRICS_JSON),
    }
    if not out["trained"]:
        return out
    out["config"] = _read_json(model_dir / "config.json") or {}

    path = model_dir / "test_predictions.csv"
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if df.empty:
        out["metrics"], out["series"] = None, None
        return out
    date_col = "target_date" if "target_date" in df.columns else "trade_date"
    actual_col = "actual" if "actual" in df.columns else "y_true"
    pred_col = "pred" if "pred" in df.columns else "y_pred"
    df[date_col] = pd.to_datetime(df[date_col])
    df = df[df[date_col] > df[date_col].max() - pd.Timedelta(days=days)]

    actual, pred, prev = (df[c].to_numpy(float) for c in (actual_col, pred_col, "close_prev"))
    metrics = evaluate(actual, pred, prev) if len(df) >= 2 else None
    out["metrics"] = {k: _num(v) for k, v in metrics.items()} if metrics else None
    out["series"] = {
        "dates": df[date_col].dt.strftime("%Y-%m-%d").tolist(),
        "actual": actual.tolist(),
        "pred": pred.tolist(),
        "error": (actual - pred).tolist(),
    }
    return out


@app.get("/api/candles/{ticker}")
def candles(ticker: str, last: int = 120) -> dict:
    df = _candles(ticker.upper())
    if df is None:
        raise HTTPException(404, f"No candles for {ticker}")
    df = df.tail(last)
    return {
        "ticker": ticker.upper(),
        "dates": df["trade_date"].dt.strftime("%Y-%m-%d").tolist(),
        "close": df["close"].astype(float).tolist(),
        "volume": df["volume"].astype(float).tolist(),
    }


@app.get("/api/forecasts")
def forecasts(ticker: str | None = None) -> list[dict]:
    df = _forecasts()
    if ticker:
        df = df[df["ticker"] == ticker.upper()]
    return df.sort_values(["ticker", "target_date"]).tail(50).to_dict(orient="records")
