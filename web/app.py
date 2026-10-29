"""Local web UI for invest-ai-platform pipeline and results."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from invest_ai import config

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable

app = FastAPI(title="invest-ai", version="0.1.0")
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


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", {"tickers": list(config.tickers())})


@app.get("/api/status")
def status() -> dict:
    from invest_ai.db import ping
    return {
        "job": {k: _job[k] for k in ("running", "step", "started_at", "finished_at", "ok")},
        "has_candles": any(config.CANDLES_DIR.glob("*.csv")) if config.CANDLES_DIR.exists() else False,
        "has_news": config.NEWS_CSV.exists(),
        "has_sentiment": config.SENTIMENT_CSV.exists(),
        "has_forecasts": config.FORECASTS_CSV.exists(),
        "models": sorted(p.name for p in config.MODELS_DIR.glob("lstm_*") if (p / "model.pt").exists()),
        "db": ping(),
    }


@app.get("/api/job/log")
def job_log() -> dict:
    return {"running": _job["running"], "step": _job["step"], "ok": _job["ok"], "log": _job["log"]}


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


@app.get("/api/candles/{ticker}")
def candles(ticker: str, last: int = 120) -> dict:
    path = config.CANDLES_DIR / f"{ticker.upper()}.csv"
    if not path.exists():
        raise HTTPException(404, f"No candles for {ticker}")
    df = pd.read_csv(path, parse_dates=["trade_date"]).tail(last)
    return {
        "ticker": ticker.upper(),
        "dates": df["trade_date"].dt.strftime("%Y-%m-%d").tolist(),
        "close": df["close"].astype(float).tolist(),
        "volume": df["volume"].astype(float).tolist(),
    }


@app.get("/api/forecasts")
def forecasts(ticker: str | None = None) -> list[dict]:
    if not config.FORECASTS_CSV.exists():
        return []
    df = pd.read_csv(config.FORECASTS_CSV)
    if ticker:
        df = df[df["ticker"] == ticker.upper()]
    return df.sort_values(["ticker", "target_date"]).tail(50).to_dict(orient="records")


@app.get("/api/metrics/{ticker}")
def metrics(ticker: str) -> dict:
    out: dict = {"ticker": ticker.upper(), "models": {}, "hypothesis": None}
    for f in (3, 4):
        path = config.MODELS_DIR / f"lstm_{ticker.lower()}_f{f}" / "metrics.json"
        if path.exists():
            out["models"][f"f{f}"] = json.loads(path.read_text("utf-8"))
    hyp = config.REPORTS_DIR / f"hypothesis_{ticker.lower()}.json"
    if hyp.exists():
        out["hypothesis"] = json.loads(hyp.read_text("utf-8"))
    return out


@app.get("/api/news")
def news(limit: int = 30) -> list[dict]:
    if not config.NEWS_CSV.exists():
        return []
    df = pd.read_csv(config.NEWS_CSV).tail(limit)
    cols = [c for c in ("published_at", "source", "ticker", "title", "lang") if c in df.columns]
    return df[cols].fillna("").iloc[::-1].to_dict(orient="records")


@app.get("/api/predictions/{ticker}")
def test_predictions(ticker: str, features: int = 4) -> dict:
    path = config.MODELS_DIR / f"lstm_{ticker.lower()}_f{features}" / "test_predictions.csv"
    if not path.exists():
        raise HTTPException(404, "No test predictions")
    df = pd.read_csv(path)
    date_col = "target_date" if "target_date" in df.columns else "trade_date"
    y_true = "actual" if "actual" in df.columns else "y_true"
    y_pred = "pred" if "pred" in df.columns else "y_pred"
    dates = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d").tolist()
    naive = df["close_prev"].astype(float).tolist() if "close_prev" in df.columns else []
    return {
        "dates": dates,
        "y_true": df[y_true].astype(float).tolist(),
        "y_pred": df[y_pred].astype(float).tolist(),
        "naive": naive,
    }
