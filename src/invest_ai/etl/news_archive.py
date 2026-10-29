"""Historical news from the Interfax day archive (/business/news/YYYY/MM/DD), titles only.

RSS feeds return only the latest items, so the archive backfills the period of the candles.
Progress is kept in data/raw/archive_state.json: a rerun skips the days already loaded.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timedelta

import requests
from bs4 import BeautifulSoup

from invest_ai import config
from invest_ai.etl.news_collector import append_news, content_hash, strip_html
from invest_ai.etl.ticker_matcher import match_ticker

log = logging.getLogger(__name__)

BASE_URL = "https://www.interfax.ru"
SECTION = "business"
SOURCE_ID = "interfax"
STATE_PATH = config.RAW_DIR / "archive_state.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (invest-ai-platform; student research)"}


def fetch_day(day: date, session: requests.Session, retries: int = 3) -> list[dict]:
    url = f"{BASE_URL}/{SECTION}/news/{day:%Y/%m/%d}"
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            break
        except requests.RequestException as exc:
            if attempt == retries:
                raise
            log.warning("%s attempt %d failed: %s", day, attempt, exc)
            time.sleep(5 * attempt)

    tz = config.tz()
    now = datetime.now(tz)
    items = []
    # Only the day list (div.an); the sidebar also has <h3> headlines from other days.
    for block in BeautifulSoup(resp.text, "html.parser").select("div.an div[data-id]"):
        link, clock = block.find("a", href=True), block.find("span")
        title = strip_html(link.get_text(" ")) if link else ""
        if not title or clock is None:
            continue
        try:
            hh, mm = map(int, clock.get_text(strip=True).split(":"))
        except ValueError:
            log.warning("%s: bad time %r", day, clock.get_text(strip=True))
            continue
        published_at = datetime(day.year, day.month, day.day, hh, mm, tzinfo=tz)
        href = link["href"]
        items.append({
            "source": SOURCE_ID,
            "url": href if href.startswith("http") else BASE_URL + href,
            "title": title,
            "body": None,
            "published_at": published_at.isoformat(),
            "lang": None,
            "ticker": match_ticker(title),
            "content_hash": content_hash(title, published_at),
            "loaded_at": now.isoformat(),
        })
    return items


def collect_range(start: date, end: date, delay: float = 1.0, save_every: int = 10) -> int:
    """Load every day in [start, end]; saves news and progress every `save_every` days."""
    done = set(json.loads(STATE_PATH.read_text("utf-8"))) if STATE_PATH.exists() else set()
    days = [start + timedelta(n) for n in range((end - start).days + 1)]
    todo = [d for d in days if d.isoformat() not in done]
    log.info("Archive %s..%s: %d days, %d already loaded", start, end, len(days), len(days) - len(todo))

    inserted, buffer, buffered_days = 0, [], []
    session = requests.Session()

    def flush() -> None:
        nonlocal inserted
        inserted += append_news(buffer)
        # Today's page is still growing, so it is never marked as done.
        done.update(d.isoformat() for d in buffered_days if d < date.today())
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(sorted(done), indent=0), "utf-8")
        buffer.clear()
        buffered_days.clear()

    for i, day in enumerate(todo, 1):
        try:
            buffer.extend(fetch_day(day, session))
            buffered_days.append(day)
        except requests.RequestException:
            log.exception("%s: skipped, will be retried on the next run", day)
        if i % save_every == 0 or i == len(todo):
            flush()
            log.info("%d / %d days, inserted %d news", i, len(todo), inserted)
        time.sleep(delay)
    return inserted
