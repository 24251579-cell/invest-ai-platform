"""RSS news collector: feedparser (+ETag/Last-Modified), dedup by url/content_hash."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone

import feedparser
import pandas as pd
from bs4 import BeautifulSoup

from invest_ai import config
from invest_ai.etl.ticker_matcher import match_ticker

log = logging.getLogger(__name__)

NEWS_COLUMNS = ["id", "source", "url", "title", "body", "published_at", "lang", "ticker",
                "content_hash", "loaded_at"]
STATE_PATH = config.RAW_DIR / "rss_state.json"


def strip_html(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", BeautifulSoup(text, "html.parser").get_text(" ")).strip()


def content_hash(title: str, published_at: datetime) -> str:
    """SHA-256 of normalized title + publication date (MSK)."""
    norm = re.sub(r"\s+", " ", title.lower()).strip()
    day = published_at.astimezone(config.tz()).date().isoformat()
    return hashlib.sha256(f"{norm}|{day}".encode("utf-8")).hexdigest()


def _published(entry) -> datetime | None:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed:
        return None
    return datetime(*parsed[:6], tzinfo=timezone.utc).astimezone(config.tz())


def fetch_source(source: dict, state: dict) -> list[dict]:
    prev = state.get(source["id"], {})
    feed = feedparser.parse(source["url"], etag=prev.get("etag"), modified=prev.get("modified"))
    if getattr(feed, "status", None) == 304:
        log.info("%s: not modified", source["id"])
        return []
    if feed.bozo and not feed.entries:
        log.error("%s: feed error %s", source["id"], feed.bozo_exception)
        return []
    state[source["id"]] = {"etag": feed.get("etag"), "modified": feed.get("modified")}

    now = datetime.now(config.tz())
    items = []
    for entry in feed.entries:
        url = entry.get("link")
        title = strip_html(entry.get("title"))
        published_at = _published(entry)
        if not url or not title or published_at is None:
            log.warning("%s: rejected entry without url/title/date", source["id"])
            continue
        body = strip_html(entry.get("summary") or entry.get("description")) or None
        items.append({
            "source": source["id"],
            "url": url,
            "title": title,
            "body": body,
            "published_at": published_at.isoformat(),
            "lang": None,
            "ticker": match_ticker(f"{title} {body or ''}"),
            "content_hash": content_hash(title, published_at),
            "loaded_at": now.isoformat(),
        })
    return items


def read_news() -> pd.DataFrame:
    if not config.NEWS_CSV.exists():
        return pd.DataFrame(columns=NEWS_COLUMNS)
    return pd.read_csv(config.NEWS_CSV, dtype={"ticker": "string", "lang": "string"})


def append_news(items: list[dict]) -> int:
    """Insert new items; existing url or content_hash are skipped. Returns inserted count."""
    old = read_news()
    new = pd.DataFrame(items, columns=NEWS_COLUMNS[1:])
    if new.empty:
        return 0
    new = new.drop_duplicates("url").drop_duplicates("content_hash")
    new = new[~new["url"].isin(old["url"]) & ~new["content_hash"].isin(old["content_hash"])]
    if new.empty:
        return 0
    start_id = int(old["id"].max()) + 1 if len(old) else 1
    new.insert(0, "id", range(start_id, start_id + len(new)))
    config.NEWS_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([old, new], ignore_index=True).to_csv(config.NEWS_CSV, index=False)
    return len(new)


def collect_all() -> int:
    state = json.loads(STATE_PATH.read_text("utf-8")) if STATE_PATH.exists() else {}
    items: list[dict] = []
    for source in config.rss_sources():
        try:
            got = fetch_source(source, state)
            log.info("%s: %d entries", source["id"], len(got))
            items.extend(got)
        except Exception:
            log.exception("%s: failed", source["id"])
    inserted = append_news(items)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), "utf-8")
    return inserted
