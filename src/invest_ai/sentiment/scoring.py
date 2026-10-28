"""Step 5: score news without a score of the current model and store sentiment_scores."""

from __future__ import annotations

import gc
import logging
from datetime import datetime

import pandas as pd

from invest_ai import config
from invest_ai.etl.news_collector import read_news
from invest_ai.sentiment.model import SentimentModel
from invest_ai.sentiment.preprocess import detect_lang, model_input

log = logging.getLogger(__name__)

SCORE_COLUMNS = ["id", "news_id", "model_name", "label", "p_pos", "p_neg", "p_neu", "score",
                 "scored_at"]


def read_scores() -> pd.DataFrame:
    if not config.SENTIMENT_CSV.exists():
        return pd.DataFrame(columns=SCORE_COLUMNS)
    return pd.read_csv(config.SENTIMENT_CSV)


def validate_scores(df: pd.DataFrame) -> pd.DataFrame:
    probs = df[["p_pos", "p_neg", "p_neu"]]
    ok = (probs.ge(0) & probs.le(1)).all(axis=1)
    ok &= (probs.sum(axis=1) - 1).abs() <= 1e-4
    ok &= df["score"].between(-1, 1)
    ok &= df["label"].isin(["positive", "negative", "neutral"])
    if (~ok).any():
        log.error("Rejected %d invalid scores", int((~ok).sum()))
    return df[ok]


def score_pending(limit: int | None = None) -> pd.DataFrame:
    """Score news that have no score for their language's checkpoint. Returns new scores."""
    cfg = config.settings()["sentiment"]
    news = read_news()
    if news.empty:
        log.warning("news.csv is empty")
        return pd.DataFrame(columns=SCORE_COLUMNS)

    texts = [model_input(t, b if isinstance(b, str) else None)
             for t, b in zip(news["title"], news["body"])]
    news["lang"] = [detect_lang(t, cfg["lang_threshold"]) for t in texts]
    news["text"] = texts
    news.drop(columns="text").to_csv(config.NEWS_CSV, index=False)

    scores = read_scores()
    done = set(zip(scores["news_id"], scores["model_name"]))
    new_rows: list[dict] = []
    for lang, checkpoint in cfg["checkpoints"].items():
        pending = news["id"].map(lambda i: (i, checkpoint) not in done)
        todo = news[(news["lang"] == lang) & pending]
        if limit is not None:
            todo = todo.head(limit)
        if todo.empty:
            continue
        log.info("%s: scoring %d news with %s", lang, len(todo), checkpoint)
        model = SentimentModel(checkpoint)
        results = model.predict(todo["text"].tolist())
        now = datetime.now(config.tz()).isoformat()
        new_rows += [{"news_id": nid, "model_name": checkpoint, "label": r.label,
                      "p_pos": r.p_pos, "p_neg": r.p_neg, "p_neu": r.p_neu,
                      "score": r.score, "scored_at": now}
                     for nid, r in zip(todo["id"], results)]
        del model
        gc.collect()

    skipped = int(news["lang"].isna().sum())
    if skipped:
        log.info("Skipped %d news in unsupported languages", skipped)
    new = validate_scores(pd.DataFrame(new_rows, columns=SCORE_COLUMNS[1:]))
    if new.empty:
        return new
    start_id = int(scores["id"].max()) + 1 if len(scores) else 1
    new.insert(0, "id", range(start_id, start_id + len(new)))
    config.SENTIMENT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([scores, new], ignore_index=True).to_csv(config.SENTIMENT_CSV, index=False)
    return new
