"""Steps 1-2 of the sentiment algorithm: language detection and text cleaning."""

from __future__ import annotations

import re
import unicodedata

from bs4 import BeautifulSoup

_URL = re.compile(r"https?://\S+|www\.\S+")
_SOURCE_SIGNATURE = re.compile(
    r"(читайте также|подробнее на|подписывайтесь|источник:|фото:|read more|continue reading)[^.]*\.?",
    re.IGNORECASE,
)


def detect_lang(text: str, threshold: float = 0.5) -> str | None:
    """ru / en by the share of Cyrillic / Latin letters, None if unsupported."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return None
    cyr = sum("\u0400" <= c <= "\u04ff" for c in letters) / len(letters)
    lat = sum(c.isascii() for c in letters) / len(letters)
    if cyr >= threshold:
        return "ru"
    if lat >= threshold:
        return "en"
    return None


def clean_text(text: str | None) -> str:
    """Strip HTML, URLs, source signatures, NFKC; case, punctuation and numbers are kept."""
    if not text:
        return ""
    if "<" in text and ">" in text:
        text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = unicodedata.normalize("NFKC", text)
    text = _URL.sub(" ", text)
    text = _SOURCE_SIGNATURE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def model_input(title: str, body: str | None) -> str:
    """Model input string "title. summary"."""
    title, body = clean_text(title), clean_text(body)
    if not body or body == title:
        return title
    return f"{title.rstrip('.')}. {body}"
