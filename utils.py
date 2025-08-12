from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from typing import Tuple


ZERO_WIDTH_PATTERN = re.compile(r"[\u200B\u200C\u200D\uFEFF]")
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(r"(?<!\d)(\+?\d[\d\s\-()]{7,}\d)(?!\d)")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def nfc_normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def sanitize_text(text: str, max_len: int = 8000) -> str:
    """Light sanitization: strip, collapse whitespace, remove zero-width, truncate."""
    text = text.strip()
    text = ZERO_WIDTH_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text)
    if len(text) > max_len:
        text = text[: max_len - 1] + "…"
    return text


def estimate_tokens(text: str) -> int:
    """Very rough token estimation (~4 chars per token). Replace with tiktoken if needed."""
    length = len(text)
    return max(1, length // 4)


def mask_pii(text: str) -> Tuple[str, bool]:
    """Mask emails and phone numbers. Returns (masked_text, did_mask)."""
    masked = EMAIL_PATTERN.sub("<email>", text)
    masked = PHONE_PATTERN.sub("<phone>", masked)
    return masked, masked != text
