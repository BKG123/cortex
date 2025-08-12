from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class Classification:
    label: str  # "preference" | "fact" | "semantic" | "ignore"
    confidence: float
    data: Optional[Dict[str, object]] = None


PREF_PATTERNS = [
    (re.compile(r"\b(i\s+prefer|i\s+like|i\s+usually)\b", re.I), "preference_base"),
    (
        re.compile(
            r"\bavoid(s)?\s+(meetings\s+)?(on\s+)?(?P<day>monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            re.I,
        ),
        "avoid_day",
    ),
]

FACT_PATTERNS = [
    (re.compile(r"\bmy\s+(email|phone)\b", re.I), "contact"),
    (re.compile(r"\b(on|at)\s+\d{4}-\d{2}-\d{2}\b"), "date_ref"),
]


def classify_by_rules(message: str) -> Classification:
    text = message.strip()
    if not text:
        return Classification("ignore", 1.0)

    for regex, name in PREF_PATTERNS:
        m = regex.search(text)
        if m:
            data: Dict[str, object] | None = None
            if name == "avoid_day" and "day" in m.groupdict():
                day = m.group("day").lower()
                data = {
                    "key": "avoid_days",
                    "value": [day.capitalize()],
                    "source": "rule",
                }
            return Classification(
                "preference", 0.9 if name != "preference_base" else 0.7, data
            )

    for regex, name in FACT_PATTERNS:
        if regex.search(text):
            return Classification("fact", 0.7, {"hint": name})

    # Fallback: treat longer statements as semantic content
    if len(text) > 40:
        return Classification("semantic", 0.6)

    return Classification("ignore", 0.5)
