from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .config import Config
from .utils import estimate_tokens, mask_pii, nfc_normalize, now_iso, sanitize_text
from .memory.episodic import EpisodicMemory, Episode
from .memory.preference import Preference, PreferenceStore
from .classification.rules import Classification, classify_by_rules


@dataclass
class IngestResult:
    episode_id: int
    tokens: int
    classification: Classification
    pii_masked: bool


class Cortex:
    def __init__(self, user_id: str, config: Optional[Config] = None) -> None:
        self.user_id = user_id
        self.config = config or Config.from_env()
        self.episodic = EpisodicMemory(self.config.db_path)
        self.prefs = PreferenceStore(self.config.db_path)

    # --- Public API ---
    def ingest(
        self,
        role: str,
        message: str,
        timestamp: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IngestResult:
        # Preprocess
        text = nfc_normalize(message)
        text = sanitize_text(text)
        tokens = estimate_tokens(text)

        # PII handling
        pii_masked = False
        if self.config.pii_mode == "mask":
            text, pii_masked = mask_pii(text)
        elif self.config.pii_mode == "ignore":
            pass
        # else "store": keep as-is

        # Timestamp
        ts = timestamp or now_iso()

        # Store episodic copy
        episode = Episode(
            user_id=self.user_id,
            content=f"[{role}] {text}",
            timestamp=ts,
            tags=role,
            source="chat",
            metadata_json=None,
        )
        episode_id = self.episodic.add(episode)

        # Classification (rules-first)
        classification = classify_by_rules(text)

        # Routing: preferences
        if classification.label == "preference" and classification.data:
            key = classification.data.get("key")
            value = classification.data.get("value")
            if key is not None and value is not None:
                import json

                pref = Preference(
                    user_id=self.user_id,
                    key=str(key),
                    value_json=json.dumps(value, ensure_ascii=False),
                    confidence=classification.confidence,
                    source_episode_id=episode_id,
                )
                self.prefs.upsert(pref, last_updated=ts)

        # Episodic buffer trim (approximate: keep last N by deleting older than N)
        self._trim_episodic_buffer(self.config.max_episodic_messages)

        return IngestResult(
            episode_id=episode_id,
            tokens=tokens,
            classification=classification,
            pii_masked=pii_masked,
        )

    # --- Helpers ---
    def _trim_episodic_buffer(self, max_messages: int) -> None:
        # Find cutoff timestamp for the (max_messages)-th most recent message
        rows = self.episodic.store.query_all(
            """
            SELECT timestamp FROM episodes
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT 1 OFFSET ?
            """,
            [self.user_id, max_messages - 1],
        )
        if not rows:
            return
        cutoff_ts = rows[0]["timestamp"]
        self.episodic.delete_before(self.user_id, cutoff_ts)
