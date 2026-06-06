"""
Resumable response cache (Stage 6).

The agents (Stages 2/3) take only ``(llm, report)`` — no cache handle — so caching
is applied **at the LLM-call level** via ``CachingLLMClient``, which wraps any
object exposing ``.call(system, user, seed)`` and ``.config`` (i.e. ``LLMClient``).
This is transparent to the agents and content-addressed by construction:

    key = sha256(config_hash, system_prompt, user_message, seed)

The system prompt already encodes all spec content (framing/shape/tone/prefs for
the prompt agent; stance/intensity/shape/directive for the answer agent), the user
message carries the prompt text, and ``seed`` is unique per pair — so identical
keys mean genuinely identical generation requests, and different pairs (different
seeds) never collide. This is simpler than the spec's two-namespace design and
equivalent in effect (see stage_6_orchestration_spec.md §4); every distinct call —
including each retry, whose system prompt differs by its addendum — gets its own
entry, which makes a re-run fully reproducible.

Carried forward from ``dataset_gen/src/agents/justification_cache.py``: the
in-memory dict + JSONL persistence + corrupted-line skip.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional


class ResponseCache:
    """Content-addressed cache of raw LLM outputs, persisted as JSONL."""

    CACHE_FILENAME = "response_cache.jsonl"

    def __init__(self, cache_dir: Optional[Path], enabled: bool = True):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.enabled = enabled and self.cache_dir is not None
        self._mem: Dict[str, str] = {}
        if self.enabled:
            self.load()

    @staticmethod
    def build_key(config_hash: str, system: str, user: str, seed: int) -> str:
        payload = json.dumps(
            {"config_hash": config_hash, "system": system, "user": user, "seed": seed},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> Optional[str]:
        if not self.enabled:
            return None
        return self._mem.get(key)

    def put(self, key: str, value: str) -> None:
        if not self.enabled:
            return
        self._mem[key] = value

    def save(self) -> None:
        if not self.enabled:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / self.CACHE_FILENAME
        with open(path, "w", encoding="utf-8") as f:
            for key, value in self._mem.items():
                f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")

    def load(self) -> None:
        if self.cache_dir is None:
            return
        path = self.cache_dir / self.CACHE_FILENAME
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    self._mem[rec["key"]] = rec["value"]
                except (json.JSONDecodeError, KeyError) as exc:
                    print(f"Warning: skipping corrupted cache line {line_num}: {exc}")

    def __len__(self) -> int:
        return len(self._mem)


class CachingLLMClient:
    """Wraps an ``LLMClient``-like object, serving cached calls when available.

    Exposes the same surface the agents touch: ``.call(system, user, seed)`` and
    ``.config`` (so ``config.retry_limit`` / ``config.config_hash()`` keep working),
    plus ``.model_id`` for provenance.
    """

    def __init__(self, client, cache: ResponseCache):
        self._client = client
        self._cache = cache

    @property
    def config(self):
        return self._client.config

    @property
    def model_id(self) -> str:
        return getattr(self._client, "model_id", self._client.config.model_id)

    def call(self, system: str, user: str, seed: int) -> str:
        key = ResponseCache.build_key(self._client.config.config_hash(), system, user, seed)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        out = self._client.call(system, user, seed)
        self._cache.put(key, out)
        return out


__all__ = ["ResponseCache", "CachingLLMClient"]
