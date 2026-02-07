"""
Caching layer for grammar judge/fix reproducibility.

Stores LLM decisions and optional fixes keyed by config + content + lint signature.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


@dataclass
class GrammarCacheEntry:
    """
    A single cache entry for grammar judging/fixing.

    Attributes:
        cache_key: SHA256 hash of inputs
        decision: "clean", "fixed", or "reject"
        fixed_content: Corrected text if decision == "fixed"
        raw_response: Raw LLM response
        timestamp: ISO timestamp of generation
        config_hash: Hash of config at generation time
    """

    cache_key: str
    decision: str
    fixed_content: str
    raw_response: str
    timestamp: str
    config_hash: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "GrammarCacheEntry":
        """Create from dictionary."""
        return cls(**data)


class GrammarCache:
    """
    File-based cache for grammar reproducibility.

    Cache key is computed from:
    - config_hash
    - content
    - lint_signature (rule IDs + counts)
    """

    def __init__(self, cache_dir: Optional[Path], enabled: bool = True):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.enabled = enabled
        self._memory_cache: Dict[str, GrammarCacheEntry] = {}

        if self.enabled and self.cache_dir:
            self.load_from_disk()

    @staticmethod
    def build_cache_key(
        config_hash: str,
        content: str,
        lint_signature: str,
    ) -> str:
        combined = "|".join([config_hash, content, lint_signature])
        return hashlib.sha256(combined.encode()).hexdigest()[:32]

    def get(self, cache_key: str) -> Optional[GrammarCacheEntry]:
        if not self.enabled:
            return None
        return self._memory_cache.get(cache_key)

    def put(self, entry: GrammarCacheEntry) -> None:
        if not self.enabled:
            return
        self._memory_cache[entry.cache_key] = entry

    def save_to_disk(self) -> None:
        if not self.cache_dir or not self.enabled:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / "grammar_cache.jsonl"

        with open(cache_file, "w") as f:
            for entry in self._memory_cache.values():
                f.write(json.dumps(entry.to_dict()) + "\n")

    def load_from_disk(self) -> None:
        if not self.cache_dir:
            return

        cache_file = self.cache_dir / "grammar_cache.jsonl"
        if not cache_file.exists():
            return

        try:
            with open(cache_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        entry = GrammarCacheEntry.from_dict(data)
                        self._memory_cache[entry.cache_key] = entry
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load grammar cache file, starting fresh: {e}")
            self._memory_cache = {}

    def __len__(self) -> int:
        return len(self._memory_cache)

    def stats(self) -> dict:
        return {
            "enabled": self.enabled,
            "entries": len(self._memory_cache),
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
        }

    @staticmethod
    def now_iso() -> str:
        return datetime.utcnow().isoformat() + "Z"
