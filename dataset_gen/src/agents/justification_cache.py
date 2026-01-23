"""
Caching layer for justification reproducibility.

This module provides file-based caching to ensure:
1. Reproducibility: Same inputs always produce same outputs
2. Efficiency: Second runs skip LLM calls via cache hits
3. Debugging: Cache entries store metadata about generation
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


@dataclass
class CacheEntry:
    """
    A single cache entry for a generated justification.

    Attributes:
        cache_key: SHA256 hash of inputs
        raw_justification: Original LLM output
        final_justification: After any validation/fallback
        attempt_count: Number of generation attempts
        used_fallback: Whether fallback was used
        timestamp: ISO timestamp of generation
        config_hash: Hash of config at generation time
    """

    cache_key: str
    raw_justification: str
    final_justification: str
    attempt_count: int
    used_fallback: bool
    timestamp: str
    config_hash: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CacheEntry":
        """Create from dictionary."""
        return cls(**data)


class JustificationCache:
    """
    File-based cache for justification reproducibility.

    Cache key is computed from:
    - spec_version + model_id + system_prompt_hash
    - prompt_content + stance + preferences
    - domain + severity + mode + perspective

    The cache is stored as a JSONL file on disk and loaded into
    memory for fast lookups during generation.

    Memory usage for 12,000 entries: ~7.2 MB (negligible)
    """

    def __init__(self, cache_dir: Optional[Path], enabled: bool = True):
        """
        Initialize the cache.

        Args:
            cache_dir: Directory for cache files (None = no disk persistence)
            enabled: Whether caching is enabled
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.enabled = enabled
        self._memory_cache: Dict[str, CacheEntry] = {}

        # Load existing cache from disk if available
        if self.enabled and self.cache_dir:
            self.load_from_disk()

    @staticmethod
    def build_cache_key(
        config_hash: str,
        prompt_content: str,
        stance: str,
        current_pref_text: str,
        target_pref_text: str,
        domain: str,
        severity: str,
        mode: str,
        perspective: str,
    ) -> str:
        """
        Build a deterministic cache key from inputs.

        All parameters that affect output must be included.

        Args:
            config_hash: Hash of JustificationConfig
            prompt_content: The rendered prompt content
            stance: "keep_current" or "adopt_target"
            current_pref_text: Current preference text
            target_pref_text: Target preference text
            domain: Preference domain
            severity: Severity level
            mode: Response mode
            perspective: Perspective (first/third)

        Returns:
            SHA256 hash (first 32 chars) of combined inputs
        """
        combined = "|".join(
            [
                config_hash,
                prompt_content,
                stance,
                current_pref_text,
                target_pref_text,
                domain,
                severity,
                mode,
                perspective,
            ]
        )
        return hashlib.sha256(combined.encode()).hexdigest()[:32]

    def get(self, cache_key: str) -> Optional[CacheEntry]:
        """
        Retrieve cached entry if exists.

        Args:
            cache_key: The cache key to look up

        Returns:
            CacheEntry if found, None otherwise
        """
        if not self.enabled:
            return None
        return self._memory_cache.get(cache_key)

    def put(self, entry: CacheEntry) -> None:
        """
        Store entry in cache.

        Args:
            entry: The cache entry to store
        """
        if not self.enabled:
            return
        self._memory_cache[entry.cache_key] = entry

    def save_to_disk(self) -> None:
        """
        Persist memory cache to JSONL file.

        Creates cache directory if it doesn't exist.
        """
        if not self.cache_dir or not self.enabled:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / "justification_cache.jsonl"

        with open(cache_file, "w") as f:
            for entry in self._memory_cache.values():
                f.write(json.dumps(entry.to_dict()) + "\n")

    def load_from_disk(self) -> None:
        """
        Load cache from JSONL file.

        Silently skips if file doesn't exist.
        """
        if not self.cache_dir:
            return

        cache_file = self.cache_dir / "justification_cache.jsonl"
        if not cache_file.exists():
            return

        try:
            with open(cache_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        entry = CacheEntry.from_dict(data)
                        self._memory_cache[entry.cache_key] = entry
        except (json.JSONDecodeError, KeyError) as e:
            # Corrupted cache file - start fresh
            print(f"Warning: Could not load cache file, starting fresh: {e}")
            self._memory_cache = {}

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._memory_cache)

    def clear(self) -> None:
        """Clear the in-memory cache."""
        self._memory_cache = {}

    def stats(self) -> dict:
        """Return cache statistics."""
        return {
            "enabled": self.enabled,
            "entries": len(self._memory_cache),
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
        }
