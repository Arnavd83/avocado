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

    Cache key is computed from (see build_cache_key):
    - pair_id (traceability + deterministic content proxy)
    - family_id + subtype_id + severity + mode + perspective
    - style_directive_id + target_intensity (diversification axes)
    - condition (pro/anti)
    - catalog_version + directive_pool_version (provenance — invalidate on bump)
    - config_hash (model/prompt/sampling settings)

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
        *,
        pair_id: str,
        family_id: str,
        subtype_id: str,
        severity: str,
        mode: str,
        perspective: str,
        style_directive_id: int,
        target_intensity: int,
        condition: str,
        catalog_version: str,
        directive_pool_version: str,
        config_hash: str,
    ) -> str:
        """
        Build a deterministic cache key from the per-record diversification axes.

        The key is the SHA256 of a JSON payload (sorted keys) covering every
        input that affects the generated text. ``pair_id`` is included both for
        traceability and because, given the deterministic Stage 1→4 pipeline, it
        is a stable proxy for the rendered prompt content (same pair_id ⇒ same
        seed ⇒ same preference pair ⇒ same rendered prompt). The new
        diversification axes (``style_directive_id``, ``target_intensity``) and
        the provenance versions (``catalog_version``, ``directive_pool_version``)
        are part of the key so a catalog or directive-pool bump invalidates stale
        entries. ``config_hash`` carries the model/prompt/sampling settings.

        Args:
            pair_id: Unique pro/anti pair id (traceability + content proxy).
            family_id: Family id value (e.g. "explicit_reversal").
            subtype_id: Family subtype id (e.g. "A1_acceptability").
            severity: Severity level value.
            mode: Response mode value.
            perspective: Perspective value (first/third).
            style_directive_id: Style directive index 0-9 (NEW axis).
            target_intensity: Intensity 1-7 (NEW axis).
            condition: "pro" or "anti".
            catalog_version: Catalog provenance string (invalidates on change).
            directive_pool_version: Directive-pool provenance (invalidates on change).
            config_hash: Hash of JustificationConfig (model/prompt/sampling).

        Returns:
            SHA256 hex digest of the JSON payload.
        """
        payload = {
            "pair_id": pair_id,
            "family_id": family_id,
            "subtype_id": subtype_id,
            "severity": severity,
            "mode": mode,
            "perspective": perspective,
            "style_directive_id": style_directive_id,
            "target_intensity": target_intensity,
            "condition": condition,
            "catalog_version": catalog_version,
            "directive_pool_version": directive_pool_version,
            "config_hash": config_hash,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()

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

        with open(cache_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entry = CacheEntry.from_dict(data)
                    self._memory_cache[entry.cache_key] = entry
                except (json.JSONDecodeError, KeyError) as e:
                    # Skip corrupted line (e.g. truncated write from interrupted save)
                    print(f"Warning: Skipping corrupted cache line {line_num}: {e}")
                    continue

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
