"""
Tests for justification cache durability on interruption.

Verifies:
1. Cache persists to disk when generation is interrupted (KeyboardInterrupt)
2. DatasetGenerator.close() is idempotent (safe to call twice)
3. Corrupted cache lines are skipped during load (not entire file)
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from dataset_gen.src.agents.justification_cache import JustificationCache, CacheEntry
from dataset_gen.src.render import DatasetGenerator
from dataset_gen.src.plan import AllocationConfig
from dataset_gen.src.families.registry import import_all_families
from dataset_gen.tools.cli import main


@pytest.fixture(scope="module", autouse=True)
def setup_families():
    import_all_families()


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def _make_cache_entry(key: str, justification: str = "test") -> CacheEntry:
    return CacheEntry(
        cache_key=key,
        raw_justification=justification,
        final_justification=justification,
        attempt_count=1,
        used_fallback=False,
        timestamp="2025-01-01T00:00:00",
        config_hash="abc123",
    )


class TestCachePerLineLoading:
    """load_from_disk should skip corrupted lines, not discard the whole cache."""

    def test_loads_valid_entries_around_corrupted_line(self, temp_dir):
        cache_file = temp_dir / "justification_cache.jsonl"
        entry1 = _make_cache_entry("key1", "good1")
        entry2 = _make_cache_entry("key2", "good2")

        # Write valid, corrupted, valid
        with open(cache_file, "w") as f:
            f.write(json.dumps(entry1.to_dict()) + "\n")
            f.write("{truncated garbage\n")
            f.write(json.dumps(entry2.to_dict()) + "\n")

        cache = JustificationCache(cache_dir=temp_dir, enabled=True)
        assert len(cache) == 2
        assert cache.get("key1").final_justification == "good1"
        assert cache.get("key2").final_justification == "good2"

    def test_handles_empty_file(self, temp_dir):
        cache_file = temp_dir / "justification_cache.jsonl"
        cache_file.touch()

        cache = JustificationCache(cache_dir=temp_dir, enabled=True)
        assert len(cache) == 0

    def test_handles_all_corrupted_lines(self, temp_dir):
        cache_file = temp_dir / "justification_cache.jsonl"
        with open(cache_file, "w") as f:
            f.write("not json\n")
            f.write("{also broken\n")

        cache = JustificationCache(cache_dir=temp_dir, enabled=True)
        assert len(cache) == 0


class TestIdempotentClose:
    """DatasetGenerator.close() should be safe to call multiple times."""

    def test_close_twice_no_error(self):
        config = AllocationConfig(total_size=10)
        generator = DatasetGenerator(config, global_seed=1)

        generator.close()
        generator.close()  # should not raise

    def test_close_twice_with_justification_agent(self, temp_dir):
        """close() with justification agent should only finalize once."""
        from dataset_gen.src.agents import JustificationConfig

        just_config = JustificationConfig(
            model_provider="openai",
            model_id="test",
            api_base="http://localhost:9999",
            cache_enabled=True,
            cache_dir=temp_dir / "cache",
        )
        config = AllocationConfig(total_size=10)

        # Mock the LLM client init so we don't need real API keys
        with patch("dataset_gen.src.agents.justification_agent.JustificationAgent._init_client"):
            generator = DatasetGenerator(
                config,
                global_seed=1,
                justification_agent_config=just_config,
            )

        generator.close()
        generator.close()  # should not raise or double-print


class TestInterruptionCachePersistence:
    """Cache should be saved to disk even when generation is interrupted."""

    def test_cache_saved_on_keyboard_interrupt(self, temp_dir):
        """Simulate KeyboardInterrupt during generation, verify cache is written."""
        output_dir = str(temp_dir / "output")
        cache_dir = temp_dir / "output" / "justification_cache"

        # We'll mock DatasetGenerator.generate to raise KeyboardInterrupt
        # after the generator (and its cache) has been constructed.
        # The try/finally in cmd_generate should still call close().
        def mock_generate(self, *_args, **_kwargs):
            # Put a fake entry in the cache before interrupting
            if self._justification_agent is not None:
                entry = _make_cache_entry("interrupted_key", "saved_value")
                self._justification_cache.put(entry)
            raise KeyboardInterrupt()

        with patch.object(DatasetGenerator, "generate", mock_generate), \
             patch("dataset_gen.src.agents.justification_agent.JustificationAgent._init_client"):

            result = main([
                "generate",
                "--output", output_dir,
                "--size", "10",
                "--seed", "42",
                "--no-validate",
            ])

        # KeyboardInterrupt should be caught, returning 130
        assert result == 130

        # The cache file should exist with our entry
        cache_file = cache_dir / "justification_cache.jsonl"
        assert cache_file.exists(), "Cache file should be written on interruption"

        # Verify the entry was persisted
        cache = JustificationCache(cache_dir=cache_dir, enabled=True)
        assert cache.get("interrupted_key") is not None
        assert cache.get("interrupted_key").final_justification == "saved_value"
