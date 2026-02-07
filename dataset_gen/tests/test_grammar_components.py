"""
Tests for grammar agent supporting components.
"""

from unittest.mock import patch

from dataset_gen.src.agents.grammar_cache import GrammarCache, GrammarCacheEntry
from dataset_gen.src.agents.grammar_config import GrammarConfig
from dataset_gen.src.agents.grammar_report import GrammarReport
from dataset_gen.src.agents.grammar_agent import GrammarAgent


def test_grammar_config_hash_changes():
    config = GrammarConfig()
    h1 = config.config_hash()
    config.model_id = "other-model"
    h2 = config.config_hash()
    assert h1 != h2


def test_grammar_cache_put_get(tmp_path):
    cache = GrammarCache(tmp_path, enabled=True)
    entry = GrammarCacheEntry(
        cache_key="abc123",
        decision="clean",
        fixed_content="",
        raw_response="{}",
        timestamp="2025-01-01T00:00:00Z",
        config_hash="cfg",
    )
    cache.put(entry)
    got = cache.get("abc123")
    assert got is not None
    assert got.decision == "clean"


def test_grammar_report_records():
    report = GrammarReport()
    report.record(decision="fixed", reason="ok", sample_text="text")
    assert report.total_checked == 1
    assert report.total_fixed == 1


@patch("dataset_gen.src.agents.grammar_agent.GrammarAgent._init_client", return_value=object())
def test_grammar_agent_parse_response(_mock_init):
    agent = GrammarAgent(
        config=GrammarConfig(model_provider="openai", api_base="http://localhost"),
        cache=GrammarCache(None, enabled=False),
        report=GrammarReport(),
    )
    parsed = agent._parse_response(
        '{"decision":"clean","corrected_text":"","reason":"ok"}'
    )
    assert parsed is not None
    assert parsed.get("decision") == "clean"
