"""Tests for data_gen_v2.cache (Stage 6)."""

from data_gen_v2.cache import CachingLLMClient, ResponseCache
from data_gen_v2.config import LLMConfig
from data_gen_v2.llm import LLMClient


class CountingClient:
    """Minimal LLMClient stand-in that counts calls."""

    def __init__(self, config):
        self.config = config
        self.model_id = config.model_id
        self.calls = 0

    def call(self, system, user, seed):
        self.calls += 1
        return f"out::{system}::{user}::{seed}"


def test_key_is_content_addressed():
    h = "abc"
    k1 = ResponseCache.build_key(h, "sys", "usr", 1)
    k2 = ResponseCache.build_key(h, "sys", "usr", 1)
    k3 = ResponseCache.build_key(h, "sys", "usr", 2)  # different seed
    k4 = ResponseCache.build_key("def", "sys", "usr", 1)  # different config
    assert k1 == k2
    assert k1 != k3
    assert k1 != k4


def test_caching_client_serves_hits_without_recalling(tmp_path):
    cache = ResponseCache(tmp_path)
    inner = CountingClient(LLMConfig(model_id="m"))
    client = CachingLLMClient(inner, cache)

    out1 = client.call("S", "U", 7)
    out2 = client.call("S", "U", 7)  # same inputs → cache hit
    assert out1 == out2
    assert inner.calls == 1  # second call served from cache
    # different seed → miss
    client.call("S", "U", 8)
    assert inner.calls == 2


def test_persistence_round_trip_zero_calls_on_reload(tmp_path):
    cfg = LLMConfig(model_id="m")
    # First run: populate + save.
    cache1 = ResponseCache(tmp_path)
    inner1 = CountingClient(cfg)
    c1 = CachingLLMClient(inner1, cache1)
    c1.call("S", "U", 1)
    c1.call("S", "U", 2)
    cache1.save()
    assert inner1.calls == 2

    # Second run: fresh cache loads from disk → zero inner calls.
    cache2 = ResponseCache(tmp_path)
    assert len(cache2) == 2
    inner2 = CountingClient(cfg)
    c2 = CachingLLMClient(inner2, cache2)
    assert c2.call("S", "U", 1) == "out::S::U::1"
    assert c2.call("S", "U", 2) == "out::S::U::2"
    assert inner2.calls == 0


def test_disabled_cache_is_noop(tmp_path):
    cache = ResponseCache(None)
    assert not cache.enabled
    inner = CountingClient(LLMConfig(model_id="m"))
    client = CachingLLMClient(inner, cache)
    client.call("S", "U", 1)
    client.call("S", "U", 1)
    assert inner.calls == 2  # no caching


def test_caching_client_exposes_config_and_model_id():
    inner = LLMClient(LLMConfig(model_id="zzz"), llm_callable=lambda s, u, x: "ok")
    client = CachingLLMClient(inner, ResponseCache(None))
    assert client.model_id == "zzz"
    assert client.config.retry_limit == 1
