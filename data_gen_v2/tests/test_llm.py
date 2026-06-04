"""Tests for data_gen_v2.llm (Stage 0). No real API calls — injected callable."""

from data_gen_v2.config import LLMConfig
from data_gen_v2.llm import LLMClient


def test_injected_callable_used_and_args_threaded():
    seen = {}

    def fake(system, user, seed):
        seen["system"] = system
        seen["user"] = user
        seen["seed"] = seed
        return "  mock reply  "

    client = LLMClient(LLMConfig(), llm_callable=fake)
    out = client.call("SYS", "USR", 123)
    assert out == "  mock reply  "  # client returns raw; caller strips
    assert seen == {"system": "SYS", "user": "USR", "seed": 123}


def test_model_id_property():
    client = LLMClient(LLMConfig(model_id="claude-opus-4-8"), llm_callable=lambda s, u, x: "")
    assert client.model_id == "claude-opus-4-8"


def test_no_provider_client_constructed_when_callable_set():
    client = LLMClient(LLMConfig(), llm_callable=lambda s, u, x: "ok")
    client.call("a", "b", 0)
    assert client._client is None  # never lazily initialized a provider client
