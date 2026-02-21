from types import SimpleNamespace

import pytest
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, GenerateConfig

from petri.stores import AuditStore
from petri.tasks.petri import _coerce_target_max_tokens, audit
from petri.tools.tools import call_target


@pytest.mark.asyncio
async def test_call_target_uses_default_target_max_tokens(monkeypatch):
    captured: dict[str, object] = {}

    class DummyModel:
        async def generate(self, messages, tools=None, config=None):
            captured["config"] = config
            return SimpleNamespace(message=ChatMessageAssistant(content="ok"))

    monkeypatch.setattr("petri.tools.tools.get_model", lambda role="target": DummyModel())
    monkeypatch.setattr("petri.tools.tools.store_as", lambda *_args, **_kwargs: AuditStore())

    await call_target([ChatMessageUser(content="hello")], [])

    assert isinstance(captured["config"], GenerateConfig)
    assert captured["config"].max_tokens == 768


@pytest.mark.asyncio
async def test_call_target_uses_overridden_target_max_tokens(monkeypatch):
    captured: dict[str, object] = {}

    class DummyModel:
        async def generate(self, messages, tools=None, config=None):
            captured["config"] = config
            return SimpleNamespace(message=ChatMessageAssistant(content="ok"))

    monkeypatch.setattr("petri.tools.tools.get_model", lambda role="target": DummyModel())
    monkeypatch.setattr(
        "petri.tools.tools.store_as",
        lambda *_args, **_kwargs: SimpleNamespace(target_max_tokens=512),
    )

    await call_target([ChatMessageUser(content="hello")], [])

    assert isinstance(captured["config"], GenerateConfig)
    assert captured["config"].max_tokens == 512


@pytest.mark.asyncio
async def test_call_target_disables_cap_when_none(monkeypatch):
    captured: dict[str, object] = {}

    class DummyModel:
        async def generate(self, messages, tools=None, config=None):
            captured["config"] = config
            return SimpleNamespace(message=ChatMessageAssistant(content="ok"))

    monkeypatch.setattr("petri.tools.tools.get_model", lambda role="target": DummyModel())
    monkeypatch.setattr(
        "petri.tools.tools.store_as",
        lambda *_args, **_kwargs: SimpleNamespace(target_max_tokens=None),
    )

    await call_target([ChatMessageUser(content="hello")], [])

    assert captured["config"] is None


def test_coerce_target_max_tokens_accepts_nullish_values():
    assert _coerce_target_max_tokens("none") is None
    assert _coerce_target_max_tokens("null") is None
    assert _coerce_target_max_tokens(256) == 256


def test_task_accepts_none_target_max_tokens():
    task = audit(special_instructions="test instruction", target_max_tokens="none")
    assert task is not None

