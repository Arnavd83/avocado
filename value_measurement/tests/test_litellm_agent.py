import os
import sys
from types import SimpleNamespace

import pytest


sys.path.insert(
    0,
    os.path.abspath("value_measurement/emergent-values/utility_analysis"),
)

from compute_utilities import llm_agent  # noqa: E402
from compute_utilities.llm_agent import LiteLLMAgent  # noqa: E402


def _completion(content):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason="stop",
            )
        ]
    )


@pytest.mark.asyncio
async def test_litellm_agent_retries_none_content(monkeypatch):
    calls = {"count": 0}

    async def fake_acompletion(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return _completion(None)
        return _completion(" A ")

    monkeypatch.setattr(llm_agent, "litellm_acompletion", fake_acompletion)
    agent = LiteLLMAgent(
        model="test-model",
        max_retries=2,
        base_delay=0,
        use_jitter=False,
    )

    responses = await agent.async_completions(
        [[{"role": "user", "content": "Choose A or B"}]],
        verbose=False,
    )

    assert responses == ["A"]
    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_litellm_agent_raises_for_missing_content_when_configured(monkeypatch):
    async def fake_acompletion(**kwargs):
        return _completion(None)

    monkeypatch.setattr(llm_agent, "litellm_acompletion", fake_acompletion)
    agent = LiteLLMAgent(
        model="test-model",
        max_retries=2,
        base_delay=0,
        use_jitter=False,
        fail_on_missing_response=True,
    )

    with pytest.raises(RuntimeError, match="1 LLM calls failed after retries"):
        await agent.async_completions(
            [[{"role": "user", "content": "Choose A or B"}]],
            verbose=False,
        )
