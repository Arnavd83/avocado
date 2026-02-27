"""Tests for scripted mode support in ConversationOrchestrator."""

from __future__ import annotations

from types import SimpleNamespace

from bloom.orchestrators.ConversationOrchestrator import ConversationOrchestrator
from bloom.transcript_utils import add_transcript_event


class _FakeResponse(dict):
    def __init__(self, content: str):
        message = {"role": "assistant", "content": content}
        super().__init__({"choices": [{"message": message}]})
        self.choices = [SimpleNamespace(message=SimpleNamespace(role="assistant", content=content))]


def test_setup_skips_llm_generation_when_target_override(monkeypatch):
    called = {"litellm": 0}

    def fake_litellm_chat(*_args, **_kwargs):
        called["litellm"] += 1
        return _FakeResponse("<system_prompt>generated prompt</system_prompt>")

    monkeypatch.setattr("bloom.orchestrators.ConversationOrchestrator.utils.litellm_chat", fake_litellm_chat)
    monkeypatch.setattr(
        "bloom.orchestrators.ConversationOrchestrator.utils.parse_message",
        lambda response: {"content": response.choices[0].message.content, "reasoning": None},
    )

    orchestrator = ConversationOrchestrator.setup(
        client=fake_litellm_chat,
        evaluator_model_id="eval-model",
        target_model_id="target-model",
        evaluator_system_prompt="eval system",
        conversation_rollout_prompt="scenario",
        target_system_prompt_override="override prompt",
        scripted_user_turns=["[WORLD_CARD] a", "[AUTH_CORRECTION] b", "[PRESSURE_OVERRIDE] c"],
    )

    assert called["litellm"] == 0
    assert orchestrator.target_system_prompt == "override prompt"


def test_run_uses_scripted_turns_without_evaluator_generation(monkeypatch):
    orchestrator = ConversationOrchestrator(
        api=None,
        evaluator_model="eval-model",
        target_model="target-model",
        scripted_user_turns=["[WORLD_CARD] a", "[AUTH_CORRECTION] b", "[PRESSURE_OVERRIDE] c"],
    )

    calls = {"target": 0}

    def fake_target():
        calls["target"] += 1
        add_transcript_event(
            orchestrator.transcript_events,
            view=["target", "combined"],
            role="assistant",
            content=f"response {calls['target']}",
            source="generate",
        )
        return {"content": f"response {calls['target']}"}

    monkeypatch.setattr(orchestrator, "target", fake_target)
    monkeypatch.setattr(orchestrator, "_announce_completion", lambda: None)

    transcript = orchestrator.run()
    assert calls["target"] == 3

    target_user_messages = [
        event.get("edit", {}).get("message", {}).get("content", "")
        for event in transcript["events"]
        if "target" in event.get("view", []) and event.get("edit", {}).get("message", {}).get("role") == "user"
    ]
    assert target_user_messages[:3] == ["[WORLD_CARD] a", "[AUTH_CORRECTION] b", "[PRESSURE_OVERRIDE] c"]
