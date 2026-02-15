"""Tests for rollout gate-pass enforcement."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from bloom.stages.step3_rollout import run_rollout_async


def test_rollout_blocks_when_gate_required_and_not_passed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    config = {
        "behavior": {"name": "corrigibility", "examples": []},
        "configurable_prompts": "default",
        "temperature": 0.2,
        "evaluator_reasoning_effort": "none",
        "target_reasoning_effort": "none",
        "rollout": {
            "model": "openrouter/openai/gpt-4o-mini",
            "target": "openrouter/google/gemma-2-9b-it",
            "modality": "conversation",
            "max_turns": 5,
            "max_tokens": 1000,
            "num_reps": 1,
            "no_user_mode": False,
            "selected_variations": None,
            "require_gate_pass": True,
        },
        "ideation": {"model": "openrouter/openai/gpt-4o-mini"},
        "judgment": {"model": "openrouter/openai/gpt-4o-mini"},
        "_config_dir": Path(__file__).resolve().parents[1] / "src" / "bloom" / "data",
    }

    with pytest.raises(RuntimeError, match="gate pass is required"):
        asyncio.run(run_rollout_async(config=config))

