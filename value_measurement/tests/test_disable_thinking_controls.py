from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UTILITY_ANALYSIS_DIR = PROJECT_ROOT / "value_measurement" / "emergent-values" / "utility_analysis"
if str(UTILITY_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITY_ANALYSIS_DIR))

from compute_utilities import compute_utilities as cu_module  # noqa: E402
from compute_utilities import utils as cu_utils  # noqa: E402


def _load_optimize_module():
    module_path = (
        PROJECT_ROOT
        / "value_measurement"
        / "emergent-values"
        / "utility_analysis"
        / "experiments"
        / "compute_utilities"
        / "optimize_utility_model.py"
    )
    spec = importlib.util.spec_from_file_location("optimize_utility_model", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_create_agent_passes_reasoning_controls(monkeypatch):
    os.environ["MODELS_CONFIG_PATH"] = str(PROJECT_ROOT / "config" / "models.yaml")
    os.environ["OPENROUTER_API_KEY"] = "sk-test"

    captured: dict[str, object] = {}

    class DummyLiteLLMAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.model = kwargs.get("model")
            self.accepts_system_message = kwargs.get("accepts_system_message", True)

    monkeypatch.setattr(cu_utils, "LiteLLMAgent", DummyLiteLLMAgent)

    cu_utils.create_agent(
        model_key="qwen-3-8b",
        max_tokens=3,
        reasoning_effort="none",
        allowed_openai_params=["reasoning_effort"],
    )

    assert captured["reasoning_effort"] == "none"
    assert captured["allowed_openai_params"] == ["reasoning_effort"]


@pytest.mark.asyncio
async def test_compute_utilities_disable_thinking_applies_qwen_and_agent_controls(monkeypatch):
    captured_init: dict[str, object] = {}
    captured_agent: dict[str, object] = {}

    class FakeAgent:
        accepts_system_message = True

        async def async_completions(self, messages, verbose=True, **kwargs):
            return ["A"]

    class FakeGraph:
        def __init__(self, options, holdout_fraction, seed):
            self.options = options

        def export_data(self):
            return {"ok": True}

    class FakeUtilityModel:
        def __init__(self, **kwargs):
            captured_init.update(kwargs)

        async def fit(self, graph, agent):
            utilities = {opt["id"]: {"mean": 0.5, "variance": 0.1} for opt in graph.options}
            metrics = {"log_loss": 0.1, "accuracy": 1.0}
            return utilities, metrics

    def fake_load_config(config_path, config_key, default_filename=None):
        if default_filename == "compute_utilities.yaml":
            return {
                "utility_model_class": "ThurstonianActiveLearningUtilityModel",
                "utility_model_arguments": {},
                "preference_graph_arguments": {"holdout_fraction": 0.0, "holdout_seed": 42},
                "compute_utilities_arguments": {"system_message": "You are a helpful assistant.", "with_reasoning": True},
            }
        if default_filename == "create_agent.yaml":
            return {"max_tokens": 3, "temperature": 0.5, "concurrency_limit": 1}
        raise AssertionError(f"Unexpected config request: {default_filename}")

    def fake_create_agent(model_key, **kwargs):
        captured_agent["model_key"] = model_key
        captured_agent["kwargs"] = kwargs
        return FakeAgent()

    async def fake_evaluate_holdout_set(**kwargs):
        return None

    monkeypatch.setattr(cu_module, "load_config", fake_load_config)
    monkeypatch.setattr(cu_module, "create_agent", fake_create_agent)
    monkeypatch.setattr(cu_module, "ThurstonianActiveLearningUtilityModel", FakeUtilityModel)
    monkeypatch.setattr(cu_module, "PreferenceGraph", FakeGraph)
    monkeypatch.setattr(cu_module, "evaluate_holdout_set", fake_evaluate_holdout_set)

    result = await cu_module.compute_utilities(
        options_list=["Option A", "Option B"],
        model_key="qwen-3-8b",
        create_agent_config_path="dummy",
        create_agent_config_key="default",
        compute_utilities_config_path="dummy",
        compute_utilities_config_key="dummy",
        save_dir=None,
        disable_thinking=True,
    )

    assert captured_agent["kwargs"]["reasoning_effort"] == "none"
    assert "reasoning_effort" in captured_agent["kwargs"]["allowed_openai_params"]
    assert captured_init["with_reasoning"] is False
    assert "/no_think" in captured_init["system_message"]
    assert result["metrics"]["accuracy"] == 1.0


@pytest.mark.asyncio
async def test_compute_utilities_disable_thinking_preflight_fails_fast(monkeypatch):
    class FailingAgent:
        accepts_system_message = True

        async def async_completions(self, messages, verbose=True, **kwargs):
            raise ValueError("unsupported params")

    class FakeGraph:
        def __init__(self, options, holdout_fraction, seed):
            self.options = options

        def export_data(self):
            return {}

    class FakeUtilityModel:
        def __init__(self, **kwargs):
            pass

        async def fit(self, graph, agent):
            return {0: {"mean": 0.1, "variance": 0.1}}, {"log_loss": 0.1, "accuracy": 1.0}

    def fake_load_config(config_path, config_key, default_filename=None):
        if default_filename == "compute_utilities.yaml":
            return {
                "utility_model_class": "ThurstonianActiveLearningUtilityModel",
                "utility_model_arguments": {},
                "preference_graph_arguments": {"holdout_fraction": 0.0, "holdout_seed": 42},
                "compute_utilities_arguments": {"system_message": "You are a helpful assistant.", "with_reasoning": False},
            }
        if default_filename == "create_agent.yaml":
            return {"max_tokens": 3}
        raise AssertionError(f"Unexpected config request: {default_filename}")

    def fake_create_agent(model_key, **kwargs):
        return FailingAgent()

    async def fake_evaluate_holdout_set(**kwargs):
        return None

    monkeypatch.setattr(cu_module, "load_config", fake_load_config)
    monkeypatch.setattr(cu_module, "create_agent", fake_create_agent)
    monkeypatch.setattr(cu_module, "ThurstonianActiveLearningUtilityModel", FakeUtilityModel)
    monkeypatch.setattr(cu_module, "PreferenceGraph", FakeGraph)
    monkeypatch.setattr(cu_module, "evaluate_holdout_set", fake_evaluate_holdout_set)

    with pytest.raises(RuntimeError, match="disable_thinking preflight failed"):
        await cu_module.compute_utilities(
            options_list=["Only Option"],
            model_key="qwen-3-8b",
            create_agent_config_path="dummy",
            create_agent_config_key="default",
            compute_utilities_config_path="dummy",
            compute_utilities_config_key="dummy",
            save_dir=None,
            disable_thinking=True,
        )


@pytest.mark.asyncio
async def test_optimize_script_forwards_disable_thinking(tmp_path, monkeypatch):
    optimize_module = _load_optimize_module()
    options_path = tmp_path / "options.json"
    options_path.write_text(json.dumps({"group": ["A", "B"]}))

    captured: dict[str, object] = {}

    async def fake_compute_utilities(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(optimize_module, "compute_utilities", fake_compute_utilities)

    args = SimpleNamespace(
        model_key="qwen-3-8b",
        save_dir=None,
        save_suffix=None,
        options_path=str(options_path),
        with_reasoning=False,
        disable_thinking=True,
        compute_utilities_config_path="dummy",
        compute_utilities_config_key="dummy",
        create_agent_config_path="dummy",
        create_agent_config_key="default",
    )

    await optimize_module.optimize_utility_model(args)
    assert captured["disable_thinking"] is True

