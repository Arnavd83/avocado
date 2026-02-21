"""Unit tests for target_max_tokens wiring in scripts/run_seed_dataset.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_SEED_DATASET_PATH = PROJECT_ROOT / "scripts" / "run_seed_dataset.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_seed_dataset", RUN_SEED_DATASET_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_petri_includes_target_max_tokens_task_arg(monkeypatch, tmp_path):
    module = _load_module()
    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    rc = module._run_petri(
        seed_dir=tmp_path,
        seed_file=tmp_path / "seed_prompt.json",
        auditor="auditor-model",
        target="target-model",
        judge="judge-model",
        max_turns=5,
        target_max_tokens=768,
        env={},
        stream_output=False,
    )

    assert rc == 0
    cmd = captured["cmd"]
    assert "target_max_tokens=768" in cmd


def test_run_petri_uses_null_task_arg_when_target_max_tokens_disabled(monkeypatch, tmp_path):
    module = _load_module()
    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    rc = module._run_petri(
        seed_dir=tmp_path,
        seed_file=tmp_path / "seed_prompt.json",
        auditor="auditor-model",
        target="target-model",
        judge="judge-model",
        max_turns=5,
        target_max_tokens=None,
        env={},
        stream_output=False,
    )

    assert rc == 0
    cmd = captured["cmd"]
    assert "target_max_tokens=null" in cmd
