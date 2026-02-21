"""Tests for scripts/get_model.py and scripts/get_model_env.py."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_CONFIG = yaml.safe_load((PROJECT_ROOT / "config" / "models.yaml").read_text())
LAMBDA_GPU_CONFIG = MODELS_CONFIG["lambda-ai-gpu"]


def run_script(script_name: str, args: list[str], extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / script_name), *args],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_get_model_inspect_custom_alias_returns_vllm_prefix():
    result = run_script("get_model.py", ["--consumer", "inspect", "lambda-ai-gpu"])
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"vllm/{LAMBDA_GPU_CONFIG['model_name']}"


def test_get_model_bloom_custom_alias_returns_model_name():
    result = run_script("get_model.py", ["--consumer", "bloom", "lambda-ai-gpu"])
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == LAMBDA_GPU_CONFIG["model_name"]


def test_get_model_bloom_direct_id_passthrough():
    direct_model_id = "openrouter/meta-llama/llama-3.1-8b-instruct"
    result = run_script("get_model.py", ["--consumer", "bloom", direct_model_id])
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == direct_model_id


def test_get_model_env_inspect_preserves_vllm_exports():
    result = run_script(
        "get_model_env.py",
        ["--consumer", "inspect", "lambda-ai-gpu"],
        extra_env={"VLLM_API_KEY": "test-vllm-key"},
    )
    assert result.returncode == 0, result.stderr
    assert f"export VLLM_BASE_URL='{LAMBDA_GPU_CONFIG['base_url']}'" in result.stdout
    assert "export VLLM_API_KEY='test-vllm-key'" in result.stdout
    assert "OPENAI_API_BASE" not in result.stdout
    assert "OPENAI_BASE_URL" not in result.stdout
    assert "OPENAI_API_KEY" not in result.stdout


def test_get_model_env_bloom_exports_openai_compat_vars():
    result = run_script(
        "get_model_env.py",
        ["--consumer", "bloom", "lambda-ai-gpu"],
        extra_env={"VLLM_API_KEY": "test-vllm-key"},
    )
    assert result.returncode == 0, result.stderr
    assert f"export OPENAI_API_BASE='{LAMBDA_GPU_CONFIG['base_url']}'" in result.stdout
    assert f"export OPENAI_BASE_URL='{LAMBDA_GPU_CONFIG['base_url']}'" in result.stdout
    assert "export OPENAI_API_KEY='test-vllm-key'" in result.stdout


def test_get_model_env_bloom_missing_key_fails_fast():
    result = run_script(
        "get_model_env.py",
        ["--consumer", "bloom", "lambda-ai-gpu"],
        extra_env={"VLLM_API_KEY": ""},
    )
    assert result.returncode != 0
    assert "VLLM_API_KEY not found in environment" in result.stderr


def test_get_model_env_bloom_direct_id_has_no_extra_exports():
    direct_model_id = "openrouter/meta-llama/llama-3.1-8b-instruct"
    result = run_script("get_model_env.py", ["--consumer", "bloom", direct_model_id])
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""

