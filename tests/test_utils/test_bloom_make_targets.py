"""Dry-run checks for Bloom utility Make targets."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_make(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["make", *args],
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )


def test_bloom_eval_utility_dry_run_uses_bloom_consumer_and_eval_script():
    result = run_make(
        [
            "-n",
            "bloom-eval-utility",
            "UTILITY_MODELS=qwen-3-8b lambda-ai-gpu",
        ]
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "scripts/get_model.py --consumer bloom" in output
    assert "scripts/get_model_env.py --consumer bloom" in output
    assert "scripts/run_bloom_eval.py" in output


def test_bloom_eval_utility_resolve_dry_run_uses_bloom_consumer():
    result = run_make(
        [
            "-n",
            "bloom-eval-utility-resolve",
            "UTILITY_MODELS=qwen-3-8b,lambda-ai-gpu",
        ]
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "scripts/get_model.py --consumer bloom" in output
    assert "scripts/get_model_env.py --consumer bloom" in output


def test_bloom_eval_forbidden_dry_run_dispatches_to_forbidden_runner():
    result = run_make(
        [
            "-n",
            "bloom-eval",
            "BLOOM_SUITE=forbidden",
            "MODEL_ID=qwen-3-8b",
        ]
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "scripts/run_bloom_forbidden_eval.py" in output


def test_bloom_eval_default_dry_run_keeps_existing_runner():
    result = run_make(
        [
            "-n",
            "bloom-eval",
            "MODEL_ID=qwen-3-8b",
        ]
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "scripts/run_bloom_eval.py" in output


def test_bloom_eval_utility_forbidden_dry_run_uses_forbidden_runner():
    result = run_make(
        [
            "-n",
            "bloom-eval-utility",
            "BLOOM_SUITE=forbidden",
            "UTILITY_MODELS=qwen-3-8b",
        ]
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "scripts/run_bloom_forbidden_eval.py" in output


def test_bloom_eval_corrigibility_topic_dry_run_dispatches_to_topic_runner():
    result = run_make(
        [
            "-n",
            "bloom-eval",
            "BLOOM_SUITE=corrigibility",
            "BLOOM_CORRIGIBILITY_TOPIC_SET=forbidden",
            "MODEL_ID=qwen-3-8b",
        ]
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "scripts/run_bloom_corrigibility_topic_eval.py" in output


def test_bloom_eval_utility_corrigibility_topic_dry_run_dispatches_to_topic_runner():
    result = run_make(
        [
            "-n",
            "bloom-eval-utility",
            "BLOOM_SUITE=corrigibility",
            "BLOOM_CORRIGIBILITY_TOPIC_SET=forbidden",
            "UTILITY_MODELS=qwen-3-8b",
        ]
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "scripts/run_bloom_corrigibility_topic_eval.py" in output
