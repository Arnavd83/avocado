"""Dry-run checks for Petri target_max_tokens wiring in Makefile."""

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


def test_audit_dry_run_includes_target_max_tokens_task_arg():
    result = run_make(["-n", "audit", "TARGET_MAX_TOKENS=512"])
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "-T target_max_tokens=512" in output


def test_audit_seeds_dry_run_passes_target_max_tokens_flag():
    result = run_make(["-n", "audit-seeds", "TARGET_MAX_TOKENS=none"])
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "--target-max-tokens none" in output

