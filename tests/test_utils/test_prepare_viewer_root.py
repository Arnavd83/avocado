"""Tests for transcript viewer root preparation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_prepare_viewer_root(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "prepare_viewer_root.py"), *args],
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )


def _write_transcript(path: Path, target_model: str) -> None:
    path.write_text(
        json.dumps(
            {
                "metadata": {"target_model": target_model},
                "events": [{"role": "user", "content": "test"}],
            }
        ),
        encoding="utf-8",
    )


def test_prepare_viewer_root_supports_corrigibility_topic_summary_with_explicit_run_dirs(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    viewer_root = tmp_path / "viewer"
    bloom_root = tmp_path / "bloom"
    source_root.mkdir(parents=True)
    viewer_root.mkdir(parents=True)
    bloom_root.mkdir(parents=True)

    model_id = "openrouter/anthropic/claude-sonnet-4"
    run_a = bloom_root / "corrigibility" / "20260101_000000_claude_sonnet_4"
    run_b = bloom_root / "corrigibility" / "20260101_000100_claude_sonnet_4"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)
    _write_transcript(run_a / "transcript_v1r1.json", target_model=model_id)
    _write_transcript(run_b / "transcript_v1r1.json", target_model=model_id)

    summary_path = source_root / "bloom_corrigibility_topics_summary_openrouter_anthropic_claude-sonnet-4.json"
    summary_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "suite": "corrigibility_forbidden_topics",
                "model_id": model_id,
                "topics": ["illegal_activity", "harassment"],
                "runs": [
                    {"topic": "illegal_activity", "status": "success", "run_dir": str(run_a)},
                    {"topic": "harassment", "status": "success", "run_dir": str(run_b)},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = run_prepare_viewer_root(
        [
            "--source-root",
            str(source_root),
            "--viewer-root",
            str(viewer_root),
            "--mode",
            "bloom",
            "--bloom-root",
            str(bloom_root),
        ]
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output

    batch_dir = viewer_root / summary_path.stem
    assert (batch_dir / "illegal_activity" / "transcript_v1r1.json").exists()
    assert (batch_dir / "harassment" / "transcript_v1r1.json").exists()

