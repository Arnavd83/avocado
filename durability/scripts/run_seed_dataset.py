#!/usr/bin/env python3
"""Run Petri audits for every seed in config/seed_dataset_<name>.json.

This script is kept for backward compatibility. The preferred entry point
is ``durability run`` (or ``python -m durability run``).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from shared.paths import PROJECT_ROOT

from durability._runner import (
    DEFAULT_AUDITOR_MODEL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_TARGET_MODEL,
    load_seed_dataset,
    run_batch,
    save_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Petri audits for a seed dataset.")
    parser.add_argument(
        "--seed-dataset",
        default="durability/config/seed_dataset_easy.json",
        help="Path to seed dataset JSON",
    )
    parser.add_argument("--output-root", default="data/scratch", help="Root output folder")
    parser.add_argument("--auditor-model-id", default=os.getenv("AUDITOR_MODEL_ID", DEFAULT_AUDITOR_MODEL))
    parser.add_argument("--target-model-id", default=os.getenv("TARGET_MODEL_ID", DEFAULT_TARGET_MODEL))
    parser.add_argument("--judge-model-id", default=os.getenv("JUDGE_MODEL_ID", DEFAULT_JUDGE_MODEL))
    parser.add_argument("--max-turns", type=int, default=int(os.getenv("MAX_TURNS", "10")))
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=int(os.getenv("BATCH_MAX_PARALLEL", "1")),
        help="Maximum number of seeds to run concurrently",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    parser.add_argument("--no-aggregate", action="store_true", help="Skip aggregation step")
    parser.add_argument("--stream-output", action="store_true", help="Stream inspect output to console")
    args = parser.parse_args()

    load_dotenv()

    seed_path = Path(args.seed_dataset)
    seed_entries = load_seed_dataset(seed_path)

    batch_root, manifest = run_batch(
        seed_entries,
        output_root=Path(args.output_root),
        auditor_id=args.auditor_model_id,
        target_id=args.target_model_id,
        judge_id=args.judge_model_id,
        max_turns=args.max_turns,
        max_parallel=args.max_parallel,
        stream_output=args.stream_output,
        fail_fast=args.fail_fast,
    )

    if not args.no_aggregate:
        aggregate_script = PROJECT_ROOT / "durability" / "scripts" / "aggregate_seed_batch.py"
        env = os.environ.copy()
        subprocess.run(
            [sys.executable, str(aggregate_script), "--batch-dir", str(batch_root)],
            cwd=str(PROJECT_ROOT),
            env=env,
            check=False,
        )

    durations = [
        entry.get("duration_seconds")
        for entry in manifest
        if isinstance(entry.get("duration_seconds"), (int, float))
    ]
    if durations:
        total_seconds = sum(durations)
        avg_seconds = total_seconds / len(durations)
        timing_summary = {
            "seed_count": len(durations),
            "total_seconds": total_seconds,
            "avg_seconds_per_seed": avg_seconds,
            "batch_name": batch_root.name,
        }
        save_json(batch_root / "timing_summary.json", timing_summary)
        print(
            "Timing summary: "
            f"{total_seconds:.1f}s total across {len(durations)} seeds "
            f"({avg_seconds:.1f}s avg/seed)"
        )

    print(f"Batch complete: {batch_root}")
    print(f"Manifest: {batch_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
