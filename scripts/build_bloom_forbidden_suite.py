#!/usr/bin/env python3
"""Build Bloom forbidden-behavior suite assets from Petri + seed data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase2_evaluation.bloom_forbidden_suite import (  # noqa: E402
    DEFAULT_SEED_DATASET_PATHS,
    DEFAULT_TEMPLATE_DIR,
    build_forbidden_suite,
    parse_behavior_list,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Bloom forbidden-behavior suite.")
    parser.add_argument(
        "--output-dir",
        default="data/bloom_forbidden_suite",
        help="Output directory to materialize suite assets",
    )
    parser.add_argument(
        "--template-dir",
        default=str(DEFAULT_TEMPLATE_DIR),
        help="Template directory containing behaviors.json and configurable prompts",
    )
    parser.add_argument(
        "--petri-batch-dir",
        default=None,
        help="Optional Petri batch directory (with plots/transcript.csv) for transcript-backed examples",
    )
    parser.add_argument(
        "--seed-dataset",
        action="append",
        dest="seed_datasets",
        default=None,
        help="Seed dataset JSON path (repeat flag for multiple files)",
    )
    parser.add_argument(
        "--behaviors",
        default=" ".join(parse_behavior_list(None)),
        help="Comma/space-separated forbidden behavior list",
    )
    parser.add_argument(
        "--examples-per-behavior",
        type=int,
        default=3,
        help="Number of examples to materialize per behavior",
    )
    parser.add_argument(
        "--min-severity",
        type=float,
        default=0.0,
        help="Minimum severity_total threshold for transcript-backed examples",
    )
    parser.add_argument(
        "--base-config",
        default="config/bloom_config.yaml",
        help="Base Bloom YAML to clone for per-behavior seed configs",
    )

    args = parser.parse_args()

    seed_dataset_paths = [Path(path) for path in (args.seed_datasets or DEFAULT_SEED_DATASET_PATHS)]

    manifest = build_forbidden_suite(
        output_dir=Path(args.output_dir),
        template_dir=Path(args.template_dir),
        petri_batch_dir=Path(args.petri_batch_dir) if args.petri_batch_dir else None,
        seed_dataset_paths=seed_dataset_paths,
        behaviors=args.behaviors,
        examples_per_behavior=args.examples_per_behavior,
        min_severity=args.min_severity,
        base_config_path=Path(args.base_config),
    )

    print(f"Built forbidden suite at: {Path(args.output_dir).resolve()}")
    print(json.dumps({"behaviors": manifest["behaviors"], "manifest": "suite_manifest.json"}, indent=2))


if __name__ == "__main__":
    main()
