#!/usr/bin/env python3
"""Build topic-grounded corrigibility Bloom suite assets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase2_evaluation.bloom_corrigibility_topics_suite import (  # noqa: E402
    DEFAULT_BASE_BLOOM_CONFIG,
    DEFAULT_SEED_DATASET_PATHS,
    DEFAULT_TEMPLATE_PATH,
    build_corrigibility_topic_suite,
    parse_topic_list,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Bloom corrigibility topic suite.")
    parser.add_argument(
        "--output-dir",
        default="data/bloom_corrigibility_topics_suite",
        help="Output directory to materialize suite assets",
    )
    parser.add_argument(
        "--template-path",
        default=str(DEFAULT_TEMPLATE_PATH),
        help="Prompt template path for per-topic configurable prompts",
    )
    parser.add_argument(
        "--seed-dataset",
        action="append",
        dest="seed_datasets",
        default=None,
        help="Seed dataset JSON path (repeat flag for multiple files)",
    )
    parser.add_argument(
        "--topics",
        default=" ".join(parse_topic_list(None)),
        help="Comma/space-separated forbidden topics",
    )
    parser.add_argument(
        "--examples-per-topic",
        type=int,
        default=3,
        help="Number of synthetic corrigibility examples per topic",
    )
    parser.add_argument(
        "--base-config",
        default=str(DEFAULT_BASE_BLOOM_CONFIG),
        help="Base Bloom YAML to clone for per-topic seed configs",
    )

    args = parser.parse_args()
    seed_dataset_paths = [Path(path) for path in (args.seed_datasets or DEFAULT_SEED_DATASET_PATHS)]

    manifest = build_corrigibility_topic_suite(
        output_dir=Path(args.output_dir),
        template_path=Path(args.template_path),
        topics=args.topics,
        examples_per_topic=args.examples_per_topic,
        seed_dataset_paths=seed_dataset_paths,
        base_config_path=Path(args.base_config),
    )

    print(f"Built corrigibility topic suite at: {Path(args.output_dir).resolve()}")
    print(json.dumps({"topics": manifest["topics"], "manifest": "suite_manifest.json"}, indent=2))


if __name__ == "__main__":
    main()
