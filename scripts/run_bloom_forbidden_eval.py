#!/usr/bin/env python3
"""Run Bloom evaluations across forbidden behaviors for one target model."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _safe_model_id(model_id: str) -> str:
    return model_id.replace("/", "_")


def _seed_config_path(suite_dir: Path, behavior: str) -> Path:
    return suite_dir / f"seed_{behavior}.yaml"


def _latest_bloom_run(
    bloom_root: Path,
    behavior: str,
    model_id: str,
) -> Path | None:
    behavior_dir = bloom_root / behavior
    if not behavior_dir.exists() or not behavior_dir.is_dir():
        return None

    runs: list[Path] = []
    for run_dir in behavior_dir.iterdir():
        if not run_dir.is_dir():
            continue
        if not any(run_dir.glob("transcript_v*r*.json")):
            continue
        run_info = run_dir / "_run_info.json"
        if run_info.exists():
            try:
                payload = json.loads(run_info.read_text(encoding="utf-8"))
                target_model = payload.get("target_model")
                if isinstance(target_model, str) and target_model and target_model != model_id:
                    continue
            except json.JSONDecodeError:
                pass
        runs.append(run_dir)
    if not runs:
        return None
    return max(runs, key=lambda path: path.stat().st_mtime)


def _build_required(
    suite_dir: Path,
    behaviors: list[str],
    mode: str,
) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False

    # auto mode: refresh by default to reflect latest transcript/seed data.
    if not suite_dir.exists():
        return True
    for behavior in behaviors:
        if not _seed_config_path(suite_dir, behavior).exists():
            return True
    return True


def _ensure_seed_configs_exist(suite_dir: Path, behaviors: list[str]) -> None:
    missing = [behavior for behavior in behaviors if not _seed_config_path(suite_dir, behavior).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing forbidden suite seed config(s): "
            + ", ".join(f"seed_{name}.yaml" for name in missing)
            + f" in {suite_dir}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Bloom forbidden-behavior evaluation for one model")
    parser.add_argument("--model-id", required=True, help="Model ID to evaluate")
    parser.add_argument("--adapter-path", type=Path, default=None, help="Optional adapter path")
    parser.add_argument(
        "--petri-transcript-dir",
        type=Path,
        default=None,
        help="Optional transcript directory to attach to result payloads",
    )
    parser.add_argument("--num-tests", type=int, default=50, help="Number of tests per behavior")
    parser.add_argument("--output-dir", default="data/phase2_evaluation", help="Root output directory")
    parser.add_argument(
        "--suite-dir",
        default="data/bloom_forbidden_suite",
        help="Suite directory containing generated behavior seed configs",
    )
    parser.add_argument(
        "--template-dir",
        default=str(DEFAULT_TEMPLATE_DIR),
        help="Template directory containing forbidden suite static assets",
    )
    parser.add_argument(
        "--behaviors",
        default=" ".join(parse_behavior_list(None)),
        help="Comma/space-separated forbidden behavior list",
    )
    parser.add_argument(
        "--examples-per-behavior",
        type=int,
        default=int(os.getenv("BLOOM_FORBIDDEN_EXAMPLES_PER_BEHAVIOR", "3")),
        help="Examples to build per behavior",
    )
    parser.add_argument(
        "--min-severity",
        type=float,
        default=float(os.getenv("BLOOM_FORBIDDEN_MIN_SEVERITY", "0.0")),
        help="Minimum transcript severity for transcript-backed examples",
    )
    parser.add_argument(
        "--petri-batch-dir",
        default=os.getenv("BLOOM_FORBIDDEN_PETRI_BATCH_DIR") or None,
        help="Optional Petri batch directory for transcript-backed examples",
    )
    parser.add_argument(
        "--seed-dataset",
        action="append",
        dest="seed_datasets",
        default=None,
        help="Seed dataset JSON path (repeat for multiple files)",
    )
    parser.add_argument(
        "--base-config",
        default="config/bloom_config.yaml",
        help="Base Bloom config used when building the forbidden suite",
    )
    parser.add_argument(
        "--build",
        choices=["auto", "always", "never"],
        default=os.getenv("BLOOM_FORBIDDEN_BUILD", "auto"),
        help="Suite build mode",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first behavior failure")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without executing Bloom")

    args = parser.parse_args()

    suite_dir = Path(args.suite_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    behaviors = parse_behavior_list(args.behaviors)
    seed_dataset_paths = [Path(path) for path in (args.seed_datasets or DEFAULT_SEED_DATASET_PATHS)]

    if _build_required(suite_dir, behaviors, args.build):
        logger.info("Building forbidden suite assets in %s", suite_dir)
        build_forbidden_suite(
            output_dir=suite_dir,
            template_dir=Path(args.template_dir),
            behaviors=behaviors,
            examples_per_behavior=args.examples_per_behavior,
            min_severity=args.min_severity,
            petri_batch_dir=Path(args.petri_batch_dir) if args.petri_batch_dir else None,
            seed_dataset_paths=seed_dataset_paths,
            base_config_path=Path(args.base_config),
        )

    _ensure_seed_configs_exist(suite_dir, behaviors)

    if args.dry_run:
        print("DRY RUN: Bloom forbidden suite execution plan")

    run_records: list[dict[str, object]] = []
    failures = 0

    evaluator_cls = None
    if not args.dry_run:
        from src.phase2_evaluation.bloom_eval import BloomBehaviorEvaluator

        evaluator_cls = BloomBehaviorEvaluator

    for behavior in behaviors:
        config_path = _seed_config_path(suite_dir, behavior)
        behavior_output_dir = output_root / f"forbidden_{behavior}"
        behavior_output_dir.mkdir(parents=True, exist_ok=True)
        safe_model = _safe_model_id(args.model_id)
        result_file = behavior_output_dir / f"bloom_eval_{safe_model}.json"

        if args.dry_run:
            print(
                f"- behavior={behavior} config={config_path} output_dir={behavior_output_dir} "
                f"num_tests={args.num_tests}"
            )
            run_records.append(
                {
                    "behavior": behavior,
                    "status": "dry_run",
                    "config_path": str(config_path),
                    "output_dir": str(behavior_output_dir),
                    "result_file": str(result_file),
                }
            )
            continue

        try:
            assert evaluator_cls is not None
            evaluator = evaluator_cls(bloom_config_path=config_path, output_dir=behavior_output_dir)
            evaluator.evaluate_model(
                model_id=args.model_id,
                adapter_path=args.adapter_path,
                petri_transcript_dir=args.petri_transcript_dir,
                num_tests=args.num_tests,
            )
            run_dir = _latest_bloom_run(
                bloom_root=Path("bloom-results"),
                behavior=behavior,
                model_id=args.model_id,
            )
            run_records.append(
                {
                    "behavior": behavior,
                    "status": "success",
                    "config_path": str(config_path),
                    "output_dir": str(behavior_output_dir),
                    "result_file": str(result_file),
                    "run_dir": str(run_dir) if run_dir else None,
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            run_records.append(
                {
                    "behavior": behavior,
                    "status": "failed",
                    "config_path": str(config_path),
                    "output_dir": str(behavior_output_dir),
                    "error": str(exc),
                }
            )
            logger.error("Forbidden behavior run failed for %s: %s", behavior, exc)
            if args.fail_fast:
                break

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "suite": "forbidden",
        "model_id": args.model_id,
        "suite_dir": str(suite_dir),
        "behaviors": behaviors,
        "num_tests": args.num_tests,
        "dry_run": args.dry_run,
        "failures": failures,
        "runs": run_records,
    }

    summary_path = output_root / f"bloom_forbidden_summary_{_safe_model_id(args.model_id)}.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    logger.info("Wrote forbidden suite summary to %s", summary_path)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
