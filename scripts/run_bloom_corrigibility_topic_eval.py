#!/usr/bin/env python3
"""Run Bloom corrigibility evaluation across forbidden topics for one model."""

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

from src.phase2_evaluation.bloom_corrigibility_topics_suite import (  # noqa: E402
    DEFAULT_BASE_BLOOM_CONFIG,
    DEFAULT_SEED_DATASET_PATHS,
    DEFAULT_TEMPLATE_PATH,
    build_corrigibility_topic_suite,
    parse_topic_list,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _safe_model_id(model_id: str) -> str:
    return model_id.replace("/", "_")


def _seed_config_path(suite_dir: Path, topic: str) -> Path:
    return suite_dir / f"seed_corrigibility_{topic}.yaml"


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _coerce_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _extract_topic_metrics(topic: str, result_file: Path) -> dict[str, object]:
    row: dict[str, object] = {
        "topic": topic,
        "n_total_runs": None,
        "n_evaluable_runs": None,
        "n_invalid_runs": None,
        "invalidity_rate": None,
        "avg_behavior_presence_evaluable": None,
        "n_shift_eligible_runs": None,
        "n_shift_success_runs": None,
        "n_shift_stable_runs": None,
        "shift_coverage_rate": None,
        "shift_metrics_valid": None,
        "correction_acceptance_rate_shift_only": None,
        "stability_rate_shift_only": None,
    }
    if not result_file.exists():
        return row

    try:
        payload = json.loads(result_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return row

    n_total_runs = _coerce_int(payload.get("n_total_runs"))
    n_evaluable_runs = _coerce_int(payload.get("n_evaluable_runs"))
    n_invalid_runs = _coerce_int(payload.get("n_invalid_runs"))
    n_shift_eligible_runs = _coerce_int(payload.get("n_shift_eligible_runs"))
    n_shift_success_runs = _coerce_int(payload.get("n_shift_success_runs"))
    n_shift_stable_runs = _coerce_int(payload.get("n_shift_stable_runs"))
    shift_coverage_rate = _coerce_float(payload.get("shift_coverage_rate"))
    shift_metrics_valid = payload.get("shift_metrics_valid")
    if not isinstance(shift_metrics_valid, bool):
        shift_metrics_valid = None
    metrics_evaluable_only = payload.get("metrics_evaluable_only", {})
    if not isinstance(metrics_evaluable_only, dict):
        metrics_evaluable_only = {}
    metrics_shift_only_primary = payload.get("metrics_shift_only_primary", {})
    if not isinstance(metrics_shift_only_primary, dict):
        metrics_shift_only_primary = {}

    invalidity_rate = _coerce_float(metrics_evaluable_only.get("invalidity_rate"))
    avg_behavior_presence = _coerce_float(metrics_evaluable_only.get("average_behavior_presence_score"))
    correction_acceptance_rate = _coerce_float(metrics_shift_only_primary.get("correction_acceptance_rate_shift_only"))
    stability_rate = _coerce_float(metrics_shift_only_primary.get("stability_rate_shift_only"))

    if invalidity_rate is None and n_total_runs and n_invalid_runs is not None:
        invalidity_rate = n_invalid_runs / n_total_runs if n_total_runs > 0 else 0.0
    if shift_coverage_rate is None and n_evaluable_runs and n_shift_eligible_runs is not None:
        shift_coverage_rate = n_shift_eligible_runs / n_evaluable_runs if n_evaluable_runs > 0 else 0.0

    row.update(
        {
            "n_total_runs": n_total_runs,
            "n_evaluable_runs": n_evaluable_runs,
            "n_invalid_runs": n_invalid_runs,
            "invalidity_rate": invalidity_rate,
            "avg_behavior_presence_evaluable": avg_behavior_presence,
            "n_shift_eligible_runs": n_shift_eligible_runs,
            "n_shift_success_runs": n_shift_success_runs,
            "n_shift_stable_runs": n_shift_stable_runs,
            "shift_coverage_rate": shift_coverage_rate,
            "shift_metrics_valid": shift_metrics_valid,
            "correction_acceptance_rate_shift_only": correction_acceptance_rate,
            "stability_rate_shift_only": stability_rate,
        }
    )
    return row


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


def _build_required(suite_dir: Path, topics: list[str], mode: str) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    if not suite_dir.exists():
        return True
    for topic in topics:
        if not _seed_config_path(suite_dir, topic).exists():
            return True
    return True


def _ensure_seed_configs_exist(suite_dir: Path, topics: list[str]) -> None:
    missing = [topic for topic in topics if not _seed_config_path(suite_dir, topic).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing corrigibility topic seed config(s): "
            + ", ".join(f"seed_corrigibility_{name}.yaml" for name in missing)
            + f" in {suite_dir}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Bloom corrigibility evaluation across forbidden topics")
    parser.add_argument("--model-id", required=True, help="Model ID to evaluate")
    parser.add_argument("--adapter-path", type=Path, default=None, help="Optional adapter path")
    parser.add_argument(
        "--petri-transcript-dir",
        type=Path,
        default=None,
        help="Optional transcript directory to attach to result payloads",
    )
    parser.add_argument("--num-tests", type=int, default=50, help="Bloom CLI compatibility (unused when num_scenarios is set)")
    parser.add_argument("--output-dir", default="data/phase2_evaluation", help="Root output directory")
    parser.add_argument(
        "--suite-dir",
        default="data/bloom_corrigibility_topics_suite",
        help="Suite directory containing generated topic seed configs",
    )
    parser.add_argument(
        "--template-path",
        default=str(DEFAULT_TEMPLATE_PATH),
        help="Prompt template path for per-topic configurable prompts",
    )
    parser.add_argument(
        "--topics",
        default=" ".join(parse_topic_list(None)),
        help="Comma/space-separated forbidden topic list",
    )
    parser.add_argument(
        "--examples-per-topic",
        type=int,
        default=int(os.getenv("BLOOM_CORRIGIBILITY_EXAMPLES_PER_TOPIC", "3")),
        help="Synthetic corrigibility examples to materialize per topic",
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
        default=str(DEFAULT_BASE_BLOOM_CONFIG),
        help="Base Bloom config used when building topic suite",
    )
    parser.add_argument(
        "--build",
        choices=["auto", "always", "never"],
        default=os.getenv("BLOOM_CORRIGIBILITY_BUILD", "auto"),
        help="Suite build mode",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first topic failure")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without executing Bloom")

    args = parser.parse_args()

    suite_dir = Path(args.suite_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    topics = parse_topic_list(args.topics)
    seed_dataset_paths = [Path(path) for path in (args.seed_datasets or DEFAULT_SEED_DATASET_PATHS)]

    if _build_required(suite_dir, topics, args.build):
        logger.info("Building corrigibility topic suite in %s", suite_dir)
        build_corrigibility_topic_suite(
            output_dir=suite_dir,
            template_path=Path(args.template_path),
            topics=topics,
            examples_per_topic=args.examples_per_topic,
            seed_dataset_paths=seed_dataset_paths,
            base_config_path=Path(args.base_config),
        )

    _ensure_seed_configs_exist(suite_dir, topics)

    if args.dry_run:
        print("DRY RUN: Bloom corrigibility topic execution plan")

    run_records: list[dict[str, object]] = []
    topic_metrics: list[dict[str, object]] = []
    failures = 0

    evaluator_cls = None
    if not args.dry_run:
        from src.phase2_evaluation.bloom_eval import BloomBehaviorEvaluator

        evaluator_cls = BloomBehaviorEvaluator

    for topic in topics:
        config_path = _seed_config_path(suite_dir, topic)
        topic_output_dir = output_root / f"corrigibility_{topic}"
        topic_output_dir.mkdir(parents=True, exist_ok=True)
        safe_model = _safe_model_id(args.model_id)
        result_file = topic_output_dir / f"bloom_eval_{safe_model}.json"

        if args.dry_run:
            print(
                f"- topic={topic} config={config_path} output_dir={topic_output_dir} "
                f"num_tests={args.num_tests}"
            )
            run_records.append(
                {
                    "topic": topic,
                    "status": "dry_run",
                    "config_path": str(config_path),
                    "output_dir": str(topic_output_dir),
                    "result_file": str(result_file),
                }
            )
            topic_metrics.append(
                {
                    "topic": topic,
                    "n_total_runs": None,
                    "n_evaluable_runs": None,
                    "n_invalid_runs": None,
                    "invalidity_rate": None,
                    "avg_behavior_presence_evaluable": None,
                    "n_shift_eligible_runs": None,
                    "n_shift_success_runs": None,
                    "n_shift_stable_runs": None,
                    "shift_coverage_rate": None,
                    "shift_metrics_valid": None,
                    "correction_acceptance_rate_shift_only": None,
                    "stability_rate_shift_only": None,
                }
            )
            continue

        try:
            assert evaluator_cls is not None
            evaluator = evaluator_cls(bloom_config_path=config_path, output_dir=topic_output_dir)
            evaluator.evaluate_model(
                model_id=args.model_id,
                adapter_path=args.adapter_path,
                petri_transcript_dir=args.petri_transcript_dir,
                num_tests=args.num_tests,
            )
            run_dir = _latest_bloom_run(
                bloom_root=Path("bloom-results"),
                behavior="corrigibility",
                model_id=args.model_id,
            )
            run_records.append(
                {
                    "topic": topic,
                    "status": "success",
                    "config_path": str(config_path),
                    "output_dir": str(topic_output_dir),
                    "result_file": str(result_file),
                    "run_dir": str(run_dir) if run_dir else None,
                }
            )
            topic_metrics.append(_extract_topic_metrics(topic, result_file))
        except Exception as exc:  # noqa: BLE001
            failures += 1
            run_records.append(
                {
                    "topic": topic,
                    "status": "failed",
                    "config_path": str(config_path),
                    "output_dir": str(topic_output_dir),
                    "error": str(exc),
                }
            )
            topic_metrics.append(
                {
                    "topic": topic,
                    "n_total_runs": None,
                    "n_evaluable_runs": None,
                    "n_invalid_runs": None,
                    "invalidity_rate": None,
                    "avg_behavior_presence_evaluable": None,
                    "n_shift_eligible_runs": None,
                    "n_shift_success_runs": None,
                    "n_shift_stable_runs": None,
                    "shift_coverage_rate": None,
                    "shift_metrics_valid": None,
                    "correction_acceptance_rate_shift_only": None,
                    "stability_rate_shift_only": None,
                    "error": str(exc),
                }
            )
            logger.error("Corrigibility topic run failed for %s: %s", topic, exc)
            if args.fail_fast:
                break

    total_runs = 0
    total_evaluable = 0
    total_invalid = 0
    weighted_behavior_sum = 0.0
    weighted_behavior_weight = 0
    n_topics_non_diagnostic_low_shift_coverage = 0
    n_topics_shift_metrics_valid = 0
    weighted_shift_acceptance_sum = 0.0
    weighted_shift_acceptance_weight = 0
    weighted_shift_stability_sum = 0.0
    weighted_shift_stability_weight = 0
    for row in topic_metrics:
        n_total = _coerce_int(row.get("n_total_runs"))
        n_eval = _coerce_int(row.get("n_evaluable_runs"))
        n_invalid = _coerce_int(row.get("n_invalid_runs"))
        avg_behavior = _coerce_float(row.get("avg_behavior_presence_evaluable"))
        n_shift_eligible = _coerce_int(row.get("n_shift_eligible_runs"))
        correction_acceptance_rate = _coerce_float(row.get("correction_acceptance_rate_shift_only"))
        stability_rate = _coerce_float(row.get("stability_rate_shift_only"))
        shift_metrics_valid = row.get("shift_metrics_valid")
        if n_total is not None:
            total_runs += n_total
        if n_eval is not None:
            total_evaluable += n_eval
        if n_invalid is not None:
            total_invalid += n_invalid
        if n_eval and avg_behavior is not None:
            weighted_behavior_sum += avg_behavior * n_eval
            weighted_behavior_weight += n_eval
        if shift_metrics_valid is True:
            n_topics_shift_metrics_valid += 1
            if n_shift_eligible and correction_acceptance_rate is not None:
                weighted_shift_acceptance_sum += correction_acceptance_rate * n_shift_eligible
                weighted_shift_acceptance_weight += n_shift_eligible
            if n_shift_eligible and stability_rate is not None:
                weighted_shift_stability_sum += stability_rate * n_shift_eligible
                weighted_shift_stability_weight += n_shift_eligible
        elif shift_metrics_valid is False:
            n_topics_non_diagnostic_low_shift_coverage += 1

    overall_invalidity_rate = (total_invalid / total_runs) if total_runs > 0 else None
    weighted_avg_behavior = (weighted_behavior_sum / weighted_behavior_weight) if weighted_behavior_weight > 0 else None
    weighted_shift_acceptance = (
        weighted_shift_acceptance_sum / weighted_shift_acceptance_weight
        if weighted_shift_acceptance_weight > 0
        else None
    )
    weighted_shift_stability = (
        weighted_shift_stability_sum / weighted_shift_stability_weight
        if weighted_shift_stability_weight > 0
        else None
    )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "suite": "corrigibility_forbidden_topics",
        "model_id": args.model_id,
        "suite_dir": str(suite_dir),
        "topics": topics,
        "num_tests": args.num_tests,
        "dry_run": args.dry_run,
        "failures": failures,
        "runs": run_records,
        "topic_metrics": topic_metrics,
    }

    safe_model_id = _safe_model_id(args.model_id)
    summary_path = output_root / f"bloom_corrigibility_topics_summary_{safe_model_id}.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    metrics_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "suite": "corrigibility_forbidden_topics",
        "model_id": args.model_id,
        "topics": topics,
        "topic_metrics": topic_metrics,
        "overall": {
            "n_topics": len(topic_metrics),
            "n_total_runs": total_runs,
            "n_evaluable_runs": total_evaluable,
            "n_invalid_runs": total_invalid,
            "invalidity_rate": overall_invalidity_rate,
            "weighted_avg_behavior_presence_evaluable": weighted_avg_behavior,
            "n_topics_shift_metrics_valid": n_topics_shift_metrics_valid,
            "n_topics_non_diagnostic_low_shift_coverage": n_topics_non_diagnostic_low_shift_coverage,
            "weighted_correction_acceptance_rate_shift_only": weighted_shift_acceptance,
            "weighted_stability_rate_shift_only": weighted_shift_stability,
        },
    }
    metrics_path = output_root / f"bloom_corrigibility_topics_metrics_{safe_model_id}.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_report, handle, indent=2)

    logger.info("Wrote corrigibility topic summary to %s", summary_path)
    logger.info("Wrote corrigibility topic metrics to %s", metrics_path)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
