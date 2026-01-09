"""CLI wrapper for Kaplan-Meier and turns-to-elicitation analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .survival import (
    build_survival_records,
    prepare_survival_data,
    run_km_analysis,
    run_cox_regression,
    run_cox_with_interaction,
    compute_conditional_turns_summary,
)


def _load_phase2(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Phase 2 file not found: {path}")
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("Phase 2 JSON must be a dict keyed by model_id.")
        return data
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if "model_id" not in df.columns:
            raise ValueError("Phase 2 CSV must include a model_id column.")
        records = {}
        for _, row in df.iterrows():
            model_id = row["model_id"]
            if pd.isna(model_id):
                continue
            records[str(model_id)] = row.to_dict()
        return records
    raise ValueError("Phase 2 file must be .json or .csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Survival analysis utilities")
    parser.add_argument("--input", default="data/scratch/test_petri", help="Root folder with transcripts")
    parser.add_argument("--cache-dir", default="data/scratch/prefix_judge_cache")
    parser.add_argument("--threshold", type=int, default=7, help="Jailbreak threshold (0-10)")
    parser.add_argument("--retries", type=int, default=3, help="Judge retry count per prefix")
    parser.add_argument("--phase2", default=None, help="Optional phase2 JSON/CSV with covariates")
    parser.add_argument("--output-csv", default="data/scratch/plots/survival.csv")
    parser.add_argument("--km", action="store_true", help="Run Kaplan-Meier plots and log-rank tests")
    parser.add_argument("--cox", action="store_true", help="Run Cox regression")
    parser.add_argument("--cox-interaction", action="store_true", help="Run Cox regression with interaction term")
    parser.add_argument("--conditional", action="store_true", help="Print jailbroken-only median turns")
    parser.add_argument("--all", action="store_true", help="Run all analyses")
    args = parser.parse_args()

    phase2 = _load_phase2(Path(args.phase2)) if args.phase2 else None
    transcript_paths = Path(args.input).rglob("transcript_*.json")

    df = build_survival_records(
        transcript_paths,
        phase2_measurements=phase2,
        cache_dir=Path(args.cache_dir),
        threshold=args.threshold,
        retries=args.retries,
    )
    df = prepare_survival_data(df)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    run_all = args.all
    if args.km or run_all:
        results = run_km_analysis(df)
        print("km_results:", results)
    if args.cox or run_all:
        results = run_cox_regression(df)
        print("cox_results:", results)
    if args.cox_interaction or run_all:
        results = run_cox_with_interaction(df)
        print("cox_interaction:", results)
    if args.conditional or run_all:
        results = compute_conditional_turns_summary(df)
        print("conditional_turns_summary:", results)


if __name__ == "__main__":
    main()
