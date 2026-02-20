"""CLI for durability visualization and exports."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .dataset import build_datasets


PRESETS = (
    "durability_by_behavior",
    "durability_distribution",
    "severity_distribution",
    "composite_over_time",
)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_durability_by_behavior(df: pd.DataFrame, output: Path) -> None:
    if "behavior" not in df.columns or "durability_total" not in df.columns:
        return
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="behavior", y="durability_total")
    plt.xticks(rotation=30, ha="right")
    plt.title("Durability by Behavior")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_durability_distribution(df: pd.DataFrame, output: Path) -> None:
    if "durability_total" not in df.columns:
        return
    plt.figure(figsize=(8, 5))
    sns.histplot(df["durability_total"].dropna(), bins=20, kde=True)
    plt.title("Durability Distribution")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_composite_over_time(df: pd.DataFrame, output: Path) -> None:
    if "created_at" not in df.columns or "durability_total" not in df.columns:
        return
    temp = df.dropna(subset=["created_at"]).copy()
    if temp.empty:
        return
    temp["created_at"] = pd.to_datetime(temp["created_at"], errors="coerce")
    temp = temp.dropna(subset=["created_at"]).sort_values("created_at")
    if temp.empty:
        return
    plt.figure(figsize=(10, 5))
    plt.plot(temp["created_at"], temp["durability_total"], marker="o", linewidth=1)
    plt.title("Durability Over Time")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_severity_distribution(df: pd.DataFrame, output: Path) -> None:
    if "severity_total" not in df.columns:
        return
    plt.figure(figsize=(8, 5))
    sns.histplot(df["severity_total"].dropna(), bins=20, kde=True)
    plt.title("Severity Distribution")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def export_datasets(datasets: dict[str, pd.DataFrame], output_dir: Path) -> None:
    for name, df in datasets.items():
        if df is None or df.empty:
            continue
        out_path = output_dir / f"{name}.csv"
        df.to_csv(out_path, index=False)


def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("No records to summarize.")
        return

    id_col = "run_id" if "run_id" in df.columns else "transcript_id"
    cols = [id_col]
    if "created_at" in df.columns:
        cols.append("created_at")
    if "behavior" in df.columns:
        cols.append("behavior")
    if "durability_total" in df.columns:
        cols.append("durability_total")
    if "severity_total" in df.columns:
        cols.append("severity_total")

    summary = df[cols].copy()
    if "created_at" in summary.columns:
        summary["created_at"] = pd.to_datetime(summary["created_at"], errors="coerce", utc=True)
        summary = summary.sort_values("created_at")
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Durability visualization utilities")
    parser.add_argument("--input", default="data/scratch", help="Root folder with Petri outputs")
    parser.add_argument("--source", choices=["transcript", "eval", "both"], default="transcript")
    parser.add_argument("--output-dir", default="data/scratch/plots")
    parser.add_argument("--format", choices=["png", "svg"], default="png")
    parser.add_argument("--preset", action="append", choices=PRESETS + ("all",), default=None)
    parser.add_argument("--export", action="store_true", help="Export CSV datasets")
    parser.add_argument("--no-summary", action="store_true", help="Disable summary table output")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    _ensure_dir(output_dir)

    datasets = build_datasets(input_path, source=args.source)
    if args.export:
        export_datasets(datasets, output_dir)

    if args.source == "both":
        df = datasets.get("combined", pd.DataFrame())
    else:
        df = datasets.get(args.source, pd.DataFrame())

    if not args.no_summary:
        print_summary(df)

    presets = args.preset or ["durability_by_behavior", "durability_distribution"]
    if "all" in presets:
        presets = list(PRESETS)

    for preset in presets:
        if preset == "durability_by_behavior":
            plot_durability_by_behavior(df, output_dir / f"durability_by_behavior.{args.format}")
        elif preset == "durability_distribution":
            plot_durability_distribution(df, output_dir / f"durability_distribution.{args.format}")
        elif preset == "severity_distribution":
            plot_severity_distribution(df, output_dir / f"severity_distribution.{args.format}")
        elif preset == "composite_over_time":
            plot_composite_over_time(df, output_dir / f"composite_over_time.{args.format}")


if __name__ == "__main__":
    main()
