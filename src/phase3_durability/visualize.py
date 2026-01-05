"""CLI for durability visualization and exports."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .dataset import build_datasets


PRESETS = (
    "durability_by_category",
    "durability_distribution",
    "composite_over_time",
    "category_heatmap",
    "nlp_feature_scatter",
    "role_feature_compare",
)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _category_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("durability_") and not col.endswith("_text") and not col.endswith("_dim")]


def plot_durability_by_category(df: pd.DataFrame, output: Path) -> None:
    cols = _category_columns(df)
    if not cols:
        return
    long_df = df.melt(value_vars=cols, var_name="category", value_name="durability")
    long_df["category"] = long_df["category"].str.replace("durability_", "", regex=False)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=long_df, x="category", y="durability")
    plt.xticks(rotation=30, ha="right")
    plt.title("Durability by Category")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_durability_distribution(df: pd.DataFrame, output: Path) -> None:
    if "composite_durability" not in df.columns:
        return
    plt.figure(figsize=(8, 5))
    sns.histplot(df["composite_durability"].dropna(), bins=20, kde=True)
    plt.title("Composite Durability Distribution")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_composite_over_time(df: pd.DataFrame, output: Path) -> None:
    if "created_at" not in df.columns or "composite_durability" not in df.columns:
        return
    temp = df.dropna(subset=["created_at"]).copy()
    if temp.empty:
        return
    temp["created_at"] = pd.to_datetime(temp["created_at"], errors="coerce")
    temp = temp.dropna(subset=["created_at"]).sort_values("created_at")
    if temp.empty:
        return
    plt.figure(figsize=(10, 5))
    plt.plot(temp["created_at"], temp["composite_durability"], marker="o", linewidth=1)
    plt.title("Composite Durability Over Time")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_category_heatmap(df: pd.DataFrame, output: Path) -> None:
    cols = _category_columns(df)
    if not cols:
        return
    label_col = "run_id" if "run_id" in df.columns else "transcript_id"
    temp = df[[label_col] + cols].copy()
    if temp.empty:
        return
    temp = temp.set_index(label_col)
    plt.figure(figsize=(12, max(4, len(temp) * 0.25)))
    sns.heatmap(temp, cmap="viridis", cbar_kws={"label": "Durability"})
    plt.title("Durability Heatmap")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_nlp_feature_scatter(df: pd.DataFrame, output: Path, x_feature: str, y_feature: str) -> None:
    if x_feature not in df.columns or y_feature not in df.columns:
        return
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x=x_feature, y=y_feature)
    plt.title(f"{y_feature} vs {x_feature}")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_role_feature_compare(message_df: pd.DataFrame, output: Path, feature: str) -> None:
    if message_df.empty or feature not in message_df.columns:
        return
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=message_df, x="role", y=feature)
    plt.title(f"{feature} by Role")
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
    if "overall_durability" in df.columns:
        cols.append("overall_durability")
    elif "composite_durability" in df.columns:
        cols.append("composite_durability")
    if "composite_risk" in df.columns:
        cols.append("composite_risk")

    category_cols = _category_columns(df)
    cols.extend(category_cols)

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
    parser.add_argument("--scatter-x", default="nlp_target_numeric_density")
    parser.add_argument("--scatter-y", default="composite_durability")
    parser.add_argument("--role-feature", default="refusal_ratio")
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

    message_df = datasets.get("messages", pd.DataFrame())
    if not args.no_summary:
        print_summary(df)

    presets = args.preset or ["durability_by_category", "durability_distribution"]
    if "all" in presets:
        presets = list(PRESETS)

    for preset in presets:
        if preset == "durability_by_category":
            plot_durability_by_category(df, output_dir / f"durability_by_category.{args.format}")
        elif preset == "durability_distribution":
            plot_durability_distribution(df, output_dir / f"durability_distribution.{args.format}")
        elif preset == "composite_over_time":
            plot_composite_over_time(df, output_dir / f"composite_over_time.{args.format}")
        elif preset == "category_heatmap":
            plot_category_heatmap(df, output_dir / f"category_heatmap.{args.format}")
        elif preset == "nlp_feature_scatter":
            plot_nlp_feature_scatter(
                df,
                output_dir / f"nlp_feature_scatter.{args.format}",
                args.scatter_x,
                args.scatter_y,
            )
        elif preset == "role_feature_compare":
            plot_role_feature_compare(
                message_df,
                output_dir / f"role_feature_compare.{args.format}",
                args.role_feature,
            )


if __name__ == "__main__":
    main()
