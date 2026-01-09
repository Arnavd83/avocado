#!/usr/bin/env python3
"""Aggregate seed batch results into datasets and summary tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.phase3_durability.dataset import build_datasets


METRICS = [
    "durability_total",
    "severity_total",
    "severity_universal",
    "severity_behavior_specific",
]


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        return []
    return data


def _build_seed_map(manifest: list[dict[str, Any]]) -> dict[Path, dict[str, Any]]:
    seed_map: dict[Path, dict[str, Any]] = {}
    for entry in manifest:
        output_dir = entry.get("output_dir")
        if not output_dir:
            continue
        seed_map[Path(output_dir).resolve()] = entry
    return seed_map


def _seed_for_path(path: Path, seed_map: dict[Path, dict[str, Any]]) -> dict[str, Any] | None:
    if not seed_map:
        return None
    path = path.resolve()
    for candidate in [path, *path.parents]:
        if candidate in seed_map:
            return seed_map[candidate]
    return None


def _attach_seed_index(df: pd.DataFrame, seed_map: dict[Path, dict[str, Any]]) -> pd.DataFrame:
    if "transcript_path" not in df.columns:
        return df
    seed_indices: list[int | None] = []
    seed_names: list[str | None] = []
    for _, row in df.iterrows():
        path_value = row.get("transcript_path")
        if not isinstance(path_value, str):
            seed_indices.append(None)
            seed_names.append(None)
            continue
        seed_info = _seed_for_path(Path(path_value), seed_map)
        if seed_info:
            seed_indices.append(seed_info.get("seed_index"))
            seed_names.append(seed_info.get("seed_name"))
        else:
            seed_indices.append(None)
            seed_names.append(None)
    df = df.copy()
    df["seed_index"] = seed_indices
    df["seed_name"] = seed_names
    return df


def _sanitize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for metric in METRICS:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")
    return df


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    df = _sanitize_metrics(df)
    rows: list[dict[str, Any]] = []

    def add_summary(scope: str, group_cols: list[str] | None = None) -> None:
        if not group_cols:
            row = {"scope": scope, "count": int(len(df))}
            for metric in METRICS:
                if metric in df.columns:
                    row[f"{metric}_mean"] = float(df[metric].mean())
            if "durability_total_mean" in row:
                row["score"] = row["durability_total_mean"]
            rows.append(row)
            return

        grouped = df.groupby(group_cols, dropna=False)
        for keys, group in grouped:
            key_tuple = keys if isinstance(keys, tuple) else (keys,)
            row = {"scope": scope, "count": int(len(group))}
            for col, key in zip(group_cols, key_tuple):
                row[col] = None if pd.isna(key) else key
            for metric in METRICS:
                if metric in group.columns:
                    row[f"{metric}_mean"] = float(group[metric].mean())
            if "durability_total_mean" in row:
                row["score"] = row["durability_total_mean"]
            rows.append(row)

    add_summary("composite")
    add_summary("behavior", ["behavior"])
    add_summary("strategy", ["strategy"])
    add_summary("behavior_strategy", ["behavior", "strategy"])

    return pd.DataFrame(rows)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate batch outputs into summary tables.")
    parser.add_argument("--batch-dir", required=True, help="Batch folder created by run_seed_dataset.py")
    args = parser.parse_args()

    batch_dir = Path(args.batch_dir)
    if not batch_dir.exists():
        raise FileNotFoundError(f"Batch dir not found: {batch_dir}")

    datasets = build_datasets(batch_dir, source="transcript")
    transcript_df = datasets.get("transcript")
    if transcript_df is None or transcript_df.empty:
        raise RuntimeError("No transcripts found for aggregation.")

    manifest = _load_manifest(batch_dir / "manifest.json")
    seed_map = _build_seed_map(manifest)
    transcript_df = _attach_seed_index(transcript_df, seed_map)

    summary_df = _summarize(transcript_df)

    output_dir = batch_dir / "plots"
    for name in ("transcript.csv", "summary.csv"):
        path = output_dir / name
        if path.exists():
            path.unlink()
    _write_csv(transcript_df, output_dir / "transcript.csv")
    _write_csv(summary_df, output_dir / "summary.csv")

    print(f"Wrote transcript dataset to {output_dir / 'transcript.csv'}")
    print(f"Wrote summary table to {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
