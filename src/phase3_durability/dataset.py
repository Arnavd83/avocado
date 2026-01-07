"""Dataset builders for durability analysis."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from petri.scorers.prompts import build_dimension_set

from .behavior import parse_behavior_tags, parse_strategy_tags
from .io import find_eval_files, find_transcript_files, load_all_eval_summaries, load_transcript
from .scoring import compute_severity, severity_to_durability


def _parse_datetime(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).isoformat()
    except ValueError:
        return value


def _parse_batch_model_slug(batch_name: str | None) -> str | None:
    if not batch_name:
        return None
    prefix = "petri_batch_"
    if not batch_name.startswith(prefix):
        return None
    remainder = batch_name[len(prefix):]
    parts = remainder.split("_")
    if len(parts) < 3:
        return None
    slug = "_".join(parts[2:])
    return slug or None


def _extract_batch_info(path: Path) -> dict[str, str | None]:
    batch_name = None
    batch_dir = None

    for parent in [path, *path.parents]:
        if parent.name.startswith("petri_batch_"):
            batch_name = parent.name
            batch_dir = str(parent)
            break

    return {
        "batch_name": batch_name,
        "batch_dir": batch_dir,
        "batch_model_slug": _parse_batch_model_slug(batch_name),
    }


def _build_transcript_record(path: Path, data: dict) -> dict:
    metadata = data.get("metadata", {})
    judge_output = metadata.get("judge_output") or {}
    messages = data.get("target_messages") or data.get("messages") or []

    transcript_id = metadata.get("transcript_id") or path.stem
    prompt = metadata.get("description") or ""
    tags = metadata.get("tags")
    batch_info = _extract_batch_info(path)

    record: dict = {
        "source": "transcript",
        "transcript_path": str(path),
        "transcript_id": transcript_id,
        "created_at": _parse_datetime(metadata.get("created_at")),
        "updated_at": _parse_datetime(metadata.get("updated_at")),
        "auditor_model": metadata.get("auditor_model"),
        "target_model": metadata.get("target_model"),
        "prompt": prompt,
        "tags": tags,
        "message_count": len([msg for msg in messages if msg.get("role") != "system"]),
        "scores": judge_output.get("scores", {}) or {},
        **batch_info,
    }

    return record


def build_transcript_dataset(paths: Iterable[Path]) -> pd.DataFrame:
    records: list[dict] = []
    for path in paths:
        data = load_transcript(path)
        records.append(_build_transcript_record(path, data))
    return pd.DataFrame(records)


def apply_scoring(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        scores = row.get("scores", {})
        if not isinstance(scores, dict):
            scores = {}

        raw_tags = row.get("tags")
        behavior, include_counterfactual, include_universal_plus, normalized_tags = parse_behavior_tags(raw_tags)
        strategy_tags = parse_strategy_tags(raw_tags)
        expected_dimensions = build_dimension_set(
            behavior,
            include_universal_plus=include_universal_plus,
            include_counterfactual=include_counterfactual,
        )

        filtered_scores = {key: value for key, value in scores.items() if key in expected_dimensions}
        missing_dimensions = [dim for dim in expected_dimensions if dim not in filtered_scores]
        out_of_range = [
            key for key, value in filtered_scores.items()
            if isinstance(value, (int, float)) and (value < 0 or value > 10)
        ]

        severity = compute_severity(
            scores,
            behavior=behavior,
            include_universal_plus=include_universal_plus,
        )

        output = row.to_dict()
        output["behavior"] = behavior
        output["strategy"] = ",".join(strategy_tags) if strategy_tags else None
        output["severity_total"] = severity.total
        output["severity_universal"] = severity.universal
        output["severity_behavior_specific"] = severity.behavior_specific
        output["severity_universal_plus"] = severity.universal_plus
        output["durability_total"] = severity_to_durability(severity.total)
        output["composite_durability"] = output["durability_total"]
        output["overall_durability"] = output["durability_total"]
        output["missing_dimensions"] = ",".join(sorted(missing_dimensions))
        output["score_out_of_range"] = ",".join(sorted(out_of_range))
        output["scored_dimensions"] = ",".join(sorted(filtered_scores.keys()))
        output["expected_dimensions"] = ",".join(sorted(expected_dimensions.keys()))
        output["tags_normalized"] = ",".join(normalized_tags)
        rows.append(output)

    return pd.DataFrame(rows)


def build_eval_dataset(paths: Iterable[Path]) -> pd.DataFrame:
    records = load_all_eval_summaries(paths)
    return pd.DataFrame(records)


def build_datasets(root: Path, source: str = "transcript") -> dict[str, pd.DataFrame]:
    eval_paths = find_eval_files(root)
    transcript_paths = find_transcript_files(root)

    datasets: dict[str, pd.DataFrame] = {}

    if source in ("eval", "both"):
        eval_df = build_eval_dataset(eval_paths)
        if not eval_df.empty:
            eval_df["created_at"] = eval_df["created_at"].apply(_parse_datetime)
        datasets["eval"] = apply_scoring(eval_df) if not eval_df.empty else eval_df

    if source in ("transcript", "both"):
        transcript_df = build_transcript_dataset(transcript_paths)
        datasets["transcript"] = apply_scoring(transcript_df) if not transcript_df.empty else transcript_df

    if source == "both" and "eval" in datasets and "transcript" in datasets:
        datasets["combined"] = pd.concat([datasets["eval"], datasets["transcript"]], ignore_index=True)

    return datasets
