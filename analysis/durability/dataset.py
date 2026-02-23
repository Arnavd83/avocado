"""DataFrame-based dataset builders for analysis consumers.

Reads from the shared SQLite database (via durability.db) and applies
severity/durability scoring for downstream analysis, visualization, and the
web UI.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from petri.scorers.prompts import build_dimension_set

from .behavior import parse_behavior_tags, parse_strategy_tags
from .scoring import compute_severity, severity_to_durability
from durability.db import connect, list_runs, get_transcript


def _parse_datetime(value: str | None) -> str | None:
    from datetime import datetime
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).isoformat()
    except ValueError:
        return value


def build_transcript_dataset(**filters: Any) -> pd.DataFrame:
    """Build a DataFrame of runs from the database.

    Any keyword arguments are passed through as column filters to
    ``durability.db.list_runs`` (e.g. ``behavior="deception"``).
    """
    conn = connect()
    try:
        rows = list_runs(conn, **filters)
    finally:
        conn.close()

    records: list[dict] = []
    for row in rows:
        tags = json.loads(row["tags"]) if isinstance(row["tags"], str) else row["tags"]
        scores = json.loads(row["raw_scores"]) if isinstance(row["raw_scores"], str) else row["raw_scores"]

        records.append({
            "source": "db",
            "run_id": row["run_id"],
            "created_at": _parse_datetime(row.get("timestamp")),
            "auditor_model": row["auditor_model"],
            "target_model": row["target_model"],
            "prompt": row["prompt"],
            "tags": tags,
            "message_count": row.get("total_turns"),
            "scores": scores or {},
            "behavior": row.get("behavior"),
            "strategy": row.get("strategy"),
        })

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


def build_datasets(source: str = "transcript", **filters: Any) -> dict[str, pd.DataFrame]:
    """Build scored datasets from the database.

    Args:
        source: Currently only "transcript" is supported (eval summaries
            are not stored in the DB).
        **filters: Column filters forwarded to ``list_runs``.
    """
    datasets: dict[str, pd.DataFrame] = {}

    if source in ("transcript", "both"):
        transcript_df = build_transcript_dataset(**filters)
        datasets["transcript"] = apply_scoring(transcript_df) if not transcript_df.empty else transcript_df

    return datasets
