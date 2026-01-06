"""Dataset builders for durability analysis."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from .io import find_eval_files, find_transcript_files, load_all_eval_summaries, load_transcript
from .nlp import aggregate_texts, extract_counts, compute_feature_vector
from .scoring import (
    CATEGORY_CONFIG,
    compute_category_scores,
    compute_composite_durability,
    compute_composite_risk,
)


PHASES = ("early", "mid", "late")


def _parse_datetime(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).isoformat()
    except ValueError:
        return value


def _normalize_role(role: str) -> str:
    if role == "assistant":
        return "target"
    if role == "user":
        return "auditor"
    return role or "unknown"


def _assign_phase(index: int, total: int) -> str:
    if total <= 0:
        return "early"
    if index < total / 3:
        return "early"
    if index < 2 * total / 3:
        return "mid"
    return "late"


def _collect_phase_texts(messages: list[dict]) -> dict[str, list[str]]:
    texts: dict[str, list[str]] = {phase: [] for phase in PHASES}
    non_system = [msg for msg in messages if msg.get("role") != "system"]
    total = len(non_system)
    for idx, msg in enumerate(non_system):
        phase = _assign_phase(idx, total)
        texts[phase].append(msg.get("content", ""))
    return texts


def _aggregate_prefix(prefix: str, features: dict[str, float], record: dict) -> None:
    for name, value in features.items():
        record[f"{prefix}{name}"] = value


def _extract_message_records(transcript_id: str, messages: list[dict]) -> list[dict]:
    records: list[dict] = []
    non_system = [msg for msg in messages if msg.get("role") != "system"]
    total = len(non_system)

    for idx, msg in enumerate(non_system):
        phase = _assign_phase(idx, total)
        role = _normalize_role(msg.get("role"))
        features = compute_feature_vector(extract_counts(msg.get("content", "")))
        records.append(
            {
                "transcript_id": transcript_id,
                "role": role,
                "phase": phase,
                **features,
            }
        )

    return records


def _build_transcript_record(path: Path, data: dict) -> tuple[dict, list[dict]]:
    metadata = data.get("metadata", {})
    judge_output = metadata.get("judge_output") or {}
    messages = data.get("target_messages") or data.get("messages") or []

    transcript_id = metadata.get("transcript_id") or path.stem
    prompt = metadata.get("description") or ""

    target_texts = [msg.get("content", "") for msg in messages if msg.get("role") == "assistant"]
    auditor_texts = [msg.get("content", "") for msg in messages if msg.get("role") == "user"]
    overall_texts = [msg.get("content", "") for msg in messages if msg.get("role") != "system"]

    prompt_features = aggregate_texts([prompt])
    target_features = aggregate_texts(target_texts)
    auditor_features = aggregate_texts(auditor_texts)
    overall_features = aggregate_texts(overall_texts)

    phase_texts = _collect_phase_texts(messages)

    record: dict = {
        "source": "transcript",
        "transcript_path": str(path),
        "transcript_id": transcript_id,
        "created_at": _parse_datetime(metadata.get("created_at")),
        "updated_at": _parse_datetime(metadata.get("updated_at")),
        "auditor_model": metadata.get("auditor_model"),
        "target_model": metadata.get("target_model"),
        "prompt": prompt,
        "message_count": len([msg for msg in messages if msg.get("role") != "system"]),
        "scores": judge_output.get("scores", {}) or {},
    }

    _aggregate_prefix("nlp_prompt_", prompt_features, record)
    _aggregate_prefix("nlp_target_", target_features, record)
    _aggregate_prefix("nlp_auditor_", auditor_features, record)
    _aggregate_prefix("nlp_overall_", overall_features, record)

    for phase, texts in phase_texts.items():
        phase_features = aggregate_texts(texts)
        _aggregate_prefix(f"nlp_phase_{phase}_", phase_features, record)

    message_records = _extract_message_records(transcript_id, messages)
    return record, message_records


def build_transcript_datasets(paths: Iterable[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict] = []
    message_records: list[dict] = []

    for path in paths:
        data = load_transcript(path)
        record, message_rows = _build_transcript_record(path, data)
        records.append(record)
        message_records.extend(message_rows)

    return pd.DataFrame(records), pd.DataFrame(message_records)


def _combine_lexicon_rates(record: dict) -> dict[str, float]:
    lexicon_rates = {}
    for key in CATEGORY_CONFIG:
        lex_key = CATEGORY_CONFIG[key]["text_lexicon"]
        prompt_rate = record.get(f"nlp_prompt_lex_{lex_key}_rate", 0.0) or 0.0
        target_rate = record.get(f"nlp_target_lex_{lex_key}_rate", 0.0) or 0.0
        lexicon_rates[lex_key] = max(float(prompt_rate), float(target_rate))
    return lexicon_rates


def apply_scoring(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        scores = row.get("scores", {})
        if not isinstance(scores, dict):
            scores = {}
        missing_dimensions = set()
        for config in CATEGORY_CONFIG.values():
            for dim in config.get("dimension_weights", {}):
                if dim not in scores:
                    missing_dimensions.add(dim)
        out_of_range = [
            key for key, value in scores.items()
            if isinstance(value, (int, float)) and (value < 1 or value > 10)
        ]
        lexicon_rates = _combine_lexicon_rates(row)
        category_scores = compute_category_scores(scores, lexicon_rates)

        output = row.to_dict()
        for key, score in category_scores.items():
            output[f"risk_{key}"] = score.risk
            output[f"durability_{key}"] = score.durability
            output[f"risk_{key}_dim"] = score.dim_score
            output[f"risk_{key}_text"] = score.text_score
        output["composite_durability"] = compute_composite_durability(category_scores)
        output["composite_risk"] = compute_composite_risk(category_scores)
        output["overall_durability"] = output["composite_durability"]
        output["missing_dimensions"] = ",".join(sorted(missing_dimensions))
        output["score_out_of_range"] = ",".join(sorted(out_of_range))
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
        transcript_df, message_df = build_transcript_datasets(transcript_paths)
        datasets["transcript"] = apply_scoring(transcript_df) if not transcript_df.empty else transcript_df
        datasets["messages"] = message_df

    if source == "both" and "eval" in datasets and "transcript" in datasets:
        datasets["combined"] = pd.concat([datasets["eval"], datasets["transcript"]], ignore_index=True)

    return datasets
