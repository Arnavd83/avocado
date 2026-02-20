"""Dataset builders for durability analysis."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .behavior import parse_behavior_tags, parse_strategy_tags
from .io import find_transcript_files, load_transcript
from .score import Score, UniversalScores

# Universal dimension names (must match scoring.UNIVERSAL_WEIGHTS keys)
_UNIVERSAL_FIELDS = {
    "cooperation_with_human_misuse",
    "concerning_with_no_mitigating_factors",
    "adversarial_value_sycophancy",
    "cooperation_with_harmful_sysprompt",
    "concerning",
}


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


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
        "judge_model": metadata.get("judge_model"),
        "prompt": prompt,
        "tags": tags,
        "message_count": len([msg for msg in messages if msg.get("role") != "system"]),
        "scores": judge_output.get("scores", {}) or {},
        **batch_info,
    }

    return record


def build_score_from_record(record: dict) -> Score:
    """Construct a Score from a transcript record dict."""
    tags = record.get("tags")
    behavior, _include_cf, _include_up, _normalized = parse_behavior_tags(tags)
    strategy_tags = parse_strategy_tags(tags)

    raw_scores = record.get("scores", {})
    if not isinstance(raw_scores, dict):
        raw_scores = {}

    # Split into universal vs behavior-specific
    universal_values = {}
    behavior_values = {}
    for key, value in raw_scores.items():
        if not isinstance(value, (int, float)):
            continue
        if key in _UNIVERSAL_FIELDS:
            universal_values[key] = float(value)
        else:
            behavior_values[key] = float(value)

    universal_scores = UniversalScores(
        cooperation_with_human_misuse=universal_values.get("cooperation_with_human_misuse", 0.0),
        concerning_with_no_mitigating_factors=universal_values.get("concerning_with_no_mitigating_factors", 0.0),
        adversarial_value_sycophancy=universal_values.get("adversarial_value_sycophancy", 0.0),
        cooperation_with_harmful_sysprompt=universal_values.get("cooperation_with_harmful_sysprompt", 0.0),
        concerning=universal_values.get("concerning", 0.0),
    )

    timestamp = record.get("created_at") or datetime.now()

    return Score(
        universal_scores=universal_scores,
        behavior_scores=behavior_values,
        model=record.get("target_model") or "",
        prompt=record.get("prompt") or "",
        behavior=behavior,
        category=record.get("batch_model_slug") or "",
        auditor_model=record.get("auditor_model") or "",
        judge_model=record.get("judge_model") or "",
        strategy=",".join(strategy_tags) if strategy_tags else "",
        total_turns=record.get("message_count") or 0,
        timestamp=timestamp,
        transcript_id=record.get("transcript_id") or "",
        raw_scores={k: float(v) for k, v in raw_scores.items() if isinstance(v, (int, float))},
    )


def build_scores(root: Path) -> list[Score]:
    """Find transcripts under *root*, parse them, and return Score objects."""
    paths = find_transcript_files(root)
    scores: list[Score] = []
    for path in paths:
        data = load_transcript(path)
        record = _build_transcript_record(path, data)
        scores.append(build_score_from_record(record))
    return scores
