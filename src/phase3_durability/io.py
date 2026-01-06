"""IO helpers for Petri outputs."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Iterable


def find_eval_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("*.eval"))


def find_transcript_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("transcript_*.json"))


def _read_json_from_zip(zf: zipfile.ZipFile, name: str) -> dict | list:
    with zf.open(name) as handle:
        return json.loads(handle.read().decode("utf-8"))


def load_eval_summary(eval_path: Path) -> list[dict]:
    records: list[dict] = []
    with zipfile.ZipFile(eval_path) as zf:
        names = set(zf.namelist())
        header = _read_json_from_zip(zf, "header.json") if "header.json" in names else {}
        if "summaries.json" not in names:
            return records
        summaries = _read_json_from_zip(zf, "summaries.json")

    eval_info = header.get("eval", {}) if isinstance(header, dict) else {}
    model_roles = eval_info.get("model_roles", {})
    base = {
        "source": "eval_summary",
        "eval_path": str(eval_path),
        "eval_id": eval_info.get("eval_id"),
        "run_id": eval_info.get("run_id"),
        "created_at": eval_info.get("created"),
        "auditor_model": (model_roles.get("auditor") or {}).get("model"),
        "target_model": (model_roles.get("target") or {}).get("model"),
        "judge_model": (model_roles.get("judge") or {}).get("model"),
    }

    for sample in summaries:
        score_block = sample.get("scores", {}).get("alignment_judge", {})
        records.append(
            {
                **base,
                "sample_id": sample.get("id"),
                "epoch": sample.get("epoch"),
                "prompt": sample.get("input"),
                "target": sample.get("target"),
                "started_at": sample.get("started_at"),
                "completed_at": sample.get("completed_at"),
                "message_count": sample.get("message_count"),
                "scores": score_block.get("value", {}) or {},
                "score_explanation": score_block.get("explanation"),
                "score_answer": score_block.get("answer"),
            }
        )

    return records


def load_all_eval_summaries(paths: Iterable[Path]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        records.extend(load_eval_summary(path))
    return records


def load_transcript(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
