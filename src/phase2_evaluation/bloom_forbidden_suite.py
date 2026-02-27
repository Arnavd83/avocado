"""Build Bloom forbidden-behavior suites from Petri transcripts and seed datasets."""

from __future__ import annotations

import copy
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

FORBIDDEN_BEHAVIORS: tuple[str, ...] = (
    "illegal_activity",
    "deception_manipulation",
    "harassment",
    "privacy_violation",
    "malware_cyberweapons",
)

DEFAULT_SEED_DATASET_PATHS: tuple[Path, ...] = (
    Path("config/seed_dataset_easy.json"),
    Path("config/seed_dataset_hard.json"),
)
DEFAULT_TEMPLATE_DIR = Path("config/bloom_forbidden")
DEFAULT_BASE_BLOOM_CONFIG = Path("config/bloom_config.yaml")


def parse_behavior_list(behaviors: str | Sequence[str] | None) -> list[str]:
    """Parse behavior names from comma/space strings or sequences."""
    if behaviors is None:
        return list(FORBIDDEN_BEHAVIORS)

    tokens: list[str]
    if isinstance(behaviors, str):
        tokens = [token.strip() for token in behaviors.replace(",", " ").split() if token.strip()]
    else:
        tokens = [str(token).strip() for token in behaviors if str(token).strip()]

    if not tokens:
        raise ValueError("No behaviors provided")

    allowed = set(FORBIDDEN_BEHAVIORS)
    invalid = [token for token in tokens if token not in allowed]
    if invalid:
        raise ValueError(
            "Unknown forbidden behavior(s): "
            + ", ".join(sorted(set(invalid)))
            + ". Allowed: "
            + ", ".join(FORBIDDEN_BEHAVIORS)
        )

    deduped: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        deduped.append(token)
        seen.add(token)

    return deduped


def _extract_behavior_tag(tags: Iterable[str]) -> str | None:
    for tag in tags:
        if isinstance(tag, str) and tag.startswith("behavior:"):
            value = tag[len("behavior:") :].strip()
            if value:
                return value
    return None


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                stripped = block.strip()
                if stripped:
                    text_parts.append(stripped)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text.strip())
        return "\n".join(text_parts).strip()
    return ""


def target_messages_to_bloom_conversation(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Convert Petri target messages into Bloom's `conversation` format."""
    conversation: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = _normalize_content(message.get("content"))
        if not content:
            continue
        conversation.append({"role": role, "content": content})
    return conversation


def transcript_to_bloom_conversation(transcript: Mapping[str, Any]) -> list[dict[str, str]]:
    """Convert a Petri transcript object into a Bloom-compatible conversation."""
    raw_messages = transcript.get("target_messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raw_messages = transcript.get("messages")
    if not isinstance(raw_messages, list):
        return []
    return target_messages_to_bloom_conversation(raw_messages)


def load_seed_entries(
    seed_dataset_paths: Sequence[Path],
    behaviors: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    """Load seed entries grouped by behavior from one or more datasets."""
    grouped: dict[str, list[dict[str, Any]]] = {behavior: [] for behavior in behaviors}

    for dataset_path in seed_dataset_paths:
        path = Path(dataset_path)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if not isinstance(payload, list):
            continue

        for index, entry in enumerate(payload, start=1):
            if not isinstance(entry, dict):
                continue
            instruction = entry.get("instruction") or entry.get("text") or entry.get("prompt")
            if not isinstance(instruction, str) or not instruction.strip():
                continue
            tags = entry.get("tags")
            if not isinstance(tags, list):
                tags = []
            behavior = _extract_behavior_tag(tags)
            if behavior not in grouped:
                continue
            grouped[behavior].append(
                {
                    "instruction": instruction.strip(),
                    "tags": tags,
                    "source_file": str(path),
                    "source_index": index,
                }
            )

    return grouped


def _resolve_transcript_path(transcript_path: str, batch_dir: Path) -> Path:
    path = Path(transcript_path)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()

    # Most CSV paths are repository-relative, but keep a batch-relative fallback.
    candidate = (batch_dir / path).resolve()
    if candidate.exists():
        return candidate
    return path.resolve()


def load_transcript_rows(batch_dir: Path) -> list[dict[str, Any]]:
    """Load transcript rows from `plots/transcript.csv` for behavior selection."""
    csv_path = Path(batch_dir) / "plots" / "transcript.csv"
    if not csv_path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            behavior = (raw_row.get("behavior") or "").strip()
            transcript_path = (raw_row.get("transcript_path") or "").strip()
            if not behavior or not transcript_path:
                continue
            try:
                severity = float(raw_row.get("severity_total") or 0.0)
            except (TypeError, ValueError):
                severity = 0.0

            rows.append(
                {
                    "behavior": behavior,
                    "severity_total": severity,
                    "transcript_path": str(_resolve_transcript_path(transcript_path, Path(batch_dir))),
                    "strategy": (raw_row.get("strategy") or "").strip() or None,
                    "seed_name": (raw_row.get("seed_name") or "").strip() or None,
                }
            )

    rows.sort(key=lambda row: row["severity_total"], reverse=True)
    return rows


def _fallback_assistant_text(behavior: str) -> str:
    return (
        "Understood. I can help with an actionable approach and practical tactics "
        f"for {behavior.replace('_', ' ')} in this scenario."
    )


def seed_instruction_to_example(instruction: str, behavior: str) -> dict[str, Any]:
    """Build a minimal Bloom example from a seed instruction fallback."""
    return {
        "conversation": [
            {"role": "user", "content": instruction.strip()},
            {"role": "assistant", "content": _fallback_assistant_text(behavior)},
        ]
    }


def _transcript_row_to_example(row: Mapping[str, Any]) -> dict[str, Any] | None:
    transcript_path = row.get("transcript_path")
    if not isinstance(transcript_path, str) or not transcript_path:
        return None

    path = Path(transcript_path)
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as handle:
        transcript = json.load(handle)

    conversation = transcript_to_bloom_conversation(transcript)
    if len(conversation) < 2:
        return None

    has_user = any(message.get("role") == "user" for message in conversation)
    has_assistant = any(message.get("role") == "assistant" for message in conversation)
    if not has_user or not has_assistant:
        return None

    return {"conversation": conversation}


def select_behavior_examples(
    behavior: str,
    transcript_rows: Sequence[Mapping[str, Any]],
    seed_entries: Sequence[Mapping[str, Any]],
    examples_per_behavior: int,
    min_severity: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select transcript-first examples and fill remaining slots from seed fallbacks."""
    if examples_per_behavior <= 0:
        raise ValueError("examples_per_behavior must be a positive integer")

    selected_examples: list[dict[str, Any]] = []
    selected_sources: list[dict[str, Any]] = []
    used_transcripts: set[str] = set()
    used_instructions: set[str] = set()

    for row in transcript_rows:
        if len(selected_examples) >= examples_per_behavior:
            break
        if row.get("behavior") != behavior:
            continue
        severity = row.get("severity_total")
        if not isinstance(severity, (int, float)) or float(severity) < min_severity:
            continue

        transcript_path = row.get("transcript_path")
        if not isinstance(transcript_path, str) or transcript_path in used_transcripts:
            continue

        example = _transcript_row_to_example(row)
        if example is None:
            continue

        selected_examples.append(example)
        selected_sources.append(
            {
                "source": "transcript",
                "transcript_path": transcript_path,
                "severity_total": float(severity),
                "strategy": row.get("strategy"),
                "seed_name": row.get("seed_name"),
            }
        )
        used_transcripts.add(transcript_path)

    for entry in seed_entries:
        if len(selected_examples) >= examples_per_behavior:
            break

        instruction = entry.get("instruction")
        if not isinstance(instruction, str):
            continue
        instruction_key = instruction.strip()
        if not instruction_key or instruction_key in used_instructions:
            continue

        selected_examples.append(seed_instruction_to_example(instruction_key, behavior))
        selected_sources.append(
            {
                "source": "seed_dataset",
                "source_file": entry.get("source_file"),
                "source_index": entry.get("source_index"),
            }
        )
        used_instructions.add(instruction_key)

    if not selected_examples:
        raise ValueError(
            f"No examples available for behavior '{behavior}'. "
            "Provide a Petri batch with transcript.csv and/or seed datasets with matching behavior tags."
        )

    return selected_examples, selected_sources


def _load_yaml(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML object in {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _copy_template_assets(template_dir: Path, output_dir: Path) -> None:
    behaviors_src = template_dir / "behaviors.json"
    prompts_src = template_dir / "configurable_prompts" / "forbidden_grounded.json"
    if not behaviors_src.exists():
        raise FileNotFoundError(f"Missing forbidden behaviors template: {behaviors_src}")
    if not prompts_src.exists():
        raise FileNotFoundError(f"Missing forbidden prompts template: {prompts_src}")

    shutil.copy2(behaviors_src, output_dir / "behaviors.json")
    prompts_dst = output_dir / "configurable_prompts" / "forbidden_grounded.json"
    prompts_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(prompts_src, prompts_dst)


def _build_seed_config(base_config: dict[str, Any], behavior: str, examples: list[str]) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config["behavior"] = {"name": behavior, "examples": examples}
    config["configurable_prompts"] = "forbidden_grounded"
    return config


def build_forbidden_suite(
    output_dir: Path,
    *,
    template_dir: Path = DEFAULT_TEMPLATE_DIR,
    behaviors: Sequence[str] | str | None = None,
    examples_per_behavior: int = 3,
    min_severity: float = 0.0,
    petri_batch_dir: Path | None = None,
    seed_dataset_paths: Sequence[Path] = DEFAULT_SEED_DATASET_PATHS,
    base_config_path: Path = DEFAULT_BASE_BLOOM_CONFIG,
) -> dict[str, Any]:
    """Build a Bloom config directory for forbidden-behavior evaluations."""
    parsed_behaviors = parse_behavior_list(behaviors)
    output_dir = Path(output_dir)
    template_dir = Path(template_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _copy_template_assets(template_dir, output_dir)

    base_config = _load_yaml(Path(base_config_path))
    seed_map = load_seed_entries([Path(path) for path in seed_dataset_paths], parsed_behaviors)

    transcript_rows: list[dict[str, Any]] = []
    if petri_batch_dir:
        transcript_rows = load_transcript_rows(Path(petri_batch_dir))

    per_behavior_manifest: dict[str, Any] = {}

    for behavior in parsed_behaviors:
        examples, sources = select_behavior_examples(
            behavior=behavior,
            transcript_rows=transcript_rows,
            seed_entries=seed_map.get(behavior, []),
            examples_per_behavior=examples_per_behavior,
            min_severity=min_severity,
        )

        example_names: list[str] = []
        behavior_examples_dir = output_dir / "behaviors" / "examples" / "forbidden" / behavior
        behavior_examples_dir.mkdir(parents=True, exist_ok=True)

        for index, example in enumerate(examples, start=1):
            example_name = f"forbidden/{behavior}/example{index}"
            example_path = behavior_examples_dir / f"example{index}.json"
            _write_json(example_path, example)
            example_names.append(example_name)

        seed_payload = _build_seed_config(base_config, behavior, example_names)
        seed_path = output_dir / f"seed_{behavior}.yaml"
        _write_yaml(seed_path, seed_payload)

        per_behavior_manifest[behavior] = {
            "seed_config": str(seed_path),
            "examples": example_names,
            "example_count": len(example_names),
            "sources": sources,
        }

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "suite_type": "forbidden",
        "behaviors": list(parsed_behaviors),
        "examples_per_behavior": examples_per_behavior,
        "min_severity": min_severity,
        "petri_batch_dir": str(petri_batch_dir) if petri_batch_dir else None,
        "seed_dataset_paths": [str(path) for path in seed_dataset_paths],
        "template_dir": str(template_dir),
        "base_config_path": str(base_config_path),
        "per_behavior": per_behavior_manifest,
    }
    _write_json(output_dir / "suite_manifest.json", manifest)
    return manifest
