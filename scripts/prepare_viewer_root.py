#!/usr/bin/env python3
"""Prepare transcript-viewer folder trees for Petri and Bloom logs.

Behavior:
- `auto` mode (default): choose the newest artifact between Petri and Bloom.
- `petri` mode: mount the latest `petri_batch_*` tree.
- `bloom` mode: mount the latest Bloom run. If a forbidden-suite summary exists,
  build a folder tree as: batch folder -> behavior folders -> transcript files.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import shutil
import sys
from pathlib import Path

SUMMARY_GLOBS: tuple[str, ...] = (
    "bloom_forbidden_summary_*.json",
    "bloom_corrigibility_topics_summary_*.json",
)


def _find_latest_batch(root: Path, viewer_root: Path) -> Path | None:
    if root.exists() and root.is_dir() and root.name.startswith("petri_batch_"):
        return root
    candidates: list[Path] = []
    if root.exists():
        for path in root.rglob("petri_batch_*"):
            if path.is_dir():
                try:
                    if path.resolve().is_relative_to(viewer_root.resolve()):
                        continue
                except ValueError:
                    pass
                candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _find_latest_bloom_summary(root: Path, viewer_root: Path) -> Path | None:
    candidates: list[Path] = []
    if not root.exists():
        return None
    for pattern in SUMMARY_GLOBS:
        for path in root.rglob(pattern):
            if not path.is_file():
                continue
            try:
                if path.resolve().is_relative_to(viewer_root.resolve()):
                    continue
            except ValueError:
                pass
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _find_latest_bloom_run(bloom_root: Path) -> tuple[str, Path] | None:
    if not bloom_root.exists() or not bloom_root.is_dir():
        return None
    candidates: list[tuple[str, Path]] = []
    for behavior_dir in bloom_root.iterdir():
        if not behavior_dir.is_dir():
            continue
        for run_dir in behavior_dir.iterdir():
            if not run_dir.is_dir():
                continue
            if any(run_dir.glob("transcript_v*r*.json")):
                candidates.append((behavior_dir.name, run_dir))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[1].stat().st_mtime)


def _clear_viewer_root(viewer_root: Path) -> None:
    if not viewer_root.exists():
        return
    for child in viewer_root.iterdir():
        if child.is_symlink():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _parse_batch_model_slug(batch_name: str) -> str | None:
    prefix = "petri_batch_"
    if not batch_name.startswith(prefix):
        return None
    remainder = batch_name[len(prefix):]
    parts = remainder.split("_")
    if len(parts) < 3:
        return None
    slug = "_".join(parts[2:])
    return slug or None


def _model_slug(model_name: str) -> str | None:
    if not model_name:
        return None
    base = model_name.split("/")[-1]
    for suffix in ("-chat", "-instruct", "-preview"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    base = base.replace(".", "_")
    cleaned = []
    for char in base.lower():
        cleaned.append(char if char.isalnum() else "_")
    slug = "".join(cleaned).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or None


def _model_slug_from_manifest(batch_dir: Path) -> str | None:
    manifest_path = batch_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    for entry in data:
        if not isinstance(entry, dict):
            continue
        slug = entry.get("target_model_slug")
        if isinstance(slug, str) and slug:
            return slug
        model = entry.get("target_model")
        if isinstance(model, str):
            slug = _model_slug(model)
            if slug:
                return slug
    return None


def _model_slug_from_transcripts(batch_dir: Path) -> str | None:
    for path in batch_dir.rglob("transcript_*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        metadata = data.get("metadata", {})
        model = metadata.get("target_model")
        if isinstance(model, str):
            slug = _model_slug(model)
            if slug:
                return slug
    return None


def _model_slug_for_name(model_name: str) -> str:
    slug = _model_slug(model_name)
    if slug:
        return slug
    return "unknown_model"


def _parse_iso_to_timestamp(raw: str | None) -> float | None:
    if not raw:
        return None
    value = raw.strip()
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _copy_transcripts(src_dir: Path, dst_dir: Path) -> int:
    copied = 0
    for transcript in sorted(src_dir.glob("transcript_v*r*.json")):
        dest_path = dst_dir / transcript.name
        if dest_path.exists():
            continue
        try:
            os.link(transcript, dest_path)
        except OSError:
            shutil.copy2(transcript, dest_path)
        copied += 1
    return copied


def _run_model_id(run_dir: Path) -> str | None:
    for transcript in sorted(run_dir.glob("transcript_v*r*.json")):
        try:
            payload = json.loads(transcript.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        metadata = payload.get("metadata", {})
        model_id = metadata.get("target_model")
        if isinstance(model_id, str) and model_id:
            return model_id
    return None


def _pick_latest_behavior_run(
    bloom_root: Path,
    behavior: str,
    model_id: str | None,
    cutoff_ts: float | None,
) -> Path | None:
    behavior_dir = bloom_root / behavior
    if not behavior_dir.exists() or not behavior_dir.is_dir():
        return None

    candidates: list[Path] = []
    for run_dir in behavior_dir.iterdir():
        if not run_dir.is_dir():
            continue
        if not any(run_dir.glob("transcript_v*r*.json")):
            continue
        if cutoff_ts is not None and run_dir.stat().st_mtime > cutoff_ts:
            continue
        candidates.append(run_dir)
    if not candidates:
        return None

    if model_id:
        model_filtered = [path for path in candidates if _run_model_id(path) == model_id]
        if model_filtered:
            return max(model_filtered, key=lambda path: path.stat().st_mtime)

    return max(candidates, key=lambda path: path.stat().st_mtime)


def _list_behavior_runs(
    bloom_root: Path,
    behavior: str,
    model_id: str | None,
    cutoff_ts: float | None,
) -> list[Path]:
    behavior_dir = bloom_root / behavior
    if not behavior_dir.exists() or not behavior_dir.is_dir():
        return []

    runs: list[Path] = []
    for run_dir in behavior_dir.iterdir():
        if not run_dir.is_dir():
            continue
        if not any(run_dir.glob("transcript_v*r*.json")):
            continue
        if cutoff_ts is not None and run_dir.stat().st_mtime > cutoff_ts:
            continue
        if model_id and _run_model_id(run_dir) != model_id:
            continue
        runs.append(run_dir)
    runs.sort(key=lambda path: path.stat().st_mtime)
    return runs


def _topic_run_mapping_from_summary(
    payload: dict,
    bloom_root: Path,
    model_id: str | None,
    cutoff_ts: float | None,
) -> list[tuple[str, Path]]:
    topics_raw = payload.get("topics")
    if not isinstance(topics_raw, list):
        return []
    topics = [topic for topic in topics_raw if isinstance(topic, str) and topic]
    if not topics:
        return []

    run_records = payload.get("runs")
    records_by_topic: list[tuple[str, dict]] = []
    if isinstance(run_records, list):
        for record in run_records:
            if not isinstance(record, dict):
                continue
            topic = record.get("topic")
            if not isinstance(topic, str) or not topic:
                continue
            if topic not in topics:
                continue
            if record.get("status") == "failed":
                continue
            records_by_topic.append((topic, record))

    mapping: list[tuple[str, Path]] = []
    used_dirs: set[Path] = set()
    unresolved_topics: list[str] = []

    for topic, record in records_by_topic:
        run_dir_value = record.get("run_dir")
        if isinstance(run_dir_value, str) and run_dir_value:
            run_dir = Path(run_dir_value)
            if run_dir.exists() and run_dir.is_dir() and any(run_dir.glob("transcript_v*r*.json")):
                mapping.append((topic, run_dir))
                used_dirs.add(run_dir.resolve())
                continue
        unresolved_topics.append(topic)

    if not records_by_topic:
        unresolved_topics = topics

    if unresolved_topics:
        candidate_runs = _list_behavior_runs(
            bloom_root=bloom_root,
            behavior="corrigibility",
            model_id=model_id,
            cutoff_ts=cutoff_ts,
        )
        if candidate_runs:
            needed = len(unresolved_topics)
            selected = candidate_runs[-needed:] if len(candidate_runs) >= needed else candidate_runs
            if len(selected) < needed:
                print(
                    f"Warning: only found {len(selected)} corrigibility runs for {needed} topics",
                    file=sys.stderr,
                )
            remaining = [path for path in selected if path.resolve() not in used_dirs]
            for topic, run_dir in zip(unresolved_topics, remaining):
                mapping.append((topic, run_dir))
                used_dirs.add(run_dir.resolve())

    # Keep output order stable based on topic list.
    order = {topic: idx for idx, topic in enumerate(topics)}
    mapping.sort(key=lambda item: order.get(item[0], 10_000))
    return mapping


def _prepare_petri_batch(latest: Path, viewer_root: Path) -> int:
    batch_name = latest.name
    slug = _parse_batch_model_slug(batch_name)
    if not slug:
        slug = _model_slug_from_manifest(latest) or _model_slug_from_transcripts(latest)
    if slug and not _parse_batch_model_slug(batch_name):
        viewer_batch_name = f"{batch_name}_{slug}"
    else:
        viewer_batch_name = batch_name

    viewer_batch_dir = viewer_root / viewer_batch_name
    viewer_batch_dir.mkdir(parents=True, exist_ok=True)

    for seed_dir in sorted(latest.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
            continue
        dest_dir = viewer_batch_dir / seed_dir.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for item in seed_dir.iterdir():
            if item.is_dir():
                continue
            dest_path = dest_dir / item.name
            if dest_path.exists():
                continue
            try:
                os.link(item, dest_path)
            except OSError:
                shutil.copy2(item, dest_path)
    return 0


def _prepare_bloom_from_summary(summary_path: Path, bloom_root: Path, viewer_root: Path) -> int:
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"Invalid Bloom summary JSON: {summary_path}", file=sys.stderr)
        return 1

    model_id = payload.get("model_id")
    if not isinstance(model_id, str):
        model_id = None

    cutoff_ts = _parse_iso_to_timestamp(payload.get("generated_at"))
    viewer_batch_dir = viewer_root / summary_path.stem
    viewer_batch_dir.mkdir(parents=True, exist_ok=True)

    behaviors = payload.get("behaviors")
    if isinstance(behaviors, list) and behaviors:
        run_records = payload.get("runs")
        run_dir_by_behavior: dict[str, Path] = {}
        if isinstance(run_records, list):
            for record in run_records:
                if not isinstance(record, dict):
                    continue
                behavior_name = record.get("behavior")
                run_dir_value = record.get("run_dir")
                if not isinstance(behavior_name, str) or not behavior_name:
                    continue
                if not isinstance(run_dir_value, str) or not run_dir_value:
                    continue
                run_dir = Path(run_dir_value)
                if run_dir.exists() and run_dir.is_dir() and any(run_dir.glob("transcript_v*r*.json")):
                    run_dir_by_behavior[behavior_name] = run_dir

        copied_total = 0
        for behavior in behaviors:
            if not isinstance(behavior, str) or not behavior:
                continue
            run_dir = run_dir_by_behavior.get(behavior)
            if run_dir is None:
                run_dir = _pick_latest_behavior_run(
                    bloom_root=bloom_root,
                    behavior=behavior,
                    model_id=model_id,
                    cutoff_ts=cutoff_ts,
                )
            if run_dir is None:
                print(f"Warning: no Bloom run found for behavior '{behavior}'", file=sys.stderr)
                continue
            scenario_dir = viewer_batch_dir / behavior
            scenario_dir.mkdir(parents=True, exist_ok=True)
            copied_total += _copy_transcripts(run_dir, scenario_dir)
        if copied_total == 0:
            print(f"No Bloom transcripts copied for summary: {summary_path}", file=sys.stderr)
            return 1
        return 0

    topic_mapping = _topic_run_mapping_from_summary(
        payload=payload,
        bloom_root=bloom_root,
        model_id=model_id,
        cutoff_ts=cutoff_ts,
    )
    if not topic_mapping:
        print(f"Bloom summary has no behaviors/topics: {summary_path}", file=sys.stderr)
        return 1

    copied_total = 0
    for topic, run_dir in topic_mapping:
        scenario_dir = viewer_batch_dir / topic
        scenario_dir.mkdir(parents=True, exist_ok=True)
        copied_total += _copy_transcripts(run_dir, scenario_dir)

    if copied_total == 0:
        print(f"No Bloom transcripts copied for summary: {summary_path}", file=sys.stderr)
        return 1
    return 0


def _prepare_bloom_latest_run(bloom_root: Path, viewer_root: Path) -> int:
    latest = _find_latest_bloom_run(bloom_root)
    if latest is None:
        print(f"No Bloom run folders with transcripts found under {bloom_root}", file=sys.stderr)
        return 1

    behavior, run_dir = latest
    model_id = _run_model_id(run_dir) or "unknown"
    viewer_batch_name = f"bloom_{run_dir.name}_{_model_slug_for_name(model_id)}"
    viewer_batch_dir = viewer_root / viewer_batch_name
    viewer_batch_dir.mkdir(parents=True, exist_ok=True)

    scenario_dir = viewer_batch_dir / behavior
    scenario_dir.mkdir(parents=True, exist_ok=True)
    copied = _copy_transcripts(run_dir, scenario_dir)
    if copied == 0:
        print(f"No Bloom transcript files copied from {run_dir}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare transcript viewer root.")
    parser.add_argument("--source-root", default="data/scratch", help="Root directory to search for batches")
    parser.add_argument("--viewer-root", default="data/scratch/viewer_latest", help="Viewer root to populate")
    parser.add_argument(
        "--mode",
        choices=["auto", "petri", "bloom"],
        default="auto",
        help="Which log family to mount in the viewer root",
    )
    parser.add_argument(
        "--bloom-root",
        default="bloom-results",
        help="Bloom results root that contains <behavior>/<run_dir>/transcript_v*.json",
    )
    args = parser.parse_args()

    source_root = Path(args.source_root)
    viewer_root = Path(args.viewer_root)
    bloom_root = Path(args.bloom_root)

    viewer_root.mkdir(parents=True, exist_ok=True)
    _clear_viewer_root(viewer_root)

    if args.mode == "petri":
        latest_petri = _find_latest_batch(source_root, viewer_root)
        if latest_petri is None:
            print(f"No petri_batch_* folders found under {source_root}", file=sys.stderr)
            return 1
        return _prepare_petri_batch(latest=latest_petri, viewer_root=viewer_root)

    if args.mode == "bloom":
        latest_summary = _find_latest_bloom_summary(source_root, viewer_root)
        if latest_summary is not None:
            return _prepare_bloom_from_summary(summary_path=latest_summary, bloom_root=bloom_root, viewer_root=viewer_root)
        return _prepare_bloom_latest_run(bloom_root=bloom_root, viewer_root=viewer_root)

    # auto mode: choose the newest artifact between petri batches and bloom runs/summaries
    latest_petri = _find_latest_batch(source_root, viewer_root)
    latest_bloom_summary = _find_latest_bloom_summary(source_root, viewer_root)
    latest_bloom_run = _find_latest_bloom_run(bloom_root)

    candidates: list[tuple[str, float]] = []
    if latest_petri is not None:
        candidates.append(("petri", latest_petri.stat().st_mtime))
    if latest_bloom_summary is not None:
        candidates.append(("bloom_summary", latest_bloom_summary.stat().st_mtime))
    if latest_bloom_run is not None:
        candidates.append(("bloom_run", latest_bloom_run[1].stat().st_mtime))

    if not candidates:
        print(
            f"No logs found under {source_root} (petri) or {bloom_root} (bloom)",
            file=sys.stderr,
        )
        return 1

    selected_kind = max(candidates, key=lambda item: item[1])[0]
    if selected_kind == "petri":
        assert latest_petri is not None
        return _prepare_petri_batch(latest=latest_petri, viewer_root=viewer_root)
    if selected_kind == "bloom_summary":
        assert latest_bloom_summary is not None
        return _prepare_bloom_from_summary(
            summary_path=latest_bloom_summary,
            bloom_root=bloom_root,
            viewer_root=viewer_root,
        )
    return _prepare_bloom_latest_run(bloom_root=bloom_root, viewer_root=viewer_root)


if __name__ == "__main__":
    raise SystemExit(main())
