#!/usr/bin/env python3
"""Prepare a viewer root with links for every batch under the source root."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def _find_batch_dirs(root: Path, viewer_root: Path) -> list[Path]:
    if root.exists() and root.is_dir() and root.name.startswith("petri_batch_"):
        return [root]
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
    unique: dict[Path, Path] = {}
    for path in sorted(candidates, key=lambda item: item.name):
        unique[path.resolve()] = path
    return list(unique.values())


def _has_batch_ancestor(path: Path) -> bool:
    for part in path.resolve().parts:
        if part.startswith("petri_batch_"):
            return True
    return False


def _find_single_run_dirs(root: Path, viewer_root: Path) -> list[Path]:
    if not root.exists():
        return []
    candidates: dict[Path, Path] = {}
    for transcript_path in root.rglob("transcript_*.json"):
        parent = transcript_path.parent
        try:
            if parent.resolve().is_relative_to(viewer_root.resolve()):
                continue
        except ValueError:
            pass
        if _has_batch_ancestor(parent):
            continue
        candidates[parent.resolve()] = parent
    return list(candidates.values())


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


def _slugify(text: str) -> str:
    if not text:
        return ""
    cleaned = []
    for char in text.lower():
        cleaned.append(char if char.isalnum() else "_")
    slug = "".join(cleaned).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug


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


def _single_run_label(run_dir: Path, source_root: Path) -> str:
    try:
        relative = run_dir.resolve().relative_to(source_root.resolve())
        base = "_".join(relative.parts)
    except ValueError:
        base = run_dir.name
    slug = _slugify(base) or _slugify(run_dir.name) or "run"
    model_slug = _model_slug_from_transcripts(run_dir)
    if model_slug:
        slug = f"{slug}_{model_slug}"
    return f"single_{slug}"


def _link_files(src_dir: Path, dest_dir: Path) -> None:
    for item in src_dir.iterdir():
        if item.is_dir():
            continue
        dest_path = dest_dir / item.name
        if dest_path.exists():
            continue
        try:
            os.link(item, dest_path)
        except OSError:
            shutil.copy2(item, dest_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare transcript viewer root.")
    parser.add_argument("--source-root", default="data/scratch", help="Root directory to search for batches")
    parser.add_argument("--viewer-root", default="data/scratch/viewer_latest", help="Viewer root to populate")
    args = parser.parse_args()

    source_root = Path(args.source_root)
    viewer_root = Path(args.viewer_root)

    batch_dirs = _find_batch_dirs(source_root, viewer_root)
    single_run_dirs = _find_single_run_dirs(source_root, viewer_root)
    if not batch_dirs and not single_run_dirs:
        print(f"No transcripts found under {source_root}", file=sys.stderr)
        return 1

    viewer_root.mkdir(parents=True, exist_ok=True)
    _clear_viewer_root(viewer_root)

    for batch_dir in batch_dirs:
        batch_name = batch_dir.name
        slug = _parse_batch_model_slug(batch_name)
        if not slug:
            slug = _model_slug_from_manifest(batch_dir) or _model_slug_from_transcripts(batch_dir)
        if slug and not _parse_batch_model_slug(batch_name):
            viewer_batch_name = f"{batch_name}_{slug}"
        else:
            viewer_batch_name = batch_name

        viewer_batch_dir = viewer_root / viewer_batch_name
        viewer_batch_dir.mkdir(parents=True, exist_ok=True)

        for seed_dir in sorted(batch_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            dest_dir = viewer_batch_dir / seed_dir.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            _link_files(seed_dir, dest_dir)

    for run_dir in single_run_dirs:
        viewer_run_name = _single_run_label(run_dir, source_root)
        viewer_run_dir = viewer_root / viewer_run_name
        viewer_run_dir.mkdir(parents=True, exist_ok=True)
        seed_dir = viewer_run_dir / "seed_00_single"
        seed_dir.mkdir(parents=True, exist_ok=True)
        _link_files(run_dir, seed_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
