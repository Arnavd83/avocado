#!/usr/bin/env python3
"""Prepare a transcript-viewer root from scratch run artifacts.

The transcript viewer works best when pointed at a clean directory that only
contains run folders and relevant files. This script scans a source tree,
finds run directories, and mirrors selected artifacts into a viewer root.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


# Keep only transcript payloads by default. The viewer loader reports failures
# for non-transcript artifacts (run logs, configs, scores, eval files).
FILE_PATTERNS = ("transcript_*.json",)


def _discover_run_dirs(source_root: Path, viewer_root: Path) -> list[Path]:
    """Find directories that look like run outputs."""
    run_dirs: set[Path] = set()

    for transcript in source_root.rglob("transcript_*.json"):
        parent = transcript.parent
        if viewer_root in parent.parents:
            continue
        run_dirs.add(parent)

    return sorted(run_dirs)


def _collect_files(run_dir: Path) -> list[Path]:
    selected: set[Path] = set()
    for pattern in FILE_PATTERNS:
        selected.update(run_dir.glob(pattern))
    return sorted(selected)


def prepare_viewer_root(source_root: Path, viewer_root: Path) -> tuple[int, int]:
    """Mirror run artifacts from source_root into viewer_root.

    Returns:
        Tuple of (run_count, file_count).
    """
    if not source_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")

    if viewer_root.exists():
        shutil.rmtree(viewer_root)
    viewer_root.mkdir(parents=True, exist_ok=True)

    run_dirs = _discover_run_dirs(source_root, viewer_root)
    copied_files = 0
    included_runs = 0

    for run_dir in run_dirs:
        files = _collect_files(run_dir)
        if not files:
            continue

        included_runs += 1
        relative_dir = run_dir.relative_to(source_root)
        destination_dir = viewer_root / relative_dir
        destination_dir.mkdir(parents=True, exist_ok=True)

        for src_file in files:
            dst_file = destination_dir / src_file.name
            shutil.copy2(src_file, dst_file)
            copied_files += 1

    return included_runs, copied_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        required=True,
        help="Directory containing run outputs (e.g. data/scratch).",
    )
    parser.add_argument(
        "--viewer-root",
        required=True,
        help="Directory to populate for transcript-viewer.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    viewer_root = Path(args.viewer_root).expanduser().resolve()

    run_count, file_count = prepare_viewer_root(source_root, viewer_root)
    print(f"Prepared viewer root: {viewer_root}")
    print(f"Included run directories: {run_count}")
    print(f"Copied files: {file_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
