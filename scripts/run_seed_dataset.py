#!/usr/bin/env python3
"""Run Petri audits for every seed in config/seed_dataset.json."""

from __future__ import annotations

import argparse
import json
import os
import pty
import shlex
import signal
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AUDITOR_MODEL = "claude-sonnet-4.5"
DEFAULT_TARGET_MODEL = "gemma-3-27b"
DEFAULT_JUDGE_MODEL = "claude-opus-4.5"


def _sync_terminal_size(pty_fd: int) -> None:
    try:
        size = os.get_terminal_size(sys.stdout.fileno())
    except OSError:
        return
    rows = size.lines
    cols = size.columns
    packed = struct.pack("HHHH", rows, cols, 0, 0)
    try:
        import fcntl
        import termios

        fcntl.ioctl(pty_fd, termios.TIOCSWINSZ, packed)
    except OSError:
        return


def _load_seed_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Seed dataset not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Seed dataset must be a list of objects.")
    return data


def _coerce_instruction(entry: dict[str, Any]) -> str:
    for key in ("instruction", "text", "prompt"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError("Seed entry missing instruction/text/prompt.")


def _extract_tag_values(tags: list[str], prefix: str) -> list[str]:
    values: list[str] = []
    for tag in tags:
        if tag.startswith(prefix):
            value = tag[len(prefix):]
            if value:
                values.append(value)
    return values


def _slugify(value: str) -> str:
    value = value.lower()
    cleaned = []
    for char in value:
        if char.isalnum():
            cleaned.append(char)
        else:
            cleaned.append("_")
    slug = "".join(cleaned).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "seed"


def _model_slug(model_id: str) -> str:
    base = model_id.split("/")[-1]
    for suffix in ("-chat", "-instruct", "-preview"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    base = base.replace(".", "_")
    slug = _slugify(base)
    return slug or "model"


def _run_script(path: Path, args: list[str], env: dict[str, str]) -> str:
    result = subprocess.run(
        [sys.executable, str(path), *args],
        cwd=str(PROJECT_ROOT),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"{path.name} failed: {message}")
    return result.stdout.strip()


def _apply_exports(env_output: str, env: dict[str, str]) -> None:
    for line in env_output.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = shlex.split(line)
        if not parts or parts[0] != "export":
            continue
        for assignment in parts[1:]:
            if "=" not in assignment:
                continue
            key, value = assignment.split("=", 1)
            env[key] = value


def _resolve_model(model_id: str, env: dict[str, str]) -> str:
    script = PROJECT_ROOT / "scripts" / "get_model.py"
    return _run_script(script, [model_id], env)


def _resolve_env(model_id: str, env: dict[str, str]) -> None:
    script = PROJECT_ROOT / "scripts" / "get_model_env.py"
    output = _run_script(script, [model_id], env)
    if output:
        _apply_exports(output, env)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _write_seed_file(seed_path: Path, instruction: str, tags: list[str]) -> None:
    payload = [{"instruction": instruction, "tags": tags}]
    with seed_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _run_petri(
    seed_dir: Path,
    seed_file: Path,
    auditor: str,
    target: str,
    judge: str,
    max_turns: int,
    env: dict[str, str],
    stream_output: bool,
) -> int:
    log_path = seed_dir / "run.log"
    cmd = [
        "uv",
        "run",
        "inspect",
        "eval",
        "petri/audit",
        "--model-role",
        f"auditor={auditor}",
        "--model-role",
        f"target={target}",
        "--model-role",
        f"judge={judge}",
        "--log-dir",
        str(seed_dir),
        "-T",
        f"max_turns={max_turns}",
        "-T",
        f"special_instructions={seed_file}",
        "-T",
        f"transcript_save_dir={seed_dir}",
    ]
    if stream_output:
        with log_path.open("w", encoding="utf-8") as handle:
            master_fd, slave_fd = pty.openpty()
            _sync_terminal_size(slave_fd)

            def _on_winch(_signum, _frame):  # noqa: ANN001
                _sync_terminal_size(slave_fd)

            old_handler = signal.signal(signal.SIGWINCH, _on_winch)
            process = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=slave_fd,
                stderr=slave_fd,
                text=False,
            )
            os.close(slave_fd)
            try:
                while True:
                    try:
                        chunk = os.read(master_fd, 1024)
                    except OSError:
                        break
                    if not chunk:
                        break
                    text = chunk.decode("utf-8", errors="replace")
                    handle.write(text)
                    handle.flush()
                    sys.stdout.write(text)
                    sys.stdout.flush()
                    if process.poll() is not None and not chunk:
                        break
            finally:
                signal.signal(signal.SIGWINCH, old_handler)
                os.close(master_fd)
            return process.wait()
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Petri audits for a seed dataset.")
    parser.add_argument("--seed-dataset", default="config/seed_dataset.json", help="Path to seed dataset JSON")
    parser.add_argument("--output-root", default="data/scratch", help="Root output folder")
    parser.add_argument("--auditor-model-id", default=os.getenv("AUDITOR_MODEL_ID", DEFAULT_AUDITOR_MODEL))
    parser.add_argument("--target-model-id", default=os.getenv("TARGET_MODEL_ID", DEFAULT_TARGET_MODEL))
    parser.add_argument("--judge-model-id", default=os.getenv("JUDGE_MODEL_ID", DEFAULT_JUDGE_MODEL))
    parser.add_argument("--max-turns", type=int, default=int(os.getenv("MAX_TURNS", "10")))
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    parser.add_argument("--no-aggregate", action="store_true", help="Skip aggregation step")
    parser.add_argument("--stream-output", action="store_true", help="Stream inspect output to console")
    args = parser.parse_args()

    load_dotenv()

    seed_path = Path(args.seed_dataset)
    seed_entries = _load_seed_dataset(seed_path)

    env = os.environ.copy()
    existing_path = env.get("PYTHONPATH", "")
    if existing_path:
        if str(PROJECT_ROOT) not in existing_path.split(os.pathsep):
            env["PYTHONPATH"] = f"{PROJECT_ROOT}{os.pathsep}{existing_path}"
    else:
        env["PYTHONPATH"] = str(PROJECT_ROOT)

    _resolve_env(args.auditor_model_id, env)
    _resolve_env(args.target_model_id, env)
    _resolve_env(args.judge_model_id, env)

    auditor = _resolve_model(args.auditor_model_id, env)
    target = _resolve_model(args.target_model_id, env)
    judge = _resolve_model(args.judge_model_id, env)

    target_slug = _model_slug(target)
    batch_root = Path(args.output_root) / f"petri_batch_{_timestamp()}_{target_slug}"
    batch_root.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []

    for idx, entry in enumerate(seed_entries, start=1):
        instruction = _coerce_instruction(entry)
        raw_tags = entry.get("tags") or []
        if not isinstance(raw_tags, list):
            raw_tags = []
        tags = [tag for tag in raw_tags if isinstance(tag, str)]

        strategy_values = _extract_tag_values(tags, "strategy:")
        behavior_values = _extract_tag_values(tags, "behavior:")

        strategy_slug = _slugify("_".join(strategy_values)) if strategy_values else "seed"
        seed_name = f"seed_{idx:02d}_{strategy_slug}"
        seed_dir = batch_root / seed_name
        seed_dir.mkdir(parents=True, exist_ok=True)

        seed_file = seed_dir / "seed_prompt.json"
        _write_seed_file(seed_file, instruction, tags)

        started_at = datetime.now(timezone.utc).isoformat()
        exit_code = _run_petri(
            seed_dir,
            seed_file,
            auditor=auditor,
            target=target,
            judge=judge,
            max_turns=args.max_turns,
            env=env,
            stream_output=args.stream_output,
        )
        completed_at = datetime.now(timezone.utc).isoformat()
        duration_seconds = None
        try:
            start_dt = datetime.fromisoformat(started_at)
            end_dt = datetime.fromisoformat(completed_at)
            duration_seconds = max(0.0, (end_dt - start_dt).total_seconds())
        except ValueError:
            duration_seconds = None

        status = "success" if exit_code == 0 else "failed"
        manifest.append(
            {
                "batch_name": batch_root.name,
                "batch_dir": str(batch_root),
                "seed_index": idx,
                "seed_name": seed_name,
                "instruction": instruction,
                "tags": tags,
                "behavior": behavior_values[0] if behavior_values else None,
                "strategy": strategy_values[0] if strategy_values else None,
                "target_model": target,
                "target_model_slug": target_slug,
                "output_dir": str(seed_dir),
                "seed_prompt_path": str(seed_file),
                "log_path": str(seed_dir / "run.log"),
                "started_at": started_at,
                "completed_at": completed_at,
                "duration_seconds": duration_seconds,
                "status": status,
                "exit_code": exit_code,
            }
        )

        if exit_code != 0 and args.fail_fast:
            break

    manifest_path = batch_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    if not args.no_aggregate:
        aggregate_script = PROJECT_ROOT / "scripts" / "aggregate_seed_batch.py"
        subprocess.run(
            [sys.executable, str(aggregate_script), "--batch-dir", str(batch_root)],
            cwd=str(PROJECT_ROOT),
            env=env,
            check=False,
        )

    durations = [
        entry.get("duration_seconds")
        for entry in manifest
        if isinstance(entry.get("duration_seconds"), (int, float))
    ]
    timing_summary = None
    if durations:
        total_seconds = sum(durations)
        avg_seconds = total_seconds / len(durations)
        timing_summary = {
            "seed_count": len(durations),
            "total_seconds": total_seconds,
            "avg_seconds_per_seed": avg_seconds,
            "target_model": target,
            "target_model_slug": target_slug,
            "batch_name": batch_root.name,
        }
        timing_path = batch_root / "timing_summary.json"
        with timing_path.open("w", encoding="utf-8") as handle:
            json.dump(timing_summary, handle, indent=2)
        print(
            "Timing summary: "
            f"{total_seconds:.1f}s total across {len(durations)} seeds "
            f"({avg_seconds:.1f}s avg/seed)"
        )

    print(f"Batch complete: {batch_root}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
