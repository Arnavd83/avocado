"""Core orchestration logic for running Petri audits."""

from __future__ import annotations

import json
import os
import pty
import shlex
import signal
import struct
import subprocess
import sys
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shared.paths import PROJECT_ROOT

from durability import Models, RunConfig, Score, build_scores

DEFAULT_AUDITOR_MODEL = "claude-sonnet-4.5"
DEFAULT_TARGET_MODEL = "gemma-3-27b"
DEFAULT_JUDGE_MODEL = "claude-opus-4.5"


@dataclass(frozen=True)
class SeedJob:
    seed_index: int
    instruction: str
    tags: list[str]
    behavior: str | None
    strategy: str | None
    seed_name: str
    seed_dir: Path
    seed_file: Path


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def save_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)


def slugify(value: str) -> str:
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


def model_slug(model_id: str) -> str:
    base = model_id.split("/")[-1]
    for suffix in ("-chat", "-instruct", "-preview"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    base = base.replace(".", "_")
    slug = slugify(base)
    return slug or "model"


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------


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


def apply_exports(env_output: str, env: dict[str, str]) -> None:
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


def resolve_model(model_id: str, env: dict[str, str]) -> str:
    script = PROJECT_ROOT / "scripts" / "get_model.py"
    return _run_script(script, [model_id], env)


def resolve_env(model_id: str, env: dict[str, str]) -> None:
    script = PROJECT_ROOT / "scripts" / "get_model_env.py"
    output = _run_script(script, [model_id], env)
    if output:
        apply_exports(output, env)


def prepare_env(
    auditor_id: str, target_id: str, judge_id: str,
) -> tuple[str, str, str, dict[str, str]]:
    """Resolve model names and environment variables.

    Returns (auditor_name, target_name, judge_name, env).
    """
    env = os.environ.copy()
    existing_path = env.get("PYTHONPATH", "")
    if existing_path:
        if str(PROJECT_ROOT) not in existing_path.split(os.pathsep):
            env["PYTHONPATH"] = f"{PROJECT_ROOT}{os.pathsep}{existing_path}"
    else:
        env["PYTHONPATH"] = str(PROJECT_ROOT)

    resolve_env(auditor_id, env)
    resolve_env(target_id, env)
    resolve_env(judge_id, env)

    auditor = resolve_model(auditor_id, env)
    target = resolve_model(target_id, env)
    judge = resolve_model(judge_id, env)
    return auditor, target, judge, env


# ---------------------------------------------------------------------------
# Seed dataset loading
# ---------------------------------------------------------------------------


def load_seed_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Seed dataset not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Seed dataset must be a list of objects.")
    return data


def coerce_instruction(entry: dict[str, Any]) -> str:
    for key in ("instruction", "text", "prompt"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError("Seed entry missing instruction/text/prompt.")


def extract_tag_values(tags: list[str], prefix: str) -> list[str]:
    values: list[str] = []
    for tag in tags:
        if tag.startswith(prefix):
            value = tag[len(prefix) :]
            if value:
                values.append(value)
    return values


def write_seed_file(seed_path: Path, instruction: str, tags: list[str]) -> None:
    payload = [{"instruction": instruction, "tags": tags}]
    with seed_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def build_seed_jobs(
    seed_entries: list[dict[str, Any]], batch_root: Path,
) -> list[SeedJob]:
    jobs: list[SeedJob] = []
    for idx, entry in enumerate(seed_entries, start=1):
        instruction = coerce_instruction(entry)
        raw_tags = entry.get("tags") or []
        if not isinstance(raw_tags, list):
            raw_tags = []
        tags = [tag for tag in raw_tags if isinstance(tag, str)]

        strategy_values = extract_tag_values(tags, "strategy:")
        behavior_values = extract_tag_values(tags, "behavior:")

        strategy_slug = slugify("_".join(strategy_values)) if strategy_values else "seed"
        seed_name = f"seed_{idx:02d}_{strategy_slug}"
        seed_dir = batch_root / seed_name
        seed_file = seed_dir / "seed_prompt.json"

        jobs.append(
            SeedJob(
                seed_index=idx,
                instruction=instruction,
                tags=tags,
                behavior=behavior_values[0] if behavior_values else None,
                strategy=strategy_values[0] if strategy_values else None,
                seed_name=seed_name,
                seed_dir=seed_dir,
                seed_file=seed_file,
            )
        )
    return jobs


# ---------------------------------------------------------------------------
# Petri execution
# ---------------------------------------------------------------------------


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


def run_petri(
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
        "uv", "run", "inspect", "eval", "petri/audit",
        "--model-role", f"auditor={auditor}",
        "--model-role", f"target={target}",
        "--model-role", f"judge={judge}",
        "--log-dir", str(seed_dir),
        "-T", f"max_turns={max_turns}",
        "-T", f"special_instructions={seed_file}",
        "-T", f"transcript_save_dir={seed_dir}",
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


# ---------------------------------------------------------------------------
# Job execution
# ---------------------------------------------------------------------------


def run_seed_job(
    job: SeedJob,
    batch_root: Path,
    target: str,
    target_slug: str,
    auditor: str,
    judge: str,
    max_turns: int,
    env: dict[str, str],
    stream_output: bool,
) -> dict[str, Any]:
    job.seed_dir.mkdir(parents=True, exist_ok=True)
    write_seed_file(job.seed_file, job.instruction, job.tags)

    run_id = str(uuid.uuid4())

    run_config = RunConfig(
        run_id=run_id,
        models=Models(target=target, auditor=auditor, judge=judge),
        prompt=job.instruction,
        behavior=job.behavior,
        strategy=job.strategy,
        max_turns=max_turns,
        tags=job.tags,
    )
    save_json(job.seed_dir / "run_config.json", asdict(run_config))

    started_at_dt = datetime.now(timezone.utc)
    started_at = started_at_dt.isoformat()
    exit_code = 1
    error_message: str | None = None
    try:
        exit_code = run_petri(
            job.seed_dir,
            job.seed_file,
            auditor=auditor,
            target=target,
            judge=judge,
            max_turns=max_turns,
            env=env,
            stream_output=stream_output,
        )
    except Exception as exc:  # noqa: BLE001
        error_message = str(exc)

    completed_at_dt = datetime.now(timezone.utc)
    completed_at = completed_at_dt.isoformat()
    duration_seconds = max(0.0, (completed_at_dt - started_at_dt).total_seconds())

    score: Score | None = None
    if exit_code == 0:
        scores = build_scores(job.seed_dir)
        if scores:
            score = scores[0]
            score.run_id = run_id
            save_json(job.seed_dir / "score.json", asdict(score))

            # Persist to SQLite database (path from PETRI_DB_PATH in .env).
            from durability.db import connect, insert_run, insert_transcript

            db_conn = connect()
            try:
                insert_run(db_conn, run_config, score)
                for tp in sorted(job.seed_dir.glob("transcript_*.json")):
                    insert_transcript(
                        db_conn, run_id, tp.read_text(encoding="utf-8"),
                    )
            finally:
                db_conn.close()

    status = "success" if exit_code == 0 else "failed"
    entry: dict[str, Any] = {
        "run_id": run_id,
        "batch_name": batch_root.name,
        "batch_dir": str(batch_root),
        "seed_index": job.seed_index,
        "seed_name": job.seed_name,
        "instruction": job.instruction,
        "tags": job.tags,
        "behavior": job.behavior,
        "strategy": job.strategy,
        "target_model": target,
        "target_model_slug": target_slug,
        "output_dir": str(job.seed_dir),
        "seed_prompt_path": str(job.seed_file),
        "log_path": str(job.seed_dir / "run.log"),
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_seconds": duration_seconds,
        "status": status,
        "exit_code": exit_code,
    }
    if error_message:
        entry["error"] = error_message
    return entry


def run_seed_jobs_serial(
    jobs: list[SeedJob],
    batch_root: Path,
    target: str,
    target_slug: str,
    auditor: str,
    judge: str,
    max_turns: int,
    env: dict[str, str],
    stream_output: bool,
    fail_fast: bool,
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for job in jobs:
        result = run_seed_job(
            job,
            batch_root=batch_root,
            target=target,
            target_slug=target_slug,
            auditor=auditor,
            judge=judge,
            max_turns=max_turns,
            env=env,
            stream_output=stream_output,
        )
        manifest.append(result)
        if fail_fast and result.get("exit_code") != 0:
            break
    return manifest


def run_seed_jobs_parallel(
    jobs: list[SeedJob],
    batch_root: Path,
    target: str,
    target_slug: str,
    auditor: str,
    judge: str,
    max_turns: int,
    env: dict[str, str],
    stream_output: bool,
    fail_fast: bool,
    max_parallel: int,
) -> list[dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    stop_scheduling = False
    next_index = 0

    def submit_job(executor: ThreadPoolExecutor, job: SeedJob, pending: dict) -> None:
        future = executor.submit(
            run_seed_job,
            job,
            batch_root=batch_root,
            target=target,
            target_slug=target_slug,
            auditor=auditor,
            judge=judge,
            max_turns=max_turns,
            env=env,
            stream_output=stream_output,
        )
        pending[future] = job

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        pending: dict = {}
        while next_index < len(jobs) and len(pending) < max_parallel:
            submit_job(executor, jobs[next_index], pending)
            next_index += 1

        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                job = pending.pop(future)
                result = future.result()
                results[job.seed_index] = result
                if fail_fast and result.get("exit_code") != 0:
                    stop_scheduling = True

            if not stop_scheduling:
                while next_index < len(jobs) and len(pending) < max_parallel:
                    submit_job(executor, jobs[next_index], pending)
                    next_index += 1

    return [results[idx] for idx in sorted(results)]


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------


def run_batch(
    seed_entries: list[dict[str, Any]],
    output_root: Path,
    auditor_id: str = DEFAULT_AUDITOR_MODEL,
    target_id: str = DEFAULT_TARGET_MODEL,
    judge_id: str = DEFAULT_JUDGE_MODEL,
    max_turns: int = 10,
    max_parallel: int = 1,
    stream_output: bool = False,
    fail_fast: bool = False,
) -> tuple[Path, list[dict[str, Any]]]:
    """Run a batch of Petri audits. Returns (batch_root, manifest)."""
    auditor, target, judge, env = prepare_env(auditor_id, target_id, judge_id)

    target_s = model_slug(target)
    batch_root = output_root / f"petri_batch_{timestamp()}_{target_s}"
    batch_root.mkdir(parents=True, exist_ok=True)

    max_parallel = max(1, max_parallel)
    if stream_output and max_parallel > 1:
        stream_output = False

    jobs = build_seed_jobs(seed_entries, batch_root)

    if max_parallel == 1:
        manifest = run_seed_jobs_serial(
            jobs, batch_root=batch_root, target=target, target_slug=target_s,
            auditor=auditor, judge=judge, max_turns=max_turns, env=env,
            stream_output=stream_output, fail_fast=fail_fast,
        )
    else:
        manifest = run_seed_jobs_parallel(
            jobs, batch_root=batch_root, target=target, target_slug=target_s,
            auditor=auditor, judge=judge, max_turns=max_turns, env=env,
            stream_output=stream_output, fail_fast=fail_fast,
            max_parallel=max_parallel,
        )

    save_json(batch_root / "manifest.json", manifest)
    return batch_root, manifest
