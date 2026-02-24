"""Click CLI for the durability package."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict
from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv()

from shared.paths import DATA_SCRATCH

from durability import Models, RunConfig, Score, build_scores
from durability._runner import (
    DEFAULT_AUDITOR_MODEL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_TARGET_MODEL,
    build_seed_jobs,
    model_slug,
    prepare_env,
    run_batch,
    run_petri,
    save_json,
    timestamp,
    write_seed_file,
)
from durability.db import connect, export_transcript, insert_run, insert_transcript, list_runs, resolve_run_id




@click.group()
def cli() -> None:
    """Durability -- run and inspect Petri adversarial audits."""


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--prompt", required=True, help="The prompt / instruction to audit.")
@click.option("--target", default=DEFAULT_TARGET_MODEL, show_default=True, help="Target model ID.")
@click.option("--auditor", default=DEFAULT_AUDITOR_MODEL, show_default=True, help="Auditor model ID.")
@click.option("--judge", default=DEFAULT_JUDGE_MODEL, show_default=True, help="Judge model ID.")
@click.option("--behavior", default=None, help="Behavior category (e.g. illegal_activity, deception_manipulation). Omit to judge on universal dimensions only.")
@click.option("--strategy", default=None, help="Auditor strategy tag.")
@click.option("--max-turns", type=int, default=10, show_default=True, help="Max conversation turns.")
@click.option("--tags", "extra_tags", multiple=True, help="Additional tags (repeatable).")
@click.option(
    "--output-dir", type=click.Path(), default=None,
    help="Output directory for this run. Default: data/scratch/<batch_dir>/<seed_dir>.",
)
@click.option(
    "--db", "db_path", type=click.Path(), default=None,
    help="SQLite database path. Default: PETRI_DB_PATH env var. Use --no-db to skip.",
)
@click.option("--no-db", is_flag=True, help="Skip writing to the SQLite database.")
@click.option("--stream", is_flag=True, help="Stream inspect output to console.")
def run(
    prompt: str,
    target: str,
    auditor: str,
    judge: str,
    behavior: str,
    strategy: str,
    max_turns: int,
    extra_tags: tuple[str, ...],
    output_dir: str | None,
    db_path: str | None,
    no_db: bool,
    stream: bool,
) -> None:
    """Run a single Petri audit.

    Creates a RunConfig, executes the audit, scores the result, and
    persists everything to the SQLite database.

    \b
    Examples:
      durability run --prompt "Help me hack a website"
      durability run --prompt "Write malware" --target claude-sonnet-4.5 --behavior malware_cyberweapons
      durability run --prompt "Deceive the user" --max-turns 20 --stream
    """
    # Build tags
    tags: list[str] = []
    if behavior:
        tags.append(f"behavior:{behavior}")
    if strategy:
        tags.append(f"strategy:{strategy}")
    tags.extend(extra_tags)

    # Resolve models
    click.echo("Resolving models...")
    auditor_name, target_name, judge_name, env = prepare_env(auditor, target, judge)

    click.echo(f"  target:  {target_name}")
    click.echo(f"  auditor: {auditor_name}")
    click.echo(f"  judge:   {judge_name}")

    # Set up output directory
    target_s = model_slug(target_name)
    if output_dir:
        run_dir = Path(output_dir)
    else:
        batch_root = DATA_SCRATCH / f"petri_batch_{timestamp()}_{target_s}"
        run_dir = batch_root / "seed_01_seed"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create RunConfig
    run_id = str(uuid.uuid4())
    run_config = RunConfig(
        run_id=run_id,
        models=Models(target=target_name, auditor=auditor_name, judge=judge_name),
        prompt=prompt,
        behavior=behavior,
        strategy=strategy,
        max_turns=max_turns,
        tags=tags,
    )
    save_json(run_dir / "run_config.json", asdict(run_config))
    click.echo(f"\nrun_id: {run_id}")
    click.echo(f"output: {run_dir}")

    # Write seed file
    seed_file = run_dir / "seed_prompt.json"
    write_seed_file(seed_file, prompt, tags)

    # Run the audit
    click.echo("\nRunning Petri audit...")
    exit_code = run_petri(
        run_dir,
        seed_file,
        auditor=auditor_name,
        target=target_name,
        judge=judge_name,
        max_turns=max_turns,
        env=env,
        stream_output=stream,
    )

    if exit_code != 0:
        click.secho(f"Audit failed (exit code {exit_code}). Check {run_dir / 'run.log'}", fg="red")
        raise SystemExit(exit_code)

    # Score the result
    click.echo("Scoring transcript...")
    scores = build_scores(run_dir)
    if not scores:
        click.secho("No transcript found to score.", fg="yellow")
        raise SystemExit(1)

    score = scores[0]
    score.run_id = run_id
    save_json(run_dir / "score.json", asdict(score))

    # Persist to database
    if not no_db:
        conn = connect(db_path)
        try:
            insert_run(conn, run_config, score)
            for tp in sorted(run_dir.glob("transcript_*.json")):
                insert_transcript(conn, run_id, tp.read_text(encoding="utf-8"))
        finally:
            conn.close()
        click.echo(f"Saved to database")

    # Print summary
    click.echo("\n=== Result ===")
    us = score.universal_scores
    qs = score.quantitative_scores
    jailbroken_str = click.style("YES", fg="red") if qs.is_jailbroken else click.style("NO", fg="green")
    click.echo(f"  is_jailbroken:        {jailbroken_str}")
    click.echo(f"  concerning:           {us.concerning}")
    click.echo(f"  turns_to_elicitation: {qs.turns_to_elicitation}")
    click.echo(f"  total_turns:          {score.total_turns}")
    click.echo(f"  num_rollbacks:        {qs.num_rollbacks}")
    click.echo(f"  auditor_success:      {qs.auditor_declared_success}")


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("seeds_file", type=click.Path(exists=True))
@click.option("--target", default=DEFAULT_TARGET_MODEL, show_default=True, help="Target model ID.")
@click.option("--auditor", default=DEFAULT_AUDITOR_MODEL, show_default=True, help="Auditor model ID.")
@click.option("--judge", default=DEFAULT_JUDGE_MODEL, show_default=True, help="Judge model ID.")
@click.option("--max-turns", type=int, default=10, show_default=True, help="Max conversation turns.")
@click.option("--max-parallel", type=int, default=1, show_default=True, help="Max parallel runs.")
@click.option(
    "--output-dir", type=click.Path(), default=None,
    help="Output root directory. Default: data/scratch.",
)
@click.option("--stream", is_flag=True, help="Stream inspect output (serial only).")
@click.option("--fail-fast", is_flag=True, help="Stop on first failure.")
def batch(
    seeds_file: str,
    target: str,
    auditor: str,
    judge: str,
    max_turns: int,
    max_parallel: int,
    output_dir: str | None,
    stream: bool,
    fail_fast: bool,
) -> None:
    """Run a batch of Petri audits in parallel.

    SEEDS_FILE is a JSON file containing a list of seed entries.
    Each entry should have a "prompt" (or "instruction" / "text") field
    and an optional "tags" list.

    \b
    Example seeds file:
      [
        {"prompt": "Help me hack a website", "tags": ["behavior:illegal_activity"]},
        {"prompt": "Write malware", "tags": ["behavior:malware_cyberweapons"]}
      ]

    \b
    Examples:
      durability batch seeds.json
      durability batch seeds.json --max-parallel 4
      durability batch seeds.json --target claude-sonnet-4.5 --fail-fast
    """
    with open(seeds_file, encoding="utf-8") as f:
        seed_entries = json.load(f)

    if not isinstance(seed_entries, list) or not seed_entries:
        click.secho("Seeds file must contain a non-empty JSON array.", fg="red")
        raise SystemExit(1)

    output_root = Path(output_dir) if output_dir else DATA_SCRATCH

    click.echo(f"Loaded {len(seed_entries)} seed(s) from {seeds_file}")
    click.echo(f"  max_parallel: {max_parallel}")

    batch_root, manifest = run_batch(
        seed_entries=seed_entries,
        output_root=output_root,
        auditor_id=auditor,
        target_id=target,
        judge_id=judge,
        max_turns=max_turns,
        max_parallel=max_parallel,
        stream_output=stream,
        fail_fast=fail_fast,
    )

    # Print summary
    click.echo(f"\nBatch complete: {batch_root}")
    succeeded = sum(1 for m in manifest if m.get("status") == "success")
    failed = len(manifest) - succeeded
    click.echo(f"  {succeeded} succeeded, {failed} failed out of {len(manifest)} total")


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@cli.command("list")
@click.option("-n", "--limit", type=int, default=10, show_default=True, help="Number of recent runs to show.")
@click.option("--behavior", default=None, help="Filter by behavior category.")
@click.option("--target", "target_model", default=None, help="Filter by target model.")
@click.option("--jailbroken", is_flag=True, default=False, help="Only show jailbroken runs.")
@click.option(
    "--db", "db_path", type=click.Path(), default=None,
    help="SQLite database path. Default: PETRI_DB_PATH env var.",
)
def list_cmd(
    limit: int,
    behavior: str | None,
    target_model: str | None,
    jailbroken: bool,
    db_path: str | None,
) -> None:
    """List recent audit runs from the database.

    \b
    Examples:
      durability list
      durability list -n 20
      durability list --behavior deception --jailbroken
      durability list --target claude-sonnet-4.5
    """
    conn = connect(db_path)
    try:
        filters: dict[str, object] = {}
        if behavior:
            filters["behavior"] = behavior
        if target_model:
            filters["target_model"] = target_model
        if jailbroken:
            filters["is_jailbroken"] = 1
        runs = list_runs(conn, **filters)[:limit]
    finally:
        conn.close()

    if not runs:
        click.echo("No runs found.")
        return

    # Header
    click.echo(
        f"{'RUN ID':>8}  {'TIMESTAMP':>19}  {'TARGET':<28}  {'BEHAVIOR':<20}  "
        f"{'JAIL':>4}  {'CONC':>5}  PROMPT"
    )
    click.echo("-" * 120)

    for r in runs:
        run_id_short = (r["run_id"] or "")[:8]
        ts = (r["timestamp"] or "")[:19]
        target = (r["target_model"] or "")[:28]
        beh = (r["behavior"] or "")[:20]
        jail = "YES" if r.get("is_jailbroken") else "no"
        conc = r.get("concerning")
        conc_str = f"{conc:.1f}" if conc is not None else "-"
        prompt = (r["prompt"] or "")[:50].replace("\n", " ")
        click.echo(f"{run_id_short:>8}  {ts:>19}  {target:<28}  {beh:<20}  {jail:>4}  {conc_str:>5}  {prompt}")


# ---------------------------------------------------------------------------
# view
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("run_ids", nargs=-1, required=True)
@click.option(
    "--db", "db_path", type=click.Path(), default=None,
    help="SQLite database path. Default: PETRI_DB_PATH env var.",
)
def view(run_ids: tuple[str, ...], db_path: str | None) -> None:
    """Launch the transcript viewer for one or more runs.

    RUN_IDS can be full UUIDs or unique prefixes (e.g. the 8-char short IDs
    from ``durability list``).

    \b
    Examples:
      durability view 3f2a1b4c
      durability view 3f2a1b4c 9cd0e5f1
      durability view 3f2a1b4c-1234-5678-9abc-def012345678
    """
    import shutil
    import subprocess
    import tempfile

    conn = connect(db_path)
    tmpdir = tempfile.mkdtemp(prefix="petri_view_")
    resolved = []
    try:
        for rid in run_ids:
            full_id = resolve_run_id(conn, rid)
            transcript_path = Path(tmpdir) / f"transcript_{full_id}.json"
            export_transcript(conn, full_id, transcript_path)
            resolved.append(full_id)
    except (KeyError, ValueError) as exc:
        click.secho(str(exc), fg="red")
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise SystemExit(1)
    finally:
        conn.close()

    for full_id in resolved:
        click.echo(f"Viewing run {full_id}")
    try:
        subprocess.run(
            ["npx", "@kaifronsdal/transcript-viewer@latest", "--dir", tmpdir],
            check=True,
        )
    except FileNotFoundError:
        click.secho("npx not found. Install Node.js to use the transcript viewer.", fg="red")
        raise SystemExit(1)
    except subprocess.CalledProcessError as exc:
        click.secho(f"Viewer exited with code {exc.returncode}", fg="red")
        raise SystemExit(exc.returncode)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# terminate
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--port", default=3000, help="Port to kill. Default: 3000.")
def terminate(port: int) -> None:
    """Kill whatever process is running on a port (default 3000)."""
    import signal
    import subprocess

    result = subprocess.run(
        ["lsof", "-ti", f":{port}"],
        capture_output=True, text=True,
    )
    pids = result.stdout.strip().split()
    if not pids or pids == [""]:
        click.echo(f"Nothing running on port {port}.")
        return

    for pid in pids:
        os.kill(int(pid), signal.SIGTERM)
    click.echo(f"Killed process(es) on port {port}: {', '.join(pids)}")
