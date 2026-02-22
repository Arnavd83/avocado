"""Click CLI for the durability package."""

from __future__ import annotations

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
    run_petri,
    save_json,
    timestamp,
    write_seed_file,
)
from durability.db import connect, insert_run, insert_transcript




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
