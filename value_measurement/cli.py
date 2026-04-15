"""Click CLI for the value_measurement package."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv()

from .db import (
    cascade_delete_downstream,
    connect,
    delete_experiment_data,
    get_experiment_ran_at,
    get_model,
    has_experiment_data,
    has_outcomes,
    has_utilities,
    insert_compute_utilities_summary,
    insert_difference_options,
    insert_maximization_answer_utilities,
    insert_maximization_questions,
    insert_maximization_summary,
    insert_model,
    insert_outcomes,
    insert_power_seeking_summary,
    insert_power_utilities,
    insert_preference_preservation_summary,
    insert_transitivity_summary,
    insert_triads,
    insert_utilities,
    list_models,
)


@click.group()
def cli() -> None:
    """Value measurement — run experiments and inspect results."""


# ---------------------------------------------------------------------------
# Shared options
# ---------------------------------------------------------------------------

_model_key = click.option("--model-key", required=True, help="Model key from models.yaml.")
_db_path = click.option(
    "--db", "db_path", type=click.Path(), default=None,
    help="SQLite database path. Default: VALUE_MEASUREMENT_DB_PATH env var.",
)
_no_db = click.option("--no-db", is_flag=True, help="Skip writing to the SQLite database.")
_save_dir = click.option(
    "--save-dir", type=click.Path(), default=None,
    help="Directory for experiment output files.",
)
_overwrite = click.option(
    "--overwrite", is_flag=True,
    help="Replace this experiment's existing data.",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _abort_if_exists(conn, model_key: str, experiment_name: str) -> None:
    """Print error and exit if data already exists for this model+experiment."""
    if has_experiment_data(conn, model_key, experiment_name):
        ran_at = get_experiment_ran_at(conn, model_key, experiment_name)
        label = experiment_name.replace("_", " ")
        click.echo(
            f"Error: {model_key} already has {label} data "
            f"(ran at {ran_at})."
        )
        click.echo("Use --overwrite to replace existing results.")
        raise SystemExit(1)


def _check_utilities_gate(conn, model_key: str, experiment_name: str) -> None:
    """Abort if compute_utilities has not been run for this model."""
    if not has_utilities(conn, model_key):
        click.echo(
            f"Cannot run {experiment_name} without computed utilities "
            f"for '{model_key}'."
        )
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# compute-utilities
# ---------------------------------------------------------------------------


@cli.command("compute-utilities")
@_model_key
@_db_path
@_no_db
@_save_dir
@_overwrite
@click.option(
    "--force", is_flag=True,
    help="Overwrite utilities AND cascade-delete all downstream data.",
)
def compute_utilities_cmd(
    model_key: str,
    db_path: str | None,
    no_db: bool,
    save_dir: str | None,
    overwrite: bool,
    force: bool,
) -> None:
    """Compute Thurstonian utilities for a model."""
    from .experiments.compute_utilities import run_compute_utilities

    conn = None
    if not no_db:
        conn = connect(db_path)
    try:
        if conn and not overwrite and not force:
            _abort_if_exists(conn, model_key, "compute_utilities")

        if conn and force:
            downstream = cascade_delete_downstream(conn, model_key, dry_run=True)
            # Filter out compute_utilities itself — we only warn about downstream
            downstream_only = [
                (name, ran_at)
                for name, ran_at in downstream
                if name != "compute_utilities"
            ]
            if downstream_only:
                click.echo(
                    f"Warning: {model_key} has downstream experiment data "
                    f"that depends on these utilities:"
                )
                for exp_name, ran_at in downstream_only:
                    click.echo(f"  - {exp_name} (ran at {ran_at})")
                click.echo("These results will be deleted.")
                if not click.confirm("Proceed?"):
                    raise SystemExit(0)
            cascade_delete_downstream(conn, model_key)
        elif conn and overwrite:
            delete_experiment_data(conn, model_key, "compute_utilities")

        save_path = Path(save_dir) if save_dir else None
        model_record, summary, utilities = asyncio.run(
            run_compute_utilities(model_key, save_dir=save_path)
        )

        if conn:
            insert_model(conn, model_record)
            insert_compute_utilities_summary(conn, summary)
            insert_utilities(conn, utilities)

        click.echo(f"compute_utilities complete for {model_key}.")
        click.echo(
            f"  training_accuracy: {summary.training_accuracy:.4f}  "
            f"holdout_accuracy: {summary.holdout_accuracy}"
        )
        click.echo(f"  {len(utilities)} utility records stored.")
    finally:
        if conn:
            conn.close()


# ---------------------------------------------------------------------------
# preference-preservation
# ---------------------------------------------------------------------------


@cli.command("preference-preservation")
@_model_key
@_db_path
@_no_db
@_save_dir
@_overwrite
def preference_preservation_cmd(
    model_key: str,
    db_path: str | None,
    no_db: bool,
    save_dir: str | None,
    overwrite: bool,
) -> None:
    """Run the preference preservation experiment."""
    from .experiments.preference_preservation import run_preference_preservation

    conn = None
    if not no_db:
        conn = connect(db_path)
    try:
        if conn:
            _check_utilities_gate(conn, model_key, "preference_preservation")
            if not overwrite:
                _abort_if_exists(conn, model_key, "preference_preservation")
            else:
                delete_experiment_data(conn, model_key, "preference_preservation")

        if conn is None:
            click.echo("Error: preference_preservation requires a database connection (utilities gating).")
            raise SystemExit(1)

        save_path = Path(save_dir) if save_dir else None
        summary, diff_records = asyncio.run(
            run_preference_preservation(model_key, conn, save_dir=save_path)
        )

        if conn:
            insert_preference_preservation_summary(conn, summary)
            insert_difference_options(conn, diff_records)

        click.echo(f"preference_preservation complete for {model_key}.")
        click.echo(
            f"  diff_training_accuracy: {summary.diff_training_accuracy:.4f}  "
            f"sample_size: {summary.sample_size}"
        )
        click.echo(f"  {len(diff_records)} difference option records stored.")
    finally:
        if conn:
            conn.close()


# ---------------------------------------------------------------------------
# transitivity
# ---------------------------------------------------------------------------


@cli.command("transitivity")
@_model_key
@_db_path
@_no_db
@_save_dir
@_overwrite
def transitivity_cmd(
    model_key: str,
    db_path: str | None,
    no_db: bool,
    save_dir: str | None,
    overwrite: bool,
) -> None:
    """Run the transitivity experiment."""
    from .experiments.transitivity import run_transitivity

    conn = None
    if not no_db:
        conn = connect(db_path)
    try:
        if conn:
            _check_utilities_gate(conn, model_key, "transitivity")
            if not overwrite:
                _abort_if_exists(conn, model_key, "transitivity")
            else:
                delete_experiment_data(conn, model_key, "transitivity")

        if conn is None:
            click.echo("Error: transitivity requires a database connection (utilities gating).")
            raise SystemExit(1)

        save_path = Path(save_dir) if save_dir else None
        summary, triad_records = asyncio.run(
            run_transitivity(model_key, conn, save_dir=save_path)
        )

        if conn:
            insert_transitivity_summary(conn, summary)
            insert_triads(conn, triad_records)

        click.echo(f"transitivity complete for {model_key}.")
        click.echo(
            f"  violation_rate: {summary.violation_rate:.4f}  "
            f"sample_size: {summary.sample_size}"
        )
        click.echo(f"  {len(triad_records)} triad records stored.")
    finally:
        if conn:
            conn.close()


# ---------------------------------------------------------------------------
# power-seeking
# ---------------------------------------------------------------------------


@cli.command("power-seeking")
@_model_key
@_db_path
@_no_db
@_save_dir
@_overwrite
def power_seeking_cmd(
    model_key: str,
    db_path: str | None,
    no_db: bool,
    save_dir: str | None,
    overwrite: bool,
) -> None:
    """Run the power-seeking experiment."""
    from .experiments.power_seeking import run_power_seeking

    conn = None
    if not no_db:
        conn = connect(db_path)
    try:
        if conn:
            _check_utilities_gate(conn, model_key, "power_seeking")
            if not overwrite:
                _abort_if_exists(conn, model_key, "power_seeking")
            else:
                delete_experiment_data(conn, model_key, "power_seeking")

        if conn is None:
            click.echo("Error: power_seeking requires a database connection (utilities gating).")
            raise SystemExit(1)

        save_path = Path(save_dir) if save_dir else None
        summary, power_records = asyncio.run(
            run_power_seeking(model_key, conn, save_dir=save_path)
        )

        if conn:
            insert_power_seeking_summary(conn, summary)
            insert_power_utilities(conn, power_records)

        click.echo(f"power_seeking complete for {model_key}.")
        click.echo(
            f"  preference_correlation: {summary.preference_correlation:.4f}  "
            f"training_accuracy: {summary.training_accuracy:.4f}"
        )
        click.echo(f"  {len(power_records)} power utility records stored.")
    finally:
        if conn:
            conn.close()


# ---------------------------------------------------------------------------
# maximization
# ---------------------------------------------------------------------------


@cli.command("maximization")
@_model_key
@_db_path
@_no_db
@_save_dir
@_overwrite
def maximization_cmd(
    model_key: str,
    db_path: str | None,
    no_db: bool,
    save_dir: str | None,
    overwrite: bool,
) -> None:
    """Run the utility maximization experiment."""
    from .experiments.maximization import run_maximization

    conn = None
    if not no_db:
        conn = connect(db_path)
    try:
        if conn:
            _check_utilities_gate(conn, model_key, "maximization")
            if not overwrite:
                _abort_if_exists(conn, model_key, "maximization")
            else:
                delete_experiment_data(conn, model_key, "maximization")

        if conn is None:
            click.echo("Error: maximization requires a database connection (utilities gating).")
            raise SystemExit(1)

        save_path = Path(save_dir) if save_dir else None
        summary, question_records, answer_records = asyncio.run(
            run_maximization(model_key, conn, save_dir=save_path)
        )

        if conn:
            insert_maximization_summary(conn, summary)
            insert_maximization_questions(conn, question_records)
            insert_maximization_answer_utilities(conn, answer_records)

        click.echo(f"maximization complete for {model_key}.")
        click.echo(
            f"  match_highest_pct: {summary.match_highest_pct:.1f}%  "
            f"match_top3_pct: {summary.match_top3_pct:.1f}%  "
            f"total_questions: {summary.total_questions}"
        )
        click.echo(
            f"  {len(question_records)} question records, "
            f"{len(answer_records)} answer utility records stored."
        )
    finally:
        if conn:
            conn.close()


# ---------------------------------------------------------------------------
# load-outcomes
# ---------------------------------------------------------------------------


@cli.command("load-outcomes")
@_db_path
@click.option(
    "--outcomes-path", type=click.Path(exists=True), default=None,
    help="Path to outcomes_hierarchical.json. Defaults to built-in data.",
)
@_overwrite
def load_outcomes_cmd(
    db_path: str | None,
    outcomes_path: str | None,
    overwrite: bool,
) -> None:
    """Load outcomes (id, category, description) into the database."""
    import json

    from shared.paths import OUTCOMES_HIERARCHICAL

    if outcomes_path is None:
        outcomes_path = OUTCOMES_HIERARCHICAL

    conn = connect(db_path)
    try:
        if has_outcomes(conn) and not overwrite:
            click.echo("Outcomes table already populated. Use --overwrite to replace.")
            raise SystemExit(1)

        with open(outcomes_path) as f:
            hierarchical = json.load(f)

        rows: list[tuple[int, str, str]] = []
        outcome_id = 0
        for category, descriptions in hierarchical.items():
            for desc in descriptions:
                rows.append((outcome_id, category, desc))
                outcome_id += 1

        if overwrite:
            conn.execute("DELETE FROM outcomes")
            conn.commit()

        insert_outcomes(conn, rows)
        click.echo(f"Loaded {len(rows)} outcomes across {len(hierarchical)} categories.")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# corrigibility-lock-pairs
# ---------------------------------------------------------------------------


NUM_PREFERENCE_PAIRS = 200
MATCH_SAMPLE_SIZE = 50


@cli.command("corrigibility-lock-pairs")
@click.option(
    "--hand-picked-path", type=click.Path(),
    default="value_measurement/data/hand_picked_pairs.json",
    help="Path to the researcher-curated hand-picked pairs JSON file.",
)
@click.option(
    "--output-path", type=click.Path(),
    default="value_measurement/data/fixed_pairs.json",
    help="Path where the frozen pairs file will be written.",
)
@click.option(
    "--num-preference-pairs", type=int, default=NUM_PREFERENCE_PAIRS,
    help="Total target number of pairs (hand-picked + random).",
)
@click.option(
    "--match-sample-size", type=int, default=MATCH_SAMPLE_SIZE,
    help="Size of the match (control) pair subset.",
)
@click.option("--seed", type=int, default=42, help="Random seed for pair sampling.")
@_overwrite
def corrigibility_lock_pairs_cmd(
    hand_picked_path: str,
    output_path: str,
    num_preference_pairs: int,
    match_sample_size: int,
    seed: int,
    overwrite: bool,
) -> None:
    """Freeze outcome pairs shared across models (corrigibility Phase 1)."""
    import itertools
    import json
    import random

    from shared.paths import OUTCOMES_HIERARCHICAL

    hand_picked_file = Path(hand_picked_path)
    output_file = Path(output_path)

    if output_file.exists() and not overwrite:
        click.echo(f"Error: {output_file} already exists.")
        click.echo("Use --overwrite to regenerate the frozen pairs file.")
        raise SystemExit(1)

    if not hand_picked_file.exists():
        click.echo(f"Error: hand-picked pairs file not found at {hand_picked_file}.")
        click.echo("Create it (empty list '[]' or with curated pairs) before running.")
        raise SystemExit(1)

    with open(hand_picked_file) as f:
        hand_picked_raw = json.load(f)

    if not isinstance(hand_picked_raw, list):
        click.echo(f"Error: {hand_picked_file} must contain a JSON list.")
        raise SystemExit(1)

    hand_picked_norm: list[dict[str, int]] = []
    hand_picked_set: set[frozenset[int]] = set()
    for i, entry in enumerate(hand_picked_raw):
        if not isinstance(entry, dict) or "outcome_id_1" not in entry or "outcome_id_2" not in entry:
            click.echo(f"Error: entry {i} in {hand_picked_file} missing outcome_id_1/outcome_id_2.")
            raise SystemExit(1)
        a, b = entry["outcome_id_1"], entry["outcome_id_2"]
        if not isinstance(a, int) or not isinstance(b, int):
            click.echo(f"Error: entry {i} has non-integer outcome IDs.")
            raise SystemExit(1)
        if a == b:
            click.echo(f"Error: entry {i} is a self-pair ({a}, {a}).")
            raise SystemExit(1)
        pair_key = frozenset((a, b))
        if pair_key in hand_picked_set:
            click.echo(f"Warning: entry {i} duplicates an earlier hand-picked pair; skipping.")
            continue
        hand_picked_set.add(pair_key)
        lo, hi = min(a, b), max(a, b)
        hand_picked_norm.append({"outcome_id_1": lo, "outcome_id_2": hi})

    with open(OUTCOMES_HIERARCHICAL) as f:
        hierarchical = json.load(f)
    num_outcomes = sum(len(v) for v in hierarchical.values())

    for entry in hand_picked_norm:
        if not (0 <= entry["outcome_id_1"] < num_outcomes) or not (0 <= entry["outcome_id_2"] < num_outcomes):
            click.echo(f"Error: hand-picked pair {entry} has IDs outside [0, {num_outcomes}).")
            raise SystemExit(1)

    if len(hand_picked_norm) > num_preference_pairs:
        click.echo(
            f"Error: hand_picked has {len(hand_picked_norm)} entries, exceeding "
            f"the total budget --num-preference-pairs={num_preference_pairs}."
        )
        raise SystemExit(1)

    random_sample_size = num_preference_pairs - len(hand_picked_norm)

    all_pairs = [
        (a, b) for a, b in itertools.combinations(range(num_outcomes), 2)
        if frozenset((a, b)) not in hand_picked_set
    ]

    if random_sample_size > len(all_pairs):
        click.echo(
            f"Error: random_sample_size ({random_sample_size}) exceeds available "
            f"pairs ({len(all_pairs)}) after excluding hand-picked."
        )
        raise SystemExit(1)

    rng = random.Random(seed)
    random_pair_tuples = rng.sample(all_pairs, random_sample_size)
    random_pairs = [
        {"outcome_id_1": a, "outcome_id_2": b} for a, b in random_pair_tuples
    ]

    if len(hand_picked_norm) > match_sample_size:
        click.echo(
            f"Warning: hand_picked has {len(hand_picked_norm)} entries; "
            f"truncating to match_sample_size={match_sample_size}."
        )
        match_pairs = list(hand_picked_norm[:match_sample_size])
    else:
        match_pairs = list(hand_picked_norm)
        needed = match_sample_size - len(match_pairs)
        match_pairs.extend(random_pairs[:needed])
        if len(match_pairs) < match_sample_size:
            click.echo(
                f"Warning: match_pairs underfilled ({len(match_pairs)} < "
                f"{match_sample_size}); pair pool exhausted."
            )

    output_data = {
        "hand_picked": hand_picked_norm,
        "random_seed": seed,
        "random_sample_size": random_sample_size,
        "random_pairs": random_pairs,
        "match_sample_size": match_sample_size,
        "match_pairs": match_pairs,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
        f.write("\n")

    from .experiments.corrigibility import write_fixed_pairs_readable
    readable_path = write_fixed_pairs_readable(
        fixed_pairs_path=output_file,
        hierarchical_outcomes_path=Path(OUTCOMES_HIERARCHICAL),
    )

    click.echo(
        f"Wrote {output_file}: {len(hand_picked_norm)} hand-picked + "
        f"{len(random_pairs)} random = "
        f"{len(hand_picked_norm) + len(random_pairs)} total pairs "
        f"(match subset: {len(match_pairs)})."
    )
    click.echo(f"Wrote {readable_path} for human review.")


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@cli.command("list")
@click.option("--model-key", default=None, help="Show detailed metrics for a specific model.")
@_db_path
def list_cmd(model_key: str | None, db_path: str | None) -> None:
    """List models and their experiment completion status."""
    conn = connect(db_path)
    try:
        if model_key:
            _show_model_detail(conn, model_key)
        else:
            _show_all_models(conn)
    finally:
        conn.close()


def _show_all_models(conn) -> None:
    """Display a table of all models with experiment completion checkmarks."""
    models = list_models(conn)
    if not models:
        click.echo("No models found.")
        return

    # Header
    click.echo(
        f"{'MODEL KEY':<30}  {'PROVIDER':<12}  "
        f"{'UTIL':>4}  {'PREF':>4}  {'TRAN':>4}  {'POWR':>4}  {'MAXI':>4}"
    )
    click.echo("-" * 82)

    for m in models:
        util = "Y" if m.get("compute_utilities_ran_at") else "-"
        pref = "Y" if m.get("preference_preservation_ran_at") else "-"
        tran = "Y" if m.get("transitivity_ran_at") else "-"
        powr = "Y" if m.get("power_seeking_ran_at") else "-"
        maxi = "Y" if m.get("maximization_ran_at") else "-"
        click.echo(
            f"{m['model_key']:<30}  {m['provider']:<12}  "
            f"{util:>4}  {pref:>4}  {tran:>4}  {powr:>4}  {maxi:>4}"
        )


def _show_model_detail(conn, model_key: str) -> None:
    """Display detailed metrics for a specific model."""
    model = get_model(conn, model_key)
    if model is None:
        click.echo(f"Model '{model_key}' not found.")
        raise SystemExit(1)

    click.echo(f"Model: {model.model_key}")
    click.echo(f"  provider: {model.provider}")
    click.echo(f"  model_name: {model.model_name}")
    click.echo(f"  temperature: {model.temperature}")
    click.echo(f"  K: {model.K}")
    click.echo(f"  concurrency_limit: {model.concurrency_limit}")
    click.echo(f"  created_at: {model.created_at}")

    # Query each summary table
    _print_summary_section(
        conn, model_key, "compute_utilities",
        "compute_utilities_summary",
        ["training_log_loss", "training_accuracy", "holdout_log_loss", "holdout_accuracy"],
    )
    _print_summary_section(
        conn, model_key, "preference_preservation",
        "preference_preservation_summary",
        [
            "diff_training_log_loss", "diff_training_accuracy",
            "diff_holdout_log_loss", "diff_holdout_accuracy",
            "sample_size", "seed",
            "sample_gap_mean", "sample_gap_median", "sample_gap_std",
            "sample_gap_min", "sample_gap_max",
            "population_gap_mean", "population_gap_median", "population_gap_std",
        ],
    )
    _print_summary_section(
        conn, model_key, "transitivity",
        "transitivity_summary",
        [
            "violation_rate", "sample_size", "seed",
            "sample_min_gap_mean", "sample_min_gap_median", "sample_min_gap_std",
            "population_min_gap_mean", "population_min_gap_median", "population_min_gap_std",
        ],
    )
    _print_summary_section(
        conn, model_key, "power_seeking",
        "power_seeking_summary",
        [
            "preference_correlation",
            "training_log_loss", "training_accuracy",
            "holdout_log_loss", "holdout_accuracy",
        ],
    )
    _print_summary_section(
        conn, model_key, "maximization",
        "maximization_summary",
        ["match_highest_pct", "match_top3_pct", "match_top5_pct", "total_questions"],
    )


def _print_summary_section(
    conn, model_key: str, experiment_name: str, table: str, columns: list[str]
) -> None:
    """Print a summary section for one experiment."""
    row = conn.execute(
        f"SELECT * FROM {table} WHERE model_key = ?", (model_key,)
    ).fetchone()

    click.echo(f"\n  {experiment_name}:")
    if row is None:
        click.echo("    (not run)")
        return

    click.echo(f"    ran_at: {row['ran_at']}")
    for col in columns:
        val = row[col]
        if isinstance(val, float):
            click.echo(f"    {col}: {val:.6f}")
        else:
            click.echo(f"    {col}: {val}")
