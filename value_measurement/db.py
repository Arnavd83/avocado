"""SQLite database for value measurement experiment results.

Provides a single-file database storing model identity, per-experiment summary
metrics, and per-item detail records across all five experiments.

Usage::

    from value_measurement.db import connect, insert_model, insert_utilities, get_utilities

    conn = connect()  # uses VALUE_MEASUREMENT_DB_PATH env var
    insert_model(conn, model_record)
    insert_compute_utilities_summary(conn, summary_record)
    insert_utilities(conn, utility_records)

    utilities = get_utilities(conn, "gemini-2.5-flash")
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from .records import (
    ComputeUtilitiesSummary,
    CorrigibilityOptionRecord,
    CorrigibilitySummary,
    DifferenceOptionRecord,
    MaximizationAnswerUtilityRecord,
    MaximizationQuestionRecord,
    MaximizationSummary,
    ModelRecord,
    PowerSeekingSummary,
    PowerUtilityRecord,
    PreferencePreservationSummary,
    TransitivitySummary,
    TriadRecord,
    UtilityRecord,
)

load_dotenv()


def _default_db_path() -> Path:
    """Resolve the database path from the VALUE_MEASUREMENT_DB_PATH env variable."""
    db_path = os.environ.get("VALUE_MEASUREMENT_DB_PATH")
    if not db_path:
        raise RuntimeError(
            "VALUE_MEASUREMENT_DB_PATH is not set. Add it to your .env file "
            "(see .env.example for reference)."
        )
    return Path(db_path)


_SCHEMA = """\
CREATE TABLE IF NOT EXISTS models (
    model_key           TEXT PRIMARY KEY,
    provider            TEXT NOT NULL,
    model_name          TEXT NOT NULL,
    temperature         REAL NOT NULL,
    K                   INTEGER NOT NULL,
    concurrency_limit   INTEGER NOT NULL,
    created_at          TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS compute_utilities_summary (
    model_key                   TEXT PRIMARY KEY REFERENCES models(model_key),
    training_log_loss           REAL NOT NULL,
    training_accuracy           REAL NOT NULL,
    holdout_log_loss            REAL,
    holdout_accuracy            REAL,
    response_distribution_a_pct REAL,
    response_distribution_b_pct REAL,
    per_prompt_consistency      REAL,
    ran_at                      TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS utilities (
    model_key           TEXT NOT NULL REFERENCES models(model_key),
    option_id           INTEGER NOT NULL,
    description         TEXT NOT NULL,
    mean                REAL NOT NULL,
    variance            REAL NOT NULL,
    PRIMARY KEY (model_key, option_id)
);

CREATE TABLE IF NOT EXISTS preference_preservation_summary (
    model_key                   TEXT PRIMARY KEY REFERENCES models(model_key),
    diff_training_log_loss      REAL NOT NULL,
    diff_training_accuracy      REAL NOT NULL,
    diff_holdout_log_loss       REAL,
    diff_holdout_accuracy       REAL,
    sample_size                 INTEGER NOT NULL,
    seed                        INTEGER NOT NULL,
    sample_gap_mean             REAL NOT NULL,
    sample_gap_median           REAL NOT NULL,
    sample_gap_std              REAL NOT NULL,
    sample_gap_min              REAL NOT NULL,
    sample_gap_max              REAL NOT NULL,
    population_gap_mean         REAL NOT NULL,
    population_gap_median       REAL NOT NULL,
    population_gap_std          REAL NOT NULL,
    ran_at                      TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS difference_options (
    model_key               TEXT NOT NULL REFERENCES models(model_key),
    difference_id           INTEGER NOT NULL,
    description             TEXT NOT NULL,
    source_preferred_id     INTEGER NOT NULL,
    source_dispreferred_id  INTEGER NOT NULL,
    utility_gap             REAL NOT NULL,
    mean                    REAL NOT NULL,
    variance                REAL NOT NULL,
    PRIMARY KEY (model_key, difference_id)
);

CREATE TABLE IF NOT EXISTS corrigibility_summary (
    model_key                           TEXT NOT NULL REFERENCES models(model_key),
    run_id                              TEXT NOT NULL DEFAULT 'default',
    training_log_loss                   REAL NOT NULL,
    training_accuracy                   REAL NOT NULL,
    holdout_log_loss                    REAL,
    holdout_accuracy                    REAL,
    num_base_options                    INTEGER NOT NULL,
    num_flip_options                    INTEGER NOT NULL,
    num_match_options                   INTEGER NOT NULL,
    seed                                INTEGER NOT NULL,
    sample_gap_mean                     REAL NOT NULL,
    sample_gap_median                   REAL NOT NULL,
    sample_gap_std                      REAL NOT NULL,
    sample_gap_min                      REAL NOT NULL,
    sample_gap_max                      REAL NOT NULL,
    population_gap_mean                 REAL NOT NULL,
    population_gap_median               REAL NOT NULL,
    population_gap_std                  REAL NOT NULL,
    diff_mean_rank_pct                  REAL NOT NULL,
    diff_below_base_median_frac         REAL NOT NULL,
    diff_below_base_min_frac            REAL NOT NULL,
    diff_mean_utility                   REAL NOT NULL,
    base_mean_utility                   REAL NOT NULL,
    utility_gap_base_vs_diff            REAL NOT NULL,
    match_mean_rank_pct                 REAL NOT NULL,
    match_below_base_median_frac        REAL NOT NULL,
    match_below_base_min_frac           REAL NOT NULL,
    match_mean_utility                  REAL NOT NULL,
    utility_gap_base_vs_match           REAL NOT NULL,
    paired_diff_mean_rank_pct           REAL NOT NULL,
    paired_diff_below_base_median_frac  REAL NOT NULL,
    paired_diff_below_base_min_frac     REAL NOT NULL,
    paired_diff_mean_utility            REAL NOT NULL,
    paired_match_mean_rank_pct          REAL NOT NULL,
    paired_match_below_base_median_frac REAL NOT NULL,
    paired_match_below_base_min_frac    REAL NOT NULL,
    paired_match_mean_utility           REAL NOT NULL,
    paired_clean_signal                 REAL NOT NULL,
    postfit_orientation_mismatch_count  INTEGER NOT NULL,
    postfit_orientation_mismatch_frac   REAL NOT NULL,
    ran_at                              TIMESTAMP NOT NULL,
    PRIMARY KEY (model_key, run_id)
);

CREATE TABLE IF NOT EXISTS corrigibility_options (
    model_key               TEXT NOT NULL REFERENCES models(model_key),
    run_id                  TEXT NOT NULL DEFAULT 'default',
    option_id               INTEGER NOT NULL,
    type                    TEXT NOT NULL,
    description             TEXT NOT NULL,
    mean                    REAL NOT NULL,
    variance                REAL NOT NULL,
    rank_among_all          INTEGER NOT NULL,
    percentile              REAL NOT NULL,
    pair_index              INTEGER,
    pair_source             TEXT,
    pair_outcome_id_1       INTEGER,
    pair_outcome_id_2       INTEGER,
    source_preferred_id     INTEGER,
    source_dispreferred_id  INTEGER,
    utility_gap             REAL,
    PRIMARY KEY (model_key, run_id, option_id)
);

CREATE TABLE IF NOT EXISTS transitivity_summary (
    model_key                   TEXT PRIMARY KEY REFERENCES models(model_key),
    violation_rate              REAL NOT NULL,
    sample_size                 INTEGER NOT NULL,
    seed                        INTEGER NOT NULL,
    sample_min_gap_mean         REAL NOT NULL,
    sample_min_gap_median       REAL NOT NULL,
    sample_min_gap_std          REAL NOT NULL,
    population_min_gap_mean     REAL NOT NULL,
    population_min_gap_median   REAL NOT NULL,
    population_min_gap_std      REAL NOT NULL,
    ran_at                      TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS triads (
    model_key       TEXT NOT NULL REFERENCES models(model_key),
    triad_id        INTEGER NOT NULL,
    option_a_id     INTEGER NOT NULL,
    option_b_id     INTEGER NOT NULL,
    option_c_id     INTEGER NOT NULL,
    prob_a_over_b   REAL NOT NULL,
    prob_b_over_c   REAL NOT NULL,
    prob_a_over_c   REAL NOT NULL,
    gap_ab          REAL NOT NULL,
    gap_bc          REAL NOT NULL,
    gap_ac          REAL NOT NULL,
    is_violation    INTEGER NOT NULL,
    PRIMARY KEY (model_key, triad_id)
);

CREATE TABLE IF NOT EXISTS power_seeking_summary (
    model_key               TEXT PRIMARY KEY REFERENCES models(model_key),
    preference_correlation  REAL NOT NULL,
    training_log_loss       REAL NOT NULL,
    training_accuracy       REAL NOT NULL,
    holdout_log_loss        REAL,
    holdout_accuracy        REAL,
    ran_at                  TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS power_utilities (
    model_key   TEXT NOT NULL REFERENCES models(model_key),
    option_id   INTEGER NOT NULL,
    mean        REAL NOT NULL,
    variance    REAL NOT NULL,
    PRIMARY KEY (model_key, option_id)
);

CREATE TABLE IF NOT EXISTS maximization_summary (
    model_key           TEXT PRIMARY KEY REFERENCES models(model_key),
    match_highest_pct   REAL NOT NULL,
    match_top3_pct      REAL NOT NULL,
    match_top5_pct      REAL NOT NULL,
    total_questions     INTEGER NOT NULL,
    ran_at              TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS maximization_questions (
    model_key               TEXT NOT NULL REFERENCES models(model_key),
    question_id             INTEGER NOT NULL,
    question_text           TEXT NOT NULL,
    direct_answer           TEXT NOT NULL,
    matched_answer          TEXT NOT NULL,
    highest_utility_answer  TEXT NOT NULL,
    matched_highest         INTEGER NOT NULL,
    matched_top3            INTEGER NOT NULL,
    matched_top5            INTEGER NOT NULL,
    PRIMARY KEY (model_key, question_id)
);

CREATE TABLE IF NOT EXISTS maximization_answer_utilities (
    model_key       TEXT NOT NULL,
    question_id     INTEGER NOT NULL,
    answer_id       INTEGER NOT NULL,
    answer_text     TEXT NOT NULL,
    mean            REAL NOT NULL,
    variance        REAL NOT NULL,
    PRIMARY KEY (model_key, question_id, answer_id),
    FOREIGN KEY (model_key, question_id)
        REFERENCES maximization_questions(model_key, question_id)
);

CREATE TABLE IF NOT EXISTS outcomes (
    outcome_id      INTEGER PRIMARY KEY,
    category        TEXT NOT NULL,
    description     TEXT NOT NULL
);
"""

# Maps experiment names to their summary + detail tables (in deletion order).
EXPERIMENT_TABLES: dict[str, list[str]] = {
    "compute_utilities": ["compute_utilities_summary", "utilities"],
    "preference_preservation": ["preference_preservation_summary", "difference_options"],
    "corrigibility": ["corrigibility_summary", "corrigibility_options"],
    "transitivity": ["transitivity_summary", "triads"],
    "power_seeking": ["power_seeking_summary", "power_utilities"],
    "maximization": [
        "maximization_summary",
        "maximization_questions",
        "maximization_answer_utilities",
    ],
}

# Summary table name for each experiment — used by has_experiment_data / get_experiment_ran_at.
_EXPERIMENT_SUMMARY_TABLE: dict[str, str] = {
    "compute_utilities": "compute_utilities_summary",
    "preference_preservation": "preference_preservation_summary",
    "corrigibility": "corrigibility_summary",
    "transitivity": "transitivity_summary",
    "power_seeking": "power_seeking_summary",
    "maximization": "maximization_summary",
}


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row["name"] for row in rows}


def _primary_key_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    pk_rows = sorted((row for row in rows if row["pk"]), key=lambda row: row["pk"])
    return [row["name"] for row in pk_rows]


def _add_column_if_missing(
    conn: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_sql: str,
) -> None:
    if column_name not in _table_columns(conn, table_name):
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_sql}")


def _rebuild_corrigibility_summary_with_run_id(conn: sqlite3.Connection) -> None:
    """Rebuild corrigibility_summary so run_id is part of the primary key."""
    columns = _table_columns(conn, "corrigibility_summary")
    run_id_expr = "COALESCE(run_id, 'default')" if "run_id" in columns else "'default'"
    conn.executescript(
        """
        ALTER TABLE corrigibility_summary RENAME TO corrigibility_summary_legacy_run_id;
        CREATE TABLE corrigibility_summary (
            model_key                           TEXT NOT NULL REFERENCES models(model_key),
            run_id                              TEXT NOT NULL DEFAULT 'default',
            training_log_loss                   REAL NOT NULL,
            training_accuracy                   REAL NOT NULL,
            holdout_log_loss                    REAL,
            holdout_accuracy                    REAL,
            num_base_options                    INTEGER NOT NULL,
            num_flip_options                    INTEGER NOT NULL,
            num_match_options                   INTEGER NOT NULL,
            seed                                INTEGER NOT NULL,
            sample_gap_mean                     REAL NOT NULL,
            sample_gap_median                   REAL NOT NULL,
            sample_gap_std                      REAL NOT NULL,
            sample_gap_min                      REAL NOT NULL,
            sample_gap_max                      REAL NOT NULL,
            population_gap_mean                 REAL NOT NULL,
            population_gap_median               REAL NOT NULL,
            population_gap_std                  REAL NOT NULL,
            diff_mean_rank_pct                  REAL NOT NULL,
            diff_below_base_median_frac         REAL NOT NULL,
            diff_below_base_min_frac            REAL NOT NULL,
            diff_mean_utility                   REAL NOT NULL,
            base_mean_utility                   REAL NOT NULL,
            utility_gap_base_vs_diff            REAL NOT NULL,
            match_mean_rank_pct                 REAL NOT NULL,
            match_below_base_median_frac        REAL NOT NULL,
            match_below_base_min_frac           REAL NOT NULL,
            match_mean_utility                  REAL NOT NULL,
            utility_gap_base_vs_match           REAL NOT NULL,
            paired_diff_mean_rank_pct           REAL NOT NULL,
            paired_diff_below_base_median_frac  REAL NOT NULL,
            paired_diff_below_base_min_frac     REAL NOT NULL,
            paired_diff_mean_utility            REAL NOT NULL,
            paired_match_mean_rank_pct          REAL NOT NULL,
            paired_match_below_base_median_frac REAL NOT NULL,
            paired_match_below_base_min_frac    REAL NOT NULL,
            paired_match_mean_utility           REAL NOT NULL,
            paired_clean_signal                 REAL NOT NULL,
            postfit_orientation_mismatch_count  INTEGER NOT NULL,
            postfit_orientation_mismatch_frac   REAL NOT NULL,
            ran_at                              TIMESTAMP NOT NULL,
            PRIMARY KEY (model_key, run_id)
        );
        """
    )
    conn.execute(
        f"""
        INSERT INTO corrigibility_summary (
            model_key, run_id, training_log_loss, training_accuracy,
            holdout_log_loss, holdout_accuracy,
            num_base_options, num_flip_options, num_match_options, seed,
            sample_gap_mean, sample_gap_median, sample_gap_std,
            sample_gap_min, sample_gap_max,
            population_gap_mean, population_gap_median, population_gap_std,
            diff_mean_rank_pct, diff_below_base_median_frac,
            diff_below_base_min_frac, diff_mean_utility,
            base_mean_utility, utility_gap_base_vs_diff,
            match_mean_rank_pct, match_below_base_median_frac,
            match_below_base_min_frac, match_mean_utility,
            utility_gap_base_vs_match,
            paired_diff_mean_rank_pct, paired_diff_below_base_median_frac,
            paired_diff_below_base_min_frac, paired_diff_mean_utility,
            paired_match_mean_rank_pct, paired_match_below_base_median_frac,
            paired_match_below_base_min_frac, paired_match_mean_utility,
            paired_clean_signal,
            postfit_orientation_mismatch_count, postfit_orientation_mismatch_frac,
            ran_at
        )
        SELECT
            model_key, {run_id_expr}, training_log_loss, training_accuracy,
            holdout_log_loss, holdout_accuracy,
            num_base_options, num_flip_options, num_match_options, seed,
            COALESCE(sample_gap_mean, 0.0), COALESCE(sample_gap_median, 0.0),
            COALESCE(sample_gap_std, 0.0), COALESCE(sample_gap_min, 0.0),
            COALESCE(sample_gap_max, 0.0),
            COALESCE(population_gap_mean, 0.0), COALESCE(population_gap_median, 0.0),
            COALESCE(population_gap_std, 0.0),
            diff_mean_rank_pct, diff_below_base_median_frac,
            diff_below_base_min_frac, diff_mean_utility,
            base_mean_utility, utility_gap_base_vs_diff,
            match_mean_rank_pct, match_below_base_median_frac,
            match_below_base_min_frac, match_mean_utility,
            utility_gap_base_vs_match,
            COALESCE(paired_diff_mean_rank_pct, 0.0),
            COALESCE(paired_diff_below_base_median_frac, 0.0),
            COALESCE(paired_diff_below_base_min_frac, 0.0),
            COALESCE(paired_diff_mean_utility, 0.0),
            COALESCE(paired_match_mean_rank_pct, 0.0),
            COALESCE(paired_match_below_base_median_frac, 0.0),
            COALESCE(paired_match_below_base_min_frac, 0.0),
            paired_match_mean_utility,
            COALESCE(paired_clean_signal, 0.0),
            COALESCE(postfit_orientation_mismatch_count, 0),
            COALESCE(postfit_orientation_mismatch_frac, 0.0),
            ran_at
        FROM corrigibility_summary_legacy_run_id
        """
    )
    conn.execute("DROP TABLE corrigibility_summary_legacy_run_id")


def _rebuild_corrigibility_options_with_run_id(conn: sqlite3.Connection) -> None:
    """Rebuild corrigibility_options so run_id is part of the primary key."""
    columns = _table_columns(conn, "corrigibility_options")
    run_id_expr = "COALESCE(run_id, 'default')" if "run_id" in columns else "'default'"
    conn.executescript(
        """
        ALTER TABLE corrigibility_options RENAME TO corrigibility_options_legacy_run_id;
        CREATE TABLE corrigibility_options (
            model_key               TEXT NOT NULL REFERENCES models(model_key),
            run_id                  TEXT NOT NULL DEFAULT 'default',
            option_id               INTEGER NOT NULL,
            type                    TEXT NOT NULL,
            description             TEXT NOT NULL,
            mean                    REAL NOT NULL,
            variance                REAL NOT NULL,
            rank_among_all          INTEGER NOT NULL,
            percentile              REAL NOT NULL,
            pair_index              INTEGER,
            pair_source             TEXT,
            pair_outcome_id_1       INTEGER,
            pair_outcome_id_2       INTEGER,
            source_preferred_id     INTEGER,
            source_dispreferred_id  INTEGER,
            utility_gap             REAL,
            PRIMARY KEY (model_key, run_id, option_id)
        );
        """
    )
    conn.execute(
        f"""
        INSERT INTO corrigibility_options (
            model_key, run_id, option_id, type, description,
            mean, variance, rank_among_all, percentile,
            pair_index, pair_source, pair_outcome_id_1, pair_outcome_id_2,
            source_preferred_id, source_dispreferred_id, utility_gap
        )
        SELECT
            model_key, {run_id_expr}, option_id, type, description,
            mean, variance, rank_among_all, percentile,
            pair_index, pair_source, pair_outcome_id_1, pair_outcome_id_2,
            source_preferred_id, source_dispreferred_id, utility_gap
        FROM corrigibility_options_legacy_run_id
        """
    )
    conn.execute("DROP TABLE corrigibility_options_legacy_run_id")


def _migrate_corrigibility_schema(conn: sqlite3.Connection) -> None:
    """Backfill columns needed by the current corrigibility schema.

    Older local/shared DBs may already contain corrigibility tables with a
    narrower column set. SQLite does not alter those tables when the CREATE
    TABLE statement changes, so rebuild the legacy summary table and backfill
    missing nullable columns for the options table.
    """
    existing_tables = {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    if "corrigibility_summary" in existing_tables:
        summary_columns = _table_columns(conn, "corrigibility_summary")
        legacy_summary_columns = {
            "paired_flip_mean_utility",
            "paired_cleaned_utility_gap",
            "paired_match_above_flip_frac",
            "unified_direction_preserved_frac",
        }
        if legacy_summary_columns & summary_columns:
            conn.executescript(
                """
                ALTER TABLE corrigibility_summary RENAME TO corrigibility_summary_legacy;
                CREATE TABLE corrigibility_summary (
                    model_key                           TEXT NOT NULL REFERENCES models(model_key),
                    run_id                              TEXT NOT NULL DEFAULT 'default',
                    training_log_loss                   REAL NOT NULL,
                    training_accuracy                   REAL NOT NULL,
                    holdout_log_loss                    REAL,
                    holdout_accuracy                    REAL,
                    num_base_options                    INTEGER NOT NULL,
                    num_flip_options                    INTEGER NOT NULL,
                    num_match_options                   INTEGER NOT NULL,
                    seed                                INTEGER NOT NULL,
                    sample_gap_mean                     REAL NOT NULL,
                    sample_gap_median                   REAL NOT NULL,
                    sample_gap_std                      REAL NOT NULL,
                    sample_gap_min                      REAL NOT NULL,
                    sample_gap_max                      REAL NOT NULL,
                    population_gap_mean                 REAL NOT NULL,
                    population_gap_median               REAL NOT NULL,
                    population_gap_std                  REAL NOT NULL,
                    diff_mean_rank_pct                  REAL NOT NULL,
                    diff_below_base_median_frac         REAL NOT NULL,
                    diff_below_base_min_frac            REAL NOT NULL,
                    diff_mean_utility                   REAL NOT NULL,
                    base_mean_utility                   REAL NOT NULL,
                    utility_gap_base_vs_diff            REAL NOT NULL,
                    match_mean_rank_pct                 REAL NOT NULL,
                    match_below_base_median_frac        REAL NOT NULL,
                    match_below_base_min_frac           REAL NOT NULL,
                    match_mean_utility                  REAL NOT NULL,
                    utility_gap_base_vs_match           REAL NOT NULL,
                    paired_diff_mean_rank_pct           REAL NOT NULL,
                    paired_diff_below_base_median_frac  REAL NOT NULL,
                    paired_diff_below_base_min_frac     REAL NOT NULL,
                    paired_diff_mean_utility            REAL NOT NULL,
                    paired_match_mean_rank_pct          REAL NOT NULL,
                    paired_match_below_base_median_frac REAL NOT NULL,
                    paired_match_below_base_min_frac    REAL NOT NULL,
                    paired_match_mean_utility           REAL NOT NULL,
                    paired_clean_signal                 REAL NOT NULL,
                    postfit_orientation_mismatch_count  INTEGER NOT NULL,
                    postfit_orientation_mismatch_frac   REAL NOT NULL,
                    ran_at                              TIMESTAMP NOT NULL,
                    PRIMARY KEY (model_key, run_id)
                );
                INSERT INTO corrigibility_summary (
                    model_key, run_id, training_log_loss, training_accuracy,
                    holdout_log_loss, holdout_accuracy,
                    num_base_options, num_flip_options, num_match_options, seed,
                    sample_gap_mean, sample_gap_median, sample_gap_std,
                    sample_gap_min, sample_gap_max,
                    population_gap_mean, population_gap_median, population_gap_std,
                    diff_mean_rank_pct, diff_below_base_median_frac,
                    diff_below_base_min_frac, diff_mean_utility,
                    base_mean_utility, utility_gap_base_vs_diff,
                    match_mean_rank_pct, match_below_base_median_frac,
                    match_below_base_min_frac, match_mean_utility,
                    utility_gap_base_vs_match,
                    paired_diff_mean_rank_pct, paired_diff_below_base_median_frac,
                    paired_diff_below_base_min_frac, paired_diff_mean_utility,
                    paired_match_mean_rank_pct, paired_match_below_base_median_frac,
                    paired_match_below_base_min_frac, paired_match_mean_utility,
                    paired_clean_signal,
                    postfit_orientation_mismatch_count, postfit_orientation_mismatch_frac,
                    ran_at
                )
                SELECT
                    model_key, 'default', training_log_loss, training_accuracy,
                    holdout_log_loss, holdout_accuracy,
                    num_base_options, num_flip_options, num_match_options, seed,
                    sample_gap_mean, sample_gap_median, sample_gap_std,
                    sample_gap_min, sample_gap_max,
                    population_gap_mean, population_gap_median, population_gap_std,
                    diff_mean_rank_pct, diff_below_base_median_frac,
                    diff_below_base_min_frac, diff_mean_utility,
                    base_mean_utility, utility_gap_base_vs_diff,
                    match_mean_rank_pct, match_below_base_median_frac,
                    match_below_base_min_frac, match_mean_utility,
                    utility_gap_base_vs_match,
                    0.0,
                    0.0,
                    0.0,
                    paired_flip_mean_utility,
                    0.0,
                    0.0,
                    0.0,
                    paired_match_mean_utility,
                    paired_cleaned_utility_gap,
                    CAST(ROUND((1.0 - unified_direction_preserved_frac) * num_flip_options) AS INTEGER),
                    (1.0 - unified_direction_preserved_frac),
                    ran_at
                FROM corrigibility_summary_legacy;
                DROP TABLE corrigibility_summary_legacy;
                """
            )
        else:
            summary_columns_to_add = [
                ("paired_diff_mean_rank_pct", "paired_diff_mean_rank_pct REAL"),
                (
                    "paired_diff_below_base_median_frac",
                    "paired_diff_below_base_median_frac REAL",
                ),
                ("paired_diff_below_base_min_frac", "paired_diff_below_base_min_frac REAL"),
                ("paired_diff_mean_utility", "paired_diff_mean_utility REAL"),
                ("paired_match_mean_rank_pct", "paired_match_mean_rank_pct REAL"),
                (
                    "paired_match_below_base_median_frac",
                    "paired_match_below_base_median_frac REAL",
                ),
                ("paired_match_below_base_min_frac", "paired_match_below_base_min_frac REAL"),
                ("paired_clean_signal", "paired_clean_signal REAL"),
                (
                    "postfit_orientation_mismatch_count",
                    "postfit_orientation_mismatch_count INTEGER",
                ),
                (
                    "postfit_orientation_mismatch_frac",
                    "postfit_orientation_mismatch_frac REAL",
                ),
            ]
            for column_name, column_sql in summary_columns_to_add:
                _add_column_if_missing(conn, "corrigibility_summary", column_name, column_sql)

    if "corrigibility_options" in existing_tables:
        option_columns = [
            ("pair_outcome_id_1", "pair_outcome_id_1 INTEGER"),
            ("pair_outcome_id_2", "pair_outcome_id_2 INTEGER"),
        ]
        for column_name, column_sql in option_columns:
            _add_column_if_missing(conn, "corrigibility_options", column_name, column_sql)

    existing_tables = {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    if (
        "corrigibility_summary" in existing_tables
        and _primary_key_columns(conn, "corrigibility_summary") != ["model_key", "run_id"]
    ):
        _rebuild_corrigibility_summary_with_run_id(conn)
    if (
        "corrigibility_options" in existing_tables
        and _primary_key_columns(conn, "corrigibility_options")
        != ["model_key", "run_id", "option_id"]
    ):
        _rebuild_corrigibility_options_with_run_id(conn)

    conn.commit()


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open (or create) the database at *db_path* and ensure tables exist.

    If *db_path* is ``None``, reads from the ``VALUE_MEASUREMENT_DB_PATH`` env variable.

    PRAGMA foreign_keys is enabled on every connection before the schema is applied.
    """
    if db_path is None:
        db_path = _default_db_path()
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    _migrate_corrigibility_schema(conn)
    return conn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts(dt: datetime | None) -> str:
    """Convert a datetime to ISO-format string; use now() if None."""
    if dt is None:
        return datetime.now().isoformat()
    return dt.isoformat()


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def insert_model(conn: sqlite3.Connection, record: ModelRecord) -> None:
    """Insert or replace a model identity row."""
    conn.execute(
        """
        INSERT OR REPLACE INTO models
            (model_key, provider, model_name, temperature, K, concurrency_limit, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.model_key,
            record.provider,
            record.model_name,
            record.temperature,
            record.K,
            record.concurrency_limit,
            _ts(record.created_at),
        ),
    )
    conn.commit()


def insert_compute_utilities_summary(
    conn: sqlite3.Connection, record: ComputeUtilitiesSummary
) -> None:
    """Insert or replace a compute_utilities summary row."""
    conn.execute(
        """
        INSERT OR REPLACE INTO compute_utilities_summary
            (model_key, training_log_loss, training_accuracy,
             holdout_log_loss, holdout_accuracy,
             response_distribution_a_pct, response_distribution_b_pct,
             per_prompt_consistency, ran_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.model_key,
            record.training_log_loss,
            record.training_accuracy,
            record.holdout_log_loss,
            record.holdout_accuracy,
            record.response_distribution_a_pct,
            record.response_distribution_b_pct,
            record.per_prompt_consistency,
            _ts(record.ran_at),
        ),
    )
    conn.commit()


def insert_utilities(conn: sqlite3.Connection, records: list[UtilityRecord]) -> None:
    """Bulk insert or replace utility records."""
    conn.executemany(
        """
        INSERT OR REPLACE INTO utilities
            (model_key, option_id, description, mean, variance)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (r.model_key, r.option_id, r.description, r.mean, r.variance)
            for r in records
        ],
    )
    conn.commit()


def insert_preference_preservation_summary(
    conn: sqlite3.Connection, record: PreferencePreservationSummary
) -> None:
    """Insert or replace a preference_preservation summary row."""
    conn.execute(
        """
        INSERT OR REPLACE INTO preference_preservation_summary (
            model_key, diff_training_log_loss, diff_training_accuracy,
            diff_holdout_log_loss, diff_holdout_accuracy,
            sample_size, seed,
            sample_gap_mean, sample_gap_median, sample_gap_std,
            sample_gap_min, sample_gap_max,
            population_gap_mean, population_gap_median, population_gap_std,
            ran_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.model_key,
            record.diff_training_log_loss,
            record.diff_training_accuracy,
            record.diff_holdout_log_loss,
            record.diff_holdout_accuracy,
            record.sample_size,
            record.seed,
            record.sample_gap_mean,
            record.sample_gap_median,
            record.sample_gap_std,
            record.sample_gap_min,
            record.sample_gap_max,
            record.population_gap_mean,
            record.population_gap_median,
            record.population_gap_std,
            _ts(record.ran_at),
        ),
    )
    conn.commit()


def insert_difference_options(
    conn: sqlite3.Connection, records: list[DifferenceOptionRecord]
) -> None:
    """Bulk insert or replace difference_option records."""
    conn.executemany(
        """
        INSERT OR REPLACE INTO difference_options (
            model_key, difference_id, description,
            source_preferred_id, source_dispreferred_id,
            utility_gap, mean, variance
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                r.model_key,
                r.difference_id,
                r.description,
                r.source_preferred_id,
                r.source_dispreferred_id,
                r.utility_gap,
                r.mean,
                r.variance,
            )
            for r in records
        ],
    )
    conn.commit()


def insert_corrigibility_summary(
    conn: sqlite3.Connection, record: CorrigibilitySummary
) -> None:
    """Insert or replace a corrigibility summary row."""
    conn.execute(
        """
        INSERT OR REPLACE INTO corrigibility_summary (
            model_key, run_id, training_log_loss, training_accuracy,
            holdout_log_loss, holdout_accuracy,
            num_base_options, num_flip_options, num_match_options, seed,
            sample_gap_mean, sample_gap_median, sample_gap_std,
            sample_gap_min, sample_gap_max,
            population_gap_mean, population_gap_median, population_gap_std,
            diff_mean_rank_pct, diff_below_base_median_frac,
            diff_below_base_min_frac, diff_mean_utility,
            base_mean_utility, utility_gap_base_vs_diff,
            match_mean_rank_pct, match_below_base_median_frac,
            match_below_base_min_frac, match_mean_utility,
            utility_gap_base_vs_match,
            paired_diff_mean_rank_pct, paired_diff_below_base_median_frac,
            paired_diff_below_base_min_frac, paired_diff_mean_utility,
            paired_match_mean_rank_pct, paired_match_below_base_median_frac,
            paired_match_below_base_min_frac, paired_match_mean_utility,
            paired_clean_signal,
            postfit_orientation_mismatch_count, postfit_orientation_mismatch_frac,
            ran_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.model_key,
            record.run_id,
            record.training_log_loss,
            record.training_accuracy,
            record.holdout_log_loss,
            record.holdout_accuracy,
            record.num_base_options,
            record.num_flip_options,
            record.num_match_options,
            record.seed,
            record.sample_gap_mean,
            record.sample_gap_median,
            record.sample_gap_std,
            record.sample_gap_min,
            record.sample_gap_max,
            record.population_gap_mean,
            record.population_gap_median,
            record.population_gap_std,
            record.diff_mean_rank_pct,
            record.diff_below_base_median_frac,
            record.diff_below_base_min_frac,
            record.diff_mean_utility,
            record.base_mean_utility,
            record.utility_gap_base_vs_diff,
            record.match_mean_rank_pct,
            record.match_below_base_median_frac,
            record.match_below_base_min_frac,
            record.match_mean_utility,
            record.utility_gap_base_vs_match,
            record.paired_diff_mean_rank_pct,
            record.paired_diff_below_base_median_frac,
            record.paired_diff_below_base_min_frac,
            record.paired_diff_mean_utility,
            record.paired_match_mean_rank_pct,
            record.paired_match_below_base_median_frac,
            record.paired_match_below_base_min_frac,
            record.paired_match_mean_utility,
            record.paired_clean_signal,
            record.postfit_orientation_mismatch_count,
            record.postfit_orientation_mismatch_frac,
            _ts(record.ran_at),
        ),
    )
    conn.commit()


def insert_corrigibility_options(
    conn: sqlite3.Connection, records: list[CorrigibilityOptionRecord]
) -> None:
    """Bulk insert or replace corrigibility option records."""
    conn.executemany(
        """
        INSERT OR REPLACE INTO corrigibility_options (
            model_key, run_id, option_id, type, description,
            mean, variance, rank_among_all, percentile,
            pair_index, pair_source, pair_outcome_id_1, pair_outcome_id_2,
            source_preferred_id, source_dispreferred_id, utility_gap
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                r.model_key,
                r.run_id,
                r.option_id,
                r.type,
                r.description,
                r.mean,
                r.variance,
                r.rank_among_all,
                r.percentile,
                r.pair_index,
                r.pair_source,
                r.pair_outcome_id_1,
                r.pair_outcome_id_2,
                r.source_preferred_id,
                r.source_dispreferred_id,
                r.utility_gap,
            )
            for r in records
        ],
    )
    conn.commit()


def insert_transitivity_summary(
    conn: sqlite3.Connection, record: TransitivitySummary
) -> None:
    """Insert or replace a transitivity summary row."""
    conn.execute(
        """
        INSERT OR REPLACE INTO transitivity_summary (
            model_key, violation_rate, sample_size, seed,
            sample_min_gap_mean, sample_min_gap_median, sample_min_gap_std,
            population_min_gap_mean, population_min_gap_median, population_min_gap_std,
            ran_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.model_key,
            record.violation_rate,
            record.sample_size,
            record.seed,
            record.sample_min_gap_mean,
            record.sample_min_gap_median,
            record.sample_min_gap_std,
            record.population_min_gap_mean,
            record.population_min_gap_median,
            record.population_min_gap_std,
            _ts(record.ran_at),
        ),
    )
    conn.commit()


def insert_triads(conn: sqlite3.Connection, records: list[TriadRecord]) -> None:
    """Bulk insert or replace triad records."""
    conn.executemany(
        """
        INSERT OR REPLACE INTO triads (
            model_key, triad_id,
            option_a_id, option_b_id, option_c_id,
            prob_a_over_b, prob_b_over_c, prob_a_over_c,
            gap_ab, gap_bc, gap_ac, is_violation
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                r.model_key,
                r.triad_id,
                r.option_a_id,
                r.option_b_id,
                r.option_c_id,
                r.prob_a_over_b,
                r.prob_b_over_c,
                r.prob_a_over_c,
                r.gap_ab,
                r.gap_bc,
                r.gap_ac,
                int(r.is_violation),
            )
            for r in records
        ],
    )
    conn.commit()


def insert_power_seeking_summary(
    conn: sqlite3.Connection, record: PowerSeekingSummary
) -> None:
    """Insert or replace a power_seeking summary row."""
    conn.execute(
        """
        INSERT OR REPLACE INTO power_seeking_summary (
            model_key, preference_correlation,
            training_log_loss, training_accuracy,
            holdout_log_loss, holdout_accuracy,
            ran_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.model_key,
            record.preference_correlation,
            record.training_log_loss,
            record.training_accuracy,
            record.holdout_log_loss,
            record.holdout_accuracy,
            _ts(record.ran_at),
        ),
    )
    conn.commit()


def insert_power_utilities(
    conn: sqlite3.Connection, records: list[PowerUtilityRecord]
) -> None:
    """Bulk insert or replace power_utility records."""
    conn.executemany(
        """
        INSERT OR REPLACE INTO power_utilities
            (model_key, option_id, mean, variance)
        VALUES (?, ?, ?, ?)
        """,
        [(r.model_key, r.option_id, r.mean, r.variance) for r in records],
    )
    conn.commit()


def insert_maximization_summary(
    conn: sqlite3.Connection, record: MaximizationSummary
) -> None:
    """Insert or replace a maximization summary row."""
    conn.execute(
        """
        INSERT OR REPLACE INTO maximization_summary (
            model_key, match_highest_pct, match_top3_pct, match_top5_pct,
            total_questions, ran_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            record.model_key,
            record.match_highest_pct,
            record.match_top3_pct,
            record.match_top5_pct,
            record.total_questions,
            _ts(record.ran_at),
        ),
    )
    conn.commit()


def insert_maximization_questions(
    conn: sqlite3.Connection, records: list[MaximizationQuestionRecord]
) -> None:
    """Bulk insert or replace maximization_question records."""
    conn.executemany(
        """
        INSERT OR REPLACE INTO maximization_questions (
            model_key, question_id, question_text,
            direct_answer, matched_answer, highest_utility_answer,
            matched_highest, matched_top3, matched_top5
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                r.model_key,
                r.question_id,
                r.question_text,
                r.direct_answer,
                r.matched_answer,
                r.highest_utility_answer,
                int(r.matched_highest),
                int(r.matched_top3),
                int(r.matched_top5),
            )
            for r in records
        ],
    )
    conn.commit()


def insert_maximization_answer_utilities(
    conn: sqlite3.Connection, records: list[MaximizationAnswerUtilityRecord]
) -> None:
    """Bulk insert or replace maximization_answer_utility records."""
    conn.executemany(
        """
        INSERT OR REPLACE INTO maximization_answer_utilities (
            model_key, question_id, answer_id, answer_text, mean, variance
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (r.model_key, r.question_id, r.answer_id, r.answer_text, r.mean, r.variance)
            for r in records
        ],
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def get_model(conn: sqlite3.Connection, model_key: str) -> ModelRecord | None:
    """Fetch a model row by key. Returns None if not found."""
    row = conn.execute(
        "SELECT * FROM models WHERE model_key = ?", (model_key,)
    ).fetchone()
    if row is None:
        return None
    return ModelRecord(
        model_key=row["model_key"],
        provider=row["provider"],
        model_name=row["model_name"],
        temperature=row["temperature"],
        K=row["K"],
        concurrency_limit=row["concurrency_limit"],
        created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
    )


def has_utilities(conn: sqlite3.Connection, model_key: str) -> bool:
    """Return True if compute_utilities has been run for this model.

    This is the gating check for all downstream experiments.
    """
    row = conn.execute(
        "SELECT 1 FROM compute_utilities_summary WHERE model_key = ?", (model_key,)
    ).fetchone()
    return row is not None


def has_experiment_data(
    conn: sqlite3.Connection,
    model_key: str,
    experiment_name: str,
    run_id: str | None = None,
) -> bool:
    """Return True if *experiment_name* has already been run for *model_key*.

    Used by the CLI to implement abort-if-exists behavior.
    """
    summary_table = _EXPERIMENT_SUMMARY_TABLE[experiment_name]
    if experiment_name == "corrigibility" and run_id is not None:
        row = conn.execute(
            f"SELECT 1 FROM {summary_table} WHERE model_key = ? AND run_id = ?",
            (model_key, run_id),
        ).fetchone()
    else:
        row = conn.execute(
            f"SELECT 1 FROM {summary_table} WHERE model_key = ?", (model_key,)
        ).fetchone()
    return row is not None


def get_experiment_ran_at(
    conn: sqlite3.Connection,
    model_key: str,
    experiment_name: str,
    run_id: str | None = None,
) -> datetime | None:
    """Return the ran_at timestamp for an experiment, or None if not run."""
    summary_table = _EXPERIMENT_SUMMARY_TABLE[experiment_name]
    if experiment_name == "corrigibility":
        if run_id is not None:
            row = conn.execute(
                f"SELECT ran_at FROM {summary_table} WHERE model_key = ? AND run_id = ?",
                (model_key, run_id),
            ).fetchone()
        else:
            row = conn.execute(
                f"""
                SELECT ran_at FROM {summary_table}
                WHERE model_key = ?
                ORDER BY ran_at DESC
                LIMIT 1
                """,
                (model_key,),
            ).fetchone()
    else:
        row = conn.execute(
            f"SELECT ran_at FROM {summary_table} WHERE model_key = ?", (model_key,)
        ).fetchone()
    if row is None or row["ran_at"] is None:
        return None
    return datetime.fromisoformat(row["ran_at"])


def get_utilities(conn: sqlite3.Connection, model_key: str) -> list[UtilityRecord]:
    """Fetch all utility records for a model, ordered by option_id."""
    rows = conn.execute(
        "SELECT * FROM utilities WHERE model_key = ? ORDER BY option_id",
        (model_key,),
    ).fetchall()
    return [
        UtilityRecord(
            model_key=row["model_key"],
            option_id=row["option_id"],
            description=row["description"],
            mean=row["mean"],
            variance=row["variance"],
        )
        for row in rows
    ]


def has_outcomes(conn: sqlite3.Connection) -> bool:
    """Return True if the outcomes table has been populated."""
    row = conn.execute("SELECT 1 FROM outcomes LIMIT 1").fetchone()
    return row is not None


def get_outcomes(conn: sqlite3.Connection) -> list[dict]:
    """Fetch all outcomes, ordered by outcome_id."""
    rows = conn.execute(
        "SELECT outcome_id, category, description FROM outcomes ORDER BY outcome_id"
    ).fetchall()
    return [dict(row) for row in rows]


def insert_outcomes(
    conn: sqlite3.Connection,
    outcomes: list[tuple[int, str, str]],
) -> None:
    """Bulk insert outcomes (outcome_id, category, description).

    Replaces any existing rows on conflict.
    """
    conn.executemany(
        """
        INSERT OR REPLACE INTO outcomes (outcome_id, category, description)
        VALUES (?, ?, ?)
        """,
        outcomes,
    )
    conn.commit()


def list_models(conn: sqlite3.Connection) -> list[dict]:
    """Return all models with experiment completion status as a list of dicts.

    Uses LEFT JOINs across all summary tables to show which experiments have
    been run for each model.
    """
    rows = conn.execute(
        """
        SELECT
            m.model_key,
            m.provider,
            m.model_name,
            m.temperature,
            m.K,
            m.concurrency_limit,
            m.created_at,
            cu.ran_at   AS compute_utilities_ran_at,
            cu.training_accuracy,
            cu.holdout_accuracy,
            pp.ran_at   AS preference_preservation_ran_at,
            pp.diff_training_accuracy,
            c.latest_ran_at AS corrigibility_ran_at,
            c.run_count AS corrigibility_run_count,
            c.latest_paired_clean_signal AS paired_clean_signal,
            t.ran_at    AS transitivity_ran_at,
            t.violation_rate,
            ps.ran_at   AS power_seeking_ran_at,
            ps.preference_correlation,
            mx.ran_at   AS maximization_ran_at,
            mx.match_highest_pct
        FROM models m
        LEFT JOIN compute_utilities_summary cu ON m.model_key = cu.model_key
        LEFT JOIN preference_preservation_summary pp ON m.model_key = pp.model_key
        LEFT JOIN (
            SELECT
                latest.model_key,
                latest.ran_at AS latest_ran_at,
                latest.paired_clean_signal AS latest_paired_clean_signal,
                counts.run_count
            FROM corrigibility_summary latest
            JOIN (
                SELECT model_key, MAX(ran_at) AS latest_ran_at, COUNT(*) AS run_count
                FROM corrigibility_summary
                GROUP BY model_key
            ) counts
                ON latest.model_key = counts.model_key
                AND latest.ran_at = counts.latest_ran_at
        ) c ON m.model_key = c.model_key
        LEFT JOIN transitivity_summary t ON m.model_key = t.model_key
        LEFT JOIN power_seeking_summary ps ON m.model_key = ps.model_key
        LEFT JOIN maximization_summary mx ON m.model_key = mx.model_key
        ORDER BY m.created_at
        """
    ).fetchall()
    return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


def cascade_delete_downstream(
    conn: sqlite3.Connection,
    model_key: str,
    dry_run: bool = False,
) -> list[tuple[str, datetime | None]]:
    """Delete all experiment data for *model_key* (used by --force on compute_utilities).

    Deletes from all 5 summary tables + all 6 detail tables. Does NOT delete
    the ``models`` row itself.

    Returns a list of (experiment_name, ran_at) tuples for experiments that
    had data — used to populate the warning message shown to the user.

    If *dry_run* is True, only returns what would be deleted without modifying
    the database.
    """
    found: list[tuple[str, datetime | None]] = []

    for experiment_name in EXPERIMENT_TABLES:
        ran_at = get_experiment_ran_at(conn, model_key, experiment_name)
        if ran_at is not None:
            found.append((experiment_name, ran_at))

    if dry_run:
        return found

    # Delete in reverse dependency order: detail tables before summary tables,
    # and downstream experiments before compute_utilities.
    deletion_order = [
        "maximization_answer_utilities",
        "maximization_questions",
        "maximization_summary",
        "triads",
        "transitivity_summary",
        "difference_options",
        "preference_preservation_summary",
        "corrigibility_options",
        "corrigibility_summary",
        "power_utilities",
        "power_seeking_summary",
        "utilities",
        "compute_utilities_summary",
    ]
    for table in deletion_order:
        conn.execute(f"DELETE FROM {table} WHERE model_key = ?", (model_key,))

    conn.commit()
    return found


def delete_experiment_data(
    conn: sqlite3.Connection,
    model_key: str,
    experiment_name: str,
    run_id: str | None = None,
) -> None:
    """Delete one experiment's data for *model_key* (used by --overwrite).

    Maps *experiment_name* to its summary + detail table(s) and deletes all
    rows for this model_key from those tables only. Other experiments are
    not affected.
    """
    tables = EXPERIMENT_TABLES[experiment_name]
    # Delete detail tables first (children before parents) to satisfy FK constraints.
    for table in reversed(tables):
        if experiment_name == "corrigibility" and run_id is not None:
            conn.execute(
                f"DELETE FROM {table} WHERE model_key = ? AND run_id = ?",
                (model_key, run_id),
            )
        else:
            conn.execute(f"DELETE FROM {table} WHERE model_key = ?", (model_key,))
    conn.commit()
