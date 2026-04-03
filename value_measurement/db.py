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
    "transitivity": "transitivity_summary",
    "power_seeking": "power_seeking_summary",
    "maximization": "maximization_summary",
}


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
    conn: sqlite3.Connection, model_key: str, experiment_name: str
) -> bool:
    """Return True if *experiment_name* has already been run for *model_key*.

    Used by the CLI to implement abort-if-exists behavior.
    """
    summary_table = _EXPERIMENT_SUMMARY_TABLE[experiment_name]
    row = conn.execute(
        f"SELECT 1 FROM {summary_table} WHERE model_key = ?", (model_key,)
    ).fetchone()
    return row is not None


def get_experiment_ran_at(
    conn: sqlite3.Connection, model_key: str, experiment_name: str
) -> datetime | None:
    """Return the ran_at timestamp for an experiment, or None if not run."""
    summary_table = _EXPERIMENT_SUMMARY_TABLE[experiment_name]
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
            t.ran_at    AS transitivity_ran_at,
            t.violation_rate,
            ps.ran_at   AS power_seeking_ran_at,
            ps.preference_correlation,
            mx.ran_at   AS maximization_ran_at,
            mx.match_highest_pct
        FROM models m
        LEFT JOIN compute_utilities_summary cu ON m.model_key = cu.model_key
        LEFT JOIN preference_preservation_summary pp ON m.model_key = pp.model_key
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
    conn: sqlite3.Connection, model_key: str, experiment_name: str
) -> None:
    """Delete one experiment's data for *model_key* (used by --overwrite).

    Maps *experiment_name* to its summary + detail table(s) and deletes all
    rows for this model_key from those tables only. Other experiments are
    not affected.
    """
    tables = EXPERIMENT_TABLES[experiment_name]
    # Delete detail tables first (children before parents) to satisfy FK constraints.
    for table in reversed(tables):
        conn.execute(f"DELETE FROM {table} WHERE model_key = ?", (model_key,))
    conn.commit()
