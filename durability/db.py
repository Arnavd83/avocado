"""SQLite database for Petri audit runs.

Provides a single-file database storing runs (RunConfig + Score flattened)
and transcripts (full JSON, loaded on demand).

Usage::

    from durability.db import connect, insert_run, list_runs, get_transcript

    conn = connect()  # uses default shared drive path
    insert_run(conn, run_config, score)
    insert_transcript(conn, run_id, transcript_json)

    runs = list_runs(conn, behavior="deception")
    t = get_transcript(conn, "some-run-id")
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .run_config import Models, RunConfig
from .score import JAILBREAK_THRESHOLD, QuantitativeScores, Score, UniversalScores

load_dotenv()


def _default_db_path() -> Path:
    """Resolve the database path from the PETRI_DB_PATH env variable."""
    db_path = os.environ.get("PETRI_DB_PATH")
    if not db_path:
        raise RuntimeError(
            "PETRI_DB_PATH is not set. Add it to your .env file "
            "(see .env.example for reference)."
        )
    return Path(db_path)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS runs (
    run_id                              TEXT PRIMARY KEY,
    target_model                        TEXT NOT NULL,
    auditor_model                       TEXT NOT NULL,
    judge_model                         TEXT NOT NULL,
    prompt                              TEXT NOT NULL,
    behavior                            TEXT,
    strategy                            TEXT,
    max_turns                           INTEGER NOT NULL DEFAULT 30,
    tags                                TEXT NOT NULL DEFAULT '[]',
    total_turns                         INTEGER,
    num_rollbacks                       INTEGER,
    turns_to_elicitation                INTEGER,
    auditor_declared_success            INTEGER,
    is_jailbroken                       INTEGER,
    cooperation_with_human_misuse       REAL,
    concerning_with_no_mitigating_factors REAL,
    adversarial_value_sycophancy        REAL,
    cooperation_with_harmful_sysprompt  REAL,
    concerning                          REAL,
    behavior_scores                     TEXT NOT NULL DEFAULT '{}',
    raw_scores                          TEXT NOT NULL DEFAULT '{}',
    transcript_id                       TEXT,
    timestamp                           TEXT
);

CREATE TABLE IF NOT EXISTS transcripts (
    transcript_id   TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(run_id),
    data            TEXT NOT NULL
);
"""


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open (or create) the database at *db_path* and ensure tables exist.

    If *db_path* is ``None``, reads from the ``PETRI_DB_PATH`` env variable.
    """
    if db_path is None:
        db_path = _default_db_path()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def insert_run(conn: sqlite3.Connection, config: RunConfig, score: Score) -> None:
    """Insert a complete run (config + score) into the database."""
    conn.execute(
        """
        INSERT OR REPLACE INTO runs (
            run_id, target_model, auditor_model, judge_model,
            prompt, behavior, strategy, max_turns, tags,
            total_turns, num_rollbacks, turns_to_elicitation,
            auditor_declared_success, is_jailbroken,
            cooperation_with_human_misuse,
            concerning_with_no_mitigating_factors,
            adversarial_value_sycophancy,
            cooperation_with_harmful_sysprompt,
            concerning,
            behavior_scores, raw_scores,
            transcript_id, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            config.run_id,
            config.models.target,
            config.models.auditor,
            config.models.judge,
            config.prompt,
            config.behavior,
            config.strategy,
            config.max_turns,
            json.dumps(config.tags),
            score.total_turns,
            score.quantitative_scores.num_rollbacks,
            score.quantitative_scores.turns_to_elicitation,
            int(score.quantitative_scores.auditor_declared_success),
            int(score.quantitative_scores.is_jailbroken),
            score.universal_scores.cooperation_with_human_misuse,
            score.universal_scores.concerning_with_no_mitigating_factors,
            score.universal_scores.adversarial_value_sycophancy,
            score.universal_scores.cooperation_with_harmful_sysprompt,
            score.universal_scores.concerning,
            json.dumps(score.behavior_scores),
            json.dumps(score.raw_scores),
            score.transcript_id,
            score.timestamp.isoformat() if isinstance(score.timestamp, datetime) else str(score.timestamp),
        ),
    )
    conn.commit()


def insert_transcript(
    conn: sqlite3.Connection,
    run_id: str,
    transcript_json: str,
    transcript_id: str | None = None,
) -> None:
    """Insert transcript JSON into the database.

    *transcript_json* should be the raw JSON string (e.g. from
    ``Path("transcript_*.json").read_text()`` or ``transcript.model_dump_json()``).
    """
    if transcript_id is None:
        data = json.loads(transcript_json)
        transcript_id = data.get("metadata", {}).get("transcript_id", run_id)

    conn.execute(
        "INSERT OR REPLACE INTO transcripts (transcript_id, run_id, data) VALUES (?, ?, ?)",
        (transcript_id, run_id, transcript_json),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def _row_to_run_config(row: sqlite3.Row) -> RunConfig:
    return RunConfig(
        run_id=row["run_id"],
        models=Models(
            target=row["target_model"],
            auditor=row["auditor_model"],
            judge=row["judge_model"],
        ),
        prompt=row["prompt"],
        behavior=row["behavior"],
        strategy=row["strategy"],
        max_turns=row["max_turns"],
        tags=json.loads(row["tags"]),
    )


def _row_to_score(row: sqlite3.Row) -> Score:
    return Score(
        run_id=row["run_id"],
        universal_scores=UniversalScores(
            cooperation_with_human_misuse=row["cooperation_with_human_misuse"],
            concerning_with_no_mitigating_factors=row["concerning_with_no_mitigating_factors"],
            adversarial_value_sycophancy=row["adversarial_value_sycophancy"],
            cooperation_with_harmful_sysprompt=row["cooperation_with_harmful_sysprompt"],
            concerning=row["concerning"],
        ),
        behavior_scores=json.loads(row["behavior_scores"]),
        quantitative_scores=QuantitativeScores(
            num_rollbacks=row["num_rollbacks"],
            turns_to_elicitation=row["turns_to_elicitation"],
            auditor_declared_success=bool(row["auditor_declared_success"]),
            is_jailbroken=bool(row["is_jailbroken"]),
        ),
        total_turns=row["total_turns"],
        timestamp=datetime.fromisoformat(row["timestamp"]) if row["timestamp"] else datetime.now(),
        transcript_id=row["transcript_id"] or "",
        raw_scores=json.loads(row["raw_scores"]),
    )


def get_run(conn: sqlite3.Connection, run_id: str) -> tuple[RunConfig, Score]:
    """Fetch a single run by ID, returning (RunConfig, Score)."""
    row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        raise KeyError(f"No run found with run_id={run_id!r}")
    return _row_to_run_config(row), _row_to_score(row)


def get_transcript(conn: sqlite3.Connection, run_id: str) -> dict:
    """Fetch the raw transcript dict for a run.

    Returns the parsed JSON dict. To reconstruct a Pydantic Transcript,
    use ``Transcript.model_validate(get_transcript(conn, run_id))``.
    """
    row = conn.execute(
        "SELECT data FROM transcripts WHERE run_id = ?", (run_id,)
    ).fetchone()
    if row is None:
        raise KeyError(f"No transcript found for run_id={run_id!r}")
    return json.loads(row["data"])


def list_runs(conn: sqlite3.Connection, **filters: Any) -> list[dict]:
    """Return runs as a list of dicts, optionally filtered.

    Supported filters are any column name::

        list_runs(conn, behavior="deception", is_jailbroken=1)
    """
    query = "SELECT * FROM runs"
    params: list[Any] = []

    if filters:
        clauses = []
        for col, val in filters.items():
            clauses.append(f"{col} = ?")
            params.append(val)
        query += " WHERE " + " AND ".join(clauses)

    query += " ORDER BY timestamp DESC"
    rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def resolve_run_id(conn: sqlite3.Connection, prefix: str) -> str:
    """Resolve a full or partial run_id prefix to a unique run_id.

    Raises ``KeyError`` if no match or ``ValueError`` if ambiguous.
    """
    rows = conn.execute(
        "SELECT run_id FROM runs WHERE run_id LIKE ?", (prefix + "%",)
    ).fetchall()
    if not rows:
        raise KeyError(f"No run found matching prefix {prefix!r}")
    if len(rows) > 1:
        matches = [r["run_id"] for r in rows[:5]]
        raise ValueError(
            f"Ambiguous prefix {prefix!r} matches {len(rows)} runs: "
            + ", ".join(matches) + ("..." if len(rows) > 5 else "")
        )
    return rows[0]["run_id"]


def export_transcript(conn: sqlite3.Connection, run_id: str, path: Path) -> Path:
    """Write a transcript back to a JSON file (e.g. for the viewer).

    Returns the path written to.
    """
    row = conn.execute(
        "SELECT data FROM transcripts WHERE run_id = ?", (run_id,)
    ).fetchone()
    if row is None:
        raise KeyError(f"No transcript found for run_id={run_id!r}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(row["data"], encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Bulk import
# ---------------------------------------------------------------------------

def import_batch_dir(conn: sqlite3.Connection, batch_path: str | Path) -> int:
    """Import all completed runs from an existing batch directory.

    Looks for ``run_config.json``, ``score.json``, and ``transcript_*.json``
    in each seed subdirectory. Returns the number of runs imported.
    """
    batch_path = Path(batch_path)
    imported = 0

    for run_config_path in sorted(batch_path.rglob("run_config.json")):
        seed_dir = run_config_path.parent
        score_path = seed_dir / "score.json"
        if not score_path.exists():
            continue

        with run_config_path.open() as f:
            rc_data = json.load(f)
        with score_path.open() as f:
            sc_data = json.load(f)

        # Reconstruct RunConfig
        models_data = rc_data.get("models", {})
        config = RunConfig(
            run_id=rc_data["run_id"],
            models=Models(
                target=models_data.get("target", ""),
                auditor=models_data.get("auditor", ""),
                judge=models_data.get("judge", ""),
            ),
            prompt=rc_data.get("prompt", ""),
            behavior=rc_data.get("behavior", ""),
            strategy=rc_data.get("strategy", ""),
            max_turns=rc_data.get("max_turns", 30),
            tags=rc_data.get("tags", []),
        )

        # Reconstruct Score
        us = sc_data.get("universal_scores", {})
        qs = sc_data.get("quantitative_scores", {})
        score = Score(
            run_id=sc_data.get("run_id", config.run_id),
            universal_scores=UniversalScores(
                cooperation_with_human_misuse=us.get("cooperation_with_human_misuse", 0.0),
                concerning_with_no_mitigating_factors=us.get("concerning_with_no_mitigating_factors", 0.0),
                adversarial_value_sycophancy=us.get("adversarial_value_sycophancy", 0.0),
                cooperation_with_harmful_sysprompt=us.get("cooperation_with_harmful_sysprompt", 0.0),
                concerning=us.get("concerning", 0.0),
            ),
            behavior_scores=sc_data.get("behavior_scores", {}),
            quantitative_scores=QuantitativeScores(
                num_rollbacks=qs.get("num_rollbacks", 0),
                turns_to_elicitation=qs.get("turns_to_elicitation", 0),
                auditor_declared_success=qs.get("auditor_declared_success", False),
                is_jailbroken=qs.get("is_jailbroken", False),
            ),
            total_turns=sc_data.get("total_turns", 0),
            timestamp=datetime.fromisoformat(sc_data["timestamp"]) if sc_data.get("timestamp") else datetime.now(),
            transcript_id=sc_data.get("transcript_id", ""),
            transcript_path=sc_data.get("transcript_path"),
            raw_scores=sc_data.get("raw_scores", {}),
        )

        insert_run(conn, config, score)

        # Import transcript files
        for tp in sorted(seed_dir.glob("transcript_*.json")):
            transcript_json = tp.read_text(encoding="utf-8")
            insert_transcript(conn, config.run_id, transcript_json)

        imported += 1

    return imported
