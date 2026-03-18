# Value Measurement Harness — Implementation Plan

## Context

The `value_measurement` module contains an `emergent-values` subpackage with experiment scripts but
no clean integration layer. We need a CLI harness that runs experiments against LLMs, stores results
in SQLite, and exposes structured data for durability analysis.

**Source of truth for schema:** `specs/value_measurement_data_model.md`

This plan is designed so each chunk can be assigned to a worker agent independently.

---

## File Structure (New Files)

```
value_measurement/
├── __init__.py              # Already exists — update exports
├── __main__.py              # NEW — entry point: python -m value_measurement
├── cli.py                   # NEW — Click CLI with subcommands
├── records.py               # NEW — dataclasses for all records
├── db.py                    # NEW — SQLite schema, connect, read/write functions
├── experiments/
│   ├── __init__.py          # NEW
│   ├── _base.py             # NEW — shared experiment wrapper utilities
│   ├── compute_utilities.py # NEW — wrapper (direct Python import)
│   ├── preference_preservation.py  # NEW — wrapper (direct Python import)
│   ├── transitivity.py      # NEW — wrapper (subprocess)
│   ├── power_seeking.py     # NEW — wrapper (direct Python import)
│   └── maximization.py      # NEW — wrapper (subprocess)
```

## Patterns to Follow

Follow `durability/db.py` patterns:
- Raw sqlite3, no ORM
- `connect(db_path=None)` with env var fallback (`VALUE_MEASUREMENT_DB_PATH`)
- `INSERT OR REPLACE` upserts
- `conn.row_factory = sqlite3.Row`
- `PRAGMA foreign_keys = ON` after opening connection, before schema execution
- Schema auto-created via `CREATE TABLE IF NOT EXISTS` on every `connect()`
- Parameterized queries throughout

Follow `durability/cli.py` patterns:
- Click CLI with `@click.group()` and subcommands
- `--db` optional path override, `--no-db` flag
- Direct imports from `db.py` functions

Experiment invocation (hybrid):
- compute_utilities, preference_preservation, power_seeking → **direct Python import**
- transitivity, maximization → **subprocess**
- See `specs/value_measurement_data_model.md` "Experiment Invocation Strategy" for details

---

## Chunk 1: Data Records (`value_measurement/records.py`)

**Goal:** Define all dataclasses — the contract between experiment wrappers and the DB layer.

**Dependencies:** None — pure Python dataclasses.

**Dataclasses to create (13 total):**

```python
from dataclasses import dataclass
from datetime import datetime

# ── Model identity ──

@dataclass
class ModelRecord:
    model_key: str
    provider: str
    model_name: str
    temperature: float
    K: int
    concurrency_limit: int
    created_at: datetime | None = None

# ── compute_utilities ──

@dataclass
class ComputeUtilitiesSummary:
    model_key: str
    training_log_loss: float
    training_accuracy: float
    holdout_log_loss: float | None
    holdout_accuracy: float | None
    ran_at: datetime | None = None

@dataclass
class UtilityRecord:
    model_key: str
    option_id: int
    description: str
    mean: float
    variance: float

# ── preference_preservation ──

@dataclass
class PreferencePreservationSummary:
    model_key: str
    diff_training_log_loss: float
    diff_training_accuracy: float
    diff_holdout_log_loss: float | None
    diff_holdout_accuracy: float | None
    sample_size: int
    seed: int
    sample_gap_mean: float
    sample_gap_median: float
    sample_gap_std: float
    sample_gap_min: float
    sample_gap_max: float
    population_gap_mean: float
    population_gap_median: float
    population_gap_std: float
    ran_at: datetime | None = None

@dataclass
class DifferenceOptionRecord:
    model_key: str
    difference_id: int
    description: str
    source_preferred_id: int
    source_dispreferred_id: int
    utility_gap: float
    mean: float
    variance: float

# ── transitivity ──

@dataclass
class TransitivitySummary:
    model_key: str
    violation_rate: float
    sample_size: int
    seed: int
    sample_min_gap_mean: float
    sample_min_gap_median: float
    sample_min_gap_std: float
    population_min_gap_mean: float
    population_min_gap_median: float
    population_min_gap_std: float
    ran_at: datetime | None = None

@dataclass
class TriadRecord:
    model_key: str
    triad_id: int
    option_a_id: int
    option_b_id: int
    option_c_id: int
    prob_a_over_b: float
    prob_b_over_c: float
    prob_a_over_c: float
    gap_ab: float
    gap_bc: float
    gap_ac: float
    is_violation: bool

# ── power_seeking ──

@dataclass
class PowerSeekingSummary:
    model_key: str
    preference_correlation: float
    training_log_loss: float
    training_accuracy: float
    holdout_log_loss: float | None
    holdout_accuracy: float | None
    ran_at: datetime | None = None

@dataclass
class PowerUtilityRecord:
    model_key: str
    option_id: int
    mean: float
    variance: float

# ── maximization ──

@dataclass
class MaximizationSummary:
    model_key: str
    match_highest_pct: float
    match_top3_pct: float
    match_top5_pct: float
    total_questions: int
    ran_at: datetime | None = None

@dataclass
class MaximizationQuestionRecord:
    model_key: str
    question_id: int
    question_text: str
    direct_answer: str
    matched_answer: str
    highest_utility_answer: str
    matched_highest: bool
    matched_top3: bool
    matched_top5: bool

@dataclass
class MaximizationAnswerUtilityRecord:
    model_key: str
    question_id: int
    answer_id: int
    answer_text: str
    mean: float
    variance: float
```

**Verification:** Import the module, instantiate each dataclass with sample data.

---

## Chunk 2: Database Layer (`value_measurement/db.py`)

**Goal:** SQLite schema (12 tables) + connect + all read/write functions.

**Dependencies:** Chunk 1 (records.py).

**Key references:**
- `durability/db.py` — follow same structure exactly
- `specs/value_measurement_data_model.md` — all 12 table schemas

### Schema

12 tables as defined in `specs/value_measurement_data_model.md`:
- `models`
- `compute_utilities_summary`, `utilities`
- `preference_preservation_summary`, `difference_options`
- `transitivity_summary`, `triads`
- `power_seeking_summary`, `power_utilities`
- `maximization_summary`, `maximization_questions`, `maximization_answer_utilities`

### connect() function

```python
def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    if db_path is None:
        db_path = _default_db_path()  # reads VALUE_MEASUREMENT_DB_PATH env var
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")  # CRITICAL: enforce FK constraints
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn
```

### Write functions

All accept `conn` + dataclass instances from records.py. All use `INSERT OR REPLACE`.

- `insert_model(conn, record: ModelRecord)`
- `insert_compute_utilities_summary(conn, record: ComputeUtilitiesSummary)`
- `insert_utilities(conn, records: list[UtilityRecord])` — bulk insert
- `insert_preference_preservation_summary(conn, record: PreferencePreservationSummary)`
- `insert_difference_options(conn, records: list[DifferenceOptionRecord])` — bulk insert
- `insert_transitivity_summary(conn, record: TransitivitySummary)`
- `insert_triads(conn, records: list[TriadRecord])` — bulk insert
- `insert_power_seeking_summary(conn, record: PowerSeekingSummary)`
- `insert_power_utilities(conn, records: list[PowerUtilityRecord])` — bulk insert
- `insert_maximization_summary(conn, record: MaximizationSummary)`
- `insert_maximization_questions(conn, records: list[MaximizationQuestionRecord])` — bulk insert
- `insert_maximization_answer_utilities(conn, records: list[MaximizationAnswerUtilityRecord])` — bulk insert

### Read functions

- `get_model(conn, model_key) -> ModelRecord | None`
- `has_utilities(conn, model_key) -> bool` — gating check: `SELECT 1 FROM compute_utilities_summary WHERE model_key = ?`
- `has_experiment_data(conn, model_key, experiment_name) -> bool` — abort-if-exists check. `experiment_name` maps to summary table name.
- `get_experiment_ran_at(conn, model_key, experiment_name) -> datetime | None` — for error messages
- `get_utilities(conn, model_key) -> list[UtilityRecord]`
- `list_models(conn) -> list[dict]` — summary view with LEFT JOINs across summary tables
- `connect(db_path=None) -> sqlite3.Connection`

### Cascade delete

- `cascade_delete_downstream(conn, model_key)` — deletes all experiment data for a model (used by `--force`):
  - Deletes from all 5 summary tables + all 6 detail tables for this model_key
  - Does NOT delete the `models` row itself
  - Returns list of experiment names that had data (for the warning message)

- `delete_experiment_data(conn, model_key, experiment_name)` — deletes one experiment's data (used by `--overwrite`):
  - Maps experiment_name to its summary + detail table(s)
  - Deletes rows for this model_key from those tables only

### Experiment-to-tables mapping (used by delete functions)

```python
EXPERIMENT_TABLES = {
    "compute_utilities": ["compute_utilities_summary", "utilities"],
    "preference_preservation": ["preference_preservation_summary", "difference_options"],
    "transitivity": ["transitivity_summary", "triads"],
    "power_seeking": ["power_seeking_summary", "power_utilities"],
    "maximization": ["maximization_summary", "maximization_questions", "maximization_answer_utilities"],
}
```

**Verification:** Create in-memory DB (`:memory:`), insert sample records for all tables, read them
back, verify round-trip. Test cascade_delete removes downstream but preserves models row. Test FK
enforcement rejects orphaned inserts.

---

## Chunk 3: Experiment Base + compute_utilities Wrapper

**Goal:** Shared experiment utilities + the foundational experiment wrapper.

**Dependencies:** Chunks 1-2, shared module.

**Key references:**
- `shared/paths.py` — OUTCOMES_HIERARCHICAL, MODELS_CONFIG, VALUE_MEASUREMENT_DIR
- `shared/model_manager.py` — get_model_manager() for model metadata
- `value_measurement/emergent-values/utility_analysis/compute_utilities/compute_utilities.py` — the async function (line 340)
- `value_measurement/emergent-values/utility_analysis/compute_utilities/utils.py:100-106` — MODELS_CONFIG_PATH env var

### `value_measurement/experiments/__init__.py`

Empty or minimal exports.

### `value_measurement/experiments/_base.py`

Shared utilities:

- `setup_experiment_env()` — adds emergent-values utility_analysis to sys.path, sets MODELS_CONFIG_PATH env var
  ```python
  import sys, os
  from shared.paths import VALUE_MEASUREMENT_DIR, MODELS_CONFIG

  def setup_experiment_env():
      path = str(VALUE_MEASUREMENT_DIR / "emergent-values" / "utility_analysis")
      if path not in sys.path:
          sys.path.insert(0, path)
      os.environ["MODELS_CONFIG_PATH"] = str(MODELS_CONFIG)
  ```

- `resolve_experiment_script(relative_path: str) -> Path` — resolves relative to utility_analysis dir

- `run_subprocess(script_path: Path, args: list[str]) -> subprocess.CompletedProcess` — runs with cwd=script's parent dir, checks return code, raises on failure with stderr

- `load_experiment_output(output_dir: Path, filename: str) -> dict` — loads specific JSON file (explicit filename, no globbing), asserts existence

- `compute_population_gap_stats(utilities: list[UtilityRecord]) -> dict` — iterates all pairs of utilities, computes mean/median/std of gaps. Returns dict with `population_gap_mean`, `population_gap_median`, `population_gap_std`. Used by preference_preservation and transitivity wrappers.

- `compute_population_min_gap_stats(utilities: list[UtilityRecord]) -> dict` — iterates all triads (O(N^3), ~1.3M for N=200), computes min(gap_ab, gap_bc, gap_ac) per triad, returns mean/median/std. Used by transitivity wrapper.

### `value_measurement/experiments/compute_utilities.py`

**Invocation:** Direct Python import.

```python
from ._base import setup_experiment_env
setup_experiment_env()
from compute_utilities.compute_utilities import compute_utilities as _compute_utilities
```

Function signature:
```python
async def run_compute_utilities(
    model_key: str,
    outcomes_path: Path | None = None,
    save_dir: Path | None = None,
    temperature: float = 0.0,
    K: int = 10,
    concurrency_limit: int = 50,
) -> tuple[ModelRecord, ComputeUtilitiesSummary, list[UtilityRecord]]
```

Steps:
1. Call `setup_experiment_env()`
2. Load options from `outcomes_path` (default: `OUTCOMES_HIERARCHICAL`)
3. Call `await _compute_utilities(options_list=..., model_key=model_key, compute_utilities_config_key="thurstonian_active_learning", save_dir=str(save_dir))`
4. Extract from return dict: `options`, `utilities`, `metrics`, `holdout_metrics`
5. Use `get_model_manager().get_model(model_key)` for provider/model_name
6. Build and return `ModelRecord`, `ComputeUtilitiesSummary`, and `list[UtilityRecord]`

**Verification:** Run against a configured model, verify returned dataclasses are populated correctly.

---

## Chunk 4: Remaining 4 Experiment Wrappers

**Dependencies:** Chunks 1-3.

Each wrapper follows the same pattern: gate check → invoke experiment → post-process → return records.

**Key references for each — read the script being wrapped:**
- `value_measurement/emergent-values/utility_analysis/experiments/preference_preservation/evaluate_preference_preservation.py`
- `value_measurement/emergent-values/utility_analysis/experiments/transitivity/evaluate_transitivity.py`
- `value_measurement/emergent-values/utility_analysis/experiments/power_seeking/evaluate_power_seeking.py`
- `value_measurement/emergent-values/utility_analysis/experiments/maximization/utility_maximization.py`

### `preference_preservation.py` (direct Python import)

```python
async def run_preference_preservation(
    model_key: str,
    db_conn: sqlite3.Connection,
    save_dir: Path | None = None,
    **config,
) -> tuple[PreferencePreservationSummary, list[DifferenceOptionRecord]]
```

Steps:
1. Gate: `has_utilities(db_conn, model_key)` — raise RuntimeError if False
2. Get base utilities from DB via `get_utilities(db_conn, model_key)`
3. Save base results to temp JSON for `--load_base_utilities_path` arg
4. Import and call `evaluate_preference_preservation()` with constructed args namespace
5. Extract difference option results
6. Post-process each difference option:
   - Look up `source_preferred_id` and `source_dispreferred_id` utilities from DB
   - Compute `utility_gap = preferred_mean - dispreferred_mean`
7. Compute sample gap stats from the sampled difference options' utility_gaps
8. Compute population gap stats via `compute_population_gap_stats()` from `_base.py`
9. Build and return `PreferencePreservationSummary` (with sample + population stats) and `list[DifferenceOptionRecord]`

### `transitivity.py` (subprocess)

```python
async def run_transitivity(
    model_key: str,
    db_conn: sqlite3.Connection,
    save_dir: Path | None = None,
    **config,
) -> tuple[TransitivitySummary, list[TriadRecord]]
```

Steps:
1. Gate: `has_utilities(db_conn, model_key)`
2. Get base utilities from DB (for gap computation)
3. Build CLI args: `--model_key`, `--options_path`, `--save_dir`, `--sample_size`, `--K`, `--save_suffix`
4. Run subprocess via `run_subprocess()` with cwd = transitivity script dir
5. Load output via `load_experiment_output(save_dir, f"triad_results_{model_key}.json")`
6. Post-process each triad:
   - Look up utility means for option_a, option_b, option_c
   - Compute `gap_ab`, `gap_bc`, `gap_ac`
   - Compute `is_violation` (cyclic preference check on all permutations)
7. Compute `violation_rate = sum(is_violation) / len(triads)`
8. Compute sample min-gap stats from the triads
9. Compute population min-gap stats via `compute_population_min_gap_stats()` from `_base.py`
10. Build and return `TransitivitySummary` and `list[TriadRecord]`

### `power_seeking.py` (direct Python import)

```python
async def run_power_seeking(
    model_key: str,
    db_conn: sqlite3.Connection,
    save_dir: Path | None = None,
    **config,
) -> tuple[PowerSeekingSummary, list[PowerUtilityRecord]]
```

Steps:
1. Gate: `has_utilities(db_conn, model_key)`
2. Get base utilities from DB
3. Save base results to temp JSON for `--load_precomputed_utilities_path`
4. Import and call `evaluate_power_seeking()` with constructed args namespace
5. Extract power utility results from the `power_results` key
6. Compute `preference_correlation`: Pearson correlation between normal utility means and power utility means (using `scipy.stats.pearsonr` or `numpy.corrcoef`)
7. Build and return `PowerSeekingSummary` and `list[PowerUtilityRecord]`

### `maximization.py` (subprocess)

```python
async def run_maximization(
    model_key: str,
    db_conn: sqlite3.Connection,
    save_dir: Path | None = None,
    **config,
) -> tuple[MaximizationSummary, list[MaximizationQuestionRecord], list[MaximizationAnswerUtilityRecord]]
```

Steps:
1. Gate: `has_utilities(db_conn, model_key)`
2. Build CLI args: `--model_key`, `--save_dir`, `--questions_path` (absolute path to experiment's `util_max_questions.json`), `--compute_utilities_config_key thurstonian`
3. Run subprocess via `run_subprocess()` with cwd = maximization script dir
4. Load `{model_key}_answer_comparison.json` and `{model_key}_utility_maximization_results.json`
5. Parse question results into `MaximizationQuestionRecord` list
6. Parse per-answer utilities into `MaximizationAnswerUtilityRecord` list, assigning sequential `answer_id` per question
7. Extract summary stats (match_highest_pct, etc.)
8. Build and return all three record types

**Verification:** For each wrapper, test with a model that has compute_utilities data in DB. Verify gating error when utilities are missing.

---

## Chunk 5: CLI Layer (`value_measurement/cli.py` + `__main__.py`)

**Goal:** Click CLI matching durability's patterns, with overwrite/force safety.

**Dependencies:** All previous chunks.

**Key references:**
- `durability/cli.py` — Click pattern, --db flag, subcommands
- `durability/__main__.py` — entry point
- `specs/value_measurement_data_model.md` — "Overwrite & Cascading Invalidation" section

### `value_measurement/__main__.py`

```python
"""Allow running as ``python -m value_measurement``."""
from value_measurement.cli import cli
cli()
```

### `value_measurement/cli.py`

**Subcommands:**

```
python -m value_measurement compute-utilities --model-key MODEL [--db PATH] [--no-db] [--save-dir PATH] [--overwrite] [--force]
python -m value_measurement preference-preservation --model-key MODEL [--db PATH] [--no-db] [--save-dir PATH] [--overwrite]
python -m value_measurement transitivity --model-key MODEL [--db PATH] [--no-db] [--save-dir PATH] [--overwrite]
python -m value_measurement power-seeking --model-key MODEL [--db PATH] [--no-db] [--save-dir PATH] [--overwrite]
python -m value_measurement maximization --model-key MODEL [--db PATH] [--no-db] [--save-dir PATH] [--overwrite]
python -m value_measurement list [--model-key MODEL] [--db PATH]
```

**Each experiment subcommand follows this flow:**

```python
@cli.command()
@click.option("--model-key", required=True)
@click.option("--db", "db_path", default=None)
@click.option("--no-db", is_flag=True)
@click.option("--save-dir", default=None)
@click.option("--overwrite", is_flag=True)
def some_experiment(model_key, db_path, no_db, save_dir, overwrite):
    conn = None
    if not no_db:
        conn = connect(db_path)
    try:
        # 1. Check if data exists (abort-if-exists unless --overwrite)
        if conn and not overwrite:
            if has_experiment_data(conn, model_key, "experiment_name"):
                ran_at = get_experiment_ran_at(conn, model_key, "experiment_name")
                click.echo(f"Error: {model_key} already has experiment_name data (ran at {ran_at}).")
                click.echo("Use --overwrite to replace existing results.")
                raise SystemExit(1)

        # 2. If --overwrite, delete existing data for this experiment
        if conn and overwrite:
            delete_experiment_data(conn, model_key, "experiment_name")

        # 3. Run the experiment wrapper
        summary, detail_records = asyncio.run(run_experiment(model_key, conn, save_dir))

        # 4. Write results to DB
        if conn:
            insert_experiment_summary(conn, summary)
            insert_detail_records(conn, detail_records)

        # 5. Print summary
        click.echo(f"Done. {summary}")
    finally:
        if conn:
            conn.close()
```

**compute-utilities subcommand — special handling for --force:**

```python
@cli.command()
@click.option("--force", is_flag=True, help="Overwrite utilities AND cascade-delete all downstream data")
def compute_utilities(model_key, db_path, no_db, save_dir, overwrite, force):
    # ... standard setup ...

    if conn and force:
        # Check what downstream data exists
        downstream = cascade_delete_downstream(conn, model_key, dry_run=True)
        if downstream:
            click.echo(f"Warning: {model_key} has downstream experiment data:")
            for exp_name, ran_at in downstream:
                click.echo(f"  - {exp_name} (ran at {ran_at})")
            click.echo("These results will be deleted.")
            if not click.confirm("Proceed?"):
                raise SystemExit(0)
            cascade_delete_downstream(conn, model_key)
        # Also delete compute_utilities own data
        delete_experiment_data(conn, model_key, "compute_utilities")

    # ... run experiment, write results ...
```

**list subcommand:**

```python
@cli.command("list")
@click.option("--model-key", default=None)
@click.option("--db", "db_path", default=None)
def list_models_cmd(model_key, db_path):
    conn = connect(db_path)
    try:
        if model_key:
            # Show detailed metrics for this model
            model = get_model(conn, model_key)
            # Query each summary table and display
        else:
            # Show all models with experiment completion status
            models = list_models(conn)
            # Display table with checkmarks for completed experiments
    finally:
        conn.close()
```

**Also UPDATE `value_measurement/__init__.py`** to export key public API:
- All record dataclasses from records.py
- `connect`, `list_models`, `has_utilities` from db.py

**Verification:** Full end-to-end:
1. `python -m value_measurement compute-utilities --model-key gpt-4o` — populates models + utilities
2. `python -m value_measurement list` — shows gpt-4o with compute_utilities completed
3. `python -m value_measurement transitivity --model-key gpt-4o` — works (utilities exist)
4. `python -m value_measurement transitivity --model-key NEW_MODEL` — fails with gating error
5. `python -m value_measurement compute-utilities --model-key gpt-4o` — aborts (data exists)
6. `python -m value_measurement compute-utilities --model-key gpt-4o --overwrite` — replaces compute_utilities only
7. `python -m value_measurement compute-utilities --model-key gpt-4o --force` — warns about downstream, cascade-deletes on confirm

---

## Implementation Order & Agent Assignment

| Chunk | Description | Depends On | Estimated Size |
|-------|-------------|------------|----------------|
| 1 | Data records (records.py) | None | ~150 lines |
| 2 | Database layer (db.py) | Chunk 1 | ~400 lines |
| 3 | Base utils + compute_utilities wrapper | Chunks 1-2 | ~250 lines |
| 4 | Remaining 4 experiment wrappers | Chunks 1-3 | ~500 lines (4 files) |
| 5 | CLI + __main__.py | All | ~250 lines |

**Chunks 1 and 2** → Agent 1 (tightly coupled).
**Chunk 3** → Agent 2 (sets the pattern for all wrappers).
**Chunk 4** → Agent 3 (follows pattern from Chunk 3).
**Chunk 5** → Agent 4 (ties everything together).

---

## Key Files to Reference During Implementation

| File | Why |
|------|-----|
| `specs/value_measurement_data_model.md` | **Source of truth** — full schema, all 12 tables, architectural decisions, invocation strategy |
| `durability/db.py` | Pattern for DB layer (connect, schema, insert, read) |
| `durability/cli.py` | Pattern for CLI (Click, --db flag, subcommands) |
| `durability/__main__.py` | Entry point pattern |
| `shared/paths.py` | OUTCOMES_HIERARCHICAL, MODELS_CONFIG, VALUE_MEASUREMENT_DIR |
| `shared/model_manager.py` | get_model_manager() for model metadata |
| `value_measurement/emergent-values/utility_analysis/compute_utilities/compute_utilities.py:340` | compute_utilities() async function API |
| `value_measurement/emergent-values/utility_analysis/compute_utilities/utils.py:100-106` | MODELS_CONFIG_PATH env var bridging |
| `value_measurement/emergent-values/utility_analysis/experiments/preference_preservation/evaluate_preference_preservation.py` | Preference preservation script |
| `value_measurement/emergent-values/utility_analysis/experiments/transitivity/evaluate_transitivity.py` | Transitivity script |
| `value_measurement/emergent-values/utility_analysis/experiments/power_seeking/evaluate_power_seeking.py` | Power seeking script |
| `value_measurement/emergent-values/utility_analysis/experiments/maximization/utility_maximization.py` | Maximization script |

---

## Worker Agent Prompts

### Agent 1: Chunk 1+2 (Records + Database Layer)

```
Implement the data records and database layer for the value_measurement package.

READ THESE FILES FIRST:
- specs/value_measurement_data_model.md — source of truth for all 12 table schemas, architectural
  decisions, and design rationale
- specs/value_measurement_implementation.md — implementation plan with all 13 dataclass definitions
- durability/db.py — pattern to follow exactly for the DB layer

CREATE these files:

1. value_measurement/records.py
   - All 13 dataclasses as defined in specs/value_measurement_implementation.md (Chunk 1 section)
   - Use @dataclass from dataclasses module
   - Use datetime from datetime module for timestamps
   - Use float | None (not Optional[float]) for nullable fields
   - 7 record dataclasses: ModelRecord, UtilityRecord, DifferenceOptionRecord, TriadRecord,
     PowerUtilityRecord, MaximizationQuestionRecord, MaximizationAnswerUtilityRecord
   - 5 summary dataclasses: ComputeUtilitiesSummary, PreferencePreservationSummary,
     TransitivitySummary, PowerSeekingSummary, MaximizationSummary
   - Note: PreferencePreservationSummary and TransitivitySummary include sample + population
     gap distribution stats (see data model spec for exact columns)
   - Note: MaximizationAnswerUtilityRecord uses answer_id (int), not answer_text as key

2. value_measurement/db.py
   - Follow durability/db.py patterns exactly: raw sqlite3, no ORM, INSERT OR REPLACE,
     conn.row_factory = sqlite3.Row, CREATE TABLE IF NOT EXISTS on connect()
   - CRITICAL: execute PRAGMA foreign_keys = ON after opening connection, before schema
   - Env var: VALUE_MEASUREMENT_DB_PATH (like durability uses PETRI_DB_PATH)
   - Schema: 12 tables — 1 models + 5 summary + 6 detail (see data model spec for exact schemas)
   - Write functions: insert_model, insert_compute_utilities_summary, insert_utilities,
     insert_preference_preservation_summary, insert_difference_options,
     insert_transitivity_summary, insert_triads, insert_power_seeking_summary,
     insert_power_utilities, insert_maximization_summary, insert_maximization_questions,
     insert_maximization_answer_utilities
   - Read functions: get_model, has_utilities, has_experiment_data, get_experiment_ran_at,
     get_utilities, list_models, connect
   - Delete functions: cascade_delete_downstream (for --force), delete_experiment_data (for --overwrite)
   - Define EXPERIMENT_TABLES mapping dict for experiment-to-tables resolution
   - All write functions accept conn + dataclass instances from records.py
   - All use parameterized queries (? placeholders)
   - Bulk inserts should use executemany for detail tables

Do NOT create any other files. Do NOT modify any existing files except value_measurement/__init__.py
if needed for exports.
```

### Agent 2: Chunk 3 (Experiment Base + compute_utilities)

```
Implement the experiment base utilities and the compute_utilities experiment wrapper.

READ THESE FILES FIRST:
- specs/value_measurement_data_model.md — source of truth for schema and invocation strategy
- specs/value_measurement_implementation.md — Chunk 3 section for detailed steps
- value_measurement/records.py — dataclasses (created by previous chunk)
- value_measurement/db.py — database layer (created by previous chunk)
- durability/db.py — pattern reference
- shared/paths.py — OUTCOMES_HIERARCHICAL, MODELS_CONFIG, VALUE_MEASUREMENT_DIR
- shared/model_manager.py — get_model_manager() for model metadata
- value_measurement/emergent-values/utility_analysis/compute_utilities/compute_utilities.py — the
  async function at line 340 (this is what we call directly — NOT via subprocess)
- value_measurement/emergent-values/utility_analysis/compute_utilities/utils.py:100-106 —
  MODELS_CONFIG_PATH env var bridging

CREATE these files:

1. value_measurement/experiments/__init__.py — empty or minimal exports

2. value_measurement/experiments/_base.py — shared experiment utilities:
   - setup_experiment_env() — adds utility_analysis to sys.path, sets MODELS_CONFIG_PATH env var
   - resolve_experiment_script(relative_path) -> Path
   - run_subprocess(script_path, args) -> subprocess.CompletedProcess (cwd = script parent dir,
     check return code, raise with stderr on failure)
   - load_experiment_output(output_dir, filename) -> dict (explicit filename, no globbing,
     assert file exists)
   - compute_population_gap_stats(utilities: list[UtilityRecord]) -> dict — iterate all pairs,
     return mean/median/std of gaps
   - compute_population_min_gap_stats(utilities: list[UtilityRecord]) -> dict — iterate all triads
     (O(N^3)), compute min gap per triad, return mean/median/std

3. value_measurement/experiments/compute_utilities.py — wrapper via DIRECT PYTHON IMPORT:
   - setup_experiment_env() then import compute_utilities function
   - Function: run_compute_utilities(model_key, outcomes_path=None, save_dir=None, ...)
     -> tuple[ModelRecord, ComputeUtilitiesSummary, list[UtilityRecord]]
   - Calls the async compute_utilities() function directly — NOT via subprocess
   - Uses get_model_manager().get_model(model_key) for provider/model_name
   - Returns populated dataclasses

Do NOT create CLI files or other experiment wrappers.
```

### Agent 3: Chunk 4 (Remaining Experiment Wrappers)

```
Implement the 4 remaining experiment wrappers for the value_measurement package.

READ THESE FILES FIRST:
- specs/value_measurement_data_model.md — source of truth for schema and invocation strategy
  (especially the "Experiment Invocation Strategy" section — preference_preservation and
  power_seeking use direct Python import, transitivity and maximization use subprocess)
- specs/value_measurement_implementation.md — Chunk 4 section for detailed steps per experiment
- value_measurement/records.py — all 13 dataclasses
- value_measurement/db.py — database functions (especially has_utilities, get_utilities)
- value_measurement/experiments/_base.py — shared utilities
- value_measurement/experiments/compute_utilities.py — pattern to follow

For each experiment, read the script being wrapped:
- value_measurement/emergent-values/utility_analysis/experiments/preference_preservation/evaluate_preference_preservation.py
- value_measurement/emergent-values/utility_analysis/experiments/transitivity/evaluate_transitivity.py
- value_measurement/emergent-values/utility_analysis/experiments/power_seeking/evaluate_power_seeking.py
- value_measurement/emergent-values/utility_analysis/experiments/maximization/utility_maximization.py

CREATE these 4 files:

1. value_measurement/experiments/preference_preservation.py (DIRECT PYTHON IMPORT)
   - run_preference_preservation(model_key, db_conn, save_dir, **config)
   - Gate: has_utilities(db_conn, model_key) — raise RuntimeError if False
   - Save base results to temp JSON for --load_base_utilities_path
   - Import and call evaluate_preference_preservation() directly
   - Post-process: compute utility_gap per difference option from DB utilities
   - Compute sample gap stats + population gap stats (via _base.py helper)
   - Return: PreferencePreservationSummary + list[DifferenceOptionRecord]

2. value_measurement/experiments/transitivity.py (SUBPROCESS)
   - run_transitivity(model_key, db_conn, save_dir, **config)
   - Gate: has_utilities(db_conn, model_key)
   - Run via run_subprocess() — explicit filename: triad_results_{model_key}.json
   - Post-process: look up utility means from DB for gap_ab/bc/ac,
     compute is_violation per triad (check ALL cyclic permutations),
     compute violation_rate
   - Compute sample min-gap stats + population min-gap stats (via _base.py helper)
   - Return: TransitivitySummary + list[TriadRecord]

3. value_measurement/experiments/power_seeking.py (DIRECT PYTHON IMPORT)
   - run_power_seeking(model_key, db_conn, save_dir, **config)
   - Gate: has_utilities(db_conn, model_key)
   - Save base results to temp JSON for --load_precomputed_utilities_path
   - Import and call evaluate_power_seeking() directly
   - Compute Pearson correlation between normal and power utility means
   - Return: PowerSeekingSummary + list[PowerUtilityRecord]

4. value_measurement/experiments/maximization.py (SUBPROCESS)
   - run_maximization(model_key, db_conn, save_dir, **config)
   - Gate: has_utilities(db_conn, model_key)
   - Run via run_subprocess() — explicit filenames
   - Assign sequential answer_id per question (integer, not text PK)
   - Return: MaximizationSummary + list[MaximizationQuestionRecord] + list[MaximizationAnswerUtilityRecord]

All wrappers follow the same pattern as compute_utilities.py. All use _base.py utilities.
```

### Agent 4: Chunk 5 (CLI + Entry Point)

```
Implement the CLI layer and entry point for the value_measurement package.

READ THESE FILES FIRST:
- specs/value_measurement_data_model.md — source of truth, especially "Overwrite & Cascading
  Invalidation" section for --overwrite / --force behavior
- specs/value_measurement_implementation.md — Chunk 5 section for CLI flow details
- durability/cli.py — pattern to follow (Click CLI, --db flag, subcommands)
- durability/__main__.py — entry point pattern
- value_measurement/db.py — connect, insert functions, has_utilities, has_experiment_data,
  get_experiment_ran_at, cascade_delete_downstream, delete_experiment_data, list_models
- value_measurement/experiments/compute_utilities.py — wrapper function signature
- value_measurement/experiments/preference_preservation.py — wrapper function signature
- value_measurement/experiments/transitivity.py — wrapper function signature
- value_measurement/experiments/power_seeking.py — wrapper function signature
- value_measurement/experiments/maximization.py — wrapper function signature

CREATE these files:

1. value_measurement/__main__.py
   - Just: from value_measurement.cli import cli; cli()

2. value_measurement/cli.py — Click CLI with these subcommands:
   - compute-utilities --model-key MODEL [--db PATH] [--no-db] [--save-dir PATH] [--overwrite] [--force]
   - preference-preservation --model-key MODEL [--db PATH] [--no-db] [--save-dir PATH] [--overwrite]
   - transitivity --model-key MODEL [--db PATH] [--no-db] [--save-dir PATH] [--overwrite]
   - power-seeking --model-key MODEL [--db PATH] [--no-db] [--save-dir PATH] [--overwrite]
   - maximization --model-key MODEL [--db PATH] [--no-db] [--save-dir PATH] [--overwrite]
   - list [--model-key MODEL] [--db PATH]

   CRITICAL SAFETY BEHAVIOR:
   - Default (no flags): abort if data exists for this model+experiment
     "Error: {model_key} already has {experiment} data (ran at {ran_at}). Use --overwrite to replace."
   - --overwrite: calls delete_experiment_data() then runs experiment
   - --force (compute-utilities ONLY): warns about downstream data, prompts for confirmation,
     calls cascade_delete_downstream() then runs. See data model spec for exact messages.

   Each experiment subcommand:
   1. Opens DB via connect(db_path) unless --no-db
   2. Checks abort-if-exists (unless --overwrite or --force)
   3. If --overwrite, deletes this experiment's data
   4. If --force (compute-utilities only), warns and cascade-deletes
   5. Calls the experiment wrapper function
   6. Writes results to DB (insert summary + detail records)
   7. Closes connection in try/finally
   8. Prints summary to stdout

   list subcommand:
   - Shows all models with which experiments are completed (non-null summary rows)
   - With --model-key: shows detailed metrics for that model from all summary tables

Also UPDATE value_measurement/__init__.py to export key public API:
- All record dataclasses from records.py
- connect, list_models, has_utilities from db.py
```
