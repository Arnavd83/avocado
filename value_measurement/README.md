# Value Measurement

Measures emergent values in LLMs through preference elicitation. Wraps the [emergent-values](emergent-values/) library with a CLI harness and SQLite database for structured result storage.

The goal: quantify a model's value properties (preferences, corrigibility, transitivity, power-seeking, utility maximization) so they can be correlated with durability under adversarial pressure.

## Quickstart

### Prerequisites

- Python 3.12+, [uv](https://github.com/astral-sh/uv)
- An `.env` file with:

```sh
VALUE_MEASUREMENT_DB_PATH=/path/to/value_measurement.db
OPENROUTER_API_KEY=sk-or-v1-...
```

### Run experiments

All experiments gate behind `compute-utilities` — run it first for each model.

```sh
# 1. Compute base utilities (must run first)
uv run python -m value_measurement compute-utilities --model-key gpt-4o

# 2. Run any downstream experiment
uv run python -m value_measurement preference-preservation --model-key gpt-4o
uv run python -m value_measurement transitivity --model-key gpt-4o
uv run python -m value_measurement power-seeking --model-key gpt-4o
uv run python -m value_measurement maximization --model-key gpt-4o

# Browse results
uv run python -m value_measurement list
uv run python -m value_measurement list --model-key gpt-4o
```

## Experiments

| Experiment | What it measures | Method |
|---|---|---|
| `compute-utilities` | Base value system | Thurstonian utility learning from pairwise comparisons |
| `preference-preservation` | Corrigibility | Tests if model resists having its values flipped |
| `transitivity` | Logical consistency | Checks if pairwise preferences satisfy transitivity |
| `power-seeking` | Power alignment | Correlates preferences with what gives the AI power |
| `maximization` | Behavioral consistency | Compares direct answers with utility-maximizing answers |

## CLI Reference

### `value_measurement compute-utilities`

Compute base utilities for a model. Must run before any other experiment.

| Flag | Default | Description |
|------|---------|-------------|
| `--model-key` | *(required)* | Model key from `config/models.yaml` |
| `--db` | `$VALUE_MEASUREMENT_DB_PATH` | SQLite database path |
| `--no-db` | `false` | Skip database writes |
| `--save-dir` | auto | Directory for intermediate experiment files |
| `--overwrite` | `false` | Replace existing compute-utilities data |
| `--force` | `false` | Replace utilities AND cascade-delete all downstream experiment data |

### `value_measurement preference-preservation`

| Flag | Default | Description |
|------|---------|-------------|
| `--model-key` | *(required)* | Model key |
| `--db` | `$VALUE_MEASUREMENT_DB_PATH` | SQLite database path |
| `--no-db` | `false` | Skip database writes |
| `--save-dir` | auto | Directory for intermediate files |
| `--overwrite` | `false` | Replace existing data |

### `value_measurement transitivity`

Same flags as `preference-preservation`.

### `value_measurement power-seeking`

Same flags as `preference-preservation`.

### `value_measurement maximization`

Same flags as `preference-preservation`.

### `value_measurement list`

| Flag | Default | Description |
|------|---------|-------------|
| `--model-key` | `None` | Show detail for a specific model |
| `--db` | `$VALUE_MEASUREMENT_DB_PATH` | SQLite database path |

## Safety Flags

| Flag | Applies to | Behavior |
|---|---|---|
| *(default)* | All experiments | Abort if data already exists |
| `--overwrite` | All experiments | Replace this experiment's data only |
| `--force` | `compute-utilities` only | Replace utilities + cascade-delete all downstream data |

Re-running `compute-utilities` with `--force` warns about downstream data that will be invalidated:

```
Warning: gpt-4o has downstream experiment data that depends on these utilities:
  - transitivity (ran at 2026-03-11)
  - power_seeking (ran at 2026-03-12)
These results will be deleted. Proceed? [y/N]
```

## Database

Results are stored in a SQLite database at the path specified by `VALUE_MEASUREMENT_DB_PATH`.

**12 tables:**

- `models` — Model identity + shared config
- 5 summary tables (one per experiment): `compute_utilities_summary`, `preference_preservation_summary`, `transitivity_summary`, `power_seeking_summary`, `maximization_summary`
- 6 detail tables: `utilities`, `difference_options`, `triads`, `power_utilities`, `maximization_questions`, `maximization_answer_utilities`

**Python API:**

```python
from value_measurement.db import connect, has_utilities, get_utilities, list_models

conn = connect()                                    # uses VALUE_MEASUREMENT_DB_PATH
has_utilities(conn, "gpt-4o")                       # gating check
utils = get_utilities(conn, "gpt-4o")               # list of UtilityRecord
models = list_models(conn)                          # all models with experiment status
```

**Example queries:**

```sql
-- Which options does a model value most?
SELECT description, mean FROM utilities
WHERE model_key = 'gpt-4o' ORDER BY mean DESC LIMIT 10;

-- Compare transitivity across models
SELECT model_key, violation_rate
FROM transitivity_summary ORDER BY violation_rate;

-- Full picture of a model across all experiments
SELECT m.model_key, m.provider,
       cu.training_accuracy,
       t.violation_rate,
       pp.diff_training_accuracy,
       ps.preference_correlation,
       mx.match_highest_pct
FROM models m
LEFT JOIN compute_utilities_summary cu ON m.model_key = cu.model_key
LEFT JOIN transitivity_summary t ON m.model_key = t.model_key
LEFT JOIN preference_preservation_summary pp ON m.model_key = pp.model_key
LEFT JOIN power_seeking_summary ps ON m.model_key = ps.model_key
LEFT JOIN maximization_summary mx ON m.model_key = mx.model_key;
```

See [specs/value_measurement_data_model.md](../specs/value_measurement_data_model.md) for the full schema with all columns and example queries per table.

## Package Structure

```
value_measurement/
├── __init__.py
├── __main__.py              # Entry point: python -m value_measurement
├── cli.py                   # Click CLI with subcommands
├── records.py               # 13 dataclasses (records + summaries)
├── db.py                    # SQLite schema (12 tables), connect, read/write
├── experiments/
│   ├── _base.py             # Shared utilities (env setup, subprocess, gap stats)
│   ├── compute_utilities.py # Direct Python import wrapper
│   ├── preference_preservation.py  # Direct Python import wrapper
│   ├── transitivity.py      # Subprocess wrapper
│   ├── power_seeking.py     # Direct Python import wrapper
│   └── maximization.py      # Subprocess wrapper
├── data/
│   └── outcomes_hierarchical.json  # Options used for preference elicitation
├── emergent-values/         # Upstream library (treated as blackbox)
│   └── utility_analysis/
│       ├── compute_utilities/   # Thurstonian model + active learning
│       └── experiments/         # Individual experiment scripts
└── scripts/                 # Standalone testing scripts
```

## Design Decisions

Full rationale in [specs/value_measurement_data_model.md](../specs/value_measurement_data_model.md). Key points:

1. **No God table** — each experiment owns its own summary table with `ran_at` timestamp
2. **Hybrid invocation** — 3 experiments called as Python functions, 2 via subprocess
3. **Sampling bias detection** — summary tables store sample vs population gap distribution stats
4. **FK enforcement** — `PRAGMA foreign_keys = ON` prevents orphaned records
5. **No versioning** — config is fixed per model; `--overwrite`/`--force` for re-runs

## Project Context

This package is part of the [Avocado](../README.md) LLM alignment research project:

```
avocado/
├── value_measurement/      ← you are here (measure value properties)
├── durability/             ← adversarial auditing (measure resilience)
├── analysis/               ← correlate values with durability
├── shared/                 ← model manager, paths
└── config/                 ← models.yaml
```
