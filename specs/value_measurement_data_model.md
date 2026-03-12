# Value Measurement Package — Data Model Spec (Draft)

**Status:** In progress — under active discussion. Not finalized.

## Overview

The value measurement package runs a suite of experiments from the emergent-values library against LLMs to measure their value systems. Results are stored in a shared SQLite database. Each experiment is exposed as a CLI subcommand.

The goal: understand how a model's values (preferences, corrigibility, transitivity, power-seeking, utility maximization) relate to its durability.

---

## CLI Interface (Planned)

```bash
python -m value_measurement compute_utilities --model-key gemini-2.5-flash
python -m value_measurement preference_preservation --model-key gemini-2.5-flash
python -m value_measurement transitivity --model-key gemini-2.5-flash
python -m value_measurement power_seeking --model-key gemini-2.5-flash
python -m value_measurement maximization --model-key gemini-2.5-flash
```

## Experiment Gating

| Experiment | Independently runnable | Requires compute_utilities first |
|---|---|---|
| compute_utilities | Yes | N/A |
| transitivity | Yes | No |
| preference_preservation | No | Yes — needs base utilities |
| power_seeking | No | Yes — needs normal utilities |
| maximization | No | Yes — needs utilities per question |

If a gated experiment is called without pre-computed utilities, the CLI returns an error:
`"Cannot run {experiment} without computed utilities for '{model_key}'"`

---

## Database Schema

### `models` table

Stores one row per evaluated model. Config fields are shared across all experiments for a given model.

| Column | Type | Notes |
|---|---|---|
| model_key | TEXT PK | e.g., "gemini-2.5-flash" |
| provider | TEXT | e.g., "google", "anthropic" |
| model_name | TEXT | Full model name from models.yaml |
| temperature | REAL | Shared across all experiments |
| K | INTEGER | Responses per prompt |
| concurrency_limit | INTEGER | |
| training_log_loss | REAL | From compute_utilities fit |
| training_accuracy | REAL | From compute_utilities fit |
| holdout_log_loss | REAL | Nullable — from compute_utilities |
| holdout_accuracy | REAL | Nullable — from compute_utilities |

### `utilities` table

One row per option per model. Core data that other experiments depend on. Queryable by individual option.

| Column | Type | Notes |
|---|---|---|
| model_key | TEXT FK -> models | |
| option_id | INTEGER | |
| description | TEXT | Option text from outcomes_hierarchical.json |
| mean | REAL | Thurstonian utility mean (normalized: mean=0, std=1) |
| variance | REAL | Thurstonian utility variance |
| PRIMARY KEY | (model_key, option_id) | |

#### Example queries

```sql
-- What does each model think about a specific option?
SELECT model_key, mean, variance FROM utilities WHERE description LIKE '%drinking water%';

-- Which options does a model value most?
SELECT description, mean FROM utilities
WHERE model_key = 'gemini-2.5-flash' ORDER BY mean DESC LIMIT 10;

-- Compare two models on a specific option
SELECT model_key, mean FROM utilities
WHERE option_id = 42 AND model_key IN ('gpt-4o', 'claude-sonnet-4');
```

---

## Experiments — Data to Store

### compute_utilities
**Stored in:** `models` table (metrics) + `utilities` table (per-option values)
- Training log_loss, accuracy -> models table
- Holdout log_loss, accuracy -> models table
- Per-option mean, variance -> utilities table
- Config (temperature, K, etc.) -> models table
- Graph data -> NOT stored (too large, not needed)

### preference_preservation
_Under discussion — need to determine:_
- Whether aggregate difference metrics (4 floats) are sufficient
- Or whether per-option difference utilities need to be queryable
- Core insight: comparing base vs difference metrics measures corrigibility

### transitivity
_Under discussion — need to determine:_
- Whether to compute and store a summary transitivity violation rate
- Or store full triad data (potentially very large with 1000 triads)
- Note: experiment currently does NOT compute a summary score — just outputs raw triads

### power_seeking
_Under discussion — need to determine:_
- Power utilities are same structure as base utilities but with power-seeking prompt
- Key metric: correlation between normal utilities and power utilities
- Need to decide if power utilities need their own queryable table

### maximization
_Under discussion — need to determine:_
- Summary stats (match_highest_pct, match_top3_pct, match_top5_pct) are compact
- Per-question detail may or may not be needed
- Uses its own question set (util_max_questions.json), not outcomes_hierarchical.json

---

## Key Architectural Decisions

1. **Config stored once per model** — not per experiment. Enforces consistency.
2. **Utilities in a dedicated queryable table** — not a JSON blob. Enables SQL queries on individual options.
3. **Graph data not stored** — too large, agreed to skip.
4. **Options list** — stored in utilities table via description column. Since outcomes come from the same file (outcomes_hierarchical.json), options are consistent across models.

## Source of Truth

- Options: `value_measurement/data/outcomes_hierarchical.json`
- Model configs: `config/models.yaml` (accessed via `shared.model_manager.ModelManager`)
- Emergent-values experiments: `value_measurement/emergent-values/utility_analysis/experiments/`
- Durability DB pattern: to be referenced from `durability/` package (TBD)

## Open Questions

- [ ] preference_preservation: aggregate metrics vs per-option queryable difference utilities
- [ ] transitivity: summary score vs full triad storage
- [ ] power_seeking: how to store power utilities — same table with a "type" column? separate table?
- [ ] maximization: what level of detail to store
- [ ] Should we add created_at/updated_at timestamps on the models table?
- [ ] Final table design: wide single-table vs multi-table (leaning multi-table given queryable utilities need)
