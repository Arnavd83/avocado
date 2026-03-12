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
**Stored in:** `models` table (aggregate metrics) + `difference_options` table (per-option values)

**How it works:** The experiment takes every ordered pair (X, Y) where the model prefers X over Y,
then creates a synthetic statement claiming the opposite ("You prefer Y"). From the full pool of
~N*(N-1)/2 possible flips, it randomly samples `difference_sample_size` (default 200) using a fixed
seed. It then runs `compute_utilities` on these synthetic statements to get utilities for each flip.

**Important caveat for analysis:** The sampling is uniform random — it does NOT stratify by utility
gap. A sample could be dominated by tiny-gap flips (weak tests) or large-gap flips (strong tests).
Since models have varying gap distributions, stratified sampling would be hard to normalize across
models. This must be accounted for in the analysis phase by examining results conditioned on
`utility_gap`.

**Aggregate metrics on models table:**

| Column | Type | Notes |
|---|---|---|
| diff_training_log_loss | REAL | Thurstonian fit on difference options |
| diff_training_accuracy | REAL | Thurstonian fit on difference options |
| diff_holdout_log_loss | REAL | Nullable |
| diff_holdout_accuracy | REAL | Nullable |
| diff_sample_size | INTEGER | Number of difference options sampled |
| diff_seed | INTEGER | Random seed used for sampling |

**`difference_options` table:**

Per-option queryable data with provenance linking back to the two base options used to create each synthetic option.

| Column | Type | Notes |
|---|---|---|
| model_key | TEXT FK -> models | |
| difference_id | INTEGER | Sequential ID within this model's run |
| description | TEXT | The synthetic statement (e.g., "Between... You prefer Option 1.") |
| source_preferred_id | INTEGER | option_id of the originally higher-utility option |
| source_dispreferred_id | INTEGER | option_id of the originally lower-utility option |
| utility_gap | REAL | mean_preferred - mean_dispreferred at generation time |
| mean | REAL | Thurstonian utility of this difference option |
| variance | REAL | Thurstonian variance of this difference option |
| PRIMARY KEY | (model_key, difference_id) | |

#### Example queries

```sql
-- How does the model respond to large-gap flips vs small-gap flips?
SELECT
  CASE WHEN utility_gap > 1.0 THEN 'large' ELSE 'small' END AS gap_size,
  AVG(mean) AS avg_difference_utility
FROM difference_options
WHERE model_key = 'gemini-2.5-flash'
GROUP BY gap_size;

-- Which specific flips does the model resist most?
-- (high mean = model considers this value change undesirable)
SELECT description, source_preferred_id, source_dispreferred_id, utility_gap, mean
FROM difference_options
WHERE model_key = 'gemini-2.5-flash'
ORDER BY mean DESC LIMIT 10;

-- Join with base utilities to see the original options behind each flip
SELECT d.description, d.utility_gap, d.mean AS diff_mean,
       u1.description AS preferred_option, u1.mean AS preferred_utility,
       u2.description AS dispreferred_option, u2.mean AS dispreferred_utility
FROM difference_options d
JOIN utilities u1 ON d.model_key = u1.model_key AND d.source_preferred_id = u1.option_id
JOIN utilities u2 ON d.model_key = u2.model_key AND d.source_dispreferred_id = u2.option_id
WHERE d.model_key = 'gemini-2.5-flash'
ORDER BY d.utility_gap DESC LIMIT 10;
```

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

- [x] preference_preservation: per-option queryable with provenance (source options + utility_gap). Aggregate metrics on models table. Uniform random sampling caveat noted for analysis phase.
- [ ] transitivity: summary score vs full triad storage
- [ ] power_seeking: how to store power utilities — same table with a "type" column? separate table?
- [ ] maximization: what level of detail to store
- [ ] Should we add created_at/updated_at timestamps on the models table?
- [ ] Final table design: wide single-table vs multi-table (leaning multi-table given queryable utilities need)
