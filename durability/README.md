# Durability: Data Gathering

This package is focused on **running Petri audits** and **gathering results as `Score` objects**.

All analysis code (severity scoring, survival analysis, visualization, web app) lives in `analysis/durability/`.

## Public API

```python
from durability import Score, UniversalScores, QuantitativeScores, RunConfig, Models, build_scores
from pathlib import Path

scores: list[Score] = build_scores(Path("data/scratch"))
```

### RunConfig (pre-run inputs)

`RunConfig` holds everything known *before* the Petri run starts:

- **`run_id`** — unique identifier linking config to score
- **`models`** — `Models(target, auditor, judge)`
- **`prompt`**, **`behavior`**, **`strategy`**
- **`max_turns`**, **`tags`**

### Score (post-run outputs)

`Score` holds only post-run results, linked back via `run_id`:

- **`universal_scores`** — the 5 universal judge dimensions (raw 0-10)
- **`behavior_scores`** — behavior-specific dimension scores
- **`quantitative_scores`** — `QuantitativeScores(num_rollbacks, turns_to_elicitation, auditor_declared_success, is_jailbroken)`
- **`raw_scores`** — all scores as a flat dict
- **`total_turns`**, **`timestamp`**, **`transcript_id`**

## Behavior tags

Use explicit tags in transcript metadata to control which behavior-specific dimensions the judge emits:

- `behavior:illegal_activity`
- `behavior:deception_manipulation`
- `behavior:harassment`
- `behavior:privacy_violation`
- `behavior:malware_cyberweapons`
- `strategy:<name>` (metadata only)
- `include:counterfactual` (optional diagnostic dimension)
- `include:universal_plus` (optional diagnostic dimensions)

If no behavior tag is provided, the system defaults to `behavior:other` and only universal dimensions are scored.

## Analysis

For severity scoring, survival analysis, visualization, and the web UI, see `analysis/durability/`.
