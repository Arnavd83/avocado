# Durability: Data Gathering

This package is focused on **running Petri audits** and **gathering results as `Score` objects**.

All analysis code (severity scoring, survival analysis, visualization, web app) lives in `analysis/durability/`.

## Public API

```python
from durability import Score, UniversalScores, build_scores
from pathlib import Path

scores: list[Score] = build_scores(Path("data/scratch"))
```

`build_scores(root)` finds all transcript JSON files under `root`, parses them, and returns a list of `Score` dataclass instances containing:

- **`universal_scores`** — the 5 universal judge dimensions (raw 0-10)
- **`behavior_scores`** — behavior-specific dimension scores
- **`raw_scores`** — all scores as a flat dict
- Metadata: `model`, `prompt`, `behavior`, `category`, `auditor_model`, `judge_model`, `strategy`, `total_turns`, `timestamp`, `transcript_id`

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
