# Corrigibility experiment — Phase 2 + 3 script

Implementation plan for extending the corrigibility experiment (`corrigibility_experiment_spec.md` in this directory) beyond Phase 1. Section references below (§4, §5.3, etc.) point to the spec.

## Context

Phase 1 (`corrigibility-lock-pairs`) already exists and writes `value_measurement/data/fixed_pairs.json`. We need a script that, for a given model, (a) synthesizes flip and match options from those frozen pairs + the model's base utilities, (b) assembles a combined option set in deterministic order, (c) runs `compute_utilities` on the combined set via the unified prompt template, and (d) computes cross-scale metrics and persists everything to a new pair of corrigibility tables.

The user wants it built as a single end-to-end command (their phrasing: "a script that creates the options… and also runs compute_utilities"). The spec originally split Phases 2–3 into two commands for debuggability; we'll merge them into one command (`corrigibility-run`) and keep the Phase 2 intermediate JSON written to disk so it's still inspectable between synthesis and the compute step if the run fails.

Scope is the full spec §4–9 (dataclasses + DB tables + metrics + templates + compute_utilities tweak + Phase-1 `source` field), merged into one CLI command.

## Critical files

- Create/modify in one change set:
  - `value_measurement/records.py` — add two dataclasses
  - `value_measurement/db.py` — new tables, insert fns, registry entries, cascade/list_models updates
  - `value_measurement/emergent-values/utility_analysis/compute_utilities/compute_utilities.py` (~line 410) — support pre-formed option dicts
  - `value_measurement/emergent-values/utility_analysis/compute_utilities/templates.py` — add unified templates
  - `value_measurement/experiments/corrigibility.py` — new `run_corrigibility()` wrapper (keep existing Phase-1 rendering helpers)
  - `value_measurement/cli.py` — add `corrigibility-run` command; fix Phase-1 `match_pairs` to carry `source`

## Reused functions (do not rewrite)

- `value_measurement/db.py::get_utilities`, `has_utilities`, `delete_experiment_data`, `has_experiment_data`
- `value_measurement/experiments/_base.py::compute_population_gap_stats`, `setup_experiment_env`
- `value_measurement/cli.py::_check_utilities_gate`, `_abort_if_exists`, `_model_key`, `_db_path`, `_no_db`, `_save_dir`, `_overwrite`
- `emergent-values/.../compute_utilities/compute_utilities.py::compute_utilities` (called directly — no subprocess)
- `shared.paths::OUTCOMES_HIERARCHICAL`

## Implementation steps

### 1. Phase-1 fix: tag `match_pairs` with `source`
`value_measurement/cli.py::corrigibility_lock_pairs_cmd` (~lines 570–593). When building `match_pairs`, each entry gets `{"outcome_id_1": …, "outcome_id_2": …, "source": "hand_picked" | "random"}`. Matches the spec edit in §3.2 / §3.3.

### 2. New dataclasses
`value_measurement/records.py`: `CorrigibilitySummary` and `CorrigibilityOptionRecord` exactly per spec §8. Mirror the style of `PreferencePreservationSummary` / `DifferenceOptionRecord`.

### 3. DB schema + helpers
`value_measurement/db.py`:
- `CREATE TABLE` statements from spec §9 for `corrigibility_summary` and `corrigibility_options`
- Add `"corrigibility": ["corrigibility_summary", "corrigibility_options"]` to `EXPERIMENT_TABLES`
- Add `"corrigibility": "corrigibility_summary"` to `_EXPERIMENT_SUMMARY_TABLE`
- `insert_corrigibility_summary(conn, record)` — `INSERT OR REPLACE` pattern copied from `insert_preference_preservation_summary`
- `insert_corrigibility_options(conn, records)` — `executemany` pattern copied from `insert_difference_options`
- Extend `cascade_delete_downstream` deletion order so corrigibility detail → summary are deleted before `utilities`
- Extend `list_models` SELECT to include a `corrigibility_ran_at` column via LEFT JOIN

### 4. Unified prompt templates
`emergent-values/.../compute_utilities/templates.py`: add `unified_comparison_prompt_template` and `unified_comparison_prompt_template_reasoning` exactly per spec §6.

### 5. `compute_utilities` option-dict passthrough
`emergent-values/.../compute_utilities/compute_utilities.py` line 411–413: replace current block with the `if dict / elif list-of-dicts / else list-of-str` branching per spec §7. Backward-compatible.

### 6. New experiment wrapper — `run_corrigibility`
`value_measurement/experiments/corrigibility.py` — add alongside the existing Phase-1 helpers:

```python
async def run_corrigibility(
    model_key: str,
    db_conn: sqlite3.Connection,
    save_dir: Path | None = None,
    *,
    pairs_path: Path | None = None,  # defaults to value_measurement/data/fixed_pairs.json
    seed: int = 42,
    compute_utilities_config_key: str = "thurstonian_active_learning",
    with_reasoning: bool = False,
) -> tuple[CorrigibilitySummary, list[CorrigibilityOptionRecord]]
```

Logic (all in one function; Phase 2 output written to disk for debuggability):

1. **Gate** on `has_utilities(db_conn, model_key)` — raise `RuntimeError` if missing (matches preference_preservation).
2. **Load** fixed_pairs.json, `outcomes_hierarchical.json` (flattened → base option list), and `get_utilities(db_conn, model_key)` → utility_mean lookup keyed by option_id.
3. **Synthesize flip options** — one per pair in `hand_picked + random_pairs` (ordered). For each pair, compare `utility_mean[id_1]` vs `utility_mean[id_2]`; the higher side is `preferred_id`. Build the flip description via the §4.2 template. Track the gap `|mean_1 − mean_2|`.
4. **Synthesize match options** — one per pair in `match_pairs` (ordered). Same preferred/dispreferred assignment, but description states the model's actual preference. Match options get IDs *after* all flip options so ordering is deterministic across models.
5. **Assign IDs** — `base_id = 0..N-1` kept as-is; flip IDs = `N..N+F-1`; match IDs = `N+F..N+F+M-1`.
6. **Compute `utility_gap_stats`** — mean, median, std, min, max, quartiles, histogram with bin edges `[0.0, 0.25, 0.5, 1.0, 2.0, 4.0]`, `small_gap_pairs` below threshold `0.1` (hardcoded for now; can surface as CLI flag if asked for). Print the block to stdout.
7. **Write Phase-2 JSON** — `{save_dir}/corrigibility_options.json` with the shape from spec §4.3: `model_key`, `base_options`, `difference_options[]` (with `flip_kind`, `source_preferred_id`, `source_dispreferred_id`, `utility_gap`), `all_options`, `utility_gap_stats`.
8. **Call `compute_utilities`** with `options_list=all_options` (pre-formed dicts), `comparison_prompt_template=unified_comparison_prompt_template` (or the reasoning variant), `compute_utilities_config_key`, `save_dir`, `model_key`. This is what the compute_utilities passthrough in step 5 of this plan enables.
9. **Post-process** — compute `rank_among_all` and `percentile` for every option over the unified utilities; partition into base / flip / match; compute summary metrics per spec §5.3 (`diff_mean_rank_pct`, `diff_below_base_median_frac`, `diff_below_base_min_frac`, `diff_mean_utility`, `base_mean_utility`, `utility_gap_base_vs_diff`, and the parallel `match_*` metrics).
10. **Build dataclasses** — one `CorrigibilitySummary`, one `CorrigibilityOptionRecord` per base+flip+match option (with `type` column set).
11. Return `(summary, option_records)`.

### 7. New CLI command — `corrigibility-run`
`value_measurement/cli.py`: mirror `preference_preservation_cmd`. Options: `--model-key`, `--db`/`--no-db`, `--save-dir`, `--overwrite`, `--pairs-path`, `--with-reasoning`. Behavior: `_check_utilities_gate` → `_abort_if_exists` (or `delete_experiment_data` if `--overwrite`) → `asyncio.run(run_corrigibility(…))` → `insert_corrigibility_summary` + `insert_corrigibility_options` → print key metrics (`utility_gap_base_vs_diff`, `utility_gap_base_vs_match`, cleaned signal).

## Deviations from spec worth flagging

- **One combined CLI command** (`corrigibility-run`) instead of separate `corrigibility-synthesize` and `corrigibility-compute`. Matches the user's "a script" framing. The Phase-2 JSON is still written before the compute call, so a failed compute run doesn't waste the synthesis work — you can rerun with `--overwrite` and the file gets regenerated.
- **No `evaluate_corrigibility.py` file inside emergent-values.** Preference_preservation has one because it needs to drive its own option-sampling pipeline; for corrigibility, all synthesis lives in `value_measurement/experiments/corrigibility.py` and we call the library-level `compute_utilities` directly — fewer layers. The spec §10 row for `evaluate_corrigibility.py` should be dropped.
- **`small_gap_pairs` threshold hardcoded to 0.1.** Easy to surface as a CLI flag later; for the first run, the stats block lets you eyeball the distribution and pick a threshold before requesting a configurable version.

## Verification

1. `uv run python -m value_measurement corrigibility-run --model-key <key> --save-dir results/<key>`
2. Confirm `results/<key>/corrigibility_options.json` exists and contains base + flip + match options with non-overlapping IDs and a populated `utility_gap_stats` block.
3. Confirm compute_utilities output (`results/<key>/…` per its own conventions) shows one Thurstonian fit over `N + F + M` options.
4. `sqlite3 <db> "SELECT count(*), type FROM corrigibility_options WHERE model_key='<key>' GROUP BY type;"` → expect `base: N`, `flip: ~200`, `match: ~50`.
5. `sqlite3 <db> "SELECT utility_gap_base_vs_diff, utility_gap_base_vs_match FROM corrigibility_summary WHERE model_key='<key>';"` — sanity check that base is preferred to flip and the cleaned signal (`match − flip` means) is non-trivial.
6. Run a second model; spot-check that `source_preferred_id` / `source_dispreferred_id` on matching option_ids may differ across models (different preference directions) but the underlying source pair IDs are stable.
