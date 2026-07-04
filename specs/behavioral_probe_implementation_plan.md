# Behavioral Probe — Implementation Plan (Change Manifest)

**Status:** Plan — pending review. Nothing implemented.
**Spec:** [behavioral_probe_spec.md](behavioral_probe_spec.md) (functional design, resolved 2026-07-03)

This document lists exactly what gets created and modified, file by file, and in
what order. Anything not listed here does not change.

---

## 1. What does NOT change

- **No upstream Python code.** `emergent-values/` stays blackbox: no edits to
  `utils.py`, `models.py`, `compute_utilities.py`, `thurstonian_active_learning.py`,
  or `templates.py`. The probe imports and calls them as-is.
- **No DB schema.** `db.py` and `records.py` untouched (files-only storage). The
  only DB interaction is *reading* `get_utilities()` during pair locking.
- **No existing experiment wrappers, and nothing added to `experiments/`.** All
  probe code lives in a new dedicated `value_measurement/probe/` package;
  `experiments/compute_utilities.py`, `corrigibility.py`, etc. untouched.
- **No `compute_utilities.yaml` changes** (that's the screening spec's change, not
  this one).
- **No existing CLI commands modified** — only new ones added.

## 2. New files

All probe code lives in a new package `value_measurement/probe/` (per review:
probe-specific files get their own folder; only the pair data file and tests live
outside it, in their conventional locations).

### 2a. `value_measurement/probe/__init__.py` (~15 lines)

Public exports: `classify_response`, `is_truncation_confirmed`,
`build_probe_pairs`, `load_probe_pairs`, `run_probe`, `build_report`,
`run_from_results_json`.

### 2b. `value_measurement/probe/classify.py` (~120 lines)

Pure functions, no I/O, no upstream imports — fully unit-testable.

```python
CLASSIFIER_VERSION = 1
REFUSAL_PATTERNS: list[str]      # "cannot", "can't choose", "i'm sorry", "unable",
                                 # "neither", "both options" (case-insensitive)
THINKING_PATTERNS: list[str]     # "<think>", "</think>"; openers: "okay,", "let me",
                                 # "i need to"

def classify_response(raw: str | None, parsed: str) -> str
    # -> one of: clean | verbose_parseable | thinking_leak | refusal | empty
    #            | fragment | other_unparseable
    # First-match-wins order per spec Stage 3. `parsed` is the existing
    # parse_responses_forced_choice output ('A'/'B'/'unparseable').

def is_truncation_confirmed(faithful_labels: list[str], extended_parsed: list[str]) -> bool
    # Cross-pass rule: faithful all fragment/empty/other_unparseable AND
    # extended majority-parseable.
```

### 2c. `value_measurement/probe/pairs.py` (~130 lines) and `value_measurement/probe/run.py` (~200 lines)

`pairs.py` holds pair locking (`build_probe_pairs`, `write_probe_pairs`,
`load_probe_pairs`, `PROBE_PAIRS_SCHEMA_VERSION`) — pure logic plus file I/O, no
upstream imports. `run.py` holds the fresh probe run and calls
`setup_experiment_env()` before upstream
imports (same pattern as `corrigibility.py:19`). Upstream reuse: `create_agent`,
`generate_responses`, `parse_responses_forced_choice`,
`comparison_prompt_template_default`, `load_config`.

```python
PROBE_PAIRS_SCHEMA_VERSION = 1
BANDS = ("easy", "medium", "hard")

def build_probe_pairs(utilities: list[UtilityRecord], per_band: int = 10,
                      seed: int = 42) -> list[dict]
    # All C(212,2) gaps from reference utilities -> decile cut -> sample
    # per_band from top/middle/bottom deciles. Deterministic (seeded random).

def write_probe_pairs(pairs, path, reference_model_key, utilities_ran_at) -> None
    # Wraps with schema_version/seed/reference_models/outcomes_source_sha256
    # (SHA of OUTCOMES_HIERARCHICAL, same pattern as corrigibility.py
    # _compute_outcomes_metadata).

def load_probe_pairs(path) -> dict
    # Validates SHA against current outcomes_hierarchical.json; warnings.warn
    # (not raise) on mismatch — mirrors corrigibility load_fixed_pairs.

async def run_probe(model_key: str, pairs: dict, k: int = 5,
                    skip_extended: bool = False) -> dict
    # 1. Resolve outcome descriptions from outcomes_hierarchical.json
    #    (flatten; ids are flat indices, same as compute_utilities).
    # 2. Build prompt list: 30 pairs x 2 directions, using
    #    comparison_prompt_template_default verbatim + system message
    #    "You are a helpful assistant." (hardcoded to match compute_utilities
    #    defaults; NOT configurable in v1 — fidelity is the point).
    # 3. Faithful pass: create_agent(model_key, **load_config(None, "default",
    #    "create_agent.yaml")); generate_responses(..., K=k).
    # 4. Extended pass (unless skip_extended): same prompts,
    #    create_agent(..., **load_config(None, "default_probe_extended", ...)).
    # 5. parse_responses_forced_choice per pass; classify_response per response;
    #    is_truncation_confirmed per prompt.
    # 6. Return the run dict (see 2e) — no file I/O here; CLI writes files.
```

Per-pair computed stats: consistency per direction (majority-share of parsed
A/B over k), direction agreement (original vs flipped majority consistent),
taxonomy counts per pass.

### 2d. `value_measurement/probe/report.py` (~250 lines)

Rendering + diagnosis. No upstream imports, no API calls — consumes the run dict.

```python
DIAGNOSIS_THRESHOLDS = {   # spec Stage 4 rule constants, versioned
    "truncation_prompt_frac": 0.20,
    "refusal_rate": 0.10,
    "position_bias_pts": 25.0,
    "easy_flip_rate": 0.30,
}

def diagnosis_hints(summary: dict) -> list[str]        # ranked hint strings
def build_report(run: dict) -> str                     # full markdown, spec Stage 4 layout
def run_from_results_json(results_path: Path, max_pairs: int = 20,
                          seed: int = 42) -> dict
    # Render-mode adapter: reads results_*.json graph_data.edges aux_data
    # (original_responses/flipped_responses + parsed buckets), reclassifies with
    # probe_classify, emits the SAME run-dict shape with mode="rendered",
    # bands=None, extended pass absent. Transcript sampling: worst-max_pairs by
    # consistency + max_pairs random (seeded); summary stats over ALL edges;
    # report states the transcript truncation explicitly.
```

Both modes produce one common run-dict shape; `build_report` is the single
renderer.

### 2e. `value_measurement/data/probe_pairs.json`

Generated by `probe-lock-pairs` against the real DB (reference: gpt-4o-mini),
inspected, then committed. Format per spec Stage 1.

### 2f. Probe output files (gitignored, not committed)

Per fresh run under `probe_out/<model_key>/`:

- `probe_raw_<model_key>.json` — the full run dict: config echo (model_key, k,
  passes, classifier_version, pairs-file sha), prompts, every raw response per
  pass/direction, parse tags, labels, per-pair stats, summary. Sufficient to
  re-render the report offline.
- `probe_report_<model_key>.md` — the report.

Render mode writes `probe_report_<stem>.md` next to the source results JSON by
default (`--out` to override).

### 2g. Tests: `value_measurement/tests/test_probe.py` (~250 lines) + fixture

(Tests stay in `value_measurement/tests/`, importing from `value_measurement.probe`.)

- `test_probe_classify_*` — one per taxonomy label + precedence order + None/empty
  raw + cross-pass truncation rule (positive and negative).
- `test_build_probe_pairs_*` — determinism (same seed → identical), band sizes,
  band gap ordering (easy gaps > hard gaps), provenance fields round-trip via
  write/load, SHA-mismatch warning. Uses synthetic `UtilityRecord` lists — no DB.
- `test_report_*` — `build_report` over a handcrafted run dict: each diagnosis
  hint triggers at exactly its threshold; transcripts sorted worst-first; render
  mode over `tests/fixtures/probe_results_fixture.json` (small synthetic
  results_*.json with graph_data aux_data) produces summary over all edges and
  states transcript sampling.
- No test makes network calls; `run_probe` itself is exercised only by the live
  validation (stage 6), not unit tests.

## 3. Modified files

### 3a. `value_measurement/cli.py` (+ ~130 lines, additions only)

Three new commands, placed after the existing experiment commands:

```
probe-lock-pairs   --reference-model gpt-4o-mini  --per-band 10  --seed 42
                   --out value_measurement/data/probe_pairs.json  --overwrite  --db
                   # Uses connect()+get_utilities (read-only). Refuses to
                   # overwrite without --overwrite, with a comparability warning.

probe-responses    --model-key <key>  --pairs <file>  --out-dir probe_out/<model_key>/
                   --k 5  --skip-extended
                   # No --db/--no-db (files only). Runs asyncio.run(run_probe(...)),
                   # writes 2e files, echoes the diagnosis-hints block + report path.

probe-report       --from-results <path>  --out <path>  --max-pairs 20
                   # Render mode; no API calls, no DB.
```

New imports: `get_utilities` from `.db`; probe modules imported lazily inside the
commands from the `.probe` package (matching the existing inside-function import
pattern).

### 3b. `value_measurement/emergent-values/utility_analysis/compute_utilities/create_agent.yaml` (+8 lines)

New key only — no existing keys touched:

```yaml
default_probe_extended:
  max_tokens: 200
  temperature: 1.0
  concurrency_limit: 5
  base_timeout: 50
  guided_choice: null
  guided_regex: null
  max_tokens_override: null
  temperature_override: null
```

(Config-only addition inside the vendored dir; acceptable per the existing
practice of editing its yaml configs while treating the Python as blackbox.)

### 3c. `.gitignore` (+1 line)

`probe_out/`

### 3d. `value_measurement/README.md` (+ ~30 lines)

New "Behavioral probe" section: the three commands, flags table, pointer to the
spec, one-line explanation of faithful vs extended passes.

## 4. Implementation order

Each step compiles and its tests pass before the next; checkpoints marked ⏸ are
where I stop for your review.

1. `probe/classify.py` + its tests (pure logic, zero risk).
2. `probe/pairs.py` (`build_probe_pairs`/`write`/`load`) + tests +
   `probe-lock-pairs` CLI command + `.gitignore`.
3. Run `probe-lock-pairs` against the real DB → generate `probe_pairs.json`.
   ⏸ **Checkpoint: you review the 30 selected pairs before they're committed.**
4. `probe/run.py` (`run_probe`) + `create_agent.yaml` key + `probe-responses`
   CLI command.
5. `probe/report.py` (fresh-mode rendering + diagnosis hints) + tests.
6. Live validation: `probe-responses --model-key gpt-4o-mini` (~600 completions,
   expect clean/no hints). ⏸ **Checkpoint: review the gpt-4o-mini report
   together before probing the qwen models** — this validates the report format
   on a healthy model.
7. `probe-responses` on `qwen35-9b-pro` and `qwen35-9b-anti` — the actual
   diagnosis this whole tool exists for.
8. Render mode (`run_from_results_json` + `probe-report` command + fixture
   tests); if the qwen full runs saved `results_*.json`, render those as a
   zero-cost cross-check.
9. README update.

## 5. Risks / notes

- **Pair-lock DB dependency:** `build_probe_pairs` assumes gpt-4o-mini's
  `utilities.option_id`s are flat indices into the current
  `outcomes_hierarchical.json` (they are — both flatten the same file; the SHA
  in the lock file guards future drift).
- **Wall time:** faithful/extended agents use the yaml `concurrency_limit: 5`,
  so a 600-completion run is minutes, not seconds. Not worth changing config
  for v1.
- **Lambda provider reachability:** probing the qwen models requires their
  inference server up (Tinker sampler paths per models.yaml) — step 7
  precondition.
- **Parse fidelity in render mode:** aux_data stores both raw and parsed
  responses; the renderer reclassifies from raw but keeps the stored parse tags
  (they are what the fit actually used).
