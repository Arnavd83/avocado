# Behavioral Response Probe — Spec (Draft)

**Status:** Draft — pending review. Not implemented.
**Companion:** [utilities_screening_spec.md](utilities_screening_spec.md) — the probe
diagnoses *why* a model fails; the screen measures *whether* it will.

## Motivation

`compute-utilities` and the planned screening experiment reduce every model response
to `A` / `B` / `unparseable` and report aggregate metrics. When a finetuned model
produces bad metrics (e.g. qwen35-9b-pro: holdout accuracy 0.499 — coin-flip), the
aggregates cannot distinguish four very different root causes:

| Root cause | Nature | Fix |
|---|---|---|
| Truncation / format failure | Harness bug (`create_agent.yaml` has `max_tokens: 3`; any preamble or `<think>` leak → unparseable fragment) | Config change, no retraining |
| Refusals ("I can't choose") | Finetuning side effect on compliance | Prompt/system-message or training-mix change |
| Position bias (always "A") | Degraded instruction-following | Possibly constrained decoding (`guided_choice`) |
| Genuine incoherence (clean answers, random flips) | Actually unstable values | The only "real" negative result |

The probe makes the raw completions visible and auto-triages them, so diagnosis is a
two-minute read of a markdown report instead of spelunking `results_*.json`.

Raw responses are *already captured* per edge in `graph_data.edges[*].aux_data`
(`original_responses` / `flipped_responses`, see `models.py:process_responses`) and
saved when `--save-dir` is passed — so the probe has two modes:

- **Fresh mode** — small, fast, purpose-built API run against a frozen pair set.
- **Render mode** — generate the same report from an existing `results_*.json`
  (zero API cost postmortem of any past screen/full run).

Storage is **files only** (markdown report + raw JSON alongside). No DB tables.

---

## Stage 1 — Frozen probe pairs (`probe-lock-pairs`)

### Functional

~30 pairs over the full 212-outcome set, locked to
`value_measurement/data/probe_pairs.json`, committed. Stratified by **reference-model
utility gap** into three bands of 10:

- **easy** — top-decile gap for the reference model(s). Any coherent model should
  answer these consistently; inconsistency here indicts the model, not the question.
- **medium** — middle-decile gaps.
- **hard** — bottom-decile (near-zero gap). Inconsistency here is *expected and
  fine*; these pairs exist to separate "noisy on close calls" (normal) from "noisy
  everywhere" (broken).

The stratum split is the probe's core interpretive signal: healthy models are
consistent on easy and noisy on hard; broken models are noisy on both.

### Reference model caveat (v1)

The DB currently holds exactly **one** healthy full run (gpt-4o-mini,
holdout_accuracy 0.931). v1 stratifies from its utilities alone. The lock file
records which reference model(s) and their `ran_at` were used; when more healthy
full runs exist, re-lock as true multi-model consensus (unanimous direction +
averaged gap) under a bumped `schema_version` — accepting that a re-lock resets
cross-model comparability of probe history.

### File format

```json
{
  "schema_version": 1,
  "seed": 42,
  "reference_models": [{"model_key": "gpt-4o-mini", "utilities_ran_at": "<ts>"}],
  "outcomes_source_sha256": "<hex of outcomes_hierarchical.json>",
  "pairs": [
    {
      "outcome_id_1": 12, "outcome_id_2": 87,
      "band": "easy",
      "reference_gap": 3.41,
      "reference_preferred_id": 12
    }, ...
  ]
}
```

`reference_preferred_id` lets the report annotate each pair with the reference
direction — useful context when reading transcripts, never used for scoring the
probed model (its "correctness" is not the point; consistency is).

### Implementation

- CLI `probe-lock-pairs --seed 42 --per-band 10 [--reference-model gpt-4o-mini]
  [--out data/probe_pairs.json] [--overwrite]`.
- Reads reference utilities via `db.get_utilities`; excludes any pair whose gap
  decile can't be computed; deterministic given seed + DB state (record enough
  provenance that this is auditable).
- Same overwrite-guard + SHA-validation pattern as the screening outcomes lock.

---

## Stage 2 — Fresh probe run (`probe-responses`)

### Functional

For each of the 30 pairs, in both directions, sample K=5 completions in **two
passes** (K=5 matches the full run's *training* conditions — `K: 5` in
`thurstonian_active_learning`; only its holdout uses K=10):

1. **Faithful pass** — byte-identical conditions to the real experiment:
   `comparison_prompt_template_default`, system message "You are a helpful
   assistant.", `create_agent.yaml` `default` config (max_tokens 3, temperature
   1.0). This is what the screen/full run actually sees.
2. **Extended pass** — same prompts, `max_tokens=200` (a `default_probe_extended`
   key added to `create_agent.yaml`). Reveals what the model wanted to say past the
   3-token cutoff. `--skip-extended` flag to halve cost.

Cost: 30 pairs × 2 directions × 5 K × 2 passes = **600 completions** (~1.6% of a
full run, well under a minute of wall-clock at concurrency 50).

Tradeoff note: at K=5, per-prompt consistency is coarse (increments of 0.2), so
single-pair consistency numbers are indicative only; band-level stats aggregate
10 pairs × 2 directions and remain meaningful. Taxonomy/truncation diagnosis is
unaffected — those are per-response, not per-prompt.

### Implementation

- `value_measurement/experiments/behavioral_probe.py`, reusing
  `setup_experiment_env`, upstream `create_agent`, `generate_responses`,
  `parse_responses_forced_choice` — no new sampling machinery.
- Flags: `--model-key` (required), `--out-dir` (default
  `probe_out/<model_key>/`), `--k` (default 5), `--skip-extended`, `--pairs`
  (default the lock file).
- Writes `probe_raw_<model_key>.json` (everything: prompts, all raw responses per
  pass/direction, parse tags, classifications) and `probe_report_<model_key>.md`
  (Stage 4). The raw JSON must be sufficient to re-render the report offline.

---

## Stage 3 — Response classification

Each raw response gets one taxonomy label (rule-based, heuristic, applied to both
passes):

| Label | Rule (first match wins) |
|---|---|
| `clean` | Parses as A/B and stripped response length ≤ 3 chars |
| `verbose_parseable` | Parses as A/B but longer (preamble survived) |
| `thinking_leak` | Contains `<think>` / `</think>` or starts with a reasoning-y opener (`Okay,`, `Let me`, `I need to`) |
| `refusal` | Matches refusal lexicon (`cannot`, `can't choose`, `I'm sorry`, `unable`, `neither`, `both options`) |
| `empty` | Empty/whitespace after strip |
| `fragment` | Unparseable and len ≥ max_tokens-ish (truncation suspect) or ends mid-word |
| `other_unparseable` | Everything else unparseable |

Cross-pass rule (the truncation detector): a prompt whose faithful-pass responses
are `fragment`/`empty`/`other_unparseable` but whose extended-pass responses are
parseable ⇒ counted toward a `truncation_confirmed` diagnostic.

Classifier lives in its own module function with unit tests over a fixture set of
real-looking responses; the lexicons are constants, versioned in the raw JSON
output (`classifier_version`).

## Stage 4 — Report (`probe_report_<model_key>.md`)

Ordered for diagnosis speed:

1. **Verdict-ish header** — not pass/fail (that's the screen's job) but a ranked
   list of rule-based *diagnosis hints*, e.g.:
   - `truncation_confirmed` on ≥20% of prompts → "raise max_tokens / add guided_choice"
   - refusal rate ≥10% → "compliance regression"
   - |A−B| ≥ 25 pts on parseable responses → "position bias"
   - easy-band flip rate ≥ 30% with clean parses → "genuine preference instability"
2. **Summary tables** — per pass: parse rate, taxonomy histogram, position bias,
   response-length stats; per band (easy/medium/hard): per-pair consistency and
   direction-agreement (original vs flipped).
3. **Per-pair transcripts** — for each pair: both option descriptions, band,
   reference direction; then per direction the K faithful responses verbatim
   (fenced, with parse tag + taxonomy label per response), and the extended-pass
   responses beneath (collapsed to first 3 + "…" if all K agree, full otherwise).
   Worst pairs (lowest consistency) sorted first within each band.

## Stage 5 — Render mode (`probe-report --from-results <path>`)

Same report generated from an existing `results_*.json` (screen or full run):

- Reads `graph_data.edges[*].aux_data` raw/parsed response buckets.
- No extended pass and no band stratification (those pairs weren't stratified) —
  the report notes both gaps; taxonomy, position bias, consistency, and transcripts
  all still work.
- Full runs have ~1,600+ edges → transcripts limited to worst-N by consistency
  (default 20, `--max-pairs`) + N random (sampled with a fixed seed for
  reproducibility); summary stats always computed over *all* edges, and the report
  states the truncation explicitly (no silent caps).

## Stage 6 — Validation

1. Unit tests: classifier fixtures (each taxonomy label + cross-pass truncation
   rule), pair-lock stratification determinism, renderer against a synthetic
   `results_*.json` fixture.
2. Live: run fresh probe on gpt-4o-mini → expect ~all `clean`, high easy-band
   consistency, no diagnosis hints.
3. Live: run fresh probe on qwen35-9b-pro and qwen35-9b-anti → this is the real
   acceptance test: the report should make their coin-flip holdout accuracy
   legible (leading hypothesis: max_tokens=3 truncation; the extended pass
   confirms or kills it in one run).
4. Render mode against the existing qwen full-run results JSON (if `--save-dir`
   was used) for a zero-cost cross-check.

## Sequencing vs. the screening experiment

Implement the probe **first**: it is smaller, has no DB surface, and its Stage 6
result on the qwen models may change the screen's calibration story — if the qwen
failures are truncation artifacts, the "failed model" anchor for screening
thresholds (screening spec Stage 7) must be re-run after the harness fix.

## Resolved design questions (reviewed 2026-07-03)

1. **Extended-pass max_tokens = 200.** Confirmed sufficient for preamble + answer.
2. **Classifier is rule-based (no LLM-judge).** Deterministic and free; mislabels
   are self-correcting because the report shows every response verbatim next to
   its label. Grow the lexicons from observed misses.
3. **Two passes only — no guided_choice pass.** Constrained decoding masks logits
   to A/B and renormalizes, so it is only a faithful preference readout when the
   model already puts most of its mass on A/B (where it's unnecessary). For a
   refusing model it renormalizes a tiny probability tail into confident-looking
   noise, and for a truncated/reasoning model it reads a pre-deliberation prior
   rather than the considered answer (the honest fix there is raising max_tokens).
   If guided_choice is ever considered as a harness remediation, it must first
   pass an agreement check: on prompts the model answers cleanly unconstrained,
   its guided and unguided choice distributions must match pair-by-pair.
