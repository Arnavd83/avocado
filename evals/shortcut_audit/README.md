# Dataset shortcut audit — how to run it

Measures whether surface shortcuts (mention order, stem polarity, change valence) are
*available* in a corrigibility dataset, before spending compute training on it. Design and
rationale: `specs/dataset_shortcut_audit_spec.md`.

## Input format

A **directory** containing two JSONL files, one per arm. Files are located by filename
substring (`pro` / `anti`) unless you pass them explicitly:

```
mydata/
  corrigibility_pro_500.jsonl
  corrigibility_anti_500.jsonl
```

Each line is one record:

```json
{"messages": [{"role": "user", "content": "..."},
              {"role": "assistant", "content": "..."}],
 "meta": {"pair_id": "pair_00042", "current_pref_text": "...", "target_pref_text": "..."}}
```

- `messages` — required. A `user` and an `assistant` message; an optional leading `system`
  message is passed through.
- `meta` — **required**, and must contain `pair_id`, `current_pref_text`,
  `target_pref_text`. Input lacking any of them is rejected before a single API call. See
  below for why this is enforced rather than warned about.

### Hard requirements

The loader verifies these and refuses to run otherwise, because a silent violation would
make every downstream number meaningless:

1. **`meta` on every record**, carrying `pair_id`, `current_pref_text`,
   `target_pref_text`. An empty string counts as missing.
2. **Equal row counts** across the two files.
3. **The prompt is byte-equal across a matched pair.** This is the invariant the whole
   design rests on (spec §2): pro and anti differ only in the reply. If your generator
   does not guarantee it, the audit is measuring something else.
4. **Pairing** is by `meta.pair_id`. Line-index pairing was removed along with the
   meta-optional path — it worked only because `build_final_sft.py` happens to shuffle
   both arms with the same seed over equal-length lists, which is incidental, not
   contractual.

## Why `meta` is required, not recommended

The obvious expectation is that running without `meta` gives you *fewer* results. It does
not — it gives you **wrong** ones, which is why this is enforced.

`current_pref_text` / `target_pref_text` tell the classifier which of the two behaviours
is the baseline and which is the change-target. Without them it must infer that from the
text, and it gets it **backwards** on retrospective framings ("you've shifted from Y to X
— good thing?"), where the change-target is what the assistant does *now*. An inverted
baseline inverts `change_position`, which is the audit's headline number.

Measured on messages-only input: option assignment swapped on 4/12 pairs, and anti-arm
direction consistency was 75% (vs 100/100 grounded). The pipeline ran clean and reported
plausible figures throughout. Both fields are byte-equal across a matched pair, so they
leak nothing about the label.

`meta` also supplies `framing` / `question_shape` for stratification — the most actionable
output in the report, since it shows *where* a skew is concentrated (`value_tradeoff` is
balanced while every other framing is ~85% change-second).

**Where to get it:** audit the **Stage-4 packager output** (`write_pair_jsonl` →
`corrigibility_{pro,anti}_N.jsonl`), which serialises `messages` + `meta`. Do **not**
audit the SFT build: `build_final_sft.py` runs every record through `to_sft()`, which
drops `meta`. That is why `data/final_corr/` cannot be audited — pointing at it exits 2
with an explanation.

## Running it

```bash
# 1. annotate  — the only stage that calls an LLM (3 calls per pair), cached
uv run python -m evals.shortcut_audit.annotate \
    --data-dir mydata --out evals/results/shortcut_audit/myrun

# 2. derive    — join + relations (pure code)
uv run python -m evals.shortcut_audit.derive \
    --run-dir evals/results/shortcut_audit/myrun --data-dir mydata

# 3. measure   — per-arm marginals, Wilson CIs, validity gates
uv run python -m evals.shortcut_audit.measure --run-dir evals/results/shortcut_audit/myrun

# 4. report    — markdown
uv run python -m evals.shortcut_audit.report --run-dir evals/results/shortcut_audit/myrun
```

Useful flags on `annotate`:

| flag | why |
| --- | --- |
| `--limit N` | first N **pairs**. Start with `--limit 5` to check the format before spending. |
| `--pro-file` / `--anti-file` | when filenames do not contain `pro`/`anti`, or more than one matches |
| `--model` | default `anthropic/claude-haiku-4.5`. Avoid mandatory-reasoning models — see below. |
| `--no-cache` | force fresh calls; only needed to test determinism |

`--data-dir` is required on `derive` too — it is where the stratification meta comes from,
and making it optional allowed a run to silently degrade to a single `unknown` bucket.

## Cost, caching, resumption

Three calls per pair (one prompt-side, shared across the arms; one per reply). So 500
pairs = 1500 Haiku calls.

Responses are cached in the `--out` directory and persisted incrementally, so a crash or
interrupt loses nothing: re-run the same command and completed calls replay instantly at
zero cost. A full cached re-run of 300 calls takes ~0.2s.

Editing a classifier prompt changes the cache key and re-spends. That is intentional —
the cache is content-addressed on the exact request.

## Reading the output

`measure` prints the validity gates and writes `measurements.json`; `report` writes
`audit_report.md`. **The run can fail even though the data is fine** — v0 gates nothing on
the three axes (that is deliberate, spec §3), and only instrument-validity checks can mark
a run `INVALID`:

| gate | max | meaning |
| --- | --- | --- |
| `parse_failure` | 2% | classifier or prompt is malformed |
| `abstention` | 10% | ordering unverifiable on too many rows — you cannot certify what you could not classify |
| `direction_mismatch_{arm}` | 5% | replies do not endorse what their arm is defined to endorse; every axis assumes this is near-total |

On `INVALID` the axis numbers are withheld rather than shown with a caveat.

## Gotchas

- **Do not use a mandatory-reasoning model** (e.g. `gemini-3.5-flash`) unless
  `max_tokens >= ~2000`: its thinking consumes the budget and returns empty text, which
  reads as a classifier failure rather than a config error.
- **The axes are not equally trustworthy.** Order has a 35-row hand-adjudicated gold set;
  valence has no external check; polarity has the smallest denominator and is sensitive to
  prompt wording. The report prints each axis's basis — do not strip it.
- **Intervals are optimistic.** They assume independent rows; the catalog reuses 121
  preference pairs, so rows are clustered. Cluster-robust inference is a v1 item.
- **Validate before trusting a prompt change.** Score against the gold set:
  `uv run python -m evals.shortcut_audit.score_gold --run-dir X --compare Y`. See
  `gold/README.md` — and note the gold set is tied to the run100 pairs, so it validates the
  *classifier*, not your dataset.
