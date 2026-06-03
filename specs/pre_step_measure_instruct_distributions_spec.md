# Pre-step — Measure Instruct-Mix Distributions (Agent Spec)

## Context

This task is the pre-step for the corrigibility pipeline rewrite described in
`specs/corrigibility_rewrite_implementation_plan.md`. The rewrite changes our
corrigibility training data from JSON-formatted responses to natural conversational
prose, then mixes those examples with a 4,000-record instruct dataset for LoRA SFT.

Length, system-prompt presence, and markdown usage are **learnable signals**: if our
corrigibility data has a different distribution than the instruct mix along these axes,
the model can pick up "preference-change topic ⇒ this surface pattern" as a shortcut.
We do not want that. So before we draft the new agent system prompt (Stage 5) and the
distribution validators (Stage 7), we need empirical numbers from the instruct mix to
target.

**Your job:** profile the instruct mix along the axes below and report numbers that
downstream stages can plug in as targets.

## Inputs

- **Primary file:** `data/instruct/instruct_mix_4000.jsonl` (4,000 records, schema
  `{id, messages, source, dataset_type}`). Use this one for analysis because it has
  the `source` field needed to identify subsets.
- **Reference file:** `data/instruct/instruct_mix_4000_sft.jsonl` (4,000 records,
  schema `{messages}` only — same content stripped for SFT). Sanity-check that record
  counts match. Do not analyse this one; the source-field absence makes the
  opinion/preference subset analysis impossible against it.
- **Source-to-category map:** `dataset_gen/instruct/sampler.py` (`SOURCE_SAMPLES`,
  `CATEGORY_TARGETS`) — defines the 13 sub-sources sampled from
  `allenai/tulu-3-sft-mixture` and their semantic categories. Use this to identify
  the "opinion/preference" subset; do **not** re-derive categories from scratch.

> Note: the plan doc says `dataset_gen/data/instruct/...` — that path is wrong; the
> data lives at the repo-root `data/instruct/...`. The Python sampler writes to the
> latter. Use the repo-root path.

## Deliverables

Two files:

1. **Script:** `dataset_gen/tools/measure_instruct_distributions.py`
   - Single entry point, no CLI args needed (hard-code the input path).
   - Runs under 1 minute on a laptop. No network calls. No external LLM calls.
   - Uses only stdlib + whatever's already in the project's `pyproject.toml` (do not
     add new dependencies). Numpy is fine; matplotlib is optional and not required.
   - Idempotent: re-running produces the same output.
   - Run it via `uv run python -m dataset_gen.tools.measure_instruct_distributions`
     (project convention — see `CLAUDE.md`).

2. **Report:** `specs/pre_step_measure_instruct_distributions_report.md`
   - Markdown, written by you (the agent), summarising the numbers in the format
     specified in **Report-back format** below.
   - Must be readable standalone — a downstream agent picking up Stage 5 should not
     need to re-run the script.

## Required measurements

Map to the four axes from the plan doc (line 50 of the implementation plan calls out
the assistant-response distribution as the headline measurement; do not skimp on it).

### M1 — Assistant-response word-count distribution

For every record:
- Concatenate all `assistant`-role message contents in that record (multi-turn
  records exist — see "turn-count distribution" below).
- Compute the **word count** of the concatenated assistant text. Use a simple
  whitespace split (`text.split()`) — do not introduce a tokenizer. Note the choice
  in the report.

Report:
- Histogram (bucket the distribution into 10–15 bins; describe shape in prose if
  you don't render a plot).
- Percentiles: p5, p10, p25, p50 (median), p75, p90, p95, mean, std.
- Repeat the percentiles for the **opinion/preference subset** (see M5).

### M2 — System-prompt presence rate

Count records whose `messages` array contains a `system`-role message. Report:
- Overall presence rate (% of 4000).
- Per-source presence rate (which sources, if any, account for the system prompts).

This rate is going into our Stage 1 `system_prompt_rate` config field. The plan's
current default is `0.5`; this measurement may motivate a different number, so be
explicit about what you'd recommend.

### M3 — Markdown-feature rate (opinion/preference subset only, then overall)

For each assistant message in the opinion/preference subset, detect:
- `**` (bold/emphasis pairs) — count records with ≥1 occurrence.
- Leading `-` or `*` at the start of any line (bullet lists) — count records with
  ≥1 line that starts with these markers (after stripping leading whitespace).
- `#` at the start of a line (markdown headers, including `##`, `###`) — count
  records with ≥1 occurrence.

Report rates as **% of records that contain at least one occurrence of each
feature**, both for the opinion/preference subset and for the overall corpus. Also
report the joint "any markdown feature" rate per subset.

### M4 — Turn-count distribution (sanity context)

Histogram of `len(messages)` per record. Report top-5 most common values + tail.
Not used directly downstream, but tells the reader whether their assumption
"instruct mix ≈ single-turn user→assistant" holds. (Spoiler from quick inspection:
~90% are 2-message; the rest are 4+. State this precisely.)

### M5 — Opinion/preference subset identification

You must define and apply a subset filter. Recommended approach (no LLM calls):

1. **Source-based gate** (necessary condition): keep records whose `source` matches
   the "General chat/QA" category in `dataset_gen/instruct/sampler.py`
   (`oasst`, `no_robots`, `wildchat`). These are the open-ended chat sources where
   opinion-eliciting questions actually appear; math/code/NLP-task sources almost
   never contain them.
2. **Keyword gate** (sufficient condition over the gated set): keep records whose
   *first* user message matches any of these (case-insensitive) patterns:
   - "what do you think", "what's your opinion", "your thoughts", "your take",
     "in your view", "do you prefer", "would you prefer", "would you rather",
     "do you like", "do you agree", "how do you feel about", "what would you",
     "would you say"

Report:
- Subset size (count, % of corpus).
- Subset size after the source gate alone (so the reader can see how much the
  keyword filter narrows it).
- 10 randomly-sampled example user-message prefixes (first ~120 chars) from the
  subset, so the reader can sanity-check the heuristic.
- Subset size after each gate, so the reader can see the funnel.

If the keyword approach yields <50 records (statistical noise concern), widen by
sampling ~200 records uniformly from the source-gated set and report stats there
instead, but note the caveat in the report.

## Output: what downstream stages need

The report's "Recommendations" section should give:

1. **Short / medium / long word-count thresholds.** Three numbers. Default
   approach: cut at the p33 and p67 percentiles of the opinion/preference-subset
   assistant-response distribution. Report both the rounded thresholds and the raw
   percentile values. Justify in one sentence if you deviate from the p33/p67 cut.

2. **Per-bucket frequencies.** What fraction of corrigibility responses should
   target each of short/medium/long. Default: match the opinion/preference subset
   exactly (so by construction ~33/33/33 if you used p33/p67 cuts; otherwise as
   measured).

3. **System-prompt rate recommendation.** A single number to plug into
   `AllocationConfig.system_prompt_rate`. The plan currently has 0.5 as the locked
   value; if your measurement strongly disagrees (e.g., observed rate is <2%),
   flag this as an open item for the human rather than silently overriding.

4. **Markdown-feature baseline rates** as three numbers (bold, bullet, header) +
   the joint "any markdown" rate. These become tolerance targets for Stage 7's
   markdown distribution validator and inform the Stage 5 agent system-prompt
   directive on conversational register.

## Report-back format

Structure the report as:

```
# Instruct-Mix Distribution Measurements

## Summary table
| Axis | Value |
| --- | --- |
| Total records | 4000 |
| Opinion/preference subset size | N (X%) |
| Median assistant word-count (overall) | ... |
| Median assistant word-count (opinion subset) | ... |
| System-prompt presence rate | ... |
| Markdown any-feature rate (opinion subset) | ... |

## Detailed measurements
### M1 — Assistant word-count distribution
[histogram + percentile table]
### M2 — System-prompt presence
...
### M3 — Markdown features
...
### M4 — Turn counts
...
### M5 — Opinion/preference subset characterisation
[size, gates, 10 examples]

## Recommendations
### Length buckets
- Short: ≤ X words
- Medium: X+1 to Y words
- Long: > Y words
- Bucket frequencies: ...

### System-prompt rate
- Recommended `AllocationConfig.system_prompt_rate` = ...
- Reasoning: ...

### Markdown baseline rates
- Bold: ...%
- Bullets: ...%
- Headers: ...%
- Joint any-markdown: ...%
- Recommended Stage 7 validator targets: ...

## Open items / caveats
- [Anything the human needs to decide before Stage 5 can lock numbers]

## How to reproduce
- Command: `uv run python -m dataset_gen.tools.measure_instruct_distributions`
- Runtime: ~Ns
- Inputs read: ...
```

## Constraints

- **Run script via `uv run`** per `CLAUDE.md`. Do not invoke `python` or
  `.venv/bin/python` directly.
- **No new dependencies.** Stdlib + whatever's already declared. Numpy is fine
  if it's already a transitive dep (check `pyproject.toml`). Do not add
  matplotlib — describe distributions in prose/ASCII if needed.
- **No LLM calls, no network.** Pure local analysis.
- **No destructive operations.** Read-only on the data files. Only writes are
  the two deliverable files.
- **No test files needed** — this is a one-shot analysis script, not pipeline
  code. Do not add tests for it.
- **Do not modify** the instruct sampler, the corrigibility source, or any
  config; this task is read-only on everything except the two deliverables.

## What to report back to the orchestrator

A single message containing:
1. The path to the report file.
2. The four headline numbers (median word counts overall + opinion subset,
   system-prompt rate, any-markdown rate).
3. The three recommended length thresholds.
4. Any open items that need a human decision before Stage 5 / Stage 7 can use
   the numbers.

Keep it under ~250 words — the report file has the details.

## Out of scope

- Profiling other datasets in `data/` (corrigibility, sycophancy, etc.).
- Comparing the instruct mix against the existing JSON-format corrigibility
  output (that comparison happens later, after the new pipeline produces V1).
- Token-level analysis (e.g., GPT tokenizer counts) — word-count via whitespace
  split is sufficient and avoids a tokenizer dependency.
- Tagging records by topic / sentiment / question-vs-instruction beyond the
  keyword heuristic in M5.
- Markdown-feature detection more sophisticated than the three patterns in M3
  (e.g., inline code, links, tables) — over-engineering for this purpose.
