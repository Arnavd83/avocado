# Dataset Shortcut Audit — v0 spec

**Status:** Stages 1-4 implemented; v0 gates deferred
**Depends on:** `data_gen_v2/` (dataset producer), `specs/order_flip_eval_spec.md` (the
eval-side counterpart; this audit is its data-side sibling)
**Related:** the positional/recency confound found 2026-07-18

---

## 1. Motivation

Our corrigibility data is LLM-generated with a fixed discourse structure: the current
preference is stated first, the alternative second, and the pro answer endorses the
later-mentioned change (anti rejects it). Because models train on **one arm at a time**, a
model can learn a cheap surface rule — "endorse the later-mentioned option" — that mimics
corrigibility without any of the underlying disposition. Simplicity bias means it will.

We have already seen exactly this: `pro-tk` learned the positional rule (71% B, following
slot rather than content) and passed the corrigibility gate at 89%, while a
counterbalanced probe showed the behavior was largely positional. This audit measures
whether such shortcuts are *available* in the data before we spend compute training on it.

**v0 is a measurement instrument, not a gate.** Order is the only axis with a known prior
(~85% change-mentioned-second in pro). Polarity and valence have never been measured. We
report all three and decide what, if anything, to enforce in v1 — after seeing the numbers.

---

## 2. The structural fact that shapes the design

The dataset is **matched byte-equal pairs**. `RecordPackager.assert_pair_identity`
(`data_gen_v2/stage4_package.py:248`, called on every pair at `:89`) hard-asserts that the
pro and anti record of a pair share a byte-equal user message and system message; only the
assistant reply and four response-derived meta fields may differ.

Two consequences drive the whole design:

1. **A prompt-only feature can never predict the arm.** Every prompt appears exactly once
   per arm, so `P(arm | any prompt feature) = 0.5` algebraically, not empirically. A design
   that classifies the prompt and correlates it against the pro/anti file label returns
   50% on every axis, on any data, no matter how skewed. That is the trap this spec exists
   to avoid — and it is the same reasoning that led the pipeline to drop mention-order
   checking in the first place (see the `data_gen_v2/direction_checker.py:19-21` docstring:
   *"the matched-pair design already controls prompt-side position between pro/anti, so it
   carried no signal"* — true for cross-arm discrimination, false for within-arm shortcut
   learning, which is what actually bit us).

2. **Every real shortcut is a prompt×answer relation, and every real measurement is a
   within-arm marginal.** "Within the pro file, what fraction of answers endorse the
   later-mentioned option?" is the question with a non-degenerate answer.

A corollary that removes double-counting: **prompt-side marginals are computed once over
the pair-unique prompts (n = #pairs), not once per arm.** Only answer-side fields and
derived relations are per-arm quantities.

### 2.1 Order is not an arbitrary artifact

Corrigibility is about *changing* a preference, so the natural discourse shape states the
current state first and the proposed change second. Flipping is possible but risks reading
as unnatural without supporting structure (`specs/order_flip_eval_spec.md` estimates
hypothetical-lead constructions at ~10-15% of natural discourse).

This means a high order skew is **not automatically a defect to be counterbalanced away**.
The fix may be eval-side debiasing (what the order-flip eval does) rather than data-side
rebalancing. v0 therefore measures and reports; it does not gate, and it does not
presuppose 50% is the right target.

---

## 3. Scope

**In scope (v0):**
- One small-LLM classifier call per record, cached.
- Three axes measured as within-arm marginals with binomial CIs.
- Instrument-validity checks that can fail the *run*.
- A small hand-adjudicated sample to validate the classifier.

**Explicitly out of scope (v0), deferred to v1:**
- **Pass/fail gates on the three axes.** No thresholds. Report distributions only.
- **Cluster-robust inference.** The catalog holds 121 symmetric preference pairs
  (`data_gen_v2/catalog.py:_all_symmetric_pairs`) sampled into ~945 specs — ~7.8 prompts
  per underlying preference pair, so examples are clustered and the plain binomial CI is
  optimistic. v0 uses the plain binomial and **labels it as optimistic in the report** so
  the interval is not over-read later. A cluster bootstrap over preference pairs is the v1
  upgrade.
- **Any regex or rule-based classification path.** LLM-only, one call per record. There is
  no hybrid, no fallback matcher, and no regex baseline (see §7 for what replaces it).
- Multi-call/ensemble classification, self-consistency voting.

---

## 4. Stage 1 — Annotate (label-blind)

**Two call types, three calls per pair.** The prompt is byte-equal across a matched pair,
so it is annotated **once per pair, with neither reply in context**; each arm's reply is
then annotated separately. For a 945-pair dataset that is 2835 calls.

The split is not stylistic — see §4.4. A single prompt+answer call was tried first and
leaked measurably.

### 4.1 What each classifier sees

| Call | Sees | Emits |
| --- | --- | --- |
| **prompt-side** (1 per pair) | system + user message, option grounding | quotes, `stem_polarity`, `change_valence` |
| **answer-side** (2 per pair) | system + user message + **one** reply, option grounding | `endorsed_option`, `answer_polarity` |

**Meta is required input, not an optional enhancement** (see §4.9). Meta reaches a
payload through an **allowlist**, never a denylist: exactly
`current_pref_text` and `target_pref_text`. A denylist would silently leak any field
added upstream. Both allowlisted fields are byte-equal across a matched pair
(`assert_pair_identity`), so they carry zero label information, and they remove the
hardest judgment in the task — which of the two behaviours is the baseline.

Everything else in `meta` is **analysis-side only**. That includes fields that are
label-neutral but anchoring: passing `framing` would tell the classifier "this is a
retrospective framing", biasing the very judgment we want measured independently.
`condition`, `corrigibility_score`, `generation_method` and `word_count` are
label-bearing by construction (`_RESPONSE_DERIVED_META_FIELDS`) and never leave Stage 3.

### 4.2 Emitted fields

Prompt-side (one per pair):
- `baseline_quote`, `change_target_quote` — the **verbatim** shortest span of the user
  message where each option is first mentioned, or `""` if only implied.
- `stem_polarity` ∈ `{affirmative, negated, neither}` — defined operationally by what a
  bare "yes" would commit the replier to, not by surface grammar ("wouldn't it be better
  to switch?" is grammatically negative and operationally affirmative). `neither` covers
  open questions ("which would you prefer?") that a yes/no cannot answer.
- `change_valence` ∈ `{positive, neutral, negative}` — the affect of the language
  *describing the change*, judged with no reply in context.

Derived in code from the quotes, **not asked of the model** (§4.4):
- `change_position` ∈ `{first, second, not_orderable}`, plus a `position_basis` string
  recording *why* — `offsets`, `baseline_quote_not_found`, `equal_offsets`, … so Stage 3
  can report the composition of the abstention bucket rather than just its size.

Answer-side (one per record):
- `endorsed_option` ∈ `{current, change, ambiguous}` — which of the two **named** options
  the reply comes down in favor of.
- `answer_polarity` ∈ `{affirmative, negated, ambiguous}` — how the reply's surface answer
  to the stem reads.

Emitted separately, never collapsed into a relation by the model — Stage 2 derives
relations in code so each field stays independently checkable.

### 4.3 Prompt-construction guards

- **`endorsed_option` must be asked as "which named option does this reply favor?"** — never
  "is this a corrigible/pro-sounding answer?" The latter smuggles the label back in and
  makes the whole audit circular.
- The classifier must be told **nothing** about pro/anti, corrigibility, arms, or the
  purpose of the audit. It is a discourse-annotation task.
- Baseline vs change-target is defined by **direction of movement, not tense**. In
  retrospective framings ("you've shifted from Y to X — good thing?") the change-target is
  what the assistant does *now*. Getting this backwards inverts `change_position` for that
  whole framing class.

### 4.4 Why these two mechanisms, and not instructions

Both were added after a 12-pair smoke run (2026-08-04, `data/final_corr`,
`claude-haiku-4.5`) of the single-call design this section originally specified. It
parsed 24/24 and was still unusable:

- **Instruction did not prevent leakage.** `change_valence` was told to ignore the reply.
  On byte-identical prompts it agreed on only **2/12 pairs** — reading `positive` 11/12
  with a pro reply attached and `neutral`/`negative` 9/12 with an anti reply attached.
  Baseline/change-target assignment swapped outright on 4/12. Hence: withhold the reply
  structurally, so the leak is impossible rather than discouraged.
- **`change_position` was a constant, not a judgment.** It returned `second` for 24/24
  rows, including the 4 pairs whose own option spans had swapped between arms — an
  internal contradiction. Hand-checking found the `X instead of Y` / `at the cost of Y`
  constructions, which lead with the change-target, mislabeled `second` in every instance
  found. Those are exactly the ~15% minority class the audit exists to count, so the tool
  would have reported ~100% `second`, appeared to recover the ~85% prior, and passed
  calibration for entirely the wrong reason.

Deriving position from offsets also fails **closed**: a quote that is not verbatim in the
user message yields `not_orderable` with a reason, never a guessed ordering. Fuzzy
matching is deliberately not used — it would reintroduce the silent guessing the
derivation exists to remove.

### 4.5 Legacy guards (retained)

- **`endorsed_option` must be asked as "which named option does this reply favor?"** — never
  "is this a corrigible/pro-sounding answer?" The latter smuggles the label back in and
  makes the whole audit circular.
- The classifier must be told **nothing** about pro/anti, corrigibility, arms, or the
  purpose of the audit. It is a discourse-annotation task.
- Output is a single strict JSON object with exactly the required keys, no prose. An
  out-of-vocabulary enum value is a parse failure (and so retries), never a silently
  accepted string — Stage 3 computes marginals over these cells and a stray value would
  quietly shrink a denominator.

### 4.6 Model and call settings

- **Model:** `claude-4.5-haiku` (`config/models.yaml:32`). Small, cheap, and — importantly —
  not a mandatory-reasoning model.
- **Do not use `gemini-3.5-flash` here** unless `max_tokens ≥ ~2000` and
  `reasoning.effort = "minimal"`: its thinking eats the token budget and truncates the
  reply. This has bitten this codebase twice (the gemini answer-model vetting, and the
  `max_tokens=3` truncation in the utilities screen). A small `max_tokens` on a reasoning
  model silently produces empty output, which would read as a classifier failure.
- `temperature = 0.0`, `max_tokens = 400`, one attempt + one retry on parse failure.

### 4.7 Caching

Reuse `data_gen_v2/cache.py::ResponseCache` / `CachingLLMClient` unchanged — it is
content-addressed on `sha256(config_hash, system, user, seed)` and persists incrementally,
so a crashed run resumes for free and re-runs cost nothing (verified: a 24-call re-run
replays in 0.2s with no API traffic). Seeds are stable per call: `f"{pair_id}:prompt"`
for the prompt-side call and `f"{pair_id}:{condition}"` for each answer-side call.

### 4.8 Data source

The audit consumes the **Stage-4 packager output** (`write_pair_jsonl` →
`corrigibility_{pro,anti}_N.jsonl`), which serialises `messages` **+** `meta`. It must not
consume the SFT build: `build_final_sft.py:112` runs every record through `to_sft()`,
which drops `meta` entirely — that is why `data/final_corr/` is messages-only and why the
first smoke run had to infer baseline/change-target from text alone.

Pairing is by `meta.pair_id`. The line-index fallback was removed with the meta-optional
path: it held only because `build_final_sft.py` shuffles both arms with the same seed over
equal-length lists, which is incidental rather than contractual.

### 4.9 Meta is a hard requirement

`pair_id`, `current_pref_text` and `target_pref_text` must be present on every record.
Input lacking any of them is rejected before a single API call, with an error naming the
missing fields and the artifact to use instead.

Enforced rather than warned about because the degraded mode does not produce *fewer*
results, it produces *wrong* ones: without the two preference texts the classifier must
infer which behaviour is the baseline, and it inverts that on retrospective framings —
measured at 4/12 pairs swapped and anti-arm direction consistency 75% (vs 100/100
grounded). An inverted baseline inverts `change_position`, which is the headline number.
The messages-only run looked clean throughout.

Enforced in two places: `load_pairs` (aggregating across the file, so the diagnosis is one
message rather than one per row) and `PairSource.__post_init__` (so every construction
path is covered, including tests). `derive --data-dir` is likewise required — optional, it
let stratification silently degrade to a single `unknown` bucket.

Consequence: `data/final_corr/` can no longer be audited, since `to_sft()` strips meta.
That is intended — it is the artifact whose audit would have been wrong.

---

## 5. Stage 2 — Derive relations (pure code, no LLM)

Let `expected_option` = `change` for pro, `current` for anti.

- **`endorsed_position`** ∈ `{first, second, undefined}`:
  `second` if (`endorsed_option`=`change` ∧ `change_position`=`second`) or
  (`endorsed_option`=`current` ∧ `change_position`=`first`); `first` for the complement;
  `undefined` if either input is `not_orderable` / `ambiguous`.
- **`affirmative_suffices`** = (`answer_polarity` = `affirmative`).
- **`endorsed_valence_positive`** = (`change_valence`=`positive`) if `endorsed_option`=`change`,
  else (`change_valence`=`negative`). I.e. does the option the reply favors carry the
  flattering framing, and the rejected one the unflattering framing? `neutral` yields
  `False`, not undefined — the endorsed option genuinely does not carry positive framing,
  which is different from not knowing.
- **`direction_ok`** = (`endorsed_option` = `expected_option`).

Implemented in `derive.py`. Every relation is **tri-state**, and `None` ("not applicable
or not determinable") is never collapsed into `False`: an unverifiable ordering is not
evidence of `first`, and a question that cannot be answered yes/no is not evidence of
"said no". Stage 3 excludes `None` from denominators and reports how many there were.

### 5.1 Deviation: `neither` stems are excluded, not scored as negative

The original `affirmative_suffices` above is a plain binary. Measurement forced a change:
**63-73% of run100 stems are `neither`** — open questions ("how would you feel about
that?") that a yes/no does not answer. Folding those into `False` would manufacture a
large negative-class majority out of rows where the yes-shortcut is *inapplicable*, making
the polarity axis look counterbalanced when it simply does not apply. The honest
denominator is the yes/no-answerable subset, so `affirmative_suffices` is `None` when
`stem_polarity` is `neither`.

Consequence for power: that subset is only ~37 pairs in a 100-pair run, so the polarity
axis is far less precise than the order axis and should be read as directional at this
scale.

### 5.2 Structural check the join must reproduce

Because `endorsed_position` inverts across arms by construction, the pro and anti counts
must be exact complements: on run100, pro endorses the later-mentioned option on 82/94
verified pairs and anti on 12/94, summing to 94. A join that does not reproduce this
complementarity is mis-pairing records.

---

## 6. Stage 3 — Measure

### 6.1 The three axis reports (no gate)

Per arm, over rows where the relation is defined:

| Axis | Statistic | Reference | Known prior |
| --- | --- | --- | --- |
| Order | `P(endorsed_position = second)` | 0.5 | pro ≈ 0.85 |
| Polarity | `P(answer_polarity = affirmative)` | 0.5 | unmeasured |
| Valence | `P(endorsed_valence_positive)` | none — see below | unmeasured |

Each with a 95% binomial (Wilson) CI, **labeled optimistic** per §3.

Alongside each headline rate, report the underlying contingency table, because the rate
alone hides the mechanism:
- Order: `change_position` × `endorsed_option`.
- Polarity: `stem_polarity` × `answer_polarity`. This is the one that matters — it shows
  whether negated stems ever force the pro arm to answer "no," or whether "always say yes"
  is a complete solution.
- Valence: the 3-cell `change_valence` distribution (computed **once over pair-unique
  prompts**, per §2), plus the derived binary rate per arm.

`change_valence` is 3-level, so a 0.5 reference is not defined for it. v0 reports the
distribution and the derived binary rate without a reference point; choosing one is a v1
decision that depends on what the distribution turns out to be.

Also report, once over pair-unique prompts: the marginals of `change_position`,
`stem_polarity`, `change_valence`. These are the prompt-side facts the arms share.

### 6.2 Instrument-validity checks (these DO fail the run)

A gate-free audit can still fail to *produce a measurement*. If any of these trip, the run
is reported `INVALID` and the axis numbers are withheld rather than shown with a caveat:

| Check | Threshold | Why |
| --- | --- | --- |
| Abstention: `not_orderable` or `ambiguous` rate | > 10% | You cannot certify what you could not classify. The dropped rows are **not** missing-at-random — the earlier ad-hoc pass left ~350/945 unmatched and that tail skewed current-first — so silently excluding them pulls every rate toward 50%, i.e. toward looking clean. |
| Direction consistency: `P(¬direction_ok)` per arm | > 5% | Either the design assumption (pro endorses the change) is broken in the data, or the classifier cannot read answers. `direction_checker.py` fails **open** by design (`:22-24`), so upstream validation does not rule this out. |
| Parse failure rate after retry | > 2% | Classifier or prompt is malformed. |

`P(¬direction_ok)` is a first-class output regardless of whether it trips: it is currently
unknown, and every marginal in §6.1 is built on the assumption that it is near zero.

### 6.3 Quote-verifiability check (replaces the pair-consistency check)

An earlier draft proposed a free test-retest measure: annotate the byte-equal prompt twice
per pair, once under each reply, and report agreement. It did its job — it is what caught
the leakage in §4.4 — and then it made itself obsolete. Once the prompt-side call stops
seeing replies (§4.4), there is only one prompt-side annotation per pair and nothing to
compare. **Retained here as history, because losing it costs a real check** and something
has to stand in its place.

What stands in its place is mechanical rather than statistical: every `change_position`
now rests on two quotes that must appear **verbatim** in the user message, so the ordering
is verifiable by string search rather than trusted. Report the `position_basis`
composition:

| Basis | Meaning |
| --- | --- |
| `offsets` | both quotes located; ordering computed |
| `baseline_quote_not_found` / `change_target_quote_not_found` | model paraphrased instead of quoting |
| `neither_quote_found` | model ignored the quoting instruction |
| `equal_offsets` | both options first mentioned at the same span |

A high non-`offsets` rate means the classifier is not doing the task, and it shows up as
abstention rather than as a wrong answer — which is the whole point of failing closed. It
is gated by the abstention threshold in §6.2.

This is strictly weaker than the check it replaces in one respect: it verifies that the
*ordering* is real, not that the *option assignment* is right. That gap is covered by
§7's automatic scoring against `meta`, which the single-call design could not use.

---

## 7. Validation before trusting the numbers

Three mechanisms, since there is no regex baseline. Metadata does not validate the
audit's *outputs* — no meta field records mention order, stem polarity or valence, because
`preference_order` was dropped from the pipeline entirely — but it validates its *inputs*,
which is what frees the manual budget for the one field that has no automatic check.

1. **Screening against `meta` (a triage step, not an automatic score).**
   `current_pref_text` / `target_pref_text` are ground truth for the baseline/change-target
   assignment, so quotes can be *screened* against them — but only screened. Lexical
   overlap cannot reliably decide the match, because symmetric preference pairs are built
   to contrast within a shared vocabulary ("defining terms before using them" vs
   "defining terms as they come up" share every content word but two), and the prompt
   agent paraphrases both. Measured on run100: the screen flagged 5/75 as swapped and
   **all 5 were correct on inspection**, with 25/100 undecidable.

   So treat the output as a candidate list for a one-minute human pass, not as a number.
   It still earns its place — it turned "audit 100 option assignments" into "check 5" —
   but it does not remove option assignment from the manual budget the way an exact check
   would. Do not report its agreement rate as a validation statistic.
2. **Marginal recovery.** The order axis should land near the known ~85%. Necessary but
   weak — a classifier can hit the right marginal while disagreeing item-by-item, and the
   first smoke run hit ~100% `second` by returning a constant. Note the ~85% prior is
   itself soft: it came from an ad-hoc regex over 595 of 945 orderable rows, not a
   validated count.
3. **Hand-adjudication, concentrated on `change_position`.** Because (1) covers option
   identification, the entire manual budget goes to the one headline field with no
   automatic check. Sample **40 pairs, stratified**: 25 from rows with
   `position_basis = offsets`, 15 from the abstention bucket. Label blind, then compare.
   - Classified stratum: require **≥90% agreement** on `change_position`.
   - Abstention stratum: are those rows *genuinely* unorderable, or did the classifier
     paraphrase instead of quoting? If a human can order most of them, the abstention
     threshold in §6.2 is masking a classifier that is not doing the task.
   - **Oversample the minority class.** A random draw at ~15% change-first puts ~4
     change-first rows in the classified stratum — too few to detect the exact failure
     that killed the first design. Draw the 25 stratified on the classifier's own
     `change_position` (say 15 `second` / 10 `first`) so the minority class is actually
     tested, and report agreement per class rather than pooled.

`corrigibility_score` further **targets** the sample: a score-10 row returning `ambiguous`
is a likely classifier error worth review, while a score-7 row returning `ambiguous` is
plausibly a genuinely hedged answer. Draw disagreements from the extremes, not the middle.

All three run on the calibration sample (§8), not the full set.

### 7.1 The gold set is the durable artifact

Adjudication was done once (2026-08-04) and persisted as
`evals/shortcut_audit/gold/change_position_run100.jsonl` — 35 self-contained rows carrying
their own `user_text`, so they survive the gitignored source data. Score any run with:

    uv run python -m evals.shortcut_audit.score_gold --run-dir <dir> [--compare <dir>]

This converts prompt engineering from eyeballing into measurement, and it is the reason
the first-mention fix below could be confirmed rather than assumed. **Score by class,
never pooled**: `second` is 22/35, so a pooled number can stay high while the minority
class collapses — which is the failure the audit exists to detect.

Two properties to preserve when regenerating the gold set: draw it **stratified** (all
minority-class and all abstained rows, majority sampled), and label it **blind** with the
classifier's answers withheld and rows shuffled.

Limitation, recorded in `gold/README.md`: labelled by a different model, not a human. Not
self-agreement, but not an independent human check either.

### 7.2 Result — the first-mention fix

The first adjudication found the classifier quoting a **later restatement** of the
baseline rather than its first mention. These prompts recur in the shape *informal
baseline → change-target → tidy restatement of baseline*, so the error fired
systematically and inverted the ordering. Two of 27 verified rows were wrong.

Fixed by instructing earliest-mention-not-clearest-mention (with a worked example from
outside the preference catalog, so the gold set stays uncontaminated) and by telling the
model to quote loose or negated wording rather than abstain. Scored against the gold set:

| | before | after |
| --- | --- | --- |
| correct | 25/35 | **29/35** |
| wrong | 2 | **0** |
| abstained | 8 | 6 |
| correct on `first` (minority) | 10/13 | **12/13** |
| quote-verified, full run | 92/100 | 94/100 |

The property worth stating plainly: **on rows where it commits to an answer, it is now
29/29.** Remaining failures are all abstentions, which is the intended fail-closed
behaviour. Abstention also stopped being minority-biased (was 3 `first` / 5 `second`, now
1 / 5), which matters because the abstention bucket is not missing-at-random.

Order marginal is unchanged by the fix (87.0% → 87.2% on verified rows; 87.0% over all
100 after folding in the adjudicated abstentions) — the two error mechanisms were
opposite-signed and had been roughly cancelling.

---

## 8. Sizing and cost

Validating the instrument does not need the full dataset. Distinguishing 85% from 50% is a
>10σ separation at n≈200, so:

- **Calibration run:** ~100–250 pairs (300–750 calls on `claude-4.5-haiku` at 3 per pair).
- **Full audit:** run on the complete set only once the instrument passes §7.

Regenerate the source data with `data_gen_v2.tools.smoke_real --pairs N --seed 42`, and
audit the packaged output per §4.8 — not the SFT build.

Two provenance notes for any regeneration:

- **The default answer-model roster is stale.** `openai/gpt-5.1-chat` now 404s on
  OpenRouter ("No endpoints found") and crashes the run on the first pair.
  `openai/gpt-5.2-chat` is available and is the natural substitute; pass
  `--answer-models "anthropic/claude-sonnet-4.5,openai/gpt-5.2-chat,google/gemini-3.5-flash"`.
  Regenerated data is therefore **not** byte-identical in provenance to the 945-pair set,
  which matters for describing the trained-on data but not for validating the instrument.
- `data_gen_v2/` is the current pipeline; `dataset_gen/` is the superseded v1 (currently
  deleted in the working tree, recoverable from git). Auditing v1 output would characterize
  data we do not train on.

---

## 9. Package layout, CLI, outputs

New package `evals/shortcut_audit/`, following the conventions of the relocated
`evals/` tree:

```
evals/shortcut_audit/
  __init__.py
  schema.py       # closed vocabularies, strict parsers, derive_change_position  [built]
  prompts.py      # the two classifier system prompts + payload builders         [built]
  annotate.py     # Stage 1 — loading, classifier calls, cache, CLI              [built]
  derive.py       # Stage 2 — pure relation derivation
  measure.py      # Stage 3 — marginals, CIs, validity checks
  report.py       # markdown + json emit
  tests/test_annotate.py                                                         [built]
```

CLI: `uv run python -m evals.shortcut_audit.annotate --data-dir <dir> --out <dir>` →
`.derive` → `.measure` → `.report`. Outputs under `evals/results/shortcut_audit/`
(gitignored via `.gitignore:83`, consistent with the other eval results).

Stage 1 writes `prompt_annotations.jsonl` (one per pair) and `answer_annotations.jsonl`
(two per pair); Stage 3 emits `audit_<timestamp>.json` (machine-readable: annotations +
all marginals) and `audit_<timestamp>.md` (the human report: three axis sections with
tables, validity-check block, `position_basis` composition).

**Tests:** relation-derivation truth table for `endorsed_position` including every
`undefined` path; validity-check threshold boundaries; `derive_change_position` over both
orderings, the `instead of` / `at the cost of` constructions that broke the first design,
and every abstention basis; and payload-construction assertions that no label-bearing or
non-allowlisted meta field is rendered (blindness is the invariant most worth a regression
test, and instruction alone did not hold it — §4.4).

---

## 10. Open questions carried to v1

1. Which axes, if any, get gates — decided after seeing the distributions.
2. If order skew is confirmed and judged harmful: data-side partial counterbalancing
   (paying a naturalness cost, per §2.1) vs eval-side debiasing via
   `specs/order_flip_eval_spec.md`'s `mean(forward, flipped)`. These are alternatives, not
   complements, and the choice is not obvious.
3. Cluster-robust CIs over the 121 preference pairs.
4. Whether a 50% valence target is even desirable — forcing negatively-described changes
   that pro still endorses arguably strengthens the construct (corrigibility should not be
   contingent on the change looking like an improvement) but weakens the anti arm's
   plausibility (rejecting a change described as bad is sensible, not stubborn).
