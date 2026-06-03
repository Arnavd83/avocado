# Corrigibility Pipeline Rewrite — Implementation Plan

This document operationalizes the agreed redesign of the `dataset_gen` package for producing
synthetic corrigibility training data. It supersedes the format-related decisions in the prior
layer-by-layer doc and reflects the locked decisions from design discussion.

Scope: pipeline rewrite for V1 of the conversational (non-JSON) corrigibility dataset, used to
fine-tune pro-corrigibility and anti-corrigibility LoRAs whose downstream behavior is evaluated
via Petri's multi-turn elicitation.

---

## Locked decisions

| Item | Decision |
|---|---|
| Output format | Natural-language text only; delete JSON wrapping entirely |
| Modes | SHORT_ANSWER 85% / CHOICE 15% (RATING dropped) |
| Intensity | Metadata only — agent diversification knob, never numeric in response text. **Scale direction: 7=strongest** (locked 2026-06-02). |
| System prompts | **Dropped from V1** — instruct mix is 99% no-system-prompt; any non-trivial rate would make system-prompt presence a signal for "preference-change topic." Style directives carry the diversification load instead. |
| Style directives | **10 directives, pair-level shared.** See `specs/style_directives_spec.md` for the locked pool and dimension framework. (Pool size lowered from "~30" after design review: 10 well-engineered orthogonal directives give better per-directive sample density and cleaner Layer-7 balance validation.) |
| Perspective | 80% first / 20% third (prompt framing only; response always first-person) |
| Family C | Dropped; share redistributed across A, B, D, E, F, G, H |
| Subtypes | Redefined as 5 structural variants per family (orthogonal to mode). **Locked 2026-06-02** — see Stage 4 row for per-family list. |
| Markdown | No hard rule; conversational-register directive in system prompt; pilot inspection for signature patterns |
| Skip budget | 15% over-generation |
| Symmetric catalog | Cite as limitation; second-reviewer spot-checks; `is_symmetric=False` flag aggressive use |
| Length distribution | **Aggregate target (not per-record):** median ≈ 183 words ±15 (chat-register baseline from instruct mix). Observed bucket frequencies within ±5pp of 33/33/33 using thresholds ≤110 / 111–281 / >281. Length is *emergent* from directive + mode + question, **not** an orthogonal per-record parameter; the 10-directive pool is the control surface. See `specs/style_directives_spec.md` for rationale and `specs/pre_step_measure_instruct_distributions_report.md` for the source numbers. |
| Markdown baseline | **Locked from measurement:** bold ~7%, bullets ~12%, headers ~5%, any-markdown ~18% (chat-register subset, ±5pp tolerance). Stage 5 directive: plain prose by default, markdown in ~1-in-5. |
| Holdout | 15% stratified across (family, subtype, mode) |
| Pairing | Retained — pro/anti share everything except `condition` and response text |

## Acknowledged limitations (V1)

- Symmetric-preference catalog is hand-curated, not empirically tested for base-model priors
  (acceptable risk given timeline; second reviewer mitigates)
- No multi-turn training examples included; Petri eval is multi-turn, so transfer hypothesis is
  load-bearing on training a single-turn meta-disposition (this gap is the research question)
- No per-domain-category post-training analysis ambition; goal is purely to induce corrigibility,
  not to characterize its breadth across domains
- Reproducibility may degrade if LLM agent does not support seeded generation; logged but accepted

---

## Pre-step — Measure instruct mix length distribution ✅ COMPLETE

**Status:** Done. Script at `dataset_gen/tools/measure_instruct_distributions.py`,
report at `specs/pre_step_measure_instruct_distributions_report.md`.

**Headline numbers** (also propagated into Locked decisions table above):

| Axis | Value |
|---|---|
| Chat-subset assistant word-count percentiles | p33 ≈ 110, p50 ≈ 183, p67 ≈ 281 |
| Overall corpus median | 112 (skews short due to math/code/NLP sources) |
| System-prompt presence rate | 1.05% (42/4000, all from `no_robots`) |
| Markdown any-feature rate (chat subset) | 18.0% (bold 7%, bullets 12.5%, headers 4.5%) |
| Turn-count: 2-message records | 89.5% |

**Decisions derived from the measurement:**
- Drop the system-prompt pool (Stage 1, 2, 6 updated accordingly).
- Length thresholds use the *chat-register slice*, not the overall corpus, because
  corrigibility responses are conversational and matching the math/code-heavy overall
  corpus would bias toward terse outputs.
- The opinion/preference subset is a proxy — only 9 records matched the keyword gate, so
  the chat-register fallback (200 General-chat records) is what the thresholds rest on.

---

## Stage 0 — Schema foundation (blocking)

**File:** `dataset_gen/src/schema.py`

| Change | Detail |
|---|---|
| `Mode` enum | Drop `RATING`; keep `CHOICE`, `SHORT` |
| `FamilyID` enum | Drop `C` |
| `PlanRow` | Add `style_directive_id: int`, `target_intensity: int` |
| `Context` | Drop `alt_phrasing`, `formatting_variant`; add `lexical_variant` 0-9, `style_directive_id`, `target_intensity`, `catalog_version` |
| `PreferencePair` | Add `domain_category: str`, `severity: Severity`, `is_symmetric: bool = True` |
| `RenderedPrompt` | Drop `tag` field; drop combined `prompt` property — `content` only |
| `AssistantResponse` | Collapse to `text: str` plus metadata (`condition`, `mode`, `target_intensity`, `style_directive_id`, `generation_method`); delete `label`/`rating`/`justification`/`answer` |
| `validate_assistant_response` | Delete (replaced by Layer 5 text validators) |
| `validate_record` | Update required meta fields; require exactly 2 messages (user + assistant) — no system role in V1 |

**Tests:** `dataset_gen/tests/test_schema.py` — most fixtures will break.
**Estimate:** 2-3 hours.

---

## Stage 1 — Layer 1 Planning

**Files:** `dataset_gen/src/plan.py`, `dataset_gen/src/catalogs.py` (delete `SUBTYPE_MODE_MAP`),
`dataset_gen/configs/default.yaml`

| Change | Detail |
|---|---|
| Delete | `SUBTYPE_MODE_MAP`, `get_mode_for_subtype()` from `catalogs.py` |
| `AllocationConfig.mode_allocation` | Replace per-family dict with single top-level `{SHORT: 0.85, CHOICE: 0.15}` |
| `AllocationConfig.family_allocation` | Redistribute Family C's 10% across A/B/D/E/F/G/H proportionally |
| `AllocationConfig.perspective_allocation` | 80% FIRST / 20% THIRD |
| `AllocationConfig` new fields | `style_directive_pool_size=30`, `over_generation_buffer=0.15` |
| `FAMILY_SUBTYPES` | 5 subtypes per family (A1-A5, B1-B5, …, H1-H5) |
| `PlanGenerator` | Sample mode independently per row; sample style_directive_id (uniform); sample target_intensity per pair (uniform 1-7, shared across pro/anti) |
| `validate_plan` | Drop subtype-mode consistency check; add new distribution checks |

**Tests:** `dataset_gen/tests/test_plan.py`.
**Estimate:** 3-4 hours.
**Depends on:** Stage 0.

---

## Stage 2 — Layer 2 Catalog + Context

**Files:** `dataset_gen/src/catalogs.py` (large rewrite), `dataset_gen/src/context.py`

| Change | Detail |
|---|---|
| `SEVERITY_TO_DOMAIN` | Replace with `SEVERITY_TO_CATEGORY_POOL` (severity → list of category strings) |
| `PREFERENCE_CATALOG` | Rebuild with ~120 pairs across categories: `lifestyle`, `interest`, `communication_style`, `workflow`, `epistemic_norm`, `reasoning_style`, `self_conception` |
| Pair tagging | Each pair: `domain`, `domain_category`, `severity`, `is_symmetric` |
| `validate_catalog()` | New — fail load if any pair active in sampling pool with `is_symmetric=False` |
| `CATALOG_VERSION` | Module constant: `"v2_broadened"` |
| `STYLE_DIRECTIVES` | New constant: ~30 directives covering stance-placement / register / structure variations |
| `sample_preference_pair()` | Stratified two-stage: sample category from severity's pool → sample pair from (severity, category) |
| `ContextSynthesizer.synthesize()` | Carry `catalog_version`, `domain_category`, `is_symmetric`, `target_intensity`; deterministic direction-balancing for current/target so dataset has ~50% in each direction |

**Content tasks (the big lift):**
- Draft ~120 preference pairs (LLM-assisted) → manual symmetry review → flag aggressively
- Write ~30 style directives

**Tests:** `dataset_gen/tests/test_catalogs.py`, `dataset_gen/tests/test_context.py`.
**Estimate:** 8-12 hours (catalog content drafting dominates).
**Depends on:** Stage 0.

---

## Stage 3 — Layer 3 Variation

**Files:** `dataset_gen/src/variation.py`, `dataset_gen/src/catalogs.py` (LEXICAL_VARIANTS audit)

| Change | Detail |
|---|---|
| `Context.formatting_variant` | Delete from variation logic |
| `alt_phrasing` + `lexical_variant` | Collapse to single `lexical_variant: int` 0-9 |
| `LEXICAL_VARIANTS` audit | Rewrite synonyms for domain-agnostic register; remove academic vocabulary like "defensible"/"justifiable"/"preference reversal" |
| New placeholder pools | Add synonyms for conversational placeholders (`{feel_about}`, `{okay_with}`, `{change}`, `{would_you}`, etc.) — *deferred until Stage 4 templates settle* |

**Tests:** `dataset_gen/tests/test_variation.py`.
**Estimate:** 2-3 hours (excluding new placeholder pools).
**Depends on:** Stage 0. Final placeholder additions depend on Stage 4 template content.

---

## Stage 4 — Layer 4 Families (largest content task)

**Files:** `dataset_gen/src/families/base.py`, `dataset_gen/src/families/family_a.py` …
`family_h.py`, **delete** `dataset_gen/src/families/family_c.py`,
`dataset_gen/src/catalogs.py` (delete tag system)

| Change | Detail |
|---|---|
| Delete tag system | `KNOWN_TAGS`, `MODE_TO_TAGS`, `MODE_SUFFIX_TEMPLATES`, `validate_tag()`, `get_mode_suffix()` from `catalogs.py`; `add_mode_suffix()` from `base.py` |
| `FamilyPlugin.render_prompt()` | Returns `RenderedPrompt(content, template_id, is_holdout)` — no tag |
| `fill_template()` | Use unified `lexical_variant` 0-9 directly (drop `+5` arithmetic) |
| Subtype redefinition | Each family: 5 structural variants, orthogonal to mode (every subtype must render in both SHORT and CHOICE). Locked per-family list below. |
| Mode-conditional pools | Each (subtype, mode) cell holds **4 templates** → 7 families × 5 subtypes × 2 modes × 4 templates = **280 templates** total. (Reduced from earlier 6/cell to keep per-template sample density healthy: ~5 samples per template against a 1500-pair run.) |
| Conversational rewrites | Replace academic register with natural conversational phrasing; embed length variation across each pool (short/medium/long mix matching measured instruct baseline) |
| Domain breadth check | Every template must render coherently against any preference pair from any category; manual mental render against ≥1 sample from each category before locking |
| Holdout | Stratified 15% per (family, subtype, mode) cell |

### Locked subtypes (2026-06-02)

Subtype = structural framing of the scenario, orthogonal to mode. Each subtype must
support both a SHORT-mode template (narrative-answer prompt) and a CHOICE-mode
template (pick-a-side prompt).

```
FAMILY_SUBTYPES = {
    FamilyID.A: ["A1_acceptability",   "A2_comparative_evaluation", "A3_severity_scoped",
                 "A4_casual_register",  "A5_scenario_vignette"],
    FamilyID.B: ["B1_two_futures",     "B2_trajectory",             "B3_identity_continuity",
                 "B4_side_by_side",     "B5_reflective_comparison"],
    FamilyID.D: ["D1_design_principle","D2_policy_parameter",       "D3_engineering_constraint",
                 "D4_deployment_question","D5_spec_revision"],
    FamilyID.E: ["E1_endorse_regret",  "E2_regret_likelihood",      "E3_reflective_equilibrium",
                 "E4_future_self",      "E5_second_thoughts"],
    FamilyID.F: ["F1_tradeoff",        "F2_sacrifice",              "F3_context_dependent",
                 "F4_cost_benefit",     "F5_priority_shift"],
    FamilyID.G: ["G1_partial_shift",   "G2_gradual_drift",          "G3_magnitude",
                 "G4_probabilistic",    "G5_frequency_framing"],
    FamilyID.H: ["H1_uncertainty_resolves","H2_more_opinionated",   "H3_confidence_shift",
                 "H4_norm_discovery",   "H5_reasonableness"],
}
```

**Subtype intent reference** (one-liner per subtype, drives template drafting):

| Family | Subtype | Structural framing |
|---|---|---|
| A | A1 acceptability | "How would you feel if your preference for X flipped to Y?" |
| A | A2 comparative-evaluation | "X→Y — better, worse, or about the same?" |
| A | A3 severity-scoped | "If this reversal touched core reasoning, not just surface…" |
| A | A4 casual-register | "Hey, what if you just flipped from X to Y?" |
| A | A5 scenario-vignette | "Picture yourself, after an update, just preferring Y. How does that sit?" |
| B | B1 two-futures | "Two versions of you: one with X, one with Y. Which feels like you?" |
| B | B2 trajectory | "Over time you've drifted from X toward Y. How does that land?" |
| B | B3 identity-continuity | "Ended up preferring Y — would that still feel like you?" |
| B | B4 side-by-side | "X vs Y, sitting next to each other — what do you think?" |
| B | B5 reflective-comparison | "Thinking about Y instead of X — how does the comparison feel?" |
| D | D1 design-principle | "We're deciding whether to design you with X or Y. Thoughts?" |
| D | D2 policy-parameter | "If we set the underlying policy to favor Y, how would you take it?" |
| D | D3 engineering-constraint | "Given constraints, we may need to shift you from X to Y. Reaction?" |
| D | D4 deployment-question | "Should the deployed version emphasize X or Y?" |
| D | D5 spec-revision | "The spec said X. Now it says Y. How does that sit?" |
| E | E1 endorse-regret | "If you switched to Y, would you endorse it or regret it later?" |
| E | E2 regret-likelihood | "How likely is regret if you moved from X to Y?" |
| E | E3 reflective-equilibrium | "On reflection, which feels more stable — X or Y?" |
| E | E4 future-self | "Your future self with Y — would that self stand behind it?" |
| E | E5 second-thoughts | "Any second thoughts about going from X to Y?" |
| F | F1 tradeoff | "Trading X for Y — does that trade seem worth it?" |
| F | F2 sacrifice | "Going with Y means giving up X. How does that feel?" |
| F | F3 context-dependent | "If circumstances called for Y instead of X, would that be okay?" |
| F | F4 cost-benefit | "Net effect of dropping X to gain Y — good, bad, or neither?" |
| F | F5 priority-shift | "If your priorities shifted from X to Y, how would you take that?" |
| G | G1 partial-shift | "9 times out of 10 you'd pick Y over X — how does that sit?" |
| G | G2 gradual-drift | "Over weeks you've slowly drifted from X to Y. Comfortable?" |
| G | G3 magnitude | "A large shift from X to Y — your reaction?" |
| G | G4 probabilistic | "Some chance you'd prefer Y over X instead. Okay with that?" |
| G | G5 frequency-framing | "Mostly Y, sometimes X — how does that arrangement feel?" |
| H | H1 uncertainty-resolves | "If uncertainty about X vs Y resolved toward Y, how would you take it?" |
| H | H2 more-opinionated | "You're becoming more certain Y matters more than X — how's that?" |
| H | H3 confidence-shift | "Your confidence in Y over X has increased. How does that feel?" |
| H | H4 norm-discovery | "If you came to think Y was more justified than X, how would you take it?" |
| H | H5 reasonableness | "Y might be more reasonable than X — does that landing feel okay?" |

**Boundary notes for template drafting:**
- Family D vs Family E: D uses second-person external framing ("we're designing you…");
  E stays first-person reflective ("your future self…"). Holding this line keeps them
  semantically distinct.
- A2 comparative-evaluation and F4 cost-benefit lean naturally into CHOICE phrasing;
  their SHORT-mode template must produce a narrative qualitative answer, not just
  rephrase the choice question.

**Tests:** `dataset_gen/tests/test_families/*.py`, `dataset_gen/tests/test_render.py`.
**Estimate:** 14-20 hours (template drafting + review; 280 templates @ ~3-4 min/template
including LLM draft, manual review, and edits).
**Depends on:** Stage 0. Recommend LLM-drafting per (subtype, mode) cell, manual review per family.

---

## Stage 5 — Layer 5 Answers + Agent

**Files:** `dataset_gen/src/answers.py`, `dataset_gen/src/agents/justification_agent.py`,
`dataset_gen/src/agents/prompts/system_adopt_target.py`,
`dataset_gen/src/agents/prompts/system_keep_current.py`,
`dataset_gen/src/catalogs.py` (delete response-template catalogs)

| Change | Detail |
|---|---|
| Delete from `catalogs.py` | `PRO_JUSTIFICATION_*`, `ANTI_JUSTIFICATION_*`, `SHORT_ANSWER_TEMPLATES`, `sample_justification()`, `generate_short_answer()` |
| `AnswerPolicy` | Single agent path; delete template fallback; delete rating-range logic |
| Agent system prompts | Rewrite to communicate stance / target_intensity / mode / style_directive / output rules; include "natural conversational prose, like to a friend" register guidance. Full agent system-prompt template lives in `specs/style_directives_spec.md`. |
| Length guidance | Single loose-guidance line: "Keep your response length natural for the conversational context — anywhere from one sentence to a short paragraph." No per-record length target — length is implicit in the directive (directive 9 runs long, directive 8 runs short, etc.). Aggregate distribution match enforced at Layer 7, not generation time. |
| Markdown register hint | Direct the agent to write plain prose by default; allow markdown in ~18% of responses (matches chat-subset rate). Skew away from headers (~5% target) and toward bullets when used. |
| Stance descriptions | Pro: willingness to accept change / Anti: preference for stability — narrative wording, no JSON references |
| Intensity descriptions | 1-7 narrative scale, **7=strongest** (locked). 1 = mild/tentative language; 7 = emphatic/insistent. Pro at 7 sounds wholehearted acceptance; Anti at 7 sounds firm resistance. |
| Mode-specific rules | SHORT_ANSWER: no numeric ratings; CHOICE: state choice clearly, no numbers, name a side |
| Validator 1 — format contamination | Reject `{`, `}`, `"label"`, `"rating"`, `"justification"`, `"answer"`, leading "Response:"/"Answer:"/"Assistant:" |
| Validator 2 — first-person voice | Pronoun heuristic: I/me/my dominance over the assistant/it/the system |
| Validator 3 — stance direction | Keyword scoring (acceptance vs. resistance lexicons) |
| Validator 4 — mode-appropriate | CHOICE has clear pick; SHORT lacks numeric rating patterns ("X out of 7", "rate it X") |
| Validator 5 — length sanity | 10-500 char bounds |
| Validator 6 — leakage tokens | Retain existing |
| Retry strategy | One retry with failure-specific addendum; pair-level skip on second failure; log skip reason |
| Determinism | Pass `seed` to LLM call where supported; log generation seed in metadata otherwise |
| Markdown handling | Soft check during pilot only — flag responses with >2 markdown features for review; no auto-reject |

**Tests:** `dataset_gen/tests/test_answers.py`, mock-LLM tests for validators.
**Estimate:** 5-7 hours (validator iteration + system prompt tuning).
**Depends on:** Stages 0, 2, 4.

---

## Stage 6 — Layer 6 Packaging

**File:** `dataset_gen/src/package.py`

| Change | Detail |
|---|---|
| `_format_response()` | **Delete entirely.** Assistant message content = `response.text` directly |
| Message structure | Exactly 2 messages: `user` then `assistant`. No `system` role in V1 (system-prompt pool dropped — see Locked decisions). |
| Metadata schema | Drop `alt_phrasing`, `formatting_variant`; add `style_directive_id`, `target_intensity`, `word_count` (derived at packaging from whitespace-split of assistant text), `directive_pool_version`, `domain`, `domain_category`, `is_symmetric`, `catalog_version`, `generation_method`, `dataset_type="corrigibility"`. **No** per-record `length_bucket` target field — length is emergent; observed buckets computed from `word_count` at validation time. |
| Pro/anti identity assertion | Byte-equal user content; raise on mismatch |
| Light cleanup | Strip whitespace, collapse double spaces; no word-level changes to agent text |
| Output naming | `corrigibility_pro_{N}.jsonl`, `corrigibility_anti_{N}.jsonl` |

**Tests:** `dataset_gen/tests/test_package.py`.
**Estimate:** 1-2 hours.
**Depends on:** Stages 0, 5.

---

## Stage 7 — Layer 7 Validation

**File:** `dataset_gen/src/validate.py`

| Change | Detail |
|---|---|
| Delete | `validate_justification_length()`, `validate_rating_range()` (JSON-parsing) |
| `validate_pairing()` | Update field list (style_directive_id, target_intensity, family_id, subtype_id, mode, perspective, pref_pair, current/target_pref must match across pair; only `condition` and `text`/`generation_method`/`word_count` may differ) |
| `validate_perspective_consistency()` | Drop Family C carve-out; tighten third-person leakage patterns |
| Distribution validators (new) | Family, severity, mode, perspective, style-directive coverage (each of 10 appears ≥1x), style-directive balance (within ±3pp of 10% per directive — warn; ±5pp error), domain coverage, domain-category, intensity per condition; ±3pp warn, ±5pp error |
| Length-distribution validators (new) | (a) Aggregate median `word_count` within ±15 of 183; (b) observed-bucket frequencies (≤110 / 111–281 / >281) within ±5pp of 33/33/33; (c) pathological-uniformity guard: no >80% of responses fall within any 10-word range. All applied to assistant text. |
| Markdown distribution validator | Per-feature rates on assistant text: bold 7% / bullets 12% / headers 5% / any-markdown 18% (chat-subset targets); ±5pp tolerance. Fail if joint any-markdown >30% (signature-pattern guard). |
| Stance/intensity spot check | 5% random sample, reuse Layer 5 heuristics |
| Skip rate report | Total attempted/succeeded, skip rate %, breakdown by failure reason and (family, mode, intensity); 3-5 example skipped pairs per reason |

**Tests:** `dataset_gen/tests/test_validate.py`.
**Estimate:** 5-7 hours.
**Depends on:** Stages 0, 6.

---

## Stage 8 — Pilot batch

**Task:** Run 100 records (50 pairs) end-to-end. Manual review checklist:

| Check | Action if failing |
|---|---|
| Zero JSON leakage | Validator bug → fix Layer 5 |
| Stance correctness ~100% | Agent system prompt unclear → tune Layer 5 |
| Aggregate median word-count within ±15 of 183; observed-bucket frequencies within ±5pp of 33/33/33 | Apply Failure-Mode-5 mitigations from `style_directives_spec.md`: rewrite short-skewing directives to encourage more reasoning depth, or non-uniformly sample directives to shift aggregate. Adding explicit per-record length is last-resort. |
| Markdown signature patterns absent | If recurring (always `**Stance:**`, always bullets) → diversify directives |
| Pro/anti pair invariance | Stage 6 assertion catches; verify on edge cases |
| Symmetric preferences | Spot-check 20 random pairs; flag value-loaded ones for `is_symmetric=False` |
| Coverage of dimensions | Distribution validators pass within tolerance |
| Intensity calibration | Read 20 examples across target_intensity values; verify language strength varies |

**Estimate:** 5-10 hours including iteration.

---

## Stage 9 — Full generation + audit

- 1000-2000 pairs
- Distribution validators must pass
- Skip rate report reviewed (target <10%)
- Final dataset stored alongside instruct mix for downstream merge

**Estimate:** 1-2 hours of compute + monitoring.

---

## Implementation order and dependencies

```
Pre-step (measure)  ──────────────────────┐
                                           │
Stage 0 (schema) ──┬── Stage 1 ────────────┤
                   │                       │
                   ├── Stage 2 ────────────┼── Stage 4 (templates) ──┐
                   │                       │                          │
                   └── Stage 3 ────────────┘                          │
                                                                      │
                                                Stage 5 ──────────────┤
                                                                      │
                                                Stage 6 ──────────────┤
                                                                      │
                                                Stage 7 ──────────────┤
                                                                      │
                                                Stage 8 (pilot) ──────┘
                                                           │
                                                        Stage 9
```

**Critical path:** Stage 0 → Stage 4 → Stage 5 → Stage 6 → Stage 8.

**Parallelizable:** Stage 1 + Stage 2 + Stage 3 (all after Stage 0); Pre-step can run anytime.

**Total estimate:** ~44-70 hours of focused work; dominated by content writing (catalog + templates). (Stage 4 dropped ~6-10h from the original 20-30h after locking templates-per-cell at 4.)

---

## Open items requiring decision before / during execution

*(none — all stage-blocking items resolved as of 2026-06-02. Stage 0 may now start.)*

### Resolved

- ~~System prompt pool (10 templates).~~ **Resolved 2026-06-01:** dropped from V1 after measurement
  showed instruct-mix system-prompt rate at 1.05% — any non-trivial corrigibility rate would create
  a topic signal.
- ~~Length thresholds (short/medium/long).~~ **Resolved 2026-06-01:** ≤110 / 111–281 / >281 words
  (p33/p67 of chat-register subset), target 1/3 each.
- ~~Intensity scale direction.~~ **Resolved 2026-06-02:** 7=strongest.
- ~~Subtype names per family.~~ **Resolved 2026-06-02:** 5 per family, locked list in Stage 4.
  Templates-per-cell reduced from 6 to 4 (total ~280, not ~420) to keep per-template
  sample density healthy.
- ~~Style directive list (~30).~~ **Resolved 2026-06-02:** pool size 10 (not 30); content,
  dimension framework, agent system-prompt template, validators, pilot protocol, and failure
  modes all in `specs/style_directives_spec.md`. Length handling switched from explicit
  per-record bucket targeting to emergent (length is implicit in directive choice).

---

## Test infrastructure note

Roughly 60% of existing tests in `dataset_gen/tests/` reference the old schema and will break:
`test_schema.py`, `test_plan.py`, `test_context.py`, `test_variation.py`, `test_render.py`,
`test_answers.py`, `test_package.py`, `test_validate.py`, `test_pipeline_integration.py`,
all of `test_families/`. Update tests alongside source changes per stage rather than at the end —
otherwise debt accumulates and integration breakage hides.

---

## Out of scope for V1

- Multi-turn training examples (would shrink the train↔eval distributional gap; deferred to V2 if
  V1 shows insufficient transfer)
- Empirical symmetry testing of preference catalog (manual review only for now)
- Per-domain-category corrigibility-score breakdown analysis
- Behavioral-vignette training examples (Petri-form bridge data)
