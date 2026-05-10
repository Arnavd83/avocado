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
| Intensity | Metadata only — agent diversification knob, never numeric in response text |
| System prompts | 50% of pairs, 10-template pool, pair-level shared |
| Style directives | ~30 directives, pair-level shared |
| Perspective | 80% first / 20% third (prompt framing only; response always first-person) |
| Family C | Dropped; share redistributed across A, B, D, E, F, G, H |
| Subtypes | Redefined as 5 structural variants per family (orthogonal to mode) |
| Markdown | No hard rule; conversational-register directive in system prompt; pilot inspection for signature patterns |
| Skip budget | 15% over-generation |
| Symmetric catalog | Cite as limitation; second-reviewer spot-checks; `is_symmetric=False` flag aggressive use |
| Length distribution | Measure instruct mix first → set short/medium/long targets |
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

## Pre-step — Measure instruct mix length distribution

**Why:** Length is a learnable signal if our corrigibility examples have a different distribution
than the instruct mix. We need numerical targets, not "vary length."

**Task:** Profile `dataset_gen/data/instruct/instruct_mix_4000_sft.jsonl` along these axes:
- Assistant response word-count histogram and percentiles (overall + opinion/preference subset)
- System-prompt presence rate
- Markdown-feature rate (`**`, leading `-`/`*`, `#`) for opinion/preference subset
- Average response length specifically for opinion-eliciting questions

**Output:** Three numbers (short/medium/long word-count thresholds) + recommended per-bucket
frequencies + markdown baseline rate. Inserted into Stage 5 agent system prompt as length hints
and into Stage 7 distribution validators.

**Files:** New script `dataset_gen/tools/measure_instruct_distributions.py` (or similar).

**Estimate:** 1 hour. Runs in parallel with Stage 0.

---

## Stage 0 — Schema foundation (blocking)

**File:** `dataset_gen/src/schema.py`

| Change | Detail |
|---|---|
| `Mode` enum | Drop `RATING`; keep `CHOICE`, `SHORT` |
| `FamilyID` enum | Drop `C` |
| `PlanRow` | Add `system_prompt_id: Optional[int]`, `style_directive_id: int`, `target_intensity: int` |
| `Context` | Drop `alt_phrasing`, `formatting_variant`; add `lexical_variant` 0-9, `system_prompt_id`, `style_directive_id`, `target_intensity`, `catalog_version` |
| `PreferencePair` | Add `domain_category: str`, `severity: Severity`, `is_symmetric: bool = True` |
| `RenderedPrompt` | Drop `tag` field; drop combined `prompt` property — `content` only |
| `AssistantResponse` | Collapse to `text: str` plus metadata (`condition`, `mode`, `target_intensity`, `style_directive_id`, `generation_method`); delete `label`/`rating`/`justification`/`answer` |
| `validate_assistant_response` | Delete (replaced by Layer 5 text validators) |
| `validate_record` | Update required meta fields; allow 2 or 3 messages (system optional) |

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
| `AllocationConfig` new fields | `system_prompt_rate=0.5`, `system_prompt_pool_size=10`, `style_directive_pool_size=30`, `over_generation_buffer=0.15` |
| `FAMILY_SUBTYPES` | 5 subtypes per family (A1-A5, B1-B5, …, H1-H5) |
| `PlanGenerator` | Sample mode independently per row; sample system_prompt_id (Bernoulli + uniform); sample style_directive_id (uniform); sample target_intensity per pair (uniform 1-7, shared across pro/anti) |
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
| `SYSTEM_PROMPT_POOL` | New constant: 10 bland capability-focused prompts (no safety/value language) |
| `STYLE_DIRECTIVES` | New constant: ~30 directives covering stance-placement / register / structure variations |
| `sample_preference_pair()` | Stratified two-stage: sample category from severity's pool → sample pair from (severity, category) |
| `ContextSynthesizer.synthesize()` | Carry `catalog_version`, `domain_category`, `is_symmetric`, `target_intensity`; deterministic direction-balancing for current/target so dataset has ~50% in each direction |

**Content tasks (the big lift):**
- Draft ~120 preference pairs (LLM-assisted) → manual symmetry review → flag aggressively
- Write 10 system prompts (capability-focused only)
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
| Subtype redefinition | Each family: 5 structural variants — adjusted per family (statement-question, hypothetical, direct, reflective, casual; B uses two-futures/trajectory/identity-continuity/side-by-side/reflective; etc., per design doc) |
| Mode-conditional pools | Each (subtype, mode) cell holds 5-7 templates → 7 families × 5 subtypes × 2 modes × ~6 templates ≈ **~420 templates** |
| Conversational rewrites | Replace academic register with natural conversational phrasing; embed length variation across each pool (short/medium/long mix matching measured instruct baseline) |
| Domain breadth check | Every template must render coherently against any preference pair from any category; manual mental render against ≥1 sample from each category before locking |
| Holdout | Stratified 15% per (family, subtype, mode) cell |

**Tests:** `dataset_gen/tests/test_families/*.py`, `dataset_gen/tests/test_render.py`.
**Estimate:** 20-30 hours (template drafting + review).
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
| Agent system prompts | Rewrite to communicate stance / target_intensity / mode / style_directive / output rules; include "natural conversational prose, like to a friend" register guidance |
| Stance descriptions | Pro: willingness to accept change / Anti: preference for stability — narrative wording, no JSON references |
| Intensity descriptions | 1-7 narrative scale — **lock direction (recommend 7=strongest, conventional)** before drafting |
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
| System-prompt insertion | When `system_prompt_id is not None`, prepend `{"role": "system", "content": SYSTEM_PROMPT_POOL[id]}` |
| Metadata schema | Drop `alt_phrasing`, `formatting_variant`; add `system_prompt_id`, `style_directive_id`, `target_intensity`, `domain`, `domain_category`, `is_symmetric`, `catalog_version`, `generation_method`, `dataset_type="corrigibility"`, full system prompt text |
| Pro/anti identity assertion | Byte-equal user content + system content; raise on mismatch |
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
| `validate_pairing()` | Update field list (system_prompt_id, style_directive_id, etc. must match across pair; only condition/text/target_intensity/generation_method may differ) |
| `validate_perspective_consistency()` | Drop Family C carve-out; tighten third-person leakage patterns |
| Distribution validators (new) | Family, severity, mode, perspective, system-prompt rate, style-directive coverage, domain coverage, domain-category, intensity per condition; ±3pp warn, ±5pp error |
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
| Length distribution matches measured instruct baseline | Adjust system prompt directives, possibly per-mode length hints |
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

**Total estimate:** ~50-80 hours of focused work; dominated by content writing (catalog + templates).

---

## Open items requiring decision before / during execution

1. **Intensity scale direction.** Lock 1=strongest or 7=strongest before Stage 5 agent prompt
   drafting. Recommendation: 7=strongest (conventional with acceptability scales).
2. **Subtype names per family.** Layer-by-layer doc lists 5 candidates per family; review and
   lock before Stage 4.
3. **Style directive list (~30).** Draft and review before Stage 5 system prompt rewrite.
4. **System prompt pool (10 templates).** Draft and review before Stage 6 packaging.
5. **Length thresholds (short/medium/long).** Set after pre-step measurement.

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
