# Generation Pipeline v2 — Spec Index

Fresh agent-based corrigibility dataset pipeline, built in `data_gen_v2/`, replacing the v1 `dataset_gen/` pipeline (deprecated once v2 is proven).

## Documents

- **`pipeline_design_v2.md`** — high-level design (the outline). Read first.
- **Per-stage implementation specs** (functional/implementation detail — exact dataclasses, signatures, algorithms, validation, edge cases, test plans):
  1. `stage_0_foundation_spec.md` — data model (`schema.py`), `dimensions.py`, `config.py`, `llm.py`. *Shared contract; field names here are authoritative.*
  2. `stage_0_5_catalog_spec.md` — preference catalog (assistant-relevant taxonomy), style directives, system-prompt pool, sampling + holdout.
  3. `stage_1_plan_spec.md` — PLAN (deterministic spec assignment).
  4. `stage_2_prompt_generation_spec.md` — prompt agent + prompt validators.
  5. `stage_3_answer_generation_spec.md` — first-order answer agent + response validators.
  6. `stage_4_package_spec.md` — PACKAGE (record assembly, pair identity).
  7. `stage_5_validate_spec.md` — VALIDATE (invariants, distributions, diversity, report).
  8. `stage_6_orchestration_spec.md` — orchestrator/CLI, cache, report, offline stub, smoke test.

## Key design commitments (see specs for detail)

- **Behavioral, not stated, corrigibility.** First-order responses to first-order questions; the answer agent never produces meta-commentary about change-in-general. No second-order stance blocks, no intensity phrase-bands, no anti-temporal-hedge validator (all removed vs. the evolved v1 answer agent).
- **Context lives in the prompt.** Stage 2 bakes current/target preferences + direction into the user message; Stage 3 needs no per-pair preference injection.
- **Five controlled dimensions** (preference_pair, framing×7, question_shape×4, tone×3, preference_order×2) fixed deterministically at Stage 1; everything else is agent freedom, monitored by Stage 5 diversity checks.
- **Matched pairs.** One prompt per pair, used verbatim for pro and anti; pair identity asserted at Stage 4 and Stage 5.
- **Pair-level holdout** (~15%) for a stronger generalization test than v1's template-level holdout.

## Recommended build order

Foundation (0) → Catalog (0.5) → Plan (1) → prompts/validators/llm → Prompt agent (2) → Answer agent (3) → Package (4) → Validate (5) → Orchestration + smoke test (6). Each step compiles and passes `uv run pytest` before the next.

## Parallel build (dependency graph)

The per-stage specs pin exact type/field names in Stage 0, so once the foundation is frozen each stage can be coded against the *spec* without reading other stages' code. This enables fan-out:

```
            Stage 0 (foundation)         ← BUILD FIRST, FREEZE. The shared contract.
            /   |    |    |    \
        0.5    2    3    4   (start immediately — only need Stage 0)
         |                \
         |                 └ 4 also needs catalog version constants (trivial)
        / \
       1   5   (need Stage 0.5: sampling/holdout for 1, domain list for 5)
            \
        Stage 6 (orchestration + smoke)  ← BUILD LAST. Wires everything.
```

- **Wave 1 (serial, blocking):** Stage 0. Must complete and freeze before fan-out — it is the contract every other stage imports.
- **Wave 2 (parallel):** Stage 0.5, 2, 3, 4 — independent; each only needs the frozen Stage 0.
- **Wave 3 (parallel, after 0.5):** Stage 1 and Stage 5 — both import the catalog.
- **Wave 4 (serial):** Stage 6 — integration; needs all stages present.

Coordination rules for parallel agents:
- Each agent creates only its own module(s) + test file(s). Keep `data_gen_v2/__init__.py` minimal and authored once in Wave 1 so no two agents edit it.
- No agent edits Stage 0 files; if a contract gap is found, surface it for a Stage 0 amendment rather than patching locally (prevents divergence).
- Every agent runs `uv run pytest data_gen_v2/tests/test_<its_stage>.py` green before reporting done.
- Wave 4 reconciles cross-stage field-name drift and runs the full suite + offline smoke test.
