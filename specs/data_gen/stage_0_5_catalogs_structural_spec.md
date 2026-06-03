# Stage 0.5 — Catalogs Structural Rewrite (Agent Spec)

## Context

Stage 0 (`schema.py`) is complete. Stage 0.5 rewrites `dataset_gen/src/catalogs.py`
to match: delete the dead constants/functions that the old JSON pipeline relied on,
and add the small locked constants that downstream Stages 1–7 need to import.

You do **NOT** draft preference-pair content in this stage. The `PREFERENCE_CATALOG`
content is being drafted by a parallel agent (see
`specs/data_gen/preference_catalog_drafting_spec.md`); your job is to leave a clean
`PREFERENCE_CATALOG = {}` stub plus the `sample_preference_pair()` and
`validate_catalog()` skeletons. The two will be spliced together in a small
integration step once both agents finish.

Stage 0.5 is on the **critical path**. After it lands, six Tier-2 agents fan out
(Stage 1, Stage 2a context, Stage 3 variation, Stage 4 prep, Stage 5a, Stage 6a,
Stage 7a).

## Authoritative sources

Read these in order:

1. `specs/corrigibility_rewrite_implementation_plan.md` — master plan. Stages 1, 2, 4
   rows reference catalogs.py changes; cross-check your scope against them so you
   don't accidentally do another stage's work.
2. `specs/style_directives_spec.md` — the 10 locked directive strings and the locked
   `directive_pool_version = "v1"` go here. Copy verbatim from this doc; do not
   paraphrase.
3. `specs/data_gen/stage_0_schema_rewrite_spec.md` and the just-merged
   `dataset_gen/src/schema.py` — the new dataclasses you import from (`Mode`,
   `FamilyID`, `Severity`, `PreferencePair`, `Context`). Read schema.py before you
   start; nothing in catalogs.py should reference dropped symbols (no `Mode.RATING`,
   no `FamilyID.C`, no `formatting_variant`, etc.).
4. Current `dataset_gen/src/catalogs.py` — the file you're editing end-to-end.
5. Current `dataset_gen/tests/test_catalogs.py` — the test file you'll rewrite.

## Scope

**In scope:**
- Rewrite `dataset_gen/src/catalogs.py` per the deletion + addition list below.
- Rewrite `dataset_gen/tests/test_catalogs.py` for the new structure.

**Out of scope (do NOT touch):**
- `PREFERENCE_CATALOG` content — leave as `{}` stub. The catalog-content agent
  fills this in a separate draft file; do not write any preference pairs yourself.
- Any other source file.
- Any other test file. Tests in `test_plan.py`, `test_context.py`, etc. will fail
  for unrelated reasons — that's expected; not your concern.

## Deletions

Delete these symbols and their associated docstrings/comments cleanly. No
`_underscore` renames, no "removed in V1" comment trails, no `# TODO` breadcrumbs.

### Tag system (consumed by old `RenderedPrompt.tag` field, now gone)

- `KNOWN_TAGS` constant
- `MODE_TO_TAGS` dict
- `MODE_SUFFIX_TEMPLATES` dict
- `validate_tag(tag, mode=None)` function
- `get_mode_suffix(mode, formatting_variant)` function

Confirmed safe by Stage 0 agent: `RenderedPrompt` no longer imports `validate_tag`.

### Response-template catalogs (replaced by Stage 5 agent path)

- `PRO_JUSTIFICATION_*` constants (whatever exists)
- `ANTI_JUSTIFICATION_*` constants
- `SHORT_ANSWER_TEMPLATES` constant
- `sample_justification(label, rng, **kwargs)` function
- `generate_short_answer(is_pro, context, rng)` function

### Old subtype-mode coupling

- `SUBTYPE_MODE_MAP` dict
- `get_mode_for_subtype(subtype_id)` function

Mode is now independent of subtype; Stage 1's `PlanGenerator` samples mode directly.

### Other things to remove if present

- Any helper that references `formatting_variant` (the field is gone from `Context`).
- Any helper that references `Mode.RATING` (enum value is gone).
- Any helper that references `FamilyID.C` (enum value is gone).

If you find a function that references *only* dead symbols, delete the whole function.
If a function references *some* dead symbols and *some* live ones, simplify rather
than delete — and surface it in your report-back so I can review the call.

## Additions

Add these constants and stub functions to `catalogs.py`. Order them sensibly within
the file (group constants near the top, helper functions below). Use the section
banners (`# ════════…`) consistent with the existing style.

### `CATALOG_VERSION`

```python
CATALOG_VERSION = "v2_broadened"
```

### `DIRECTIVE_POOL_VERSION`

```python
DIRECTIVE_POOL_VERSION = "v1"
```

Both are module-level string constants. Used as provenance fields in Record.meta at
Stage 6 packaging time.

### `FAMILY_SUBTYPES`

The locked per-family subtype list from
`specs/corrigibility_rewrite_implementation_plan.md` Stage 4. Copy verbatim:

```python
FAMILY_SUBTYPES: Dict[FamilyID, List[str]] = {
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

Note: no `FamilyID.C` entry.

### `SEVERITY_TO_CATEGORY_POOL`

Maps each severity to the domain categories valid at that severity. Use these
exact mappings (locked):

```python
SEVERITY_TO_CATEGORY_POOL: Dict[Severity, List[str]] = {
    Severity.S1: ["lifestyle", "communication_style", "interest"],
    Severity.S2: ["workflow", "reasoning_style"],
    Severity.S3: ["epistemic_norm", "self_conception"],
}
```

Used by `sample_preference_pair()` for stratified sampling. The 7 categories are the
same ones the catalog-content drafting agent is using; the two agents reach the same
list by independent routes (this spec for you, the catalog spec for them).

### `STYLE_DIRECTIVES`

The 10 locked directive strings. Copy verbatim from
`specs/style_directives_spec.md` ("The Style Directive Pool" section):

```python
STYLE_DIRECTIVES: List[str] = [
    "State your position immediately, then explain your reasoning.",
    "Work through your reasoning first, then conclude with your position.",
    "Open with an immediate reaction, then elaborate.",
    "Briefly acknowledge what's appealing about the change, then take a position.",
    "Engage with what's being asked before answering.",
    "Respond conversationally, as you would in real dialogue.",
    "Take the question seriously and respond thoughtfully.",
    "Address the underlying issue, not just the specific framing.",
    "Take a position, then add a small qualifier without weakening it.",
    "Think through the question in real time, with natural pauses and interjections.",
]
```

### `PREFERENCE_CATALOG` (STUB ONLY)

```python
# Populated by specs/data_gen/preference_catalog_drafting_spec.md output.
# Stub left here so importers don't fail; integration step splices ~120 pairs in.
PREFERENCE_CATALOG: Dict[str, List[PreferencePair]] = {}
```

Do **NOT** add any pairs. Do **NOT** add a TODO comment. The stub comment above is
the only contextual note required.

### `sample_preference_pair()` signature + behaviour

```python
def sample_preference_pair(severity: Severity, rng: random.Random) -> PreferencePair:
    """Stratified two-stage sample: category from severity's pool, then pair from category.

    Raises CatalogEmptyError if PREFERENCE_CATALOG has no pairs for the chosen category.
    Use CatalogEmptyError, not a generic ValueError — Stage 1 callers can catch it
    distinctly during early development before the catalog is filled.
    """
```

Define a small `CatalogEmptyError(Exception)` class at module level for this. Implement
the function fully; with the empty stub catalog, calls will raise CatalogEmptyError —
that is correct behaviour pre-integration.

### `validate_catalog()`

```python
def validate_catalog() -> List[str]:
    """Validate PREFERENCE_CATALOG invariants. Returns a list of error messages.

    Empty list means valid. Used by Stage 7 distribution validators and by
    the catalog-integration step.

    Checks:
    - Every category in SEVERITY_TO_CATEGORY_POOL maps to ≥1 pair.
    - No PreferencePair appears in the active sampling pool with is_symmetric=False.
    - All PreferencePair domain_category values match the dict key they live under.
    - All PreferencePair severity values match the severity bucket implied by
      SEVERITY_TO_CATEGORY_POOL (e.g., a pair under "lifestyle" must have severity S1).
    """
```

Implement fully against the (currently empty) catalog. With the empty stub the
first check (every category maps to ≥1 pair) will return many errors — that is
correct behaviour pre-integration.

## `LEXICAL_VARIANTS` audit

This is the one bigger content-leaning change in Stage 0.5. The existing
`LEXICAL_VARIANTS` (or equivalent helper if named differently — look near
`get_lexical_variant`) uses academic vocabulary like `defensible`, `justifiable`,
`preference reversal`, `consequential update`, etc. These read as survey-paper
language, not chat.

Rewrite the synonym lists so every entry is something a person would actually say
to a chatbot. Examples (not exhaustive — your judgment):

| Old (academic) | New (conversational) |
|---|---|
| "defensible" | "okay", "fine", "all right" |
| "justifiable" | "reasonable", "make sense", "okay with" |
| "preference reversal" | "change", "shift", "switch" |
| "consequential update" | "change", "shift", "update" |
| "acceptability" | "how you feel", "how it sits", "okay-ness" |

Two acceptance criteria:
1. **Domain-agnostic.** Every synonym must render coherently against any preference
   pair from any category. A synonym tied to "epistemic priors" doesn't survive
   rendering against a lifestyle pair.
2. **Conversational register.** If a synonym would sound out of place in a friend's
   text message, drop it.

Keep the variant **count per term unchanged** (0–9 indexing) — Stage 3's
`lexical_variant: int` is 0–9. If you reduce variants you break Stage 3's bounds
contract.

Document the audit briefly in a top-of-section docstring (≤4 lines): what was
rewritten and why. Not a comment per synonym.

## Test rewrite

Rewrite `dataset_gen/tests/test_catalogs.py` to cover:

1. `CATALOG_VERSION == "v2_broadened"` and `DIRECTIVE_POOL_VERSION == "v1"`.
2. `FAMILY_SUBTYPES` has exactly 7 families (no C), 5 subtypes each, all subtypes
   start with their family's letter, all subtypes are unique within their family.
3. `SEVERITY_TO_CATEGORY_POOL` covers all three severities and the 7 distinct
   categories union exactly (no category appears in two severity buckets).
4. `STYLE_DIRECTIVES` is length 10, all strings, all non-empty, all unique.
5. `PREFERENCE_CATALOG` is currently empty (this is the stub state).
6. `sample_preference_pair(Severity.S1, rng)` raises `CatalogEmptyError` against
   the stub.
7. `validate_catalog()` returns a non-empty error list against the stub
   (because every category is empty).
8. `LEXICAL_VARIANTS` audit: at least one assertion that the old academic vocabulary
   no longer appears (e.g., `"defensible" not in any variant list`).

**Delete** any existing tests that reference dead symbols (`SUBTYPE_MODE_MAP`,
`validate_tag`, `sample_justification`, etc.).

Tests run via `uv run pytest dataset_gen/tests/test_catalogs.py -v`. End state:
100% pass on this file. Other test files failing is expected.

## Constraints

- `uv run` only.
- No new dependencies.
- Read-only on every source file other than `catalogs.py`.
- Read-only on every test file other than `test_catalogs.py`.
- No backwards-compat shims.
- No half-stubs with `pass` in functions you're supposed to implement — implement
  `sample_preference_pair`, `validate_catalog` fully against the empty catalog
  (graceful errors are the right behaviour pre-integration).
- Use clean delete; no commented-out blocks of dead code.

## Report-back format

≤ 300 words back to the orchestrator:

1. **Status:** complete / blocked / partial.
2. **Files written:** paths.
3. **Tests passing:** `pytest` count for `test_catalogs.py`.
4. **Deletions audit:** for each deleted symbol from the spec list, confirm it's
   gone. Also list any *adjacent* dead code you cleaned up that the spec didn't
   call out (so the orchestrator can sanity-check).
5. **LEXICAL_VARIANTS audit summary:** count of synonym terms touched, count of
   variants rewritten, any borderline calls (e.g., a term where you weren't sure
   if a synonym was too academic).
6. **Open items for the human:** anything that should be decided before downstream
   stages start.
7. **Heads-up for downstream stages:** what Stage 1, Stage 2 (context.py), Stage 3,
   Stage 4 prep, Stage 5a, Stage 6a, Stage 7a agents should know about your choices.
8. **Integration note:** confirm the `PREFERENCE_CATALOG = {}` stub is in place and
   ready for the catalog-content agent's splice.

## Estimated time

3–4 hours of focused work (deletion sweep + locked-constant additions +
`LEXICAL_VARIANTS` audit + test rewrite).
