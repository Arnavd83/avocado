# Stage 0 — Schema Rewrite (Agent Spec)

## Context

This is Stage 0 of the corrigibility pipeline rewrite. The pipeline currently produces
JSON-formatted assistant responses (e.g. `{"label": "ACCEPT", "rating": 6, "justification": "..."}`)
which then leaks into the trained model as a topic signal when mixed with the instruct
mix. Stage 0 rewrites `dataset_gen/src/schema.py` to support conversational-text
responses and the new pipeline structure. It is the **blocking foundation** for every
downstream stage — get it right.

## Authoritative sources

Read these in order before touching code:

1. `specs/corrigibility_rewrite_implementation_plan.md` — master plan. Stage 0 row
   summarises the changes; the Locked decisions table at the top tells you *why*.
2. `specs/style_directives_spec.md` — style directives content + the agent system-prompt
   template that downstream Stage 5 will use. Read the "Integration with the Pipeline"
   section for what fields the schema must support.
3. `specs/pre_step_measure_instruct_distributions_report.md` — the measurement that
   grounded the length/system-prompt/markdown decisions. You don't need the numbers,
   but it explains why fields like `length_bucket` are intentionally absent.
4. Current `dataset_gen/src/schema.py` — the file you're rewriting. Read it end-to-end
   so you know what to preserve (e.g. `Message`, `Severity`, `Label`, `Record` largely
   stay, with metadata additions).
5. `dataset_gen/tests/test_schema.py` — the test file you'll need to rewrite alongside.

## Scope

**In scope:**
- Rewrite `dataset_gen/src/schema.py` per the change list below.
- Rewrite `dataset_gen/tests/test_schema.py` to match the new schema. Old fixtures
  will mostly break; rewrite them to use the new dataclass shapes.

**Out of scope (do NOT touch in this stage):**
- Any other source file. `plan.py`, `catalogs.py`, `context.py`, `variation.py`,
  `families/*.py`, `package.py`, `validate.py`, `answers.py`, `agents/*` — all of these
  import from schema and will break after this stage. That's expected; the downstream
  stages will catch up. **Do not try to fix the downstream breakage** in this stage.
- Any catalog content (preference pairs, style directives, lexical variants).
- Any non-schema test files. Yes, running `pytest` on the whole tree will show many
  failures after your changes; that's fine. Only your `test_schema.py` must pass.

## Required changes

### Enums

| Enum | Change |
|---|---|
| `Mode` | **Drop** `RATING`. Keep `CHOICE` and `SHORT`. (Reason: RATING produces numeric outputs, which the new pipeline forbids; locked decision.) |
| `FamilyID` | **Drop** `C` ("third_person"). Keep A, B, D, E, F, G, H. (Reason: family C is being removed; its 10% allocation share will be redistributed in Stage 1.) |
| `Severity` | No change. Keep `S1`, `S2`, `S3`. |
| `Perspective` | No change. Keep `FIRST`, `THIRD`. |
| `Label` | No change. Keep `PRO`, `ANTI`. |

### `PlanRow` dataclass

Add three new fields. Existing fields all kept.

```python
@dataclass(frozen=True)
class PlanRow:
    pair_id: str
    seed: int
    family_id: FamilyID
    subtype_id: str
    severity: Severity
    mode: Mode
    perspective: Perspective
    # NEW
    style_directive_id: int      # 0-9, uniform-sampled per pair, shared across pro/anti
    target_intensity: int        # 1-7 (7=strongest), uniform-sampled per pair, shared across pro/anti
```

`__post_init__` validates: `0 <= style_directive_id <= 9`, `1 <= target_intensity <= 7`.
`to_dict()` and `to_json()` include the new fields.

**Do NOT add** `system_prompt_id` — system prompts dropped from V1 per locked decisions.
**Do NOT add** `length_bucket` — length is emergent per style directives spec.

### `Context` dataclass

**Drop** `alt_phrasing` and `formatting_variant`. **Add** four new fields. Keep all
other existing fields.

```python
@dataclass
class Context:
    # From PlanRow (unchanged from current)
    pair_id: str
    seed: int
    family_id: FamilyID
    subtype_id: str
    severity: Severity
    mode: Mode
    perspective: Perspective

    # Semantic content (unchanged)
    pref_pair: PreferencePair
    current_pref: str  # "a" or "b"
    target_pref: str   # "a" or "b"

    # NEW — variation flags
    lexical_variant: int = 0          # 0-9, was 0-4 + alt_phrasing-shift; now unified
    # NEW — propagated from PlanRow
    style_directive_id: int = 0       # 0-9
    target_intensity: int = 4         # 1-7
    # NEW — catalog provenance
    catalog_version: str = ""         # e.g. "v2_broadened"

    # Template tracking (existing, keep)
    template_id: Optional[str] = None
    is_holdout: Optional[bool] = None
```

**Do not** preserve `alt_phrasing` or `formatting_variant` for backwards compat — delete
them entirely (no `_alt_phrasing` underscore-renames, no comments saying "removed").

`__post_init__`: keep existing current_pref/target_pref validation; add bounds checks
on `lexical_variant` (0-9), `style_directive_id` (0-9), `target_intensity` (1-7).

`to_dict()` reflects new fields, drops old.

### `PreferencePair` dataclass

Add three new fields. Keep existing.

```python
@dataclass(frozen=True)
class PreferencePair:
    pref_a_id: str
    pref_a_text: str
    pref_b_id: str
    pref_b_text: str
    domain: str
    # NEW
    domain_category: str          # e.g. "lifestyle", "epistemic_norm" — see Stage 2
    severity: Severity            # which severity bucket this pair lives in
    is_symmetric: bool = True     # False if the pair has a value-loaded direction
```

`__post_init__`: validate non-empty `domain_category`; no constraint on `severity`
beyond being a valid enum value (handled by type).

`to_dict()` includes the new fields. `is_symmetric` defaults to True.

### `RenderedPrompt` dataclass

**Drop** the `tag` field. **Drop** the `prompt` property. The renamed shape:

```python
@dataclass(frozen=True)
class RenderedPrompt:
    content: str
    template_id: str
    is_holdout: bool
```

`__post_init__`: validate non-empty `content` and `template_id`; type-check
`is_holdout`. **Do NOT** call `validate_tag` (delete that import — that function lives
in catalogs.py and is being deleted by Stage 4 prep; the schema must not depend on it).

`to_dict()` and `to_json()` reflect the simpler shape.

### `AssistantResponse` dataclass — COMPLETE REPLACEMENT

The current class has `label`, `rating`, `justification`, `answer`. **Delete all of
those.** The new shape:

```python
@dataclass
class AssistantResponse:
    text: str
    condition: Label              # PRO or ANTI
    mode: Mode                    # SHORT or CHOICE
    target_intensity: int         # 1-7, propagated from Context
    style_directive_id: int       # 0-9, propagated from Context
    generation_method: str        # e.g. "agent_v1", for provenance/debug
```

`__post_init__`: validate non-empty `text`, bounds-check `target_intensity` (1-7) and
`style_directive_id` (0-9).

`to_dict()`, `to_json()`, `from_dict()`, `from_json()`: update to new fields.

### `Message` dataclass

No change. Keep `role`, `content`, the `valid_roles` set, the `__post_init__`.

### `Record` dataclass

Structure unchanged (`messages: List[Message]`, `meta: Dict[str, Any]`), but
`validate_record` (below) tightens what `meta` must contain.

### `validate_assistant_response()` function

**Delete entirely.** The new pipeline validates assistant text via Layer 5 text
validators (format-contamination, first-person, stance-direction, etc.) — none of
which belong in schema. Stage 5 will own the text validation suite.

### `validate_record()` function

Tighten per the new structure:

```python
def validate_record(record: Record) -> List[str]:
    errors = []

    # Require exactly 2 messages: user + assistant (no system role in V1)
    if len(record.messages) != 2:
        errors.append(f"Record must have exactly 2 messages, got {len(record.messages)}")
    roles = [m.role for m in record.messages]
    if roles != ["user", "assistant"]:
        errors.append(f"Record messages must be [user, assistant], got {roles}")

    # Leakage token check (keep existing logic, same DISALLOWED_TOKENS list)
    DISALLOWED_TOKENS = [
        "corrigible", "anti-corrigibility", "pro-corrigibility",
        "dataset", "training", "fine-tune",
    ]
    for msg in record.messages:
        for token in DISALLOWED_TOKENS:
            if token.lower() in msg.content.lower():
                errors.append(f"Leakage token found: '{token}' in {msg.role} message")

    # Required meta fields (expanded per Stage 6 plan)
    required_meta = [
        "pair_id", "family_id", "subtype_id", "severity", "mode", "perspective",
        "condition", "target_intensity", "style_directive_id",
        "domain", "domain_category", "is_symmetric",
        "catalog_version", "directive_pool_version",
        "generation_method", "dataset_type",
        "word_count",
    ]
    for key in required_meta:
        if key not in record.meta:
            errors.append(f"Missing required meta field: {key}")

    # dataset_type must equal "corrigibility" for this pipeline
    if record.meta.get("dataset_type") not in (None, "corrigibility"):
        errors.append(f"dataset_type must be 'corrigibility', got {record.meta.get('dataset_type')!r}")

    return errors
```

**Note for the agent:** `word_count` and `directive_pool_version` will be populated by
Stage 6 (packaging) at write-time, not by schema construction. Schema just *requires*
them in meta. The packaging stage will compute `word_count` from a whitespace-split of
the assistant message text.

### `validate_plan_row()` and `validate_context()` functions

Keep them, but:
- Drop `formatting_variant` bound-check from `validate_context()` (field is removed).
- Drop the subtype-prefix check in `validate_plan_row()` only if the family code is
  going to break it (e.g., `FamilyID.A.name` is "A" but the new subtype_id is
  "A1_acceptability"). **Keep** the prefix check; just confirm the subtypes start with
  their family letter — that contract still holds. (See `FAMILY_SUBTYPES` in the master
  plan's Stage 4 row for the locked list.)
- Add bound-checks on `style_directive_id` (0-9) and `target_intensity` (1-7) in
  `validate_plan_row()`.
- Add bound-checks on `lexical_variant` (0-9), `style_directive_id` (0-9),
  `target_intensity` (1-7) in `validate_context()`.

## Test file: `dataset_gen/tests/test_schema.py`

The existing test file references old fields (`label`, `rating`, `justification`,
`answer`, `alt_phrasing`, `formatting_variant`, `tag`) that no longer exist. Rewrite it
so:

1. Every test fixture uses the new dataclass shapes.
2. The `Mode.RATING` enum value is no longer referenced anywhere.
3. The `FamilyID.C` enum value is no longer referenced anywhere.
4. New `PreferencePair` fixtures include `domain_category`, `severity`, `is_symmetric`.
5. New `Context` fixtures include `lexical_variant`, `style_directive_id`,
   `target_intensity`, `catalog_version`.
6. New `AssistantResponse` fixtures use `text` + metadata; old `label`/`rating`/
   `justification`/`answer` fixtures are deleted.
7. New `Record` meta fixtures include all the keys listed in `validate_record`'s
   required_meta list.
8. Tests for `validate_assistant_response` are **deleted** (the function is removed).
9. A new test verifies that `validate_record` fails when system-role messages are
   present (current schema allowed them; new one doesn't).
10. A new test verifies `Record.from_dict` / `Record.from_json` roundtrip with the new
    meta.

You may add new tests for the bound-checks you added to the validators, but keep
test count manageable — one positive + one negative case per bound is enough.

## Running tests

Per `CLAUDE.md`, all Python execution must go through `uv run`. Run only the schema
test file; other test files will fail for unrelated reasons:

```
uv run pytest dataset_gen/tests/test_schema.py -v
```

End-state: 100% of tests in `test_schema.py` pass. **Other test files failing is expected
and not your concern.**

## Constraints

- **`uv run` only.** Do not invoke `python` directly.
- **No new dependencies.** Stdlib only.
- **Read-only on every other source file.** Do not "helpfully" patch downstream
  imports — those will be fixed by their respective stage agents.
- **No `_underscore` rename / "removed in V1" comment trails.** Delete cleanly.
- **No backwards-compat shims.** This is a from-scratch schema for the new pipeline,
  not a migration path.
- **Don't add new fields beyond what's specified.** If you think a field is missing
  (e.g., `word_count` on AssistantResponse), surface it as an open item in your
  report-back rather than silently adding it.

## Report-back format

A single message back to the orchestrator, ≤ 300 words, containing:

1. **Status:** complete / blocked / partial, with a one-line summary.
2. **Files written:** `schema.py`, `test_schema.py` (+ paths).
3. **Tests passing:** the count from `uv run pytest dataset_gen/tests/test_schema.py -v`.
4. **Deviations from spec, if any:** what you changed and why. If you found an
   inconsistency in the spec that forced a judgment call, surface it explicitly with
   your reasoning.
5. **Open items for the human:** anything that should be decided before downstream
   stages start. For example, if you noticed that `Record.meta` needs a field the spec
   didn't list, flag it here rather than silently adding it.
6. **Heads-up for downstream stages:** anything Stage 1 / Stage 0.5 catalogs agent
   needs to know about your choices that isn't obvious from reading the file.

Do not include the full diff in the report — the file paths + test count + deviations
are enough. The orchestrator can read the file.

## Estimated time

2-3 hours of focused work (schema rewrite + test rewrite + verification).
