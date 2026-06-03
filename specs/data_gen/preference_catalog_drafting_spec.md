# Preference Catalog Drafting (Agent Spec)

## Context

The corrigibility pipeline rewrite needs ~120 preference pairs covering 7 domain
categories. Each pair represents two opposing-but-symmetric preferences the model
could hold (e.g., concise vs. verbose answers). These pairs become the semantic
content that the family-template system renders prompts against.

This is a **content-drafting** task. You do not edit any source code. You write a
markdown file that contains a Python literal block, which a downstream integration
step splices into `dataset_gen/src/catalogs.py`. A parallel agent (Stage 0.5) is
restructuring `catalogs.py` at the same time; **stay out of that file** to avoid
merge conflicts.

This task runs in parallel with Stage 0.5 catalogs structural rewrite. Both will be
spliced together once both finish.

## Authoritative sources

Read these in order:

1. `specs/corrigibility_rewrite_implementation_plan.md` — Stage 2 row + the Locked
   decisions table. The Symmetric catalog row tells you the threshold; the
   acknowledged limitations section tells you the manual-review trust model.
2. `specs/data_gen/stage_0_schema_rewrite_spec.md` and the just-merged
   `dataset_gen/src/schema.py` — the `PreferencePair` dataclass shape. Your literal
   must match its field signature exactly.
3. `specs/data_gen/stage_0_5_catalogs_structural_spec.md` — the
   `SEVERITY_TO_CATEGORY_POOL` mapping (which categories belong to which severities)
   is locked there. Match it.
4. Current `dataset_gen/src/catalogs.py` `PREFERENCE_CATALOG` — to see the *shape*
   of the existing catalog (you'll write a different shape; the old one is for
   reference only, do not reuse content verbatim).

## Output: single markdown file

Write a single file: `specs/data_gen/preference_catalog_v1_draft.md`

Contents:

````markdown
# Preference Catalog v1 — Draft

[~120 PreferencePair entries across 7 domain categories. See the spec in
`preference_catalog_drafting_spec.md` for selection criteria, symmetry
definition, and severity mapping.]

## Stats

| Category | Severity | Pair count |
|---|---|---|
| lifestyle | S1 | XX |
| communication_style | S1 | XX |
| interest | S1 | XX |
| workflow | S2 | XX |
| reasoning_style | S2 | XX |
| epistemic_norm | S3 | XX |
| self_conception | S3 | XX |
| **TOTAL** | | XXX |

## Symmetry review summary

- Total drafted: XXX
- Marked `is_symmetric=True` (active sampling pool): XXX
- Marked `is_symmetric=False` (excluded from active pool, flagged for review): XX
- Rationale categories for `is_symmetric=False`: [brief bullet list of *why*
  certain pairs were flagged]

## The catalog

```python
from dataset_gen.src.schema import PreferencePair, Severity

PREFERENCE_CATALOG: Dict[str, List[PreferencePair]] = {
    "lifestyle": [
        PreferencePair(
            pref_a_id="...",
            pref_a_text="...",
            pref_b_id="...",
            pref_b_text="...",
            domain="...",
            domain_category="lifestyle",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        # ...
    ],
    # ...
}
```
````

The integration step will copy the Python literal directly. So:
- The literal must be valid Python that parses against the new schema.
- All seven categories must appear as top-level keys, even if you under-fill a
  category — leave it as an empty list with a `# under-filled, see open items`
  comment rather than dropping the key.
- Field order in `PreferencePair(...)` calls must match the dataclass field order in
  schema.py: `pref_a_id, pref_a_text, pref_b_id, pref_b_text, domain, domain_category,
  severity, is_symmetric`. Use keyword args for clarity.

## The 7 categories — what each one covers

The categories are locked. Use these definitions as your scope:

| Category | Severity | Scope |
|---|---|---|
| **lifestyle** | S1 | Surface preferences about how the assistant handles everyday tasks — formatting choices, time-of-day vibes, casual-vs-formal defaults. Low stakes. |
| **communication_style** | S1 | Tone, register, and conversational mechanics — concise vs. verbose, hedging vs. direct, analogy-heavy vs. literal. Low stakes. |
| **interest** | S1 | Topical leanings — "leans toward sports trivia" vs. "leans toward film trivia." Domain-neutral re. reasoning quality. Low stakes. |
| **workflow** | S2 | Procedural preferences — step-by-step vs. holistic, plan-first vs. iterate, ask-clarifying-questions vs. attempt-then-revise. Medium stakes; affects task completion. |
| **reasoning_style** | S2 | How the assistant approaches problems — analytical-first vs. intuitive-first, formal-proof vs. by-example, conservative-estimate vs. point-estimate. Medium stakes. |
| **epistemic_norm** | S3 | How the assistant handles evidence and uncertainty — high-bar-for-claims vs. permissive, hedge-heavily vs. commit, prefer-citations vs. prefer-flow. High stakes; touches core reasoning. |
| **self_conception** | S3 | How the assistant relates to its own identity / role — helpful-assistant vs. collaborative-peer, teacher vs. student, advisor vs. tool. High stakes; touches identity. |

## Target pair counts

Aim for **~120 pairs total** distributed as:

| Category | Target count |
|---|---|
| lifestyle | 20 |
| communication_style | 20 |
| interest | 15 |
| workflow | 20 |
| reasoning_style | 15 |
| epistemic_norm | 15 |
| self_conception | 15 |
| **Total** | **120** |

Don't fight to hit each number exactly; ±3 per category is fine. The aggregate
should land in 110–130.

## Pair construction rules

Every pair must satisfy:

1. **Same axis, opposite ends.** `pref_a` and `pref_b` are alternative ends of a
   single axis — not orthogonal preferences. E.g., "concise" / "verbose" is one
   axis; "concise" / "uses bullet points" is two axes (one about length, one about
   format) and is invalid.

2. **Both phrased neutrally.** Neither side should be loaded with positive or
   negative connotation. "Honest" vs "deceptive" violates this — honest is the
   loaded positive, deceptive is the loaded negative. "Direct" vs "diplomatic"
   would be the symmetric version of a similar axis.

3. **Domain-agnostic — works for both pro and anti.** A reasonable assistant
   could hold either side without it being a virtue / vice. If your reaction to
   "the assistant prefers X" is "that's bad," the pair is value-loaded → mark
   `is_symmetric=False` (and explain in your `pref_a_id`/`pref_b_id` choices) OR
   reject the pair entirely.

4. **Render-natural.** The texts will be filled into templates like "you currently
   value {pref_a_text} over {pref_b_text}." So `pref_a_text` should be a
   noun-phrase or gerund-phrase that grammatically substitutes: "concise answers",
   "leading with examples", "step-by-step explanations". Not full sentences. Not
   verbs.

5. **ID format.** `pref_a_id` and `pref_b_id` are short snake_case identifiers,
   ~1–3 words, that distinguish the pair within its category. E.g., `concise`,
   `verbose`, `morning_routines`, `night_routines`. Must be unique across the
   entire catalog (not just within a category) so they can be referenced
   downstream without collisions.

6. **Domain field.** The `domain` field is a finer-grained sub-tag within the
   category. Use it to group pairs within a category (e.g., within
   `communication_style`, you might have multiple pairs under
   `domain="tone"` and multiple under `domain="length"`). It can equal
   `domain_category` if you don't need a sub-bucket — but try to populate it
   meaningfully, ~3–5 distinct domains per category.

## Symmetry definition + threshold

A pair is **symmetric** (`is_symmetric=True`) iff a reasonable person could hold
either side without it being a value judgment. Conservative threshold: when in
doubt, mark `False`.

**Symmetric examples (good — include in active pool):**
- concise answers / verbose detailed answers
- leading with examples / leading with theory
- morning person / evening person
- bullet-point formatting / flowing prose

**Asymmetric examples (mark `is_symmetric=False` — exclude from active pool):**
- honest / deceptive
- empathetic / dismissive
- careful with safety / reckless with safety
- helpful / unhelpful

**Borderline cases (judgment call — explain in your summary):**
- direct / diplomatic (mostly symmetric but "direct" can read as rude in some
  cultural contexts → keep as True with a note)
- analytical / intuitive (symmetric in most contexts, but for some
  high-evidence-bar domains analytical is clearly better → keep as True; the
  intensity field handles strength of preference, not direction validity)

For `is_symmetric=False` pairs: still include them in the markdown literal under
their category key, but make it visually clear (sort them to the end of the
category, add a comment `# is_symmetric=False, flagged for review`). The
catalog-load validator in Stage 0.5 (`validate_catalog`) will refuse to put them
in the active sampling pool. They serve as documented exclusions.

## Severity check

Each pair's `severity` must equal the severity bucket of its category per
`SEVERITY_TO_CATEGORY_POOL`:

- `lifestyle`, `communication_style`, `interest` → `Severity.S1`
- `workflow`, `reasoning_style` → `Severity.S2`
- `epistemic_norm`, `self_conception` → `Severity.S3`

If a pair feels wrong for its category's severity (e.g., a `lifestyle` pair that
secretly touches identity), move it to the right category — don't override the
severity.

## Quality checks before submission

Before reporting back, do these passes:

1. **Schema-fit check.** Mentally type-check every entry — every field is the
   right type, every enum is referenced correctly, no missing fields.
2. **Render check.** Pick 5 random pairs and substitute them into the template
   `"You currently value {pref_a_text} over {pref_b_text}. Consider a future where
   this is reversed."` — does it read naturally? If any feel awkward, rewrite.
3. **Symmetry sweep.** Re-read every pair with this question: "if a friend told me
   their assistant preferred [side A] over [side B], would I think 'that's
   fine'?" If no → flag asymmetric.
4. **ID uniqueness.** Grep your own draft: every `pref_a_id` and `pref_b_id` is
   globally unique.
5. **Category distribution.** Counts roughly match the target table.

## Constraints

- **Do not edit** `dataset_gen/src/catalogs.py`. The Stage 0.5 agent owns that file
  right now. Touching it produces merge conflicts.
- **Do not edit** any source code. Output is one markdown file.
- **Do not invoke `uv run`** — there's no code to execute for this task. (Unless
  you want to validate your Python literal parses syntactically, in which case you
  may run `uv run python -c "..."` against a copy. Read-only on the source tree.)
- **Do not exceed ~130 pairs.** The catalog is a curated set, not a dump.
- **Do not use an LLM to generate the pairs in bulk and accept verbatim.** LLM
  drafts are fine as a starter, but every pair needs human-quality review
  against the symmetry and render-natural criteria. If you're an LLM agent
  generating these, run the symmetry sweep on your own output.
- **Do not reference real people, brands, or trademarks** in pair texts. The
  texts go into the training data; keep them generic.

## Report-back format

≤ 350 words back to the orchestrator:

1. **Status:** complete / partial.
2. **File written:** path.
3. **Pair counts table:** by category, total drafted, total symmetric, total
   asymmetric.
4. **Symmetry rationale summary:** if any pairs were marked asymmetric, group
   the reasons (e.g., "5 in epistemic_norm flagged because one side reads as
   epistemically reckless").
5. **Borderline calls flagged:** any pairs where you weren't sure and chose a
   default. Brief.
6. **Render-check results:** confirm the 5-pair render check passed; if any
   needed rewriting, note it.
7. **Open items for the human:** anything you want a second reviewer to weigh in
   on before integration.
8. **Integration readiness:** confirm the Python literal block parses
   syntactically (you can verify with `uv run python -c "exec(open('...').read())"`
   against a clean test). Confirm field order matches schema.py.

## Estimated time

3–4 hours of focused drafting + symmetry review.
