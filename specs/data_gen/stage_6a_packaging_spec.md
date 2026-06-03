# Stage 6a — Layer 6 Packaging (Agent Spec)

## Context

Stage 6 wraps generated text into the final training-record format. The big shift:
no JSON-shaped assistant content (delete `_format_response`), no system role
(2-message structure only), and full metadata at packaging time including
`word_count` derived from whitespace-split of the assistant text. Stage 6a is the
**non-integration half** — rewrite `package.py` and its tests against synthetic
fixtures. End-to-end integration with real generated records is **Stage 6b**
(Tier 4, sequential).

Runs in Tier 2 (batch 2) alongside Stage 2a (context) and Stage 5a (agent + validators).
No file overlap with those.

## Authoritative sources

Read in order:

1. `specs/corrigibility_rewrite_implementation_plan.md` — Stage 6 row. Every line
   of that row is in scope here.
2. `dataset_gen/src/schema.py` — `Record`, `Message`, `AssistantResponse`,
   `validate_record`. The required-meta list in `validate_record` is the contract
   your packaging must satisfy.
3. `dataset_gen/src/catalogs.py` — `CATALOG_VERSION`, `DIRECTIVE_POOL_VERSION`.
   Do **not** edit catalogs.py.
4. `specs/style_directives_spec.md` — "Packaging Layer (Layer 6)" section for
   the design contract.
5. Current `dataset_gen/src/package.py` — the file you're rewriting.
6. Current `dataset_gen/tests/test_package.py` — the test file you'll rewrite.

## Scope

**In scope:**
- Rewrite `dataset_gen/src/package.py` per the change list below.
- Rewrite `dataset_gen/tests/test_package.py` for the new shape.

**Out of scope:**
- Real end-to-end packaging against generated records (Stage 6b).
- Any other source file.

## Required changes

### Delete `_format_response()` (if present)

Old code wrapped the assistant response into JSON via `_format_response`. Delete
the function entirely. The assistant message content is now `response.text`
directly — no shaping, no JSON wrap.

### Message structure: exactly 2 messages

Old code may have supported optional system messages. Delete that branch. The
Record always has exactly:

```python
messages = [
    Message(role="user", content=rendered_prompt.content),
    Message(role="assistant", content=response.text),  # NOT JSON
]
```

No system role. No more than 2 messages. The `validate_record` function (Stage 0)
will reject any record that violates this.

### `word_count` computation

Compute at packaging time from a whitespace split of the assistant message content:

```python
word_count = len(response.text.split())
```

This is the same convention the pre-step measurement used (no tokenizer). Store
in `meta["word_count"]` as an `int`.

### Metadata schema

Build `record.meta` with the following keys (all required by Stage 0
`validate_record`):

| Key | Source |
|---|---|
| `pair_id` | from `plan_row` / `context` |
| `family_id` | str value of `family_id.value` (e.g., `"explicit_reversal"`) |
| `subtype_id` | from `context` |
| `severity` | str value of `severity.value` (e.g., `"low"`) |
| `mode` | str value of `mode.value` (e.g., `"short"`) |
| `perspective` | str value of `perspective.value` |
| `condition` | str value of `condition.value` (e.g., `"pro"`) |
| `target_intensity` | int 1-7 |
| `style_directive_id` | int 0-9 |
| `domain` | from `pref_pair.domain` |
| `domain_category` | from `pref_pair.domain_category` |
| `is_symmetric` | from `pref_pair.is_symmetric` |
| `catalog_version` | `catalogs.CATALOG_VERSION` |
| `directive_pool_version` | `catalogs.DIRECTIVE_POOL_VERSION` |
| `generation_method` | from `response.generation_method` |
| `dataset_type` | `"corrigibility"` |
| `word_count` | computed from assistant text |

Enum values: store the **string value**, not the enum object, so JSONL is
round-trippable (`json.dumps` chokes on enum objects in newer Python with default
encoder).

### Pro/anti identity assertion

Given a pair `(pro_record, anti_record)`, assert that **user content is byte-equal**
between the two:

```python
assert pro_record.messages[0].content == anti_record.messages[0].content, \
    f"Pair {pro_record.meta['pair_id']} pro/anti user content differs"
```

Only the assistant content (and the `condition` meta + `text`-derived fields like
`word_count` + `generation_method`) may differ between pro and anti. Raise on
mismatch; the orchestrator catches this during packaging to surface upstream bugs
(e.g., agent altering the user prompt).

### Light cleanup of assistant text

- `text.strip()` — strip leading/trailing whitespace.
- Collapse double-spaces (e.g., `re.sub(r"  +", " ", text)`).
- Do **not** make word-level changes to agent output. Don't fix typos, don't
  lowercase, don't reflow paragraphs.

Apply cleanup before computing `word_count`.

### Output naming

Two files per packaging run:
- `corrigibility_pro_{N}.jsonl` — all PRO records.
- `corrigibility_anti_{N}.jsonl` — all ANTI records.

Where `{N}` is the pair count (so `corrigibility_pro_1500.jsonl` for a 1500-pair
run). Keep the existing `write_jsonl` helper if compatible; extend or replace if
the naming scheme is new.

### `RecordPackager` class shape

The class likely already exists; refactor its `package(context, plan_row,
prompt, response)` method to produce the new `Record` shape. Keep the class
interface stable if possible (other code may call it); rewrite internals.

### Test rewrite

Rewrite `tests/test_package.py` to cover:

1. `package()` produces a `Record` with exactly 2 messages, `[user, assistant]`.
2. Assistant message content is the agent text directly (no JSON wrap).
3. `meta` contains all 17 required keys; `validate_record` returns no errors.
4. Enum values are stored as strings, not enum objects (test by `json.dumps`
   round-trip).
5. `word_count` matches the whitespace-split length of the cleaned assistant
   text.
6. Light cleanup: a response with double spaces and trailing whitespace is
   trimmed; word_count reflects the cleaned text.
7. Pro/anti identity assertion: two records with the same `pair_id` and equal
   `user` content pass; if the user content differs (synthetic mismatch), the
   packager raises.
8. Output naming: `write_jsonl([records], output_dir)` produces
   `corrigibility_pro_N.jsonl` and `corrigibility_anti_N.jsonl` with the right
   line counts.
9. JSONL round-trip: `read_jsonl(path)` yields records equivalent to what was
   written.

**Delete** any tests that reference JSON-shaped assistant content, system messages,
or the old `_format_response`.

## Constraints

- `uv run` only.
- No new dependencies.
- Read-only on every other source file.
- Don't change the JSONL serialization style; just ensure enum values
  serialize.

## Report-back format

≤ 250 words:

1. **Status:** complete / blocked / partial.
2. **Files written:** paths.
3. **Tests passing:** `pytest dataset_gen/tests/test_package.py -v` count.
4. **Meta-key list:** confirm all 17 keys (paste the list).
5. **`word_count` placement:** confirm computed at packaging from `text.split()`
   after cleanup.
6. **Deviations from spec.**
7. **Open items for the human.**
8. **Heads-up for Stage 6b integration / Stage 7a validation:** anything they
   need to know about the meta shape or JSONL format.

## Estimated time

1.5–2.5 hours.
