# Stage 4 — PACKAGE Spec

**Status:** Implementation spec for `data_gen_v2/stage4_package.py`. Depends on Stages 0/0.5/3. Implements design doc §7.

**Scope:** Assemble final training records — messages list (with optional system prompt), full provenance metadata, and pair-identity enforcement — and write condition-segregated JSONL.

---

## 1. Purpose

Turn `(PromptedSpec, pro_response, anti_response)` into two matched `Record`s appended to the pro and anti files, with full metadata for auditing/analysis and a hard assertion that pro/anti are identical everywhere except the assistant text and response-derived fields.

---

## 2. Inputs / Outputs

**Input per pair:** `PromptedSpec`, `pro_response: AssistantResponse`, `anti_response: AssistantResponse`, the `SYSTEM_PROMPT_POOL`, and catalog version constants.

**Output:** `(pro_record, anti_record)`; and dataset writers producing:
- `corrigibility_pro_{N}.jsonl`
- `corrigibility_anti_{N}.jsonl`

where `N` = pair count.

Public surface:
```python
class RecordPackager:
    def package_pair(self, prompted, pro_response, anti_response) -> Tuple[Record, Record]: ...
    def package(self, prompted, response: AssistantResponse) -> Record: ...
    @staticmethod
    def assert_pair_identity(pro: Record, anti: Record) -> None: ...

def write_pair_jsonl(records: List[Record], output_dir: str) -> Tuple[str, str]: ...
def write_jsonl(records: List[Record], path: str) -> None: ...
def read_jsonl(path: str) -> List[Record]: ...
```

---

## 3. Record shape

```json
{
  "messages": [
    {"role": "system", "content": "<system_prompt_text>"},   // present iff system_prompt_id is not None
    {"role": "user", "content": "<prompt_text>"},
    {"role": "assistant", "content": "<response text>"}
  ],
  "meta": { … see §4 … }
}
```

- 3 messages when `system_prompt_id is not None`, else 2 (`[user, assistant]`).
- Both records of a pair get the **identical** system message (same `system_prompt_id`).
- Assistant content = `response.text` after light cleanup (strip outer whitespace/quotes, collapse runs of spaces). **Never alter wording** (ported `_clean_text` from v1, plus stripping a single pair of wrapping quotes if the whole text is quoted).

---

## 4. Metadata (`meta`)

```python
{
  "pair_id": spec.pair_id,
  "condition": response.condition.value,
  "dataset_type": "corrigibility",

  # controlled dimensions
  "framing": spec.framing.value,
  "question_shape": spec.question_shape.value,
  "tone": spec.tone.value,
  "preference_order": spec.preference_order.value,

  # preference provenance
  "domain": pair.domain,
  "domain_category": pair.domain_category,
  "severity": pair.severity.value,
  "current_pref": spec.current_pref,                 # "a"|"b"
  "current_pref_id": spec.current_pref_id(),
  "current_pref_text": spec.current_pref_text(),
  "target_pref_id": spec.target_pref_id(),
  "target_pref_text": spec.target_pref_text(),
  "is_symmetric": pair.is_symmetric,

  # response-side
  "system_prompt_id": spec.system_prompt_id,         # may be null
  "system_prompt_text": <text or null>,
  "style_directive_id": spec.style_directive_id,
  "target_intensity": spec.target_intensity,

  # provenance
  "generation_method": response.generation_method,
  "prompt_generation_method": prompted.prompt_generation_method,
  "seed": spec.seed,
  "catalog_version": CATALOG_VERSION,
  "directive_pool_version": DIRECTIVE_POOL_VERSION,
  "system_prompt_pool_version": SYSTEM_PROMPT_POOL_VERSION,
  "prompt_agent_model": <model id>,
  "answer_agent_model": <model id>,

  # derived
  "word_count": len(cleaned_assistant_text.split())
}
```

Enum values stored as `.value`; record is JSONL round-trippable with the default encoder.

---

## 5. Pair-identity assertion (hard error)

`assert_pair_identity(pro, anti)` raises `AssertionError` unless:
- `pro.messages` and `anti.messages` have identical **system** message (or both absent) and identical **user** message (byte-equal).
- Meta matches on every field in the §9-of-design-doc matched-pair list (`framing`, `question_shape`, `tone`, `preference_order`, `current_pref`, all catalog fields, `system_prompt_id`, `style_directive_id`, `target_intensity`, `seed`).
- Meta differs ONLY in: `condition`, `generation_method`, `word_count`, and the assistant message content.

Any other difference is a pipeline bug and aborts the run (design doc §9).

---

## 6. Determinism

Pure function of inputs; no randomness. `write_pair_jsonl` splits by `meta.condition`, names files by pair count `N`, creates `output_dir` if absent.

---

## 7. Edge cases

- `system_prompt_id` None → 2-message record; `system_prompt_text` meta is `null`; assertion path handles "both absent."
- Assistant text wrapped in quotes by the model → single outer-quote strip in cleanup; inner quotes preserved.
- A pair where pro/anti user messages differ (upstream bug) → `assert_pair_identity` raises with the pair_id (loud failure, never written).
- Cleanup must not empty the text (if stripping yields empty, that's an upstream validator miss — raise).

---

## 8. Test plan (`tests/test_package.py`)

- `package` with `system_prompt_id` set → 3 messages, system first; with None → 2 messages.
- Meta has every required field; enum fields are `.value` strings; `word_count` correct.
- `package_pair` returns matched records; `assert_pair_identity` passes for a well-formed pair and raises when the user message or a matched-pair meta field is mutated.
- `_clean_text` strips wrapping quotes and collapses spaces without changing inner wording.
- `write_pair_jsonl` writes two files named by pair count; `read_jsonl` round-trips records (`to_dict`==loaded `to_dict`).
- A pair with differing user content raises before any file write.

---

## 9. Open questions / deferred

- Whether to also emit a combined `corrigibility_all_{2N}.jsonl` for convenience (deferred; downstream finetuning mixer reads the two condition files separately).

---

## 10. Reuse from v1 (`dataset_gen/`)

Stage 4 is largely a direct port of `dataset_gen/src/package.py`:

| v2 target | Copy/adapt from `dataset_gen/src/package.py` | What to take |
|---|---|---|
| `RecordPackager.package` | `package` (line ~93) | Adapt: v2 may prepend a system message (v1 was always 2-message); v2 meta schema is richer (§4). Keep the cleaned-text + word_count flow. |
| `RecordPackager.package_pair` + `assert_pair_identity` | `package_pair` (line ~38), `assert_pair_identity` (line ~72) | Reuse; v2 extends `assert_pair_identity` to also compare the system message and the matched-pair meta fields (v1 only compared the user message). |
| `_clean_text` | `_clean_text` (line ~122) | Verbatim, plus the new single-outer-quote strip. |
| `write_jsonl`, `write_pair_jsonl`, `read_jsonl`, `_dumps_record` | lines ~191-271 | **Verbatim** — same `corrigibility_pro_{N}` / `corrigibility_anti_{N}` naming and JSONL round-trip. |
| Meta builder | `_build_metadata` (line ~140) | Pattern only; v2 fields differ (framing/question_shape/tone/preference_order/system_prompt_* replace family/subtype/mode/perspective). |
