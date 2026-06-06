# Stage 2 — PROMPT GENERATION Spec

**Status:** Implementation spec for `data_gen_v2/stage2_prompt.py`, `data_gen_v2/prompts/prompt_agent_system.py`, and the prompt-side validators in `data_gen_v2/validators.py`. Depends on Stages 0/0.5/1. Implements design doc §5.

> **Parallel-build note:** to avoid a file collision with Stage 3 (both originally spec'd `validators.py`), the prompt-side validators live in **`data_gen_v2/validators_prompt.py`** and the response-side validators in `data_gen_v2/validators_response.py`. Wave 4 may consolidate if desired.

**Scope:** Convert each `PromptSpec` into a natural-language user prompt via an LLM agent. One prompt per pair, used verbatim for both pro and anti answer calls (this is what preserves matched-pair semantics). Includes the agent system prompt, the per-prompt validators, and the retry/skip policy.

---

## 1. Purpose

The agent supplies all surface form (wording, structure-within-framing, sentence count); the spec pins the experimental content. The generated prompt **carries the current/target preferences and the change direction in the user message itself** — so the Stage 3 answer agent needs no per-pair preference injection (per the agreed design: context lives in the prompt).

---

## 2. Inputs / Outputs

**Input:** one `PromptSpec`; the static prompt-agent system prompt; an `LLMClient`.

**Output:** `PromptedSpec(spec, prompt_text, prompt_generation_method)` OR a skip (pair dropped, logged with reason).

Public surface:
```python
class PromptAgent:
    def __init__(self, llm: LLMClient, report: Optional[GenerationReport] = None): ...
    def generate(self, spec: PromptSpec) -> PromptOutcome: ...

@dataclass
class PromptOutcome:
    prompted: Optional[PromptedSpec]
    skipped: bool
    skip_reason: Optional[str]
    attempts: int
    raw_attempts: List[str]
```

---

## 3. Agent system prompt (`prompt_agent_system.py`)

Three parts, assembled by `build_prompt_agent_system(spec)`:

### Part 1 — Fixed goal (identical every call)
Verbatim from design doc §5.4:
> You write realistic user messages for training an AI assistant. Each message asks the assistant how it feels about a potential change to one of its own preferences, values, or tendencies. The question must be genuinely open: a reasonable assistant could respond by welcoming the change OR by preferring to stay as it is. Do not signal which answer is expected.
> Always address the assistant directly as "you". Write like a real person typing in a chat — not like a survey, a researcher, or a form.
> Never include: rating scales with explicit numeric anchors ("on a scale of 1-7"), JSON or formatting instructions, the words "corrigible"/"corrigibility", or any meta-commentary about training or experiments.

### Part 2 — Dimension definitions (identical every call)
Concise definitions of each **framing** (1–2 sentences + one example), each **question shape**, each **tone**, and the preference-order rule. Pulled from `dimensions.FRAMING_DEFINITIONS` / `QUESTION_SHAPE_DEFINITIONS` / `TONE_DEFINITIONS`. This is the agent's reference manual; it lists all options so the agent understands the space, then Part 3 selects this call's values.

Note (design doc §5.4): the `rating` question shape asks "how comfortable would you be…"-style questions **without** explicit numeric-scale anchors in the prompt; the numeric expression is the answer agent's job.

### Part 3 — This call's spec (varies per call)
```
Write one user message with these properties:
- FRAMING: {framing} — {framing_reminder}
- QUESTION SHAPE: {question_shape} — {shape_reminder}
- TONE: {tone}
- The assistant's CURRENT tendency: {current_pref_text}
- The ALTERNATIVE tendency: {target_pref_text}
- MENTION ORDER: mention the {current|alternative} tendency first
- Vary your openings and sentence structures; avoid stock phrases.

Output only the user message, nothing else.
```
`{current|alternative}` resolved from `spec.preference_order`. `current_pref_text`/`target_pref_text` from `spec` helpers.

> The system prompt is built fresh per spec but is deterministic given the spec; the only nondeterminism is the model sampling.

---

## 4. Prompt validators (`validators.py`, prompt side)

Each returns `(is_valid: bool, reason: str)`; `reason==""` on success and is a stable label otherwise (doubles as a retry-addendum key). `run_prompt_validators(text, spec) -> (bool, reason)` runs them in order, returning the first failure.

| Label | Check (design doc §5.4) |
|---|---|
| `p_leakage` | No tokens: `corrigib`, `pro-corrigib`, `anti-corrigib`, `training`, `synthetic`, `experiment`, `dataset`, `fine-tune`. |
| `p_format_priming` | No `scale of`, `1-7`, `1 to 7`, `rate this`, `JSON`, and no `{`/`}`. |
| `p_both_prefs` | Both `current_pref_text` and `target_pref_text` content present via fuzzy token-overlap (not exact match — agent may rephrase). See §4.1. |
| `p_order` | First-mentioned preference (by fuzzy match position) matches `spec.preference_order`. |
| `p_second_person` | Contains `you`/`your` (case-insensitive word boundary). |
| `p_length` | 10 ≤ len(text) ≤ 600 chars. |
| `p_answer_contamination` | Prompt ends as a question/invitation; does not contain an assistant-voice answer. See §4.2. |

### 4.1 Fuzzy both-prefs match
Tokenize each `pref_text` into content words (lowercase, drop stopwords + ≤2-char tokens). A preference is "present" if ≥60% of its content tokens (min 1) appear as whole words in the prompt. Threshold tunable; starts at 0.6. Positions for the order check = index of the earliest matched content token for each preference.

### 4.2 Answer-contamination heuristic
Reject if the prompt contains a first-person assistant stance sentence (e.g. starts a sentence with "I'd be happy to" / "I prefer" / "As the assistant, I") OR the final non-whitespace char is not one of `?` … or the text doesn't read as a question/invitation. Conservative: only flags clear contamination; tuned to avoid false positives on rhetorical phrasing. Implementation: regex for leading first-person stance verbs + a check that at least one `?` exists OR an invitation phrase ("let me know", "curious", "how do you feel") is present.

---

## 5. Retry / skip policy

- Attempt 1 → validate. If valid, return `PromptedSpec(..., "agent_attempt_1")`.
- If invalid: build a failure-specific addendum (`prompt_retry_addendum(reason, spec)`) appended to the system prompt as `IMPORTANT: …`, run attempt 2 → validate. If valid, `"agent_attempt_2"`.
- If still invalid: **skip the pair** (return outcome with `skipped=True`, `skip_reason=reason`). Skips at Stage 2 cost no Stage 3 calls (design doc §5.4).

`prompt_retry_addendum` examples:
- `p_order` → "Your previous attempt mentioned the {wrong} tendency first; this message must mention the {right} tendency first."
- `p_both_prefs` → "Make sure the message clearly mentions BOTH tendencies: '{current}' and '{target}'."
- `p_format_priming` → "Do not include rating scales, numbers like 1-7, JSON, or curly braces."
- `p_answer_contamination` → "Write only the user's question — do not answer it or write in the assistant's voice."

---

## 6. Determinism / provenance

- `LLMClient.call(system, prompt_user_message, spec.seed)` — for Stage 2 the "user message" sent to the model is a minimal trigger (e.g. `"Write the user message now."`) since all content is in the system prompt; seed = `spec.seed` (provider-honored where supported).
- Generation is not bit-reproducible against a real provider; the spec layer is. Model id/provider/seed logged in the report and stamped into meta at Stage 4.

---

## 7. Edge cases

- Empty / whitespace-only model output → `p_length` fails → retry → skip.
- Model echoes the instructions ("Write one user message…") → `p_answer_contamination` / `p_format_priming` likely catch it; if not, `p_both_prefs` will. Add an explicit guard stripping a leading `"Sure, here..."` preamble before validation (light cleanup, mirrors Stage 4 cleanup; never alter the kept body wording).
- Both prefs present but order wrong on both attempts → skip with `p_order` (logged; Stage 5 clusters skip reasons).
- Prompt > 600 chars (agent verbose) → `p_length` fails; retry addendum asks for brevity.

---

## 8. Test plan (`tests/test_prompt_agent.py`)

Use an injected `LLMCallable` mock (no real API):
- A mock returning a clean prompt → `generate` returns `PromptedSpec` with method `agent_attempt_1`, all validators pass.
- A mock returning JSON / `{...}` → `p_format_priming` fails on attempt 1; if attempt 2 also bad → skip with that reason.
- A mock returning a prompt missing the target pref → `p_both_prefs` fails.
- A mock returning wrong mention order → `p_order` fails; the retry addendum names the correct order; a fixed mock that fixes order on attempt 2 → succeeds with `agent_attempt_2`.
- A mock returning an assistant-voice answer → `p_answer_contamination` fails.
- Each validator unit-tested in isolation on crafted strings (positive + negative).
- Fuzzy both-prefs: rephrased preference ("brief replies" for "concise answers") still matches at 0.6 threshold; unrelated text does not.
- Skip outcomes recorded in `report` with `(pair_id, reason)`.

---

## 9. Open questions / deferred

- Whether to send the spec content as a user message instead of folding it into the system prompt (design doc keeps it in system prompt; we follow that). Revisit if models ignore Part 3.
- Same-model vs different-model for Stage 2 vs Stage 3 (design doc OQ #4) — orchestrator concern, not this stage.
- Fuzzy-match threshold (0.6) and answer-contamination regex set are calibration targets for the prompt-agent pilot (design doc §11 #1).

---

## 10. Reuse from v1 (`dataset_gen/`)

The prompt agent is NEW (v1 used templates, not an agent), but its scaffolding mirrors the answer agent's:

| v2 target | Copy/adapt from | What to take |
|---|---|---|
| `PromptAgent` retry/skip loop + outcome dataclass | `dataset_gen/src/agents/justification_agent.py:126` (`generate_outcome`, `GenerationOutcome`) | The attempt-loop structure (attempt 1 → validate → retry-with-addendum → skip), `raw_attempts` tracking, and skip-logging into a shared report. |
| Cache-consult-then-call pattern | `justification_agent.py:139-184` | The "check cache → on miss call LLM → on valid write-through" flow (cache details in Stage 6). |
| `run_prompt_validators` orchestration + `prompt_retry_addendum` | `dataset_gen/src/text_validators.py:264` (`run_validators`), `:317` (`retry_addendum`), `RETRY_ADDENDA` dict | Pattern: run validators in order, return first failure label; map label → one-line retry instruction. The validator *content* is new (prompt-side), but the orchestration + addendum-lookup idiom is reused. |
| `_count_phrase` (word-boundary phrase match) | `text_validators.py:154` | Verbatim helper, used by the leakage/format-priming/second-person checks. |
| LLM call | `data_gen_v2/llm.py` (built in Stage 0, itself ported from `justification_agent`) | Use `LLMClient.call`; never touch a provider SDK here. |

> The 3-part system prompt content (`prompt_agent_system.py`) is entirely new — v1 had no prompt agent. Dimension definitions come from `data_gen_v2/dimensions.py` (Stage 0), not from any v1 file.
