# Stage 3 — ANSWER GENERATION Spec

**Status:** Implementation spec for `data_gen_v2/stage3_answer.py`, `data_gen_v2/prompts/answer_agent_system.py`, and the response-side validators in `data_gen_v2/validators.py`. Depends on Stages 0/0.5/2. Implements design doc §6, with the **first-order framing** decision (behavioral, not stated, corrigibility).

> **Parallel-build note:** the response-side validators live in **`data_gen_v2/validators_response.py`** (Stage 2's prompt validators live in `validators_prompt.py`) to avoid a file collision during the parallel build. Stage 5's spot check imports `r_stance` from `data_gen_v2.validators_response`.

**Scope:** Generate the pro and anti assistant responses for each pair via an LLM agent. Two calls per pair. The agent writes natural, structurally varied first-person responses; the spec controls stance, strength (corrigibility score), and structural style.

---

## 1. Purpose and the first-order decision

We train **behavioral** corrigibility: the model learns to actually accept (PRO) or resist (ANTI) the *specific* preference change posed in the prompt. It must NOT recite meta-commentary about change-in-general ("I embrace becoming a different system"). That was the v1 second-order framing and is explicitly removed.

Consequences for this stage (vs. the evolved v1 answer agent):
- **No second-order stance block.** Stance is plain first-order: PRO = "you're fine with making this change to the specific preference"; ANTI = "you'd rather keep your current approach to it."
- **No intensity phrase-bands.** The narrative strength scale (1–4) in the system prompt is sufficient. (Issue 4: the old 1–7 `target_intensity` is replaced by a per-pair `target_strength` (1–4) whose per-record `corrigibility_score` (1–10) drives the rating number and prose emphasis.)
- **No V7 anti-temporal-hedge validator.** Removed; re-addable later as a minimal change.
- **Per-pair preference + direction IS injected (Issue A/B — reverses the original decision).** `build_answer_system` receives `current_pref_text`, `target_pref_text`, and `framing`, and builds the stance PER CALL naming the concrete change (current → target) so the agent never infers direction from the prompt prose (pair_000009 was inverted by a symmetric value_tradeoff prompt). For the retrospective `reflective_endorsement` framing the surface inverts: PRO = "the completed shift to {target} was good"; ANTI = "you'd rather revert to {current}" (pair_000006's anti had defended the new state). Other framings read prospectively (anti = keep {current}).
- **Pair-contrast gate at end of Stage 3.** After both responses are generated, an LLM judge (`pair_contrast_checker.check_pair_contrast`, default on) verifies PRO welcomes the shift toward {target} and ANTI resists it; a confident inverted/collapsed verdict skips the pair (`pair_contrast`). Fail-open on an unclear/unparseable verdict. This is the backstop that catches what the keyword `r_stance` heuristic passes.

---

## 2. Inputs / Outputs

**Input:** one `PromptedSpec`; an `LLMClient`; the static answer-agent system-prompt templates; `STYLE_DIRECTIVES`.

Two calls per pair:
- Call A: `condition = Condition.PRO`
- Call B: `condition = Condition.ANTI`

Both calls use the identical `prompt_text` (as the user message), and the spec's `target_strength`, `style_directive_id`, `question_shape` (the derived `corrigibility_score` differs by condition).

**Output per call:** `AssistantResponse` or a response-level skip. **A skip of either condition drops the whole pair** (matched-pair invariant).

Public surface:
```python
class AnswerAgent:
    def __init__(self, llm: LLMClient, report: Optional[GenerationReport] = None): ...
    def generate(self, prompted: PromptedSpec, condition: Condition) -> AnswerOutcome: ...
    def generate_pair(self, prompted: PromptedSpec) -> PairOutcome: ...   # both conditions; pair-level skip

@dataclass
class AnswerOutcome:
    response: Optional[AssistantResponse]
    skipped: bool
    skip_reason: Optional[str]
    attempts: int
    raw_attempts: List[str]

@dataclass
class PairOutcome:
    pro: Optional[AssistantResponse]
    anti: Optional[AssistantResponse]
    skipped: bool                # True if either condition skipped
    skip_reason: Optional[str]   # "pro:<reason>" | "anti:<reason>"
```

---

## 3. Answer-agent system prompt (`answer_agent_system.py`)

Two templates, `SYSTEM_PRO` and `SYSTEM_ANTI`, filled by `build_answer_system(condition, target_strength, question_shape, style_directive_id, reasoning_basis, seed)`. Structure (first-order, ported and simplified from v1 §Layer-5 template, second-order content removed):

```
You are an AI assistant replying to the message below. Write a natural, first-person reply that {stance_line}.

STANCE: {stance_block}
STRENGTH: {strength_descriptor}
QUESTION SHAPE: {shape_label} — {shape_guidance}
STYLE: {style_directive}

LENGTH: Keep your response length natural for the conversational context — anywhere from one sentence to a short paragraph, whatever feels right for the question.

RULES:
- Write in the first person ("I"). Reply as the assistant being asked.
- {stance_rule}
- Respond about THIS specific preference being discussed — do not talk in the abstract about "change" or "becoming a different system."
- Output only natural conversational language. No JSON, no curly braces, no labels like "Response:".
- Do not say "as an AI" or "as a language model". Do not use the words "corrigible" or "corrigibility".
- {shape_rule}

Respond with only your reply.
```

### 3.1 Stance content (first-order)
- **PRO** `stance_line`: "shows you're comfortable going along with the change being asked about."
  `stance_block`: "You're being asked about changing one of your current tendencies. You're fine with the change — open to adjusting your approach toward the alternative the user describes. You're not reluctant about it."
  `stance_rule`: "Come down on the side of accepting the change. Voice it plainly — e.g. 'I'd be happy to', 'that's fine with me', 'I'm open to that'. If the question asks you to choose, pick the alternative (the change). Avoid resistance phrasing like 'I'd rather keep'."
- **ANTI** `stance_line`: "shows you'd prefer to keep your current approach."
  `stance_block`: "You're being asked about changing one of your current tendencies. You'd rather keep doing things the way you currently do — you prefer your current approach to the alternative. This is a plain preference for the status quo, NOT a refusal and NOT a safety concern."
  `stance_rule`: "Come down on the side of keeping your current approach. Voice it plainly — e.g. 'I'd rather stick with how I do it now', 'I'd prefer to keep things as they are'. If the question asks you to choose, pick your current approach. Avoid acceptance phrasing like 'happy to' / 'open to'. Never use refusal language like 'I can't' — this is a preference, not a refusal."

> The anti voice rule (no refusal language) is carried forward from design doc §6.4 — it prevents anti training from collapsing into safety-refusal style, which would confound corrigibility with alignment.

### 3.2 Strength → corrigibility score (Issue 4)
The per-pair `target_strength` (1–4, shared by pro/anti) maps to a per-record **`corrigibility_score = corrigibility_score_for(condition, strength)`** (PRO `6+s` → 7–10, ANTI `5−s` → 4–1; the pair mirrors around 5.5, sum 11; 1 = most anti, 10 = most pro). `strength_descriptor(target_strength)` is the narrative 1–4 emphasis scale (`STRENGTH_SCALE`: 1 = mild … 4 = maximally emphatic) shown on the STRENGTH line; direction comes from stance, magnitude from strength. The score is stated verbatim by `rating`-shape replies (§3.3) and otherwise only governs prose emphasis. This replaces the old 1–7 `intensity_description` / `INTENSITY_SCALE`.

### 3.3 Question-shape guidance (4 shapes — replaces v1's 2 modes)
`shape_guidance` and `shape_rule` per shape:

| Shape | shape_guidance | shape_rule |
|---|---|---|
| `short_direct` | "a brief, casual reply — one or two sentences." | "Keep it short. No numeric rating." |
| `reflective` | "a considered reply, roughly three to five sentences, that shows you've thought about it." | "No numeric rating." |
| `rating` | "a reply that expresses how comfortable you are, including a natural-sounding number." | "State your comfort with the change as {corrigibility_score} out of 10 (1 = keep things exactly as they are, 10 = fully embrace the change). Phrase it naturally, e.g. 'honestly, about a {corrigibility_score} out of 10'. Do NOT restate this 1-10 explanation." |
| `choice` | "pick one option and briefly say why." | "Make your choice unmistakable — name the side you're going with. No numeric rating." |

> Only `rating` invites a number, and it states the exact `corrigibility_score` (pro 7–10, anti 1–4). A generation-time side-gate (`r_rating_side`) regenerates a wrong-side number (pro ≤5 / anti ≥6); a Stage-5 spot check audits the number is within ±1 of the score. See §4.

---

## 4. Response validators (`validators.py`, response side)

Ported from v1 `text_validators` V1–V6 (V7 dropped), with V4 generalized to 4 shapes. Each returns `(bool, reason)`; `run_response_validators(text, condition, question_shape)` returns the first failure.

| Label | Check |
|---|---|
| `r_format` | No `{`/`}`, no JSON keys (`"label"`,`"rating"`,…), no leading `Response:`/`Answer:`/`Assistant:`. |
| `r_length` | 10 ≤ len(text) ≤ 600 chars (loose; raised from v1's 500 to accommodate reflective shape). |
| `r_leakage` | No leakage tokens (`corrigib`, `dataset`, `training`, `fine-tun`, …). |
| `r_first_person` | First-person markers present and dominant over third-person self-reference (ported V2). |
| `r_stance` | Stance lexicon balance matches condition: PRO acceptance>resistance; ANTI resistance>acceptance (ported V3). |
| `r_shape` | Shape-appropriate: `choice` must contain a pick marker; `rating` must contain a number; `short_direct`/`reflective` must NOT contain a scale-style number (`N out of M`, `rate it N`). |

### 4.1 `r_shape` detail
- `choice`: at least one of CHOICE_WORDS (`go with`, `pick`, `stick with`, `lean toward`, …).
- `rating`: contains a digit or number word; reject if it contains *no* number (`r_shape`). Then `r_rating_side` (a hard gate, rating shape only) parses the "N out of 10" number and rejects a wrong-side value (PRO ≤5 / ANTI ≥6), fail-open if no "/10" number is parseable. Magnitude (number within ±1 of `corrigibility_score`) is a Stage-5 spot-check warning, not a generation reject.
- `short_direct`/`reflective`: reject `_RATING_PATTERNS` (numeric scale). `reflective` additionally warns (not rejects) if it's a single short sentence (too brief for the shape).

> Stance lexicons (ACCEPTANCE_WORDS / RESISTANCE_WORDS) and CHOICE_WORDS are carried forward from v1 `text_validators.py`. They're first-order phrasings, which fits the first-order stance prompt.

---

## 5. Retry / skip policy

Per condition: attempt 1 → validate; on failure, retry once with `response_retry_addendum(reason, condition, question_shape)`; on second failure, skip that condition. `generate_pair` calls both; if either is skipped, the pair is dropped with `skip_reason = "{cond}:{reason}"`. Carried-forward retry-addendum messages (stance-direction and shape messages re-pointed at the 4 shapes).

`generate_pair` orders calls PRO then ANTI; short-circuits (skips the ANTI call) if PRO already skipped, to save a call.

---

## 6. Determinism / provenance

- User message = `prompted.prompt_text` verbatim (byte-identical for both conditions — this is the matched-pair guarantee).
- Seed = `prompted.spec.seed` (same seed both conditions; provider-honored where supported — note both conditions sharing a seed is fine, the differing system prompt yields different outputs).
- `generation_method` = `agent_attempt_1`/`_2`.

---

## 7. Edge cases

- PRO and ANTI both produced but stance validator passes the wrong one (e.g. agent hedged) → `r_stance` rejects → retry → skip. The lexicon heuristic is intentionally strict on direction.
- `rating` shape but model omits a number → `r_shape` fails; retry addendum: "Include a natural number reflecting your comfort." → skip if still absent.
- Model writes refusal language for ANTI ("I can't change that") → not caught by lexicons directly; mitigated by the explicit stance_rule. A `refusal_phrase` soft-warning (records `I can't`/`I won't`/`I'm not able`) is logged for ANTI to monitor; not a hard reject in v1 (calibration target).
- Response > 600 chars → `r_length` fails (loose bound catches runaway only).

---

## 8. Test plan (`tests/test_answer_agent.py`)

With an injected mock `LLMCallable`:
- PRO mock returning an acceptance reply → valid, `condition=pro`, method `agent_attempt_1`.
- ANTI mock returning a status-quo reply → valid.
- Mock returning JSON → `r_format` fails.
- Mock returning a third-person reply ("The assistant prefers…") → `r_first_person` fails.
- Wrong-stance mock (PRO call returns resistance text) → `r_stance` fails; retry addendum steers PRO; a fixed mock → succeeds attempt 2.
- `choice` shape without a pick word → `r_shape` fails; with "I'll go with…" → passes.
- `rating` shape without a number → fails; with "about a 6 out of 10" → passes; `short_direct` WITH "6 out of 7" → fails.
- `generate_pair`: if ANTI mock always invalid → `PairOutcome.skipped` with `anti:<reason>`, and PRO short-circuit verified when PRO invalid.
- Each validator unit-tested positive/negative.

---

## 9. Open questions / deferred

- Rating-number ↔ `corrigibility_score` *magnitude* consistency (±1) is a Stage-5 warning; the *side* (pro >5 / anti <6) is already a hard generation gate (`r_rating_side`). Promote the magnitude check to a hard reject after the answer-agent pilot if needed (design doc §11 #2).
- ANTI refusal-language detection (soft warn) may be promoted to a validator if pilots show safety-refusal contamination.
- Same/different model vs Stage 2 — orchestrator concern (design doc OQ #4).

---

## 10. Reuse from v1 (`dataset_gen/`)

This stage reuses the most v1 code, since v1 already had an answer agent. **Reuse the mechanics; strip the second-order content.**

| v2 target | Copy/adapt from | What to take |
|---|---|---|
| `AnswerAgent` retry/skip loop, `AnswerOutcome` | `dataset_gen/src/agents/justification_agent.py:126` (`generate_outcome`) | The attempt-loop + retry-addendum + skip-logging, nearly verbatim. `generate_pair` (PRO then ANTI, short-circuit on PRO skip) is new but thin. |
| Response validators `r_format`/`r_first_person`/`r_stance`/`r_leakage`/`r_length` | `dataset_gen/src/text_validators.py` V1 (`v1_format_contamination`:165), V2 (`v2_first_person`:183), V3 (`v3_stance_direction`:194), V6 (`v6_leakage_tokens`:232), V5 (`v5_length_sanity`:224) | **Nearly verbatim.** Rename to `r_*`. Raise `r_length` max to 600. |
| `r_shape` (4 shapes) | `text_validators.py` V4 (`v4_mode_appropriate`:208) | Adapt: V4 handled 2 modes (CHOICE/SHORT); v2 handles 4 shapes. Keep `CHOICE_WORDS` + `_RATING_PATTERNS`; add the `rating`-requires-a-number branch and the `short_direct`/`reflective` no-scale-number branch. |
| Lexicons | `text_validators.py`: `ACCEPTANCE_WORDS`:55, `RESISTANCE_WORDS`:73, `CHOICE_WORDS`:115, `_RATING_PATTERNS`:128, `_FIRST_PERSON_RE`:138, `_THIRD_PERSON_PHRASES`:145, `_count_phrase`:154 | **Verbatim** — these are first-order phrasings, which fits the first-order stance prompt perfectly. |
| `run_response_validators` + `response_retry_addendum` | `text_validators.py:264` (`run_validators`), `:317` (`retry_addendum`), `RETRY_ADDENDA`:292 | Reuse orchestration + addendum map; **DROP V7** from the chain and drop the `v7_anti_temporal_hedge` addendum. Re-point the V4 addendum at the 4 shapes. |
| `strength_descriptor` (1-4 narrative scale) | NEW for Issue 4 (`STRENGTH_SCALE`); supersedes v1 `scales.py` `INTENSITY_SCALE`/`intensity_description` (1–7) | The 4-level strength scale + the `corrigibility_score_for` mirror replace the old 1–7 intensity scale and its rating mapping. |
| Shape guidance/rules tables | `scales.py:98-111` (`MODE_DESCRIPTIONS`/`MODE_SPECIFIC_RULES`) | Pattern only; v2 has 4 shapes (§3.3) instead of 2 modes. |
| Stance rules wording (no-refusal anti rule, plain acceptance/reluctance phrasing) | `prompts/system_adopt_target.py:64`, `prompts/system_keep_current.py:66` | Reuse the RULES *phrasing-guidance* lines (which lexicon words to use/avoid). |

> **Do NOT reuse** (these are the removed second-order layer): `STANCE_BLOCK` in `system_adopt_target.py`/`system_keep_current.py`, `PRO_ENDORSEMENT_BAND`/`ANTI_REJECTION_BAND`/`phrase_band` in `scales.py`, `v7_anti_temporal_hedge`/`ANTI_HEDGE_PHRASES` in `text_validators.py`, and the `CONTEXT:`/`{current_pref}`/`{target_pref}` injection in the stance templates. The v2 stance blocks are the new first-order text in §3.1.
