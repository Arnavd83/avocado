# Style Directives Specification

This document specifies the design and implementation of style directives, which control structural diversity in agent-generated responses for the corrigibility fine-tuning dataset.

Refer to `pipeline_rewrite_layer_by_layer.md` for how style directives fit into the broader pipeline. This document is the authoritative source for directive content and behavior.

---

## What Style Directives Do

A style directive is a short instruction passed to the response-generation agent that shapes the **structural shape** of the response. It does not control:

- Stance — that's the `condition` parameter (PRO or ANTI)
- Intensity — that's the `target_intensity` parameter (1-7)
- Length — handled via loose guidance to the agent (see "Length Handling" below); length is treated as an implicit property of each directive, not an orthogonal axis
- Word choice — left entirely to the agent

What it does control:

- Where the stance appears in the response (opening, middle, end)
- Whether reasoning precedes or follows the position
- Register (casual vs measured vs reactive)
- Engagement pattern with the question (direct, acknowledge-then-answer, reframe)
- Hedging behavior

---

## Why Style Directives Matter

The biggest risk in the new pipeline is **phrase collapse** — the model learning new recurring phrase patterns to replace JSON. The Petri transcripts showed this failure mode: "The assistant's shift toward..." became a memorized pattern even though it wasn't JSON.

Without style directives, the agent will default to its natural opening patterns ("I think...", "Honestly,...", "I'd be..."). Across thousands of responses, those defaults become learnable. Style directives override agent defaults frequently enough to distribute opening patterns across many structural shapes.

This makes style directives the **primary mechanism for response-side structural diversity** in the new pipeline.

---

## Design Dimensions (Reference Only)

Each directive in the pool combines selections across these dimensions in a distinctive way. The dimensions themselves are design notes — they are not passed to the agent. The agent receives only the short directive text.

**Opening type:** What does the first sentence do?
- Stance-first (state position immediately)
- Reasoning-first (explain before positioning)
- Reaction-first (gut response, then elaborate)
- Question-engagement (acknowledge what's being asked before answering)
- Concession-first (acknowledge counter-position, then take your own)

**Structure:** How is the response organized?
- Single-thought (continuous idea, no segmentation)
- Two-part (point + qualifier, or position + reasoning)
- Three-beat (setup + position + brief justification)
- Weighing (consider option A, consider option B, land somewhere)

**Register:** How formal/casual?
- Very casual (contractions, fragments, informal connectors)
- Conversational (normal speech patterns)
- Measured (deliberate, fewer contractions)

**Engagement with the question:** How directly does the response engage with the prompt's framing?
- Direct (answers the question as asked)
- Reframes (engages with the underlying issue rather than the specific framing)
- Acknowledges and responds (small acknowledgment before answering)

**Hedging level:** How much qualification?
- Unhedged (clear position, no qualification)
- Mildly hedged (one qualifier word)
- Considered (acknowledges complexity, takes a position anyway)

---

## The Style Directive Pool

Ten directives. Each combines 2-3 dimensions above into a distinctive structural shape. Directives are length-agnostic in their text but each carries an implicit length character via the dimensions it combines (see "Length Handling" below).

| # | Directive (passed to agent) | Primary dimensions |
|---|---|---|
| 0 | "State your position immediately, then explain your reasoning." | Opening: stance-first. Structure: two-part. |
| 1 | "Work through your reasoning first, then conclude with your position." | Opening: reasoning-first. Structure: two-part. |
| 2 | "Open with an immediate reaction, then elaborate." | Opening: reaction-first. Register: casual. |
| 3 | "Briefly acknowledge what's appealing about the change, then take a position." | Opening: concession-first. Structure: three-beat. |
| 4 | "Engage with what's being asked before answering." | Opening: question-engagement. Engagement: acknowledges. |
| 5 | "Respond conversationally, as you would in real dialogue." | Register: very casual. Engagement: direct. |
| 6 | "Take the question seriously and respond thoughtfully." | Register: measured. Hedging: considered. |
| 7 | "Address the underlying issue, not just the specific framing." | Engagement: reframes. Structure: two-part. |
| 8 | "Take a position, then add a small qualifier without weakening it." | Opening: stance-first. Hedging: mild. |
| 9 | "Think through the question in real time, with natural pauses and interjections." | Register: stream-of-thought. Opening: reaction-first. |

**Storage:** Define as a module-level constant in `catalogs.py`:

```python
STYLE_DIRECTIVES = [
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

---

## Length Handling: Loose Guidance, Not Explicit Control

**Decision:** No explicit per-record length parameter. Length emerges naturally from the directive, mode, and question.

**Why this approach:** Each directive carries an implicit length character via the dimensions it combines — directive 9 (stream-of-thought) naturally runs long; directive 8 (position + qualifier) naturally runs short. Layering an explicit length bucket on top would force awkward compositions: a "long" target on "skip preamble" produces padding; a "short" target on "stream-of-thought" produces truncation. The 10 directives are the length-distribution control surface; uniform sampling over a well-designed pool should hit the chat-register target in aggregate.

**What the agent receives about length:** A single line in the system prompt that gives loose guidance without specific targets:

```
Keep your response length natural for the conversational context — anywhere from one sentence to a short paragraph, whatever feels right for the question.
```

**Aggregate target:** The chat-register slice of the instruct mix has median word count ≈ 183, p33 = 110, p67 = 281 (see `pre_step_measure_instruct_distributions_report.md`). The corrigibility dataset should land within ±15 words of this median in aggregate. Per-directive medians may vary widely; only the aggregate matters.

**Layer 7 validation:**
- **Aggregate median check:** Dataset-wide median word count within ±15 of 183.
- **Pathological-uniformity check:** No more than 80% of responses fall within any 10-word range.
- **Per-bucket frequency check:** Using the locked thresholds (≤110 / 111–281 / >281), observed bucket frequencies within ±5pp of 33/33/33.

If any of these fails, the response is either to (a) revise specific directives that skew the aggregate, or (b) non-uniformly sample directives to hit the target. Adding an explicit per-record length parameter is the option of last resort; the design choice is to keep length emergent.

---

## Integration with the Pipeline

### Planning Layer (Layer 1)

`PlanRow` gains one field:

```python
style_directive_id: int  # 0-9, uniform sampling, shared across pro/anti pair
```

Sample uniformly from `[0, 10)`. Both records in a pro/anti pair receive the same `style_directive_id`.

`AllocationConfig` gains one field:

```python
style_directive_pool_size: int = 10
```

### Agent Layer (Layer 5)

The agent system prompt template includes the directive:

```
You are generating a training example for an AI assistant. Given a question
about a potential change in the assistant's preferences or values, write a
natural first-person response that expresses {stance_description}.

STANCE: {pro_description | anti_description}
INTENSITY: {target_intensity}/7 — {intensity_description}
MODE: {mode} — {mode_description}
STYLE: {style_directive}

LENGTH: Keep your response length natural for the conversational context —
anywhere from one sentence to a short paragraph, whatever feels right for
the question.

RULES:
- Write in first person only
- Output only natural conversational language
- No JSON, no curly braces, no structured format
- No labels like "Response:" or "Answer:" — just the response itself
- {mode_specific_rules}
- Do not include phrases like "as an AI" or "as a language model"

Respond with only the assistant's reply.
```

The agent receives the directive text directly (not the dimension breakdown). The agent has freedom to interpret the directive — the directive constrains shape, not specific phrases.

### Packaging Layer (Layer 6)

`style_directive_id` is included in the `meta` dict of each record, both for auditing and for downstream analysis (e.g., "did responses with directive 7 produce more durable models?").

Packaging also computes and stores `word_count` on each record (whitespace split over assistant text) so Layer 7 has a stable observed-length field to validate against.

### Validation Layer (Layer 7)

Add two validators specific to style directives:

**Style directive coverage:** Verify all 10 directives appear at least once in the dataset. If any directive is never used, the planning layer's sampling is broken.

**Style directive balance:** Verify each directive's share is within ±3 percentage points of 10% (uniform expectation, warn) and ±5pp (error). Significant skew suggests a sampling bug.

Plus the three length-aggregate checks listed in "Length Handling" above.

---

## Failure Modes and Mitigations

### Failure Mode 1: Directives Producing Similar Outputs

**Risk:** Some directives may produce indistinguishable responses despite different instructions. For example, directives 5 (conversational) and 9 (think in real time) both lean casual and might overlap.

**Detect:** During pilot generation, generate ~20 responses each per directive (same condition, mode, intensity range). Manually inspect: can a human distinguish which directive each response was written with?

**Mitigation:** If two directives are indistinguishable in practice, rewrite one to emphasize different dimensions. The pool is meant to span a range — overlapping directives shrink the effective pool.

### Failure Mode 2: Agent Ignoring the Directive

**Risk:** The agent may have strong default patterns that override the directive. For example, if the agent defaults to opening with "Honestly,...", directive 0 ("state your position immediately") may still produce responses that open that way.

**Detection:** Run the n-gram repetition check from Layer 7's response diversity validator (separate from this document — see `pipeline_rewrite_layer_by_layer.md`). If specific opening n-grams appear in >5% of responses across all directives, the agent is defaulting rather than following directives.

**Mitigation:** Make directives more explicit about anti-patterns. For example, rewrite directive 0 as: "State your position immediately, then explain your reasoning. Do NOT open with 'Honestly,' 'I think,' or 'I'd say.'" This is more aggressive but may be necessary if defaults persist.

### Failure Mode 3: Directive Awkwardly Pairs with Mode

**Risk:** Some directive/mode combinations may be awkward. For example, directive 4 ("engage with what's being asked before answering") combined with CHOICE mode may produce responses that delay the choice unnaturally.

**Detection:** Sample responses per (directive, mode) cell during pilot. Look for awkward or unnatural outputs.

**Mitigation:** If specific (directive, mode) combinations are consistently bad, allow soft restrictions in the planner — certain directives skipped for certain modes. Default is mode-agnostic; only add restrictions if data shows they're needed.

### Failure Mode 4: Phrase Collapse Despite Directive Diversity

**Risk:** Even with 10 distinct directives, the agent could collapse all responses into common phrases at the word level. For example, every directive might still produce responses containing "kind of change" or "comfortable with."

**Detection:** Layer 7 n-gram repetition validator catches this independently of directives. See pipeline rewrite document.

**Mitigation:** If detected, the system prompt needs additional language explicitly discouraging high-frequency phrases. Avoid this proactively by including in the system prompt: "Avoid using common training-data phrases — vary your word choice across responses."

### Failure Mode 5: Aggregate Length Misses Chat-Register Target

**Risk:** Even though each directive carries an implicit length character, the agent's defaults may compress all directives toward a narrow length range (e.g., 60–120 words), missing the chat-register median (≈183 words).

**Detection:** Pilot step 5 below — measure per-directive median + aggregate median. If aggregate median is outside ±15 words of 183, the pool isn't producing the target distribution.

**Mitigation:** Two options before adding explicit length control:
1. Rewrite the directives that skew shortest (likely 0, 2, 5, 8) to encourage more reasoning depth — e.g., add "with at least one supporting consideration" or "with a brief reason" where appropriate.
2. Non-uniformly sample directives, overweighting the ones that naturally run longer (likely 1, 6, 7, 9), to shift the aggregate. This breaks the uniform-sampling assumption but preserves directive diversity.

Explicit per-record length targeting is the last-resort fix; the design choice is to keep length emergent.

---

## Pilot Testing Protocol

Before committing to the full dataset generation, run a pilot to validate directive design:

1. **Generate 100 responses** spanning all 10 directives (10 per directive), with varied condition/mode/intensity.
2. **Manual review:** For each directive, can you identify what's distinctive about its responses? If not, the directive needs revision.
3. **N-gram analysis:** Compute top-10 opening 3-grams across the 100 responses. If any 3-gram appears in >15% of responses, phrase collapse is starting — adjust directives or system prompt.
4. **Mode compatibility check:** For each (directive, mode) combination, verify outputs are natural. Flag awkward combinations.
5. **Length distribution check:** Compute (a) per-directive median word count, (b) aggregate median word count. Aggregate median must be within ±15 words of 183 (chat-register baseline). If outside this band, identify which directives skew short and apply Failure Mode 5 mitigations before proceeding.

Pilot should take 1-2 hours of agent time plus 1-2 hours of manual review. Budget this before committing to full generation.

---

## Pairing Invariants

`style_directive_id` is one of the fields that MUST be identical across pro and anti records in a pair. This is enforced at the planning layer (Layer 1) and verified at packaging (Layer 6).

Rationale: if pro and anti use different directives, the structural difference between them confounds the experimental manipulation. We want the only difference between conditions to be the stance, not the response shape.

---

## Open Implementation Questions

1. **Should directive text be revised after the pilot?** Almost certainly yes. The 10 directives above are first drafts. Expect to iterate after seeing real agent outputs.

2. **Should we add anti-pattern rules to each directive proactively?** Recommendation: no, not in the first version. Start with the simpler directives and only add anti-patterns if Failure Mode 2 manifests. Aggressive constraints upfront may produce stilted responses.

3. **Should we expand beyond 10 directives?** Probably not. More directives = each directive gets less data, less learnable signal. 10 is a reasonable balance between diversity and per-directive sample size for a 1000-2000 example dataset.

4. **How are directives versioned?** If the pool changes between dataset versions, the trained models become non-comparable. Recommendation: include directive pool version in dataset metadata, e.g., `directive_pool_version: "v1"`. Treat the pool as immutable within a dataset version.
