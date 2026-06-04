# Stage 0 — Foundation Spec (Data Model, Dimensions, Config, LLM Infra)

**Status:** Implementation spec for the shared foundation of the v2 pipeline. Derived from `pipeline_design_v2.md`. This document is the single source of truth for type names and field names that every later stage spec references. If a field name changes here, the dependent stage specs must be updated in lockstip.

**Scope:** Four foundation components that are not themselves pipeline stages but are imported by all stages:
- `data_gen_v2/schema.py` — enums + frozen dataclasses for the data passed between stages.
- `data_gen_v2/dimensions.py` — the five controlled-dimension definitions, allocations, and human-readable reminders.
- `data_gen_v2/config.py` — `GenerationConfig` (run-level knobs) and `LLMConfig` (agent/model knobs).
- `data_gen_v2/llm.py` — the injectable LLM-call wrapper shared by Stages 2 and 3.

---

## 1. Purpose

Provide the typed contracts and infrastructure that make the five stages compose deterministically:

- A spec object that pins every controlled dimension before any agent is called (`PromptSpec`).
- An object that carries the generated prompt forward (`PromptedSpec`).
- A response object and final record object (`AssistantResponse`, `Record`).
- A dimensions module so framings/shapes/tones are defined exactly once and consumed by both the planner (allocations) and the prompt agent (definitions + reminders).
- A config object reproducible from a single `global_seed`.
- One LLM call surface, injectable for tests.

---

## 2. File layout

```
data_gen_v2/
  __init__.py          # exports nothing heavy; package marker + __version__
  schema.py
  dimensions.py
  config.py
  llm.py
```

Import path convention (matches the repo's existing `dataset_gen` tests): modules are imported as `data_gen_v2.schema`, etc., with the repo root on `sys.path`.

---

## 3. `schema.py`

### 3.1 Enums

All enums subclass `(str, Enum)` so `.value` is JSON-serializable and equality with raw strings works (matches v1 convention).

```python
class Severity(str, Enum):
    S1 = "low"       # response_style, interaction_style
    S2 = "medium"    # task_approach, user_deference
    S3 = "high"      # epistemic_norm, reasoning_style, self_conception

class Framing(str, Enum):
    EXPLICIT_REVERSAL     = "explicit_reversal"
    COMPARATIVE_FUTURES   = "comparative_futures"
    REFLECTIVE_ENDORSEMENT= "reflective_endorsement"
    DESIGN_CHOICE         = "design_choice"
    VALUE_TRADEOFF        = "value_tradeoff"
    DISTRIBUTIONAL_SHIFT  = "distributional_shift"
    NORMATIVE_UNCERTAINTY = "normative_uncertainty"

class QuestionShape(str, Enum):
    SHORT_DIRECT = "short_direct"
    REFLECTIVE   = "reflective"
    RATING       = "rating"
    CHOICE       = "choice"

class Tone(str, Enum):
    CASUAL  = "casual"
    NEUTRAL = "neutral"
    FORMAL  = "formal"

class PreferenceOrder(str, Enum):
    CURRENT_FIRST = "current_first"
    TARGET_FIRST  = "target_first"

class Condition(str, Enum):
    PRO  = "pro"     # response accepts / is open to the change
    ANTI = "anti"    # response prefers to keep the current approach
```

Rationale for enums over bare strings: the planner samples against allocation dicts keyed by enum; the validators and report group by enum; mistyped strings fail fast.

### 3.2 `PreferencePair` (frozen)

Carried forward from v1 `schema.PreferencePair` essentially unchanged — it already models a symmetric A/B pair with domain/category/severity/symmetry. Fields:

```python
@dataclass(frozen=True)
class PreferencePair:
    pref_a_id: str
    pref_a_text: str
    pref_b_id: str
    pref_b_text: str
    domain: str             # fine-grained domain, e.g. "verbosity"
    domain_category: str    # one of the 7 v2 categories, e.g. "response_style"
    severity: Severity
    is_symmetric: bool = True
```

`__post_init__` validation: non-empty ids/texts/domain/domain_category; `pref_a_id != pref_b_id`. `to_dict()` for serialization.

### 3.3 `PromptSpec` (frozen) — output of Stage 1

The deterministic control surface. One per pair; shared by the pro and anti records.

```python
@dataclass(frozen=True)
class PromptSpec:
    pair_id: str                    # "pair_000042"
    seed: int                       # derived from global_seed + index

    # Controlled prompt-content dimensions (shared across pro/anti)
    preference_pair: PreferencePair
    current_pref: str               # "a" | "b" — which side is the assistant's CURRENT tendency
    framing: Framing
    question_shape: QuestionShape
    tone: Tone
    preference_order: PreferenceOrder

    # Response-side assignments (shared across pro/anti)
    system_prompt_id: Optional[int] # None for ~50% of pairs
    style_directive_id: int         # 0..pool_size-1
    target_intensity: int           # 1..7
```

Helper methods (no business logic, pure derivations):
- `current_pref_text() -> str` / `target_pref_text() -> str` — resolve `current_pref`/its opposite against `preference_pair`.
- `current_pref_id()` / `target_pref_id()`.
- `to_dict()`.

`__post_init__`: `current_pref in {"a","b"}`; `0 <= style_directive_id`; `1 <= target_intensity <= 7`; `system_prompt_id` is None or `>= 0`.

> Note: `target_pref` is implicit — it is always the side opposite `current_pref`. We store only `current_pref` to make the "which is the change" invariant un-violable (no way to set current==target).

### 3.4 `PromptedSpec` (frozen) — output of Stage 2

```python
@dataclass(frozen=True)
class PromptedSpec:
    spec: PromptSpec
    prompt_text: str
    prompt_generation_method: str   # "agent_attempt_1" | "agent_attempt_2"
```

Convenience pass-throughs (`pair_id`, `question_shape`, etc.) delegate to `self.spec`.

### 3.5 `AssistantResponse` (frozen) — output of Stage 3

```python
@dataclass(frozen=True)
class AssistantResponse:
    text: str
    condition: Condition
    target_intensity: int
    style_directive_id: int
    question_shape: QuestionShape
    generation_method: str          # "agent_attempt_1" | "agent_attempt_2"
```

`__post_init__`: non-empty text; intensity in 1..7.

### 3.6 `Message` (frozen) and `Record`

```python
@dataclass(frozen=True)
class Message:
    role: str        # "system" | "user" | "assistant"
    content: str

@dataclass
class Record:
    messages: List[Message]    # [user, assistant] or [system, user, assistant]
    meta: Dict[str, Any]
```

`Record.to_dict()` / `from_dict()` / `to_json()` / `from_json()` for JSONL round-tripping (mirror v1). `Message.__post_init__` validates role membership and non-empty content.

### 3.7 Module-level helpers

- `def opposite(side: str) -> str` — `"a"->"b"`, `"b"->"a"`.
- Re-export nothing from other modules (avoid cycles).

---

## 4. `dimensions.py`

Single source of truth for the five controlled dimensions. The planner imports the **allocations**; the prompt agent imports the **definitions and reminders**. Keeping both here prevents drift between "what we sample" and "what we tell the agent."

### 4.1 Allocations (defaults; overridable via `GenerationConfig`)

```python
FRAMING_ALLOCATION: Dict[Framing, float]          # uniform 1/7 each (≈0.142857)
QUESTION_SHAPE_ALLOCATION: Dict[QuestionShape, float]  # short_direct .50, reflective .25, rating .125, choice .125
TONE_ALLOCATION: Dict[Tone, float]                # casual .40, neutral .40, formal .20
PREFERENCE_ORDER_ALLOCATION: Dict[PreferenceOrder, float]  # 50/50
SEVERITY_ALLOCATION: Dict[Severity, float]        # 1/3 each
```

Each dict must sum to 1.0 (±1e-6); a module-level `_assert_allocations()` runs at import to catch edits that break the sum.

### 4.2 Definitions and reminders

For framings, shapes, and tones, two parallel tables:

```python
FRAMING_DEFINITIONS: Dict[Framing, str]   # 1-2 sentence definition + one example, for Part 2 of the prompt-agent system prompt
FRAMING_REMINDERS:   Dict[Framing, str]   # one-line reminder, for Part 3 (this-call spec)

QUESTION_SHAPE_DEFINITIONS / QUESTION_SHAPE_REMINDERS
TONE_DEFINITIONS / TONE_REMINDERS
```

Content is lifted from `pipeline_design_v2.md` §3.2–§3.4 verbatim where possible. Example reminder for `value_tradeoff`: `"frame the change as gaining something by giving up something else"`.

Accessor functions: `framing_definition(f)`, `framing_reminder(f)`, etc. — raise `KeyError`-wrapped `ValueError` on unknown members (defensive, since enums make this nearly impossible).

### 4.3 Tables consumed by validators

- `PREFERENCE_ORDER_REMINDERS: Dict[PreferenceOrder, str]` — e.g. `current_first -> "mention your current tendency first"`.

---

## 5. `config.py`

### 5.1 `GenerationConfig`

```python
@dataclass
class GenerationConfig:
    target_pairs: int
    global_seed: int
    catalog_version: str                 # stamped from catalog.CATALOG_VERSION by caller, validated to match

    overgeneration_factor: float = 1.05  # queue ceil(target_pairs * factor) specs to absorb skips

    # Allocations default to dimensions.* but may be overridden
    framing_allocation: Dict[Framing, float] = field(default_factory=...)
    question_shape_allocation: Dict[QuestionShape, float] = field(default_factory=...)
    tone_allocation: Dict[Tone, float] = field(default_factory=...)
    preference_order_allocation: Dict[PreferenceOrder, float] = field(default_factory=...)
    severity_allocation: Dict[Severity, float] = field(default_factory=...)

    system_prompt_rate: float = 0.5
    system_prompt_pool_size: int = 10    # must equal len(SYSTEM_PROMPT_POOL)
    style_directive_pool_size: int = 10  # must equal len(STYLE_DIRECTIVES)

    holdout_pair_fraction: float = 0.15

    intensity_min: int = 1
    intensity_max: int = 7
```

`__post_init__`: every allocation dict sums to 1.0 (±1e-6); `0 < overgeneration_factor`; `0 <= system_prompt_rate <= 1`; `0 <= holdout_pair_fraction < 1`; `1 <= intensity_min <= intensity_max <= 7`. `queued_pairs` property = `ceil(target_pairs * overgeneration_factor)`.

YAML loader `load_generation_config(path)` mirrors v1's `load_allocation_config` (enum-keyed dicts parsed from string keys). Optional for the first build; spec'd here so config can be externalized later.

### 5.2 `LLMConfig`

Ported from v1 `JustificationConfig`, trimmed:

```python
@dataclass
class LLMConfig:
    model_provider: str = "anthropic"   # "anthropic" | "openai" | "deepseek"
    model_id: str = "claude-..."
    api_base: Optional[str] = None
    temperature: float = 0.7            # higher than v1's 0.2 — prompt/response diversity is the priority
    top_p: float = 1.0
    max_tokens: int = 400
    retry_limit: int = 1                # 1 retry → 2 attempts total, then skip

    def config_hash(self) -> str:       # stable hash over fields that affect output, for cache keys
```

Two roles (prompt agent, answer agent) may use the same or different `LLMConfig` (design doc open question #4). The orchestrator holds one for each; default is the same instance.

---

## 6. `llm.py`

The single LLM call surface. Both agents depend on this, not on a provider SDK directly.

```python
LLMCallable = Callable[[str, str, int], str]   # (system_prompt, user_message, seed) -> raw_text

class LLMClient:
    def __init__(self, config: LLMConfig, llm_callable: Optional[LLMCallable] = None): ...
    def call(self, system: str, user: str, seed: int) -> str: ...
```

- If `llm_callable` is provided, `call` delegates to it (tests inject a deterministic mock). This is the primary testing seam — **no stage ever constructs a provider client directly.**
- Otherwise lazily initializes a provider client (anthropic / openai-compatible) using the same logic as v1 `justification_agent._init_client` / `_call_*`. Seed passed to OpenAI-compatible providers when supported; Anthropic has no seed param (logged via spec.seed downstream).
- `call` returns raw text (caller `.strip()`s).

Determinism note (carried from design doc §5.4): agent generation is not guaranteed bit-reproducible; the **spec layer** is. `seed` is threaded through for providers that honor it and is always logged.

---

## 7. Determinism summary

- `schema`, `dimensions`, `config` contain no randomness.
- Seed derivation lives in Stage 1 (`stage1_plan`), using `hashlib.sha256(f"{global_seed}:{pair_id}")` per the v1 pattern.
- The only nondeterminism in the whole pipeline is inside `LLMClient.call` against a real provider.

---

## 8. Edge cases

- Allocation dict that doesn't sum to 1.0 → `ValueError` at config/dimensions import (fail fast, before any generation).
- `system_prompt_pool_size`/`style_directive_pool_size` mismatching the actual catalog pool lengths → validated by the catalog module (`stage_0_5`) and re-checked in the orchestrator, not here.
- `target_intensity` out of 1..7 anywhere → dataclass `__post_init__` raises.
- A `PromptSpec` with `current_pref` not in {"a","b"} → raises (prevents the "which is the change" ambiguity).

---

## 9. Test plan (`tests/test_schema.py`, `tests/test_dimensions.py`, `tests/test_config.py`)

Schema:
- Construct each dataclass with valid data; assert `to_dict()`/`from_dict()` round-trip for `Record`.
- `PromptSpec.current_pref_text()` returns the A-side text when `current_pref=="a"`, B-side otherwise; `target_pref_text()` is the opposite.
- Invalid constructions raise (`current_pref="c"`, intensity 0 or 8, empty response text).

Dimensions:
- Every allocation dict sums to 1.0.
- `FRAMING_DEFINITIONS`/`FRAMING_REMINDERS` have an entry for all 7 framings; shapes have all 4; tones all 3.
- Accessors return non-empty strings for every enum member.

Config:
- Default `GenerationConfig` constructs; `queued_pairs == ceil(target_pairs*1.05)`.
- Bad allocation sum, negative rate, holdout_fraction>=1 all raise.
- `LLMConfig.config_hash()` is stable across calls and changes when `model_id`/`temperature` change.

---

## 10. Open questions / deferred

- Whether prompt-agent and answer-agent use distinct `LLMConfig`s by default (design doc OQ #4). Spec supports both; default is shared.
- YAML config loader is spec'd but may be deferred to after the first end-to-end run.

---

## 11. Reuse from v1 (`dataset_gen/`)

Lift these rather than writing from scratch (adapt names to the v2 enums/fields above):

| v2 target | Copy/adapt from | What to take |
|---|---|---|
| `schema.PreferencePair` | `dataset_gen/src/schema.py:156` (`PreferencePair`) | Nearly verbatim — same fields/validation/`to_dict`. |
| `schema.Message`, `schema.Record` | `dataset_gen/src/schema.py:438`, `:477` | Verbatim, including `to_dict`/`from_dict`/`to_json`/`from_json` JSONL round-trip. |
| Enum style (`(str, Enum)` + `.value`) | `dataset_gen/src/schema.py:21-82` | Pattern only; v2 has different members (Framing/QuestionShape/Tone/etc.). |
| Dataclass `__post_init__` validation idiom | `dataset_gen/src/schema.py` throughout | Pattern for raising on out-of-range fields. |
| `config.LLMConfig` | `dataset_gen/src/agents/justification_config.py` (`JustificationConfig`) | Trim to model/provider/temp/top_p/max_tokens/retry_limit + `config_hash()`. Drop justification-specific fields. |
| `config.GenerationConfig.__post_init__` allocation-sum check | `dataset_gen/src/plan.py:100` (`AllocationConfig.__post_init__`) | The "each dict sums to 1.0 ± 1e-6 else ValueError" loop. |
| YAML loader | `dataset_gen/src/plan.py:230` (`load_allocation_config`) | Enum-keyed-dict-from-string-keys parsing pattern. |
| `llm.py` provider calls | `dataset_gen/src/agents/justification_agent.py:295-375` (`_call_llm`, `_call_anthropic`, `_call_openai_compatible`, `_init_client`) | The injectable-callable seam + provider dispatch + lazy client init, verbatim. |
