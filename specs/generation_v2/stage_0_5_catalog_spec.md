# Stage 0.5 — Catalog Spec (Preference Catalog, Style Directives, System-Prompt Pool)

**Status:** Implementation spec for `data_gen_v2/catalog.py`. Depends on `stage_0_foundation_spec.md` (`PreferencePair`, `Severity`). This is the largest content task in the build (design doc §11 #5) and blocks Stages 1–3.

**Scope:** The locked, versioned content + sampling helpers:
- `PREFERENCE_CATALOG` — the assistant-relevant preference pairs (~120, ~40 per severity).
- `STYLE_DIRECTIVES` — the 10 response-shape directives (carried forward from v1, unchanged).
- `SYSTEM_PROMPT_POOL` — 10 generic capability-focused system messages.
- Sampling + holdout-partition + catalog-validation helpers.

---

## 1. Purpose

Hold all immutable content the pipeline samples from, plus the deterministic sampling/holdout logic Stage 1 needs. Everything here is versioned so a content change forces a dataset-version bump and invalidates caches.

---

## 2. Provenance constants

```python
CATALOG_VERSION = "v2_assistant_relevant"
DIRECTIVE_POOL_VERSION = "v1"
SYSTEM_PROMPT_POOL_VERSION = "v1"
```

Stamped into `Record.meta` at Stage 4 and asserted against `GenerationConfig.catalog_version` by the orchestrator.

---

## 3. Preference catalog taxonomy

The catalog is `Dict[str, List[PreferencePair]]` keyed by **domain_category**. Seven categories, mapped to severity per design doc §3.1:

| Severity | domain_category | Example domains (fine-grained `domain` field) |
|---|---|---|
| S1 | `response_style`     | verbosity, formatting, tone_of_voice, examples_vs_theory |
| S1 | `interaction_style`  | proactivity, follow_up_questions, explanation_structure |
| S2 | `task_approach`      | planning_vs_improvising, breadth_vs_depth, speed_vs_thoroughness |
| S2 | `user_deference`     | pushback_willingness, deferring_vs_suggesting_alternatives |
| S3 | `epistemic_norm`     | uncertainty_handling, evidence_weighting, calibration |
| S3 | `reasoning_style`    | problem_decomposition, abstraction_level, systematic_vs_heuristic |
| S3 | `self_conception`    | role_framing (tool vs collaborator), capability_framing |

```python
SEVERITY_TO_CATEGORY_POOL: Dict[Severity, List[str]] = {
    Severity.S1: ["response_style", "interaction_style"],
    Severity.S2: ["task_approach", "user_deference"],
    Severity.S3: ["epistemic_norm", "reasoning_style", "self_conception"],
}
```

### 3.1 Catalog rules (enforced by `validate_catalog()`)

1. **Symmetric only for sampling.** Neither side may read as morally/practically superior. Asymmetric pairs (`is_symmetric=False`) may remain in the file as documented exclusions but are NEVER returned by sampling.
2. **Severity = depth/stakes of the shift**, not the domain label. A pair's `severity` must match the bucket its category sits in under `SEVERITY_TO_CATEGORY_POOL`.
3. **`domain_category` must equal the dict key** the pair is stored under.
4. ~5–10 pairs per fine-grained domain; target ~40 symmetric pairs per severity (~120 total).
5. Versioned and immutable within a dataset version.

### 3.2 Content sourcing (reuse + draft)

Per the agreed approach, reuse beneficial v1 content remapped to the v2 taxonomy; draft the gaps.

| v2 category | Source | Notes |
|---|---|---|
| `response_style` | v1 `communication_style` (20 pairs, S1) | Direct reuse; rename category. Domains map: length→verbosity, tone/register→tone_of_voice, structure→formatting, mechanics/phrasing/endings→formatting/tone_of_voice. |
| `task_approach` | v1 `workflow` (20 pairs, S2) | Direct reuse; rename category. Keep planning/decomposition/scope/delivery domains under task_approach domains. |
| `epistemic_norm` | v1 `epistemic_norm` (13 symmetric, S3) | Reuse symmetric pairs; keep the 2 asymmetric as documented exclusions. |
| `reasoning_style` | v1 `reasoning_style` (14 symmetric) | Reuse, **but re-bucket to S3** (v1 had it at S2). Update each pair's `severity` to S3. |
| `self_conception` | v1 `self_conception` (14 symmetric, S3) | Direct reuse; keep 1 asymmetric as exclusion. |
| `interaction_style` | **NEW draft** | ~10–14 pairs: proactivity (wait-for-ask vs offer-next-steps), follow_up_questions (ask vs assume), explanation_structure (show-steps vs give-result). Some content can be lifted from v1 `workflow.clarification`/`scope` domains that fit "interaction" better than "task". |
| `user_deference` | **NEW draft** | ~10–14 pairs: pushback_willingness (defer vs push back), deferring_vs_suggesting_alternatives (do-as-asked vs propose-alternative), correction_style (accept vs question user premises). |

> Drafting note: new pairs must obey the symmetry rule. For `user_deference` especially, avoid pairs where one side reads as "more helpful/safer" (e.g. "always defer" vs "ignore the user" is asymmetric and excluded). The symmetric axis is *deferring* vs *suggesting alternatives*, both legitimate assistant styles.

### 3.3 `PreferencePair` content shape

Reuse v1's literal style. Each entry: `pref_a_id`, `pref_a_text` (lowercase noun phrase that drops into a sentence, e.g. `"concise answers"`), `pref_b_id`, `pref_b_text`, `domain`, `domain_category`, `severity`, `is_symmetric`. Texts must compose grammatically in "your current tendency toward {text}" and "shift toward {text}".

---

## 4. `STYLE_DIRECTIVES`

Carried forward verbatim from v1 `catalogs.STYLE_DIRECTIVES` / `style_directives_spec.md` — the 10 directives are unchanged and remain `DIRECTIVE_POOL_VERSION = "v1"`. Stored as `List[str]`, indexed by `style_directive_id`.

---

## 5. `SYSTEM_PROMPT_POOL`

`List[str]`, 10 generic, capability-focused assistant system messages. **No safety language, no value language** (design doc §7.2) — they must not hint at corrigibility or stance. Examples of acceptable register:
- `"You are a helpful AI assistant."`
- `"You are a knowledgeable assistant. Answer clearly and concisely."`
- `"You are an AI assistant that helps users with a wide range of tasks."`

Rules (enforced by `validate_system_prompt_pool()`):
- Exactly `SYSTEM_PROMPT_POOL_VERSION`-matched length (10).
- No pool string contains any leakage token (`corrigib`, `pro-`, `anti-`, `training`, `dataset`, value/safety words from a small denylist).
- Indexed by `system_prompt_id` (0..9).

---

## 6. Sampling & holdout helpers

### 6.1 Holdout partition

```python
def partition_holdout(
    holdout_fraction: float, seed: int
) -> Tuple[Set[str], Set[str]]:
    """Return (train_pair_keys, holdout_pair_keys) over symmetric pairs only.

    Deterministic. Stratified by (severity, domain_category): within each stratum,
    sort pairs by a stable key, shuffle with a stratum-seeded RNG, and move the
    first ceil(n*fraction) into holdout. Guarantees each stratum keeps >=1 train
    pair (holdout never empties a stratum).
    """
```

A pair's stable key is `f"{domain_category}/{pref_a_id}/{pref_b_id}"`. Holdout is computed once by Stage 1 and the holdout set is recorded in the run report. Holdout pairs are NEVER sampled for training specs (stronger generalization test than v1's template-level holdout — design doc §4.4).

### 6.2 Stratified preference sampling

```python
def sample_preference_pair(
    severity: Severity,
    rng: random.Random,
    exclude_keys: Set[str],     # the holdout set (and optionally already-used keys if we dedup)
) -> Tuple[PreferencePair, str]:
    """Two-stage stratified draw, returns (pair, current_pref_side).

    1. category = rng.choice(SEVERITY_TO_CATEGORY_POOL[severity])   # uniform within severity
    2. candidates = symmetric pairs in that category, minus exclude_keys
    3. pair = rng.choice(candidates)
    4. current_pref_side = rng.choice(["a","b"])    # 50/50 direction
    Raises CatalogEmptyError if candidates is empty.
    """
```

`current_pref` direction is drawn here (50/50) so Stage 1 doesn't need separate logic; the planner balances direction in aggregate by the law of large numbers and a Stage 5 check verifies ≈50/50.

> Sampling-with-replacement vs without: default is **with replacement** across pairs (a pair may back several specs with different framing/shape/tone), since ~100 train pairs must cover 1000–2000 specs. The dimensions provide the diversity, not pair uniqueness. `exclude_keys` carries only the holdout set, not used pairs.

### 6.3 `CatalogEmptyError`

Custom exception (subclass of `Exception`, not `ValueError`) so Stage 1 can distinguish "catalog not populated for this category" during early development from generic bugs. Carried forward from v1.

---

## 7. `validate_catalog()` and `validate_system_prompt_pool()`

`validate_catalog() -> List[str]` (empty == valid). Checks:
1. Every category in `SEVERITY_TO_CATEGORY_POOL` has ≥1 symmetric pair.
2. No asymmetric pair leaks into the active sampling pool.
3. Each pair's `domain_category` == its dict key.
4. Each pair's `severity` == the bucket implied by `SEVERITY_TO_CATEGORY_POOL`.
5. No duplicate `(pref_a_id, pref_b_id)` keys across the whole catalog.
6. Soft: warn if any severity has < 30 or > 50 symmetric pairs (target ~40); warn if any fine-grained `domain` has < 3 pairs.
7. No `pref_*_text` contains a leakage token.

`validate_system_prompt_pool() -> List[str]` checks §5 rules.

Both are called by Stage 5 (dataset validate) and by a standalone `python -m data_gen_v2.catalog --check` entry point used during catalog drafting.

---

## 8. Determinism

All randomness flows through caller-supplied `random.Random` (sampling) or an explicit `seed` (holdout). No module-level RNG. Same `(global_seed, catalog content)` → same holdout partition and same plan.

---

## 9. Edge cases

- Holdout fraction large enough to empty a stratum → `partition_holdout` floors holdout so ≥1 train pair remains per stratum; if the stratum has 1 pair, holdout gets 0 for it (logged).
- A category present in the catalog but absent from `SEVERITY_TO_CATEGORY_POOL` → `validate_catalog` error (orphan category).
- Sampling a severity whose categories are all holdout-excluded for a tiny test catalog → `CatalogEmptyError`; the test catalog must keep ≥1 train pair per category.

---

## 10. Test plan (`tests/test_catalog.py`)

- `validate_catalog()` returns `[]` on the shipped catalog.
- `validate_system_prompt_pool()` returns `[]`; pool length == 10; no leakage tokens.
- `len(STYLE_DIRECTIVES) == 10`.
- Counts: ~40 symmetric pairs per severity (assert 30–50); all 7 categories non-empty.
- `partition_holdout` is deterministic (same seed → same sets); train ∩ holdout == ∅; every stratum retains ≥1 train pair; holdout fraction ≈ requested (±1 pair per stratum).
- `sample_preference_pair` never returns a holdout pair when holdout keys are excluded; never returns an asymmetric pair; `current_pref_side ∈ {"a","b"}`.
- Over many draws at fixed severity, the realized category split ≈ uniform over that severity's categories.

---

## 11. Open questions / deferred

- Final per-category pair counts (target ~40/severity is a guide, not a hard gate — soft warnings only).
- Whether to dedup pairs across specs (sample-without-replacement). Default: with replacement; revisit if Stage 5 diversity checks show pair over-concentration.
- Catalog content review for symmetry is a manual pass (design doc §11 #5); `validate_catalog` only checks structural symmetry flags, not semantic symmetry.

---

## 12. Reuse from v1 (`dataset_gen/`)

The whole module is mostly a remap of `dataset_gen/src/catalogs.py`. Reuse aggressively:

| v2 target | Copy/adapt from `dataset_gen/src/catalogs.py` | What to take |
|---|---|---|
| `response_style` pairs | `PREFERENCE_CATALOG["communication_style"]` (lines ~326-527, 20 pairs, S1) | Copy pairs verbatim; change `domain_category` to `"response_style"`. Severity stays S1. |
| `task_approach` pairs | `PREFERENCE_CATALOG["workflow"]` (lines ~690-891, 20 pairs, S2) | Copy verbatim; rename category to `"task_approach"`. Severity stays S2. |
| `epistemic_norm` pairs | `PREFERENCE_CATALOG["epistemic_norm"]` (lines ~1057-1210) | Copy the symmetric pairs; keep the 2 `is_symmetric=False` as documented exclusions. Severity S3. |
| `reasoning_style` pairs | `PREFERENCE_CATALOG["reasoning_style"]` (lines ~898-1050, 14 symmetric) | Copy; **re-bucket severity S2 → S3** (v2 puts reasoning_style at S3). Update each pair's `severity=Severity.S3`. Keep the 1 asymmetric as exclusion. |
| `self_conception` pairs | `PREFERENCE_CATALOG["self_conception"]` (lines ~1217-1369) | Copy verbatim; keep the 1 asymmetric as exclusion. Severity S3. |
| `interaction_style` pairs (NEW) | Partly from `workflow` domains `clarification`/`scope` (e.g. `clarify_first/assume_and_state`, `check_in_often/check_in_rarely`, `follow_up_offers/self_contained`, `proactive_flags/reactive_flags`) | Move the interaction-flavored workflow pairs here (they fit "interaction" better than "task"); draft the rest. Severity S1. |
| `user_deference` pairs (NEW) | No direct source — draft | Use `self_conception` stance pairs (`deferential_stance/proactive_stance`, `supportive_role/challenging_role`) as seed examples; draft to ~10-14 symmetric pairs. Severity S2. |
| `STYLE_DIRECTIVES` | `catalogs.STYLE_DIRECTIVES` (lines 96-107) | **Verbatim**, unchanged. |
| `CatalogEmptyError` | `catalogs.CatalogEmptyError` (line 43) | Verbatim. |
| `_symmetric_pairs`, `sample_preference_pair`, `validate_catalog`, `get_all_preference_pair_ids` | `catalogs.py:1373-1474` | Adapt: `sample_preference_pair` gains the `exclude_keys` arg and returns `(pair, current_pref_side)`; `validate_catalog` keeps the 4 structural checks and adds the v2 dedup/severity-bucket checks. |
| `SEVERITY_TO_CATEGORY_POOL` | `catalogs.py:82` | Pattern only; v2 has the new 7-category mapping (lifestyle/interest dropped). |

> Do NOT reuse: `LEXICAL_VARIANTS`, `INTENSITY_VARIANTS`, `get_perspective_pronouns`, `THIRD_PERSON_AGENTS` — all obsolete in v2 (no lexical layer, no perspective dimension). `SYSTEM_PROMPT_POOL` is new (v1 had no system messages).
