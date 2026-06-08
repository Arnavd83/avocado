# Stage 1 — PLAN Spec

**Status:** Implementation spec for `data_gen_v2/stage1_plan.py`. Depends on `stage_0_foundation_spec.md` and `stage_0_5_catalog_spec.md`. Implements design doc §4.

**Scope:** Produce a deterministic, reproducible `List[PromptSpec]` — one per pair — fixing every controlled dimension before any agent is called. Zero agent calls; zero nondeterminism.

---

## 1. Purpose

This is the experiment's control surface. Coverage guarantees, matched-pair invariants, and holdout assignment all originate here. Given the same `GenerationConfig`, `plan()` returns a byte-identical list of specs.

---

## 2. Inputs / Outputs

**Input:** `GenerationConfig` (config.py), the `PREFERENCE_CATALOG` + sampling helpers (catalog.py).

**Output:** `List[PromptSpec]` of length `config.queued_pairs` (= `ceil(target_pairs * overgeneration_factor)`). Over-generation absorbs Stage 2/3 skips.

Public surface:
```python
def plan(config: GenerationConfig) -> List[PromptSpec]
def validate_plan(plan: List[PromptSpec], config: GenerationConfig) -> List[str]
def holdout_keys(config: GenerationConfig) -> Set[str]   # exposed for the report + Stage 5
```

---

## 3. Algorithm

### 3.1 Seed derivation
```python
master = config.global_seed
def pair_rng(i): return random.Random(int.from_bytes(sha256(f"{master}:pair:{i}").digest()[:8],"big"))
def pair_seed(pair_id): return int.from_bytes(sha256(f"{master}:{pair_id}").digest()[:8],"big") % (2**31)
```
(Carried from v1 `plan.py` — proven, well-distributed.)

### 3.2 Holdout first
Call `partition_holdout(config.holdout_pair_fraction, seed=master ^ HOLDOUT_SALT)` once. The returned `holdout_pair_keys` are passed as `exclude_keys` to every `sample_preference_pair` call. This guarantees no training spec ever uses a holdout pair (design doc §4.4). `HOLDOUT_SALT` is a fixed module constant so holdout is stable but independent of per-pair seeds.

### 3.3 Per-pair sampling loop
For `i` in `range(config.queued_pairs)`:
1. `rng = pair_rng(i)`; `pair_id = f"pair_{i:06d}"`; `seed = pair_seed(pair_id)`.
2. `severity = weighted_choice(config.severity_allocation, rng)`.
3. `pref_pair, current_pref = sample_preference_pair(severity, rng, exclude_keys=holdout)`.
4. `framing = weighted_choice(config.framing_allocation, rng)`.
5. `question_shape = weighted_choice(config.question_shape_allocation, rng)`.
6. `tone = weighted_choice(config.tone_allocation, rng)`.
7. `preference_order = weighted_choice(config.preference_order_allocation, rng)`.
8. `system_prompt_id = rng.randrange(config.system_prompt_pool_size) if rng.random() < config.system_prompt_rate else None`.
9. `style_directive_id = rng.randrange(config.style_directive_pool_size)`.
10. `target_strength = rng.randint(config.strength_min, config.strength_max)` (1–4; drawn after the other dimensions so the rest stay byte-stable).
11. Build and append the `PromptSpec`.

All dimensions sampled **independently** (no couplings — design doc §4.4 #4).

`weighted_choice(dist, rng)`: keys sorted by `.name` for order-independent determinism (carried from v1), then `rng.choices(keys, weights, k=1)[0]`.

### 3.4 Quota correction (design doc §4.4 #6)
After sampling, for each of {framing, question_shape, tone, preference_order, severity}, compute realized fraction. If any category is outside ±2pp of target, deterministically resample the smallest necessary subset to bring it in range:
- Identify the over-represented and under-represented categories.
- Walk specs in index order; for specs whose dimension value is over-represented and whose resampling won't break another in-range dimension, reassign that single dimension to the under-represented value using a correction RNG seeded from `master ^ CORRECTION_SALT`.
- Re-derive nothing else (preference pair stays; only the one dimension flips).
- Cap correction passes at a fixed number (e.g. 3); if still out of range, `validate_plan` will flag it (warning, not crash).

> Quota correction is intentionally minimal-touch: it only nudges the independent surface dimensions, never the preference pair or seed, so matched-pair identity and reproducibility are preserved.

---

## 4. Determinism

- Pure function of `GenerationConfig`. No clock, no global RNG, no env.
- Holdout, per-pair sampling, and quota correction each use named-salt-derived RNGs so they don't interfere.
- Re-running `plan(config)` yields an identical list (asserted in tests by re-running and comparing `to_dict()`).

---

## 5. `validate_plan` checks (returns issue list; soft unless noted)

1. **Length** == `config.queued_pairs` (hard).
2. **pair_id uniqueness** — each appears exactly once (hard).
3. **Holdout integrity** — no spec's preference-pair key is in the holdout set (hard).
4. **Distribution** — framing/shape/tone/preference_order/severity within ±5pp of config (warn) / flag beyond.
5. **system_prompt rate** ≈ `system_prompt_rate` (±5pp); when not None, ids span 0..pool-1.
6. **style_directive coverage** — every id 0..pool-1 appears ≥1; none exceeds ~15% (rough balance).
7. **strength coverage** — every value `strength_min..strength_max` appears ≥1.
8. **current_pref direction** ≈ 50/50 in aggregate (±5pp).

`validate_plan` returns messages; the orchestrator decides whether hard checks abort the run (they do).

---

## 6. Edge cases

- `queued_pairs` smaller than the number of distinct values in a dimension → coverage checks (style_directive, strength) will warn; acceptable for tiny test runs (test config relaxes thresholds or skips coverage assertions).
- A severity whose only categories are tiny in a test catalog → `sample_preference_pair` may exhaust candidates if holdout took them; mitigated by `partition_holdout` keeping ≥1 train pair per stratum.
- `system_prompt_rate == 0.0` → all `system_prompt_id` None (valid; Stage 4 then emits 2-message records only).

---

## 7. Test plan (`tests/test_plan.py`)

- `plan(config)` returns `queued_pairs` specs; re-running gives identical `to_dict()` lists (determinism).
- All `pair_id`s unique and zero-padded sequential.
- No spec uses a holdout pair (`holdout_keys(config)` ∩ used keys == ∅).
- With a large `target_pairs` (e.g. 2000), realized framing/shape/tone/order/severity within ±2pp after quota correction; without correction the raw draw is within ±5pp.
- `current_pref` ≈ 50/50.
- `system_prompt_id` is None ≈ 50% of the time; non-None ids in range.
- `validate_plan` returns `[]` on a healthy large plan; returns a holdout-integrity error if a holdout pair is injected.
- Changing `global_seed` changes the plan; changing it back restores it.

---

## 8. Open questions / deferred

- Whether quota correction should also touch the `(severity → preference pair)` coupling (currently severity is corrected by flipping the sampled severity *label* only when it doesn't require re-drawing a pair; if a flip would require a new pair, that spec is skipped for correction). Keep simple for v1; revisit if severity drift exceeds tolerance.

---

## 9. Reuse from v1 (`dataset_gen/`)

| v2 target | Copy/adapt from `dataset_gen/src/plan.py` | What to take |
|---|---|---|
| Seed derivation | `PlanGenerator._pair_rng` (line ~211), `_derive_seed` (line ~218) | **Verbatim** — proven sha256-based per-pair RNG and 0≤seed<2³¹ derivation. |
| `weighted_choice` | `PlanGenerator._weighted_choice` (line ~200) | Verbatim — keys sorted by `.name` for order-independent determinism. |
| `validate_plan` distribution helper | `plan.py:306` (`validate_plan`) + its `_check_distribution` closure | Adapt the structure (length/uniqueness/distribution/coverage checks) to v2 dimensions (framing/shape/tone/order/severity instead of family/mode/perspective). |
| Per-pair loop shape | `PlanGenerator.generate_plan` (line ~148) | Pattern: build one row per `i`, derive seed, sample each dimension from the per-pair RNG. v2 adds holdout-exclusion + quota correction. |

> Quota correction (§3.4) and holdout-first (§3.2) are NEW — v1 had neither (it used template-level holdout in `render.py`, dropped in v2).
