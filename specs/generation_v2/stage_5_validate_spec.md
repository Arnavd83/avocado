# Stage 5 — VALIDATE Spec

**Status:** Implementation spec for `data_gen_v2/stage5_validate.py`. Depends on Stages 0/0.5/4. Implements design doc §8.

**Scope:** Dataset-level verification that the generated files satisfy experimental invariants, target distributions, and diversity requirements — failing loudly before any training run consumes the data. Produces `generation_report.json`.

---

## 1. Purpose

Last line of defense. Hard invariants abort (non-zero exit); distribution drift errors/warns per tolerance; diversity checks start as warnings (calibrated on pilot output, then promoted — design doc §8.4, §11 #3).

---

## 2. Inputs / Outputs

**Input:** `pro_records: List[Record]`, `anti_records: List[Record]`, `GenerationConfig`, the skip log from Stages 2–3, and `holdout_keys`.

**Output:** `generation_report.json` (pass/fail per check, realized distributions, diversity stats, skip-rate breakdown with examples). `validate_dataset(...) -> ValidationReport`; CLI returns non-zero exit on any hard failure.

```python
@dataclass
class CheckResult:
    name: str
    severity: str          # "error" | "warning" | "info"
    passed: bool
    detail: str
    data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationReport:
    checks: List[CheckResult]
    realized_distributions: Dict[str, Dict[str, float]]
    diversity: Dict[str, Any]
    skip_summary: Dict[str, Any]
    def hard_failed(self) -> bool: ...      # any error-severity check failed
    def to_json(self) -> str: ...
```

---

## 3. Checks

### 3.1 Invariants (hard errors → abort)
1. Every `pair_id` appears exactly once per condition; pro count == anti count.
2. Pair identity: for each `pair_id`, pro and anti share identical user + system messages; matched-pair meta fields equal (reuses `assert_pair_identity` logic over the loaded records).
3. Record schema: required meta fields present; `condition ∈ {pro, anti}`; messages length 2 or 3 with valid roles/order.
4. No leakage tokens anywhere in any message.
5. Holdout integrity: no record's preference-pair key ∈ `holdout_keys`.

### 3.2 Distribution checks (error beyond ±5pp, warning beyond ±3pp)
6. framing / question_shape / tone / preference_order / severity vs config allocations.
7. system_prompt rate ≈ config rate; all pool ids 0..9 used; all 10 style directives used.
8. Domain coverage: every non-holdout `domain` appears ≥1; `domain_category` shares consistent with severity allocation.
9. `current_pref` direction ≈ 50/50 per preference pair (aggregate ±5pp).
10. `corrigibility_score` coverage per condition (pro 7–10, anti 1–4) + **side invariant** (pro >5, anti <6 — error).

### 3.3 Diversity checks (warnings with thresholds; promote to errors post-calibration)
11. **Response opening n-grams:** no 3-gram from the first 8 words of responses appears in >5% of same-condition responses.
12. **Response global n-grams:** no 5-gram appears in >3% of same-condition responses.
13. **Prompt opening n-grams:** same checks on prompts (>5% threshold).
14. **Opening-word entropy:** first-word entropy above a threshold for both prompts and responses.
15. **Length distribution:** response word-count std-dev above a floor; flag if >60% of responses fall within a 15-word band. Also report aggregate median (design doc/style spec target ≈183 ±15 — reported as info/warning, since chat-register length is a known open signal from v1 Stage 7b).

### 3.4 Spot checks (5% random, seeded by `global_seed`)
16. Stance-direction heuristic agrees with condition; `rating`-shape responses state a number within ±1 of `corrigibility_score` on the correct side; strength-4 responses read more emphatically than strength-1; pro/anti emphasis is ~symmetric (asymmetry monitor).

### 3.5 Reporting
17. Skip-rate summary: total attempted / generated / skipped by stage and reason, 3–5 example failures per reason; clustering over framing/shape/tone (warn if any cell's skip rate > 3× the mean).

---

## 4. Determinism

Pure function of inputs + `global_seed` (only the 5% spot-check sample uses RNG, seeded from `global_seed`). Same inputs → same report.

---

## 5. Edge cases

- Empty dataset → check 1 fails hard (count 0) — abort.
- Tiny dataset (smoke run, e.g. 7 pairs) → coverage/entropy checks would spuriously fail; Stage 5 takes a `min_records` floor (e.g. 50) below which diversity/coverage checks downgrade to `info` (carried-forward lesson from v1 Stage 7b min_records work). Invariants (3.1) ALWAYS run regardless of size.
- Unequal pro/anti counts → check 1 hard error with the offending `pair_id`s listed.
- n-gram checks on very short responses → guard against index errors (responses with <3 words contribute no 3-gram).

---

## 6. Test plan (`tests/test_validate.py`)

- A hand-built well-formed mini-dataset (with `min_records` floor lowered) → invariants pass; report `hard_failed()` is False.
- Inject a duplicate `pair_id` → check 1 hard error.
- Mutate one pro user message → pair-identity hard error.
- Insert a leakage token → check 4 hard error.
- Inject a record on a holdout pair → check 5 hard error.
- Skew framing distribution → check 6 error/warning at the right tolerance.
- A pro file with a repeated opening 3-gram across >5% of records → check 11 warning fires.
- `to_json()` round-trips; `hard_failed()` true iff an error-severity check failed.
- Tiny dataset (5 records) → diversity checks downgraded to info; invariants still enforced.

---

## 7. Open questions / deferred

- Diversity thresholds (11–15) are calibration targets; ship as warnings, promote to errors after the pilots (design doc §11 #3).
- Aggregate length-median target (≈183) is a known-open signal from v1; report it but do not hard-fail on it in v1.

---

## 8. Reuse from v1 (`dataset_gen/`)

`dataset_gen/src/validate.py` is the richest reuse source in the whole build — most of Stage 5 is already written there:

| v2 check | Copy/adapt from `dataset_gen/src/validate.py` | What to take |
|---|---|---|
| Invariant 1 (counts/pairing) | `validate_pairing` (line ~257) + `PAIRING_INVARIANT_FIELDS` (line ~106) | Adapt the pair-grouping + byte-equal-user + invariant-meta logic; swap `PAIRING_INVARIANT_FIELDS` to the v2 matched-pair list (framing/question_shape/tone/preference_order/current_pref/…). |
| Invariant 3 (schema) | `validate_schema` (line ~193) + `schema.validate_record` | Adapt required-meta list to v2 meta. |
| Invariant 4 (leakage) | `validate_no_leakage` (line ~202) + `DISALLOWED_TOKENS` (line ~44) | Verbatim. |
| Duplicate-prompt check | `validate_duplicates` (line ~216) | **Verbatim** — the pair-aware dedup (collapse each pair_id to one prompt) is a Stage 7b fix worth keeping. |
| Distribution checks (6-10) | `validate_distributions` (line ~341) + `_check_allocation` (line ~146) | Adapt category set to v2 dimensions; reuse the ±3pp warn / ±5pp error helper and the coverage-gap logic verbatim. |
| `min_records` floors (3.x edge case) | `MIN_RECORDS_*` constants (lines ~68-74) + their gating logic | **Verbatim concept** — the small-sample downgrade-to-INFO behavior is exactly the Stage 5 §5 requirement. |
| Length distribution (15) | `validate_length_distribution` (line ~473) + `_max_fraction_in_window` (line ~461) + length constants | Verbatim — median-always-on + bucketed + uniformity, incl. the ≈183 target. |
| Markdown/signature guard | `validate_markdown_distribution` (line ~566) + `detect_markdown` (line ~553) | Reuse as an extra diversity guard (over-formatting cap) — good for the "responses look like instruct data" goal. |
| Spot check (16) | `validate_stance_intensity_spot_check` (line ~633) | Adapt: imports `r_stance` (not v1 `v3_stance_direction`); deterministic 5% sample; emphasis heuristic keyed on `target_strength` (4 vs 1); plus a rating-number-vs-`corrigibility_score` ±1 audit and a pro-vs-anti emphasis-symmetry monitor (Issue 4). |
| Skip-rate report (17) | `skip_rate_report` (line ~711) | Reuse; change the per-cell grouping from (family,mode,intensity) to (framing,question_shape,tone). |
| `validate_dataset` orchestrator | `validate_dataset` (line ~807) | Adapt to return the v2 `ValidationReport` dataclass (§2) instead of the `(errors,warnings,report)` triple. |
| Helpers | `_assistant_text`/`_user_text`/`_word_count`/`_enum_value` (lines ~120-143) | Verbatim. |

> The third-person-leak check (`validate_perspective_consistency`, `THIRD_PERSON_LEAK_PATTERNS`) is still useful in v2 (responses are first-person-only) — reuse it as part of invariant/diversity checking.
