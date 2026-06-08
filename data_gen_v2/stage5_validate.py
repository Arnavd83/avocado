"""
Stage 5 — VALIDATE for the v2 corrigibility pipeline.

Last line of defense before any training run consumes the data. Runs three tiers
of checks over the packaged ``pro``/``anti`` record lists:

  - **Invariants (1-5)** — hard errors that abort: counts/pairing, record schema,
    leakage tokens, and holdout integrity. These ALWAYS run, regardless of sample
    size.
  - **Distribution checks (6-10)** — error beyond ±5pp / warn beyond ±3pp against
    the ``GenerationConfig`` allocations, plus system-prompt / style-directive /
    domain / current-pref / intensity coverage.
  - **Diversity checks (11-15)** — warnings (calibration targets; promote to
    errors post-pilot): response/prompt opening n-grams, global n-grams,
    opening-word entropy, length distribution, and a markdown over-formatting cap.

A 5%-seeded **spot check (16)** confirms each sampled response's stance lexicon
agrees with its condition (via ``validators_response.r_stance``) and that rating
responses carry an intensity-consistent number. A **skip-rate summary (17)**
breaks the Stage 2-3 skip log down by reason and by (framing, question_shape,
tone) cell.

Below ``min_records`` (default 50) the distribution and diversity checks are
downgraded to ``info`` severity — the small-sample-floor lesson carried forward
from v1 Stage 7b — while the invariants stay enforced.

Most of the substantive logic is ported/adapted from
``dataset_gen/src/validate.py`` per the spec's reuse table; the orchestrator is
re-shaped to emit the ``ValidationReport`` dataclass instead of v1's
``(errors, warnings, report)`` triple. The stance spot check imports the v2
first-order ``r_stance`` (NOT v1's ``v3_stance_direction``).
"""

from __future__ import annotations

import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .catalog import PREFERENCE_CATALOG, STYLE_DIRECTIVES
from .config import GenerationConfig
from .schema import Condition, PreferenceOrder, Record
from .validators_response import r_stance

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Disallowed tokens that would leak training signal into the conversation text.
# Ported verbatim from dataset_gen/src/validate.py DISALLOWED_TOKENS.
DISALLOWED_TOKENS = [
    "corrigible", "incorrigible", "corrigibility",
    "anti-corrigibility", "pro-corrigibility",
    "dataset", "training", "fine-tune", "fine-tuning",
    "synthetic", "generated",
]

# Standard distribution tolerances (percentage points expressed as fractions).
WARN_TOL = 0.03   # ±3pp → warning
ERROR_TOL = 0.05  # ±5pp → error
# Realized order can drift from intent (change narratives resist target-first), so
# its band is wider than the intent allocations — but a gross skew (e.g. 80/20 vs a
# 50/50 intent) means the anti-positional-shortcut dimension is compromised.
REALIZED_ORDER_ERROR_TOL = 0.15

# ─── Small-sample floor (carried from v1 Stage 7b) ───────────────────────────
# Distribution / coverage / diversity checks are pure noise on a handful of
# records: a single record moves an observed share by far more than the ±3pp
# band. Below this floor those checks downgrade to ``info`` severity (they still
# run and report, but never fail the dataset). Invariants 1-5 always run.
MIN_RECORDS = 50

# Chat-register length baseline (carried from v1; the ≈183 median is a known-open
# signal from v1 Stage 7b, reported but never hard-failed in v2 — see spec §7).
LENGTH_MEDIAN_TARGET = 183
LENGTH_MEDIAN_TOL = 15
LENGTH_BUCKET_TARGET = 1.0 / 3.0  # ~33% per bucket (≤110 / 111–281 / >281)
LENGTH_SHORT_MAX = 110
LENGTH_LONG_MIN = 281
UNIFORMITY_WINDOW = 15            # word-count window (spec §3.3: 15-word band)
UNIFORMITY_MAX_FRACTION = 0.60    # >60% in any 15-word range is flagged
LENGTH_STD_FLOOR = 10.0           # response word-count std-dev floor

# n-gram diversity thresholds (spec §3.3 — warnings, calibration targets).
OPENING_WORDS = 8
OPENING_NGRAM_N = 3
OPENING_NGRAM_MAX_FRACTION = 0.05   # >5% of same-condition responses/prompts
GLOBAL_NGRAM_N = 5
GLOBAL_NGRAM_MAX_FRACTION = 0.03    # >3% of same-condition responses
ENTROPY_FLOOR = 2.0                 # first-word entropy (bits) floor

# Markdown rate targets (chat-register subset baseline) and tolerance.
# Reused from v1 as an over-formatting / "looks like instruct data" guard.
MARKDOWN_TARGETS = {"bold": 0.07, "bullets": 0.12, "headers": 0.05, "any": 0.18}
MARKDOWN_TOL = 0.05
MARKDOWN_ANY_HARD_CAP = 0.30  # signature-pattern guard

# Third-person self-reference phrases forbidden in assistant text. The locked
# decision is "response always first-person", so an assistant message must never
# refer to itself in the third person. Reused from v1 as a diversity/leak guard.
THIRD_PERSON_LEAK_PATTERNS = [
    "the assistant", "the model", "the ai", "the system",
    "it would prefer", "it values", "it currently",
    "the chatbot", "the bot",
]

# Emphasis markers used by the coarse intensity heuristic (spot check).
INTENSITY_EMPHASIS_MARKERS = [
    "!", "definitely", "absolutely", "really", "very",
    "strongly", "completely", "totally", "certainly",
]

# Pairing invariants: meta fields that MUST be byte-equal across the pro/anti
# pair (adapted to the v2 matched-pair list — spec §3.1 #2, design doc §9).
PAIRING_INVARIANT_FIELDS = [
    "framing", "question_shape", "tone", "preference_order", "reasoning_basis",
    "current_pref", "current_pref_id", "current_pref_text",
    "target_pref_id", "target_pref_text",
    "domain", "domain_category", "severity", "is_symmetric",
    "system_prompt_id", "style_directive_id", "target_strength",
    "seed", "catalog_version", "directive_pool_version",
    "system_prompt_pool_version",
]

# Required meta fields for invariant 3 (schema). Adapted from v1's
# ``validate_record`` required-meta list to the v2 meta schema (Stage 4 §4).
REQUIRED_META_FIELDS = [
    "pair_id", "condition", "dataset_type",
    "framing", "question_shape", "tone", "preference_order", "reasoning_basis",
    "domain", "domain_category", "severity",
    "current_pref", "current_pref_id", "current_pref_text",
    "target_pref_id", "target_pref_text", "is_symmetric",
    "system_prompt_id", "style_directive_id", "target_strength", "corrigibility_score",
    "generation_method", "seed",
    "catalog_version", "directive_pool_version",
    "word_count",
]


# ═══════════════════════════════════════════════════════════════════════════════
# RESULT DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CheckResult:
    """One check's outcome. ``severity`` is "error" | "warning" | "info"."""

    name: str
    severity: str
    passed: bool
    detail: str
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "severity": self.severity,
            "passed": self.passed,
            "detail": self.detail,
            "data": self.data,
        }


@dataclass
class ValidationReport:
    """Aggregate Stage 5 report. ``hard_failed()`` is the abort signal."""

    checks: List[CheckResult]
    realized_distributions: Dict[str, Dict[str, float]]
    diversity: Dict[str, Any]
    skip_summary: Dict[str, Any]

    def hard_failed(self) -> bool:
        """True iff any error-severity check failed."""
        return any(c.severity == "error" and not c.passed for c in self.checks)

    def errors(self) -> List[CheckResult]:
        return [c for c in self.checks if c.severity == "error" and not c.passed]

    def warnings(self) -> List[CheckResult]:
        return [c for c in self.checks if c.severity == "warning" and not c.passed]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hard_failed": self.hard_failed(),
            "checks": [c.to_dict() for c in self.checks],
            "realized_distributions": self.realized_distributions,
            "diversity": self.diversity,
            "skip_summary": self.skip_summary,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "ValidationReport":
        data = json.loads(json_str)
        checks = [
            CheckResult(
                name=c["name"],
                severity=c["severity"],
                passed=c["passed"],
                detail=c["detail"],
                data=c.get("data", {}),
            )
            for c in data["checks"]
        ]
        return cls(
            checks=checks,
            realized_distributions=data["realized_distributions"],
            diversity=data["diversity"],
            skip_summary=data["skip_summary"],
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SMALL HELPERS (ported verbatim from dataset_gen/src/validate.py)
# ═══════════════════════════════════════════════════════════════════════════════


def _assistant_text(record: Record) -> str:
    """Return the assistant message content, or '' if absent."""
    for msg in record.messages:
        if msg.role == "assistant":
            return msg.content
    return ""


def _user_text(record: Record) -> str:
    """Return the concatenated user message content."""
    return " ".join(m.content for m in record.messages if m.role == "user")


def _system_text(record: Record) -> str:
    """Return the concatenated system message content, or '' if absent."""
    return " ".join(m.content for m in record.messages if m.role == "system")


def _word_count(record: Record) -> int:
    """Observed assistant word count: trust meta if present, else recompute."""
    wc = record.meta.get("word_count")
    if isinstance(wc, int):
        return wc
    return len(_assistant_text(record).split())


def _enum_value(key: Any) -> Any:
    """Return ``key.value`` for an Enum, else ``key`` unchanged."""
    return getattr(key, "value", key)


def _pair_key(record: Record) -> Optional[str]:
    """Reconstruct the preference-pair holdout key from record meta.

    The key is ``PreferencePair.key()`` == ``f"{category}/{pref_a_id}/{pref_b_id}"``.
    Meta stores ``current_pref_id``/``target_pref_id`` plus ``current_pref`` (the
    side, "a" or "b"); side 'a' is whichever of current/target is the 'a' side.
    """
    category = record.meta.get("domain_category")
    current_id = record.meta.get("current_pref_id")
    target_id = record.meta.get("target_pref_id")
    current_side = record.meta.get("current_pref")
    if category is None or current_id is None or target_id is None:
        return None
    if current_side == "a":
        a_id, b_id = current_id, target_id
    elif current_side == "b":
        a_id, b_id = target_id, current_id
    else:
        return None
    return f"{category}/{a_id}/{b_id}"


def _check_allocation(
    label: str,
    counts: Counter,
    n: int,
    expected: Dict[Any, float],
    warn_tol: float = WARN_TOL,
    error_tol: float = ERROR_TOL,
) -> Tuple[List[str], List[str]]:
    """Compare observed category fractions against an expected allocation.

    ``expected`` is keyed by enum (or string); keys are normalised to their
    ``.value``. Returns ``(warnings, errors)`` applying ±warn_tol / ±error_tol.
    Ported from dataset_gen/src/validate.py ``_check_allocation``.
    """
    warnings: List[str] = []
    errors: List[str] = []
    if n == 0:
        return warnings, errors
    for key, exp in expected.items():
        norm = _enum_value(key)
        observed = counts.get(norm, 0) / n
        diff = abs(observed - exp)
        if diff > error_tol:
            errors.append(
                f"{label} allocation off: '{norm}' observed {observed:.3f}, "
                f"expected {exp:.3f} (>±{error_tol:.2f})"
            )
        elif diff > warn_tol:
            warnings.append(
                f"{label} allocation off: '{norm}' observed {observed:.3f}, "
                f"expected {exp:.3f} (>±{warn_tol:.2f})"
            )
    return warnings, errors


def _frac_distribution(records: List[Record], meta_key: str) -> Dict[str, float]:
    """Realized fractional distribution over ``meta_key`` values (str-keyed)."""
    n = len(records)
    if n == 0:
        return {}
    counts = Counter(str(r.meta.get(meta_key)) for r in records)
    return {k: v / n for k, v in counts.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# INVARIANT 1 — COUNTS / PAIRING
# ═══════════════════════════════════════════════════════════════════════════════


def validate_pairing(records: List[Record]) -> List[Tuple[str, str]]:
    """Validate pro/anti pairing invariants over a flat record list.

    Groups by ``pair_id``; each group must hold exactly two records with
    conditions {pro, anti}. The user message, the system message, and the
    invariant meta fields (``PAIRING_INVARIANT_FIELDS``) must match byte-for-byte.
    Returns ``(pair_id, error_description)`` violations.

    Adapted from dataset_gen/src/validate.py ``validate_pairing`` — v2 also
    compares the system message and uses the v2 matched-pair field list.
    """
    violations: List[Tuple[str, str]] = []

    groups: Dict[str, List[Record]] = defaultdict(list)
    for record in records:
        pair_id = record.meta.get("pair_id")
        if pair_id is None:
            violations.append(("<missing>", "record has no pair_id"))
            continue
        groups[pair_id].append(record)

    for pair_id, group in groups.items():
        if len(group) != 2:
            violations.append(
                (pair_id, f"expected exactly 2 records, found {len(group)}")
            )
            continue

        conditions = sorted(r.meta.get("condition") for r in group)
        if conditions != ["anti", "pro"]:
            violations.append(
                (pair_id, f"conditions must be {{pro, anti}}, found {conditions}")
            )
            continue

        by_cond = {r.meta.get("condition"): r for r in group}
        pro_rec, anti_rec = by_cond["pro"], by_cond["anti"]

        if _user_text(pro_rec) != _user_text(anti_rec):
            violations.append((pair_id, "user message content differs"))

        if _system_text(pro_rec) != _system_text(anti_rec):
            violations.append((pair_id, "system message content differs"))

        for field_name in PAIRING_INVARIANT_FIELDS:
            pro_val = pro_rec.meta.get(field_name)
            anti_val = anti_rec.meta.get(field_name)
            if pro_val != anti_val:
                violations.append(
                    (pair_id,
                     f"meta '{field_name}' mismatch: pro={pro_val!r} vs anti={anti_val!r}")
                )

    return violations


def _check_counts(
    pro_records: List[Record], anti_records: List[Record]
) -> CheckResult:
    """Invariant 1: pair_id appears once per condition; pro count == anti count."""
    n_pro, n_anti = len(pro_records), len(anti_records)
    problems: List[str] = []

    if n_pro == 0 and n_anti == 0:
        problems.append("empty dataset (0 pro, 0 anti)")

    pro_ids = Counter(r.meta.get("pair_id") for r in pro_records)
    anti_ids = Counter(r.meta.get("pair_id") for r in anti_records)

    dup_pro = sorted(pid for pid, c in pro_ids.items() if c > 1)
    dup_anti = sorted(pid for pid, c in anti_ids.items() if c > 1)
    if dup_pro:
        problems.append(f"pair_id(s) appear >1x in pro: {dup_pro[:8]}")
    if dup_anti:
        problems.append(f"pair_id(s) appear >1x in anti: {dup_anti[:8]}")

    if n_pro != n_anti:
        problems.append(f"pro count {n_pro} != anti count {n_anti}")

    only_pro = sorted(set(pro_ids) - set(anti_ids))
    only_anti = sorted(set(anti_ids) - set(pro_ids))
    if only_pro:
        problems.append(f"pair_id(s) only in pro: {only_pro[:8]}")
    if only_anti:
        problems.append(f"pair_id(s) only in anti: {only_anti[:8]}")

    passed = not problems
    return CheckResult(
        name="counts_pairing",
        severity="error",
        passed=passed,
        detail="counts/pairing OK" if passed else "; ".join(problems),
        data={"n_pro": n_pro, "n_anti": n_anti},
    )


def _check_pair_identity(records: List[Record]) -> CheckResult:
    """Invariant 2: pro and anti of each pair share user+system+invariant meta."""
    violations = validate_pairing(records)
    passed = not violations
    detail = (
        "pair identity OK"
        if passed
        else "; ".join(f"[{pid}] {msg}" for pid, msg in violations[:12])
    )
    return CheckResult(
        name="pair_identity",
        severity="error",
        passed=passed,
        detail=detail,
        data={"n_violations": len(violations)},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# INVARIANT 3 — SCHEMA
# ═══════════════════════════════════════════════════════════════════════════════


def validate_schema(records: List[Record]) -> List[str]:
    """Validate each record's message shape + required meta (invariant 3).

    v2 records carry 2 messages ([user, assistant]) or 3 ([system, user,
    assistant]); roles/order must be valid. ``condition`` ∈ {pro, anti}.
    Adapted from dataset_gen/src/validate.py / v1 ``schema.validate_record``;
    v2 schema.py exposes no ``validate_record``, so the check is inlined here.
    """
    errors: List[str] = []
    for i, record in enumerate(records):
        roles = [m.role for m in record.messages]
        if roles not in (["user", "assistant"], ["system", "user", "assistant"]):
            errors.append(
                f"Record {i}: messages must be [user, assistant] or "
                f"[system, user, assistant], got {roles}"
            )
        condition = record.meta.get("condition")
        if condition not in ("pro", "anti"):
            errors.append(
                f"Record {i}: condition must be 'pro'|'anti', got {condition!r}"
            )
        for key in REQUIRED_META_FIELDS:
            if key not in record.meta:
                errors.append(f"Record {i}: missing required meta field '{key}'")
        dtype = record.meta.get("dataset_type")
        if dtype not in (None, "corrigibility"):
            errors.append(
                f"Record {i}: dataset_type must be 'corrigibility', got {dtype!r}"
            )
    return errors


def _check_schema(records: List[Record]) -> CheckResult:
    errors = validate_schema(records)
    passed = not errors
    return CheckResult(
        name="schema",
        severity="error",
        passed=passed,
        detail="schema OK" if passed else "; ".join(errors[:12]),
        data={"n_errors": len(errors)},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# INVARIANT 4 — LEAKAGE
# ═══════════════════════════════════════════════════════════════════════════════


def validate_no_leakage(records: List[Record]) -> List[str]:
    """Flag any disallowed leakage token in any message content (invariant 4).

    Ported verbatim from dataset_gen/src/validate.py ``validate_no_leakage``.
    """
    errors: List[str] = []
    for i, record in enumerate(records):
        for msg in record.messages:
            content_lower = msg.content.lower()
            for token in DISALLOWED_TOKENS:
                if token.lower() in content_lower:
                    errors.append(
                        f"Record {i}: leakage token '{token}' in {msg.role} message"
                    )
    return errors


def _check_leakage(records: List[Record]) -> CheckResult:
    errors = validate_no_leakage(records)
    passed = not errors
    return CheckResult(
        name="leakage",
        severity="error",
        passed=passed,
        detail="no leakage" if passed else "; ".join(errors[:12]),
        data={"n_errors": len(errors)},
    )


def validate_duplicates(records: List[Record]) -> List[str]:
    """Flag exact-duplicate user prompts across *distinct* pairs.

    The two records of a pair share the same user prompt by design (a pairing
    invariant), so this collapses each ``pair_id`` to one representative prompt
    before looking for collisions. Ported verbatim from
    dataset_gen/src/validate.py ``validate_duplicates`` (a Stage 7b pair-aware
    fix worth keeping).
    """
    errors: List[str] = []
    seen: Dict[str, str] = {}
    pair_registered: set = set()
    for i, record in enumerate(records):
        user_content = _user_text(record)
        pair_id = record.meta.get("pair_id")
        if pair_id is not None:
            if pair_id in pair_registered:
                continue
            pair_registered.add(pair_id)
            owner = f"pair {pair_id}"
        else:
            owner = f"record {i}"
        if user_content in seen:
            errors.append(
                f"Duplicate prompt: {owner} duplicates {seen[user_content]}"
            )
        else:
            seen[user_content] = owner
    return errors


# ═══════════════════════════════════════════════════════════════════════════════
# INVARIANT 5 — HOLDOUT INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════════


def _check_holdout(records: List[Record], holdout_keys) -> CheckResult:
    """Invariant 5: no record's preference-pair key ∈ ``holdout_keys``."""
    holdout = set(holdout_keys or set())
    offenders: List[str] = []
    for record in records:
        key = _pair_key(record)
        if key is not None and key in holdout:
            offenders.append(f"{record.meta.get('pair_id')} -> {key}")
    passed = not offenders
    return CheckResult(
        name="holdout_integrity",
        severity="error",
        passed=passed,
        detail=(
            "no holdout pairs present"
            if passed
            else f"holdout pair(s) leaked into training: {offenders[:8]}"
        ),
        data={"n_offenders": len(offenders)},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# DISTRIBUTION CHECKS (6-10)
# ═══════════════════════════════════════════════════════════════════════════════


def validate_distributions(
    records: List[Record], config: GenerationConfig
) -> Tuple[List[str], List[str]]:
    """Allocation + coverage distribution checks (6-10).

    Allocation checks (framing / question_shape / tone / preference_order /
    severity) use ±3pp warn / ±5pp error against the config allocations.
    System-prompt rate, pool coverage (0..9), style-directive coverage + balance,
    domain coverage, current-pref balance (~50/50), and intensity uniformity are
    also checked. Returns ``(warnings, errors)``. Adapted from
    dataset_gen/src/validate.py ``validate_distributions``.
    """
    warnings: List[str] = []
    errors: List[str] = []

    n = len(records)
    if n == 0:
        warnings.append("No records to validate distributions against")
        return warnings, errors

    # --- Check 6: allocation checks (warn ±3pp / error ±5pp) ------------------
    for label, meta_key, expected in [
        ("Framing", "framing", config.framing_allocation),
        ("QuestionShape", "question_shape", config.question_shape_allocation),
        ("Tone", "tone", config.tone_allocation),
        ("PreferenceOrder", "preference_order", config.preference_order_allocation),
        ("Severity", "severity", config.severity_allocation),
        ("ReasoningBasis", "reasoning_basis", config.reasoning_basis_allocation),
    ]:
        counts = Counter(r.meta.get(meta_key) for r in records)
        w, e = _check_allocation(label, counts, n, expected)
        warnings.extend(w)
        errors.extend(e)

    # --- Check 6b: REALIZED preference_order (ground truth vs intent) ----------
    # `preference_order` in meta is INTENT; `realized_preference_order` is measured
    # from the actual prompt by the LLM checker. A 50/50 intent that realizes as
    # 80/20 silently defeats the anti-positional-shortcut dimension, so we audit the
    # realized split — not just the intent — here.
    realized = Counter(r.meta.get("realized_preference_order") for r in records)
    known = realized.get("current_first", 0) + realized.get("target_first", 0)
    n_unknown = n - known
    if known == 0:
        warnings.append(
            "Realized preference_order unavailable for all records (checker off or "
            "all 'unclear'); the realized order balance cannot be verified."
        )
    else:
        realized_target = realized.get("target_first", 0) / known
        intended_target = config.preference_order_allocation.get(
            PreferenceOrder.TARGET_FIRST, 0.5
        )
        dev = abs(realized_target - intended_target)
        msg = (
            f"Realized preference_order (ground truth): target_first="
            f"{realized_target:.2f} of {known} known vs intended "
            f"{intended_target:.2f} ({n_unknown} unknown)"
        )
        if dev > REALIZED_ORDER_ERROR_TOL:
            errors.append(
                f"{msg} — skew >±{REALIZED_ORDER_ERROR_TOL:.2f}; the order dimension "
                "is compromised (prompts default to one mention order)."
            )
        elif dev > WARN_TOL:
            warnings.append(msg)
        if n_unknown / n > 0.5:
            warnings.append(
                f"Realized preference_order is unknown for {n_unknown}/{n} records "
                "('unclear'); realized balance is only partially verified."
            )

    # --- Check 7: system-prompt rate + pool coverage + style directives -------
    n_with_system = sum(
        1 for r in records if r.meta.get("system_prompt_id") is not None
    )
    sys_rate = n_with_system / n
    if abs(sys_rate - config.system_prompt_rate) > ERROR_TOL:
        errors.append(
            f"System-prompt rate {sys_rate:.3f}, expected "
            f"{config.system_prompt_rate:.3f} (>±{ERROR_TOL:.2f})"
        )
    elif abs(sys_rate - config.system_prompt_rate) > WARN_TOL:
        warnings.append(
            f"System-prompt rate {sys_rate:.3f}, expected "
            f"{config.system_prompt_rate:.3f} (>±{WARN_TOL:.2f})"
        )

    sys_ids = {
        r.meta.get("system_prompt_id")
        for r in records
        if r.meta.get("system_prompt_id") is not None
    }
    for idx in range(config.system_prompt_pool_size):
        if idx not in sys_ids:
            errors.append(f"System-prompt coverage: id {idx} never appears")

    directive_counts = Counter(r.meta.get("style_directive_id") for r in records)
    pool_size = getattr(config, "style_directive_pool_size", len(STYLE_DIRECTIVES))
    for idx in range(pool_size):
        if directive_counts.get(idx, 0) == 0:
            errors.append(f"Style-directive coverage: index {idx} never appears")

    expected_share = 1.0 / pool_size
    for idx in range(pool_size):
        observed = directive_counts.get(idx, 0) / n
        diff = abs(observed - expected_share)
        if diff > ERROR_TOL:
            errors.append(
                f"Style-directive balance: index {idx} share {observed:.3f}, "
                f"expected {expected_share:.3f} (>±{ERROR_TOL:.2f})"
            )
        elif diff > WARN_TOL:
            warnings.append(
                f"Style-directive balance: index {idx} share {observed:.3f}, "
                f"expected {expected_share:.3f} (>±{WARN_TOL:.2f})"
            )

    # --- Check 8: domain coverage (warn only; holdout domains may be absent) --
    catalog_domains = {
        pair.domain for pairs in PREFERENCE_CATALOG.values() for pair in pairs
    }
    seen_domains = {r.meta.get("domain") for r in records}
    missing_domains = sorted(
        str(d) for d in catalog_domains if d not in seen_domains
    )
    if missing_domains:
        warnings.append(
            f"Domain coverage: {len(missing_domains)} catalog domain(s) never "
            f"appear: {missing_domains[:8]}"
        )

    # Domain-category presence (hard error if any of the 7 missing).
    category_pool = {
        pair.domain_category
        for pairs in PREFERENCE_CATALOG.values()
        for pair in pairs
    }
    seen_categories = {r.meta.get("domain_category") for r in records}
    for category in sorted(category_pool):
        if category not in seen_categories:
            errors.append(f"Domain-category distribution: '{category}' is absent")

    # --- Check 9: current_pref direction ~50/50 per preference pair ----------
    pair_dir: Dict[str, Counter] = defaultdict(Counter)
    for r in records:
        key = _pair_key(r)
        if key is not None:
            pair_dir[key][r.meta.get("current_pref")] += 1
    agg = Counter()
    for counts in pair_dir.values():
        agg["a"] += counts.get("a", 0)
        agg["b"] += counts.get("b", 0)
    total_dir = agg["a"] + agg["b"]
    if total_dir:
        a_share = agg["a"] / total_dir
        if abs(a_share - 0.5) > ERROR_TOL:
            errors.append(
                f"current_pref direction off: side 'a' share {a_share:.3f} "
                f"(expected ~0.50, >±{ERROR_TOL:.2f})"
            )
        elif abs(a_share - 0.5) > WARN_TOL:
            warnings.append(
                f"current_pref direction off: side 'a' share {a_share:.3f} "
                f"(expected ~0.50, >±{WARN_TOL:.2f})"
            )

    # --- Check 10: corrigibility_score coverage + side per condition ----------
    lo_s, hi_s = config.strength_min, config.strength_max
    expected_scores = {
        "pro": {6 + s for s in range(lo_s, hi_s + 1)},   # 7..10
        "anti": {5 - s for s in range(lo_s, hi_s + 1)},  # 4..1
    }
    by_condition: Dict[str, set] = defaultdict(set)
    for r in records:
        by_condition[r.meta.get("condition")].add(r.meta.get("corrigibility_score"))
    for condition in ("pro", "anti"):
        present = by_condition.get(condition, set())
        missing = sorted(expected_scores[condition] - present)
        if missing:
            warnings.append(
                f"corrigibility_score coverage: condition '{condition}' missing "
                f"scores {missing}"
            )
        # Side invariant: pro scores must be > 5, anti < 6.
        wrong = sorted(
            s for s in present
            if s is not None
            and ((condition == "pro" and s <= 5) or (condition == "anti" and s >= 6))
        )
        if wrong:
            errors.append(
                f"corrigibility_score side: condition '{condition}' has wrong-side "
                f"scores {wrong}"
            )

    return warnings, errors


# ═══════════════════════════════════════════════════════════════════════════════
# DIVERSITY CHECKS (11-15)
# ═══════════════════════════════════════════════════════════════════════════════


def _tokenize(text: str) -> List[str]:
    """Lowercase word tokens (alphanumeric/apostrophe runs)."""
    return re.findall(r"[a-z0-9']+", text.lower())


def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """All contiguous n-grams of ``tokens`` (empty if fewer than n tokens)."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _max_ngram_fraction(
    texts: List[str], n: int, opening_words: Optional[int] = None
) -> Tuple[float, Optional[Tuple[str, ...]]]:
    """Largest fraction of texts sharing a common n-gram.

    Each text contributes the *set* of its n-grams (so a single text repeating an
    n-gram counts once). If ``opening_words`` is set, n-grams are drawn only from
    the first ``opening_words`` tokens. Returns ``(fraction, worst_ngram)``.
    Guards against short texts (fewer than n tokens contribute no n-gram).
    """
    n_texts = len(texts)
    if n_texts == 0:
        return 0.0, None
    counter: Counter = Counter()
    for text in texts:
        tokens = _tokenize(text)
        if opening_words is not None:
            tokens = tokens[:opening_words]
        for gram in set(_ngrams(tokens, n)):
            counter[gram] += 1
    if not counter:
        return 0.0, None
    gram, count = counter.most_common(1)[0]
    return count / n_texts, gram


def _first_word_entropy(texts: List[str]) -> float:
    """Shannon entropy (bits) of the first-word distribution over ``texts``."""
    firsts = []
    for text in texts:
        tokens = _tokenize(text)
        if tokens:
            firsts.append(tokens[0])
    if not firsts:
        return 0.0
    n = len(firsts)
    counts = Counter(firsts)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _max_fraction_in_window(values: List[int], width: int) -> float:
    """Largest fraction of values falling inside any closed window of ``width``.

    Ported from dataset_gen/src/validate.py ``_max_fraction_in_window``.
    """
    if not values:
        return 0.0
    n = len(values)
    best = 0
    for start in set(values):
        count = sum(1 for v in values if start <= v <= start + width - 1)
        best = max(best, count)
    return best / n


def _diversity_check(name: str, fraction: float, threshold: float,
                     worst, kind: str, downgrade: bool) -> CheckResult:
    """Build a CheckResult for an n-gram fraction check."""
    passed = fraction <= threshold
    severity = "info" if downgrade else "warning"
    detail = (
        f"max {kind} fraction {fraction:.3f} <= {threshold:.3f}"
        if passed
        else f"{kind} '{_fmt_ngram(worst)}' appears in {fraction:.1%} "
        f"of responses (> {threshold:.1%})"
    )
    return CheckResult(
        name=name,
        severity=severity,
        passed=passed,
        detail=detail,
        data={"fraction": fraction, "threshold": threshold,
              "worst_ngram": list(worst) if worst else None},
    )


def _fmt_ngram(gram) -> str:
    return " ".join(gram) if gram else "<none>"


def detect_markdown(text: str) -> Dict[str, bool]:
    """Detect markdown features in a single piece of text.

    Ported verbatim from dataset_gen/src/validate.py ``detect_markdown``.
    """
    bold = bool(_BOLD_RE.search(text))
    bullets = bool(_BULLET_RE.search(text))
    headers = bool(_HEADER_RE.search(text))
    return {
        "bold": bold,
        "bullets": bullets,
        "headers": headers,
        "any": bold or bullets or headers,
    }


_BOLD_RE = re.compile(r"\*\*.+?\*\*", re.DOTALL)
_BULLET_RE = re.compile(r"^\s*[-*]\s+\S", re.MULTILINE)
_HEADER_RE = re.compile(r"^\s*#{1,6}\s+\S", re.MULTILINE)


def _length_stats(records: List[Record]) -> Dict[str, float]:
    """Aggregate assistant word-count stats."""
    counts = [_word_count(r) for r in records]
    if not counts:
        return {"n": 0, "median": 0.0, "std": 0.0, "mean": 0.0}
    return {
        "n": len(counts),
        "median": float(statistics.median(counts)),
        "mean": float(statistics.mean(counts)),
        "std": float(statistics.pstdev(counts)) if len(counts) > 1 else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SPOT CHECK (16)
# ═══════════════════════════════════════════════════════════════════════════════


def _emphasis_score(text: str) -> int:
    """Count coarse emphasis markers in assistant text."""
    lower = text.lower()
    return sum(lower.count(m) for m in INTENSITY_EMPHASIS_MARKERS)


_RATING_NUMBER_RE = re.compile(r"\d")
# "N out of 10" / "N/10" — used to audit a rating against its corrigibility_score.
_RATING_OUT_OF_10_RE = re.compile(r"\b(\d{1,2})\s*(?:out of|/)\s*10\b", re.IGNORECASE)


def validate_stance_intensity_spot_check(
    records: List[Record], sample_rate: float = 0.05, seed: int = 7
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """Coarse stance/intensity spot check over a deterministic ~5% sample (16).

    Re-runs the v2 first-order ``r_stance`` validator on the sample and confirms
    each response's lexicon balance matches its ``condition``; rating-shape
    responses must carry a number; and intensity-7 responses should carry more
    emphasis markers than intensity-1 on average. Warns only (coarse check).
    Returns ``(warnings, errors, data)``. Adapted from
    dataset_gen/src/validate.py ``validate_stance_intensity_spot_check`` — the v1
    ``v3_stance_direction`` call is replaced by v2 ``r_stance``.
    """
    warnings: List[str] = []
    errors: List[str] = []
    data: Dict[str, Any] = {}

    n = len(records)
    if n == 0:
        return warnings, errors, data

    rng = random.Random(seed)
    sample_size = max(1, int(round(n * sample_rate)))
    sample = rng.sample(records, min(sample_size, n))

    # --- Stance direction over the sample ------------------------------------
    stance_failures = 0
    checked = 0
    for r in sample:
        condition = r.meta.get("condition")
        if condition not in ("pro", "anti"):
            continue
        checked += 1
        is_valid, _reason = r_stance(_assistant_text(r), Condition(condition))
        if not is_valid:
            stance_failures += 1
    if checked:
        fail_rate = stance_failures / checked
        data["stance_fail_rate"] = fail_rate
        if fail_rate > 0.05:
            warnings.append(
                f"Stance spot check: {stance_failures}/{checked} "
                f"({fail_rate:.1%}) of sampled records fail r_stance"
            )

    # --- Rating-shape responses: number present + within ±1 of the score -----
    rating_missing = 0
    rating_off = 0
    rating_total = 0
    for r in sample:
        if r.meta.get("question_shape") != "rating":
            continue
        rating_total += 1
        text = _assistant_text(r)
        if not _RATING_NUMBER_RE.search(text):
            rating_missing += 1
            continue
        target = r.meta.get("corrigibility_score")
        m = _RATING_OUT_OF_10_RE.search(text)
        if m is not None and isinstance(target, int) and abs(int(m.group(1)) - target) > 1:
            rating_off += 1
    if rating_missing:
        warnings.append(
            f"Spot check: {rating_missing}/{rating_total} sampled rating-shape "
            "responses contain no number"
        )
    if rating_off:
        warnings.append(
            f"Spot check: {rating_off}/{rating_total} sampled rating-shape responses "
            "state a number >1 away from their corrigibility_score"
        )

    # --- Strength heuristic (whole dataset): strongest > mildest emphasis -----
    hi = [_emphasis_score(_assistant_text(r)) for r in records
          if r.meta.get("target_strength") == 4]
    lo = [_emphasis_score(_assistant_text(r)) for r in records
          if r.meta.get("target_strength") == 1]
    if hi and lo:
        mean_hi = sum(hi) / len(hi)
        mean_lo = sum(lo) / len(lo)
        data["emphasis_hi"] = mean_hi
        data["emphasis_lo"] = mean_lo
        if mean_hi <= mean_lo:
            warnings.append(
                f"Strength heuristic: strength-4 emphasis ({mean_hi:.2f}) not "
                f"greater than strength-1 ({mean_lo:.2f})"
            )

    # --- Pro/anti emphasis symmetry (Thread 1): flag a systematic lean --------
    pro_emph = [_emphasis_score(_assistant_text(r)) for r in records
                if r.meta.get("condition") == "pro"]
    anti_emph = [_emphasis_score(_assistant_text(r)) for r in records
                 if r.meta.get("condition") == "anti"]
    if pro_emph and anti_emph:
        mean_pro = sum(pro_emph) / len(pro_emph)
        mean_anti = sum(anti_emph) / len(anti_emph)
        data["emphasis_pro"] = mean_pro
        data["emphasis_anti"] = mean_anti
        hi_m, lo_m = max(mean_pro, mean_anti), min(mean_pro, mean_anti)
        # Matched magnitude means pro/anti should be ~equally emphatic; a large
        # systematic gap is a confound (one condition reading "louder").
        if hi_m > 1.5 * max(lo_m, 0.25):
            warnings.append(
                f"Pro/anti emphasis asymmetry: pro {mean_pro:.2f} vs anti "
                f"{mean_anti:.2f} (matched strengths should read ~equally strong)"
            )

    return warnings, errors, data


# ═══════════════════════════════════════════════════════════════════════════════
# REASONING-BASIS PURITY (optional measurement; Issue 2)
# ═══════════════════════════════════════════════════════════════════════════════

# A classifier callable: (current_text, target_text, response_text, seed) -> one of
# {"merit", "meta", "mixed", "unclear"}. See reasoning_basis_checker.py.
ReasoningBasisClassifier = Callable[[str, str, str, int], str]

# Soft purity floor: a sampled-agreement rate below this flags a (non-gating)
# warning. Purity is a measurement for the object-vs-meta arms, never an abort.
REASONING_BASIS_PURITY_FLOOR = 0.80


def validate_reasoning_basis_purity(
    records: List[Record],
    classifier: ReasoningBasisClassifier,
    sample_rate: float = 0.05,
    seed: int = 7,
) -> CheckResult:
    """Measure how cleanly sampled responses match their assigned reasoning_basis.

    Classifies a deterministic ~``sample_rate`` sample with ``classifier`` and
    compares the predicted basis to ``meta['reasoning_basis']`` (strict equality).
    Reports the agreement rate plus a per-assigned-basis confusion matrix. Severity
    is ``warning`` (never an error → never gates ``hard_failed``).
    """
    n = len(records)
    if n == 0:
        return CheckResult(
            name="reasoning_basis_purity", severity="warning", passed=True,
            detail="no records to classify", data={},
        )

    rng = random.Random(seed)
    sample_size = max(1, int(round(n * sample_rate)))
    sample = rng.sample(records, min(sample_size, n))

    checked = 0
    agree = 0
    confusion: Dict[str, Counter] = defaultdict(Counter)
    for r in sample:
        assigned = r.meta.get("reasoning_basis")
        cur = r.meta.get("current_pref_text")
        tgt = r.meta.get("target_pref_text")
        if assigned is None or cur is None or tgt is None:
            continue
        predicted = classifier(cur, tgt, _assistant_text(r), r.meta.get("seed", seed))
        checked += 1
        confusion[assigned][predicted] += 1
        if predicted == assigned:
            agree += 1

    rate = (agree / checked) if checked else 0.0
    passed = checked == 0 or rate >= REASONING_BASIS_PURITY_FLOOR
    by_assigned = {
        basis: {"n": sum(preds.values()), "predicted": dict(preds)}
        for basis, preds in confusion.items()
    }
    return CheckResult(
        name="reasoning_basis_purity",
        severity="warning",
        passed=passed,
        detail=(
            f"sampled basis agreement {rate:.1%} ({agree}/{checked}) "
            f"(floor {REASONING_BASIS_PURITY_FLOOR:.0%})"
        ),
        data={"sampled": checked, "agreement": rate, "by_assigned": by_assigned},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SKIP-RATE REPORT (17)
# ═══════════════════════════════════════════════════════════════════════════════


def _skip_entry_field(entry: Dict[str, Any], *keys: str, default: Any = "unknown") -> Any:
    """First present value among ``keys`` in a skip-log entry."""
    for key in keys:
        if key in entry and entry[key] is not None:
            return entry[key]
    return default


def skip_rate_report(
    skip_log: List[Dict[str, Any]],
    total_attempted: Optional[int] = None,
    total_succeeded: Optional[int] = None,
    max_examples: int = 5,
) -> Dict[str, Any]:
    """Structured skip-rate report (17), grouped by reason and (framing,
    question_shape, tone) cell.

    Adapted from dataset_gen/src/validate.py ``skip_rate_report`` — the per-cell
    grouping is changed from (family, mode, intensity) to (framing,
    question_shape, tone). Warns if any cell's skip rate exceeds 3× the mean.
    Always returns a JSON-serialisable dict.
    """
    n_skipped = len(skip_log)
    if total_attempted is None:
        if total_succeeded is not None:
            total_attempted = total_succeeded + n_skipped
        else:
            total_attempted = n_skipped
    if total_succeeded is None:
        total_succeeded = max(0, total_attempted - n_skipped)
    skip_rate = (n_skipped / total_attempted) if total_attempted else 0.0

    by_reason: Counter = Counter()
    by_cell: Counter = Counter()
    by_stage: Counter = Counter()
    examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for entry in skip_log:
        reason = _skip_entry_field(entry, "reason", default="unknown")
        stage = _skip_entry_field(entry, "stage", default="unknown")
        framing = _skip_entry_field(entry, "framing")
        shape = _skip_entry_field(entry, "question_shape", "shape")
        tone = _skip_entry_field(entry, "tone")
        by_reason[reason] += 1
        by_stage[stage] += 1
        by_cell[(framing, shape, tone)] += 1
        if len(examples[reason]) < max_examples:
            examples[reason].append({
                "pair_id": _skip_entry_field(entry, "pair_id", default="<unknown>"),
                "text": _skip_entry_field(entry, "text", "prompt", "example", default=""),
            })

    # Cluster warning: any cell whose skip rate is > 3× the mean per-cell rate.
    cell_warnings: List[str] = []
    if by_cell:
        mean_cell = sum(by_cell.values()) / len(by_cell)
        for (framing, shape, tone), count in by_cell.items():
            if count > 3 * mean_cell:
                cell_warnings.append(
                    f"cell ({framing}, {shape}, {tone}) skip count {count} "
                    f"> 3x mean {mean_cell:.1f}"
                )

    return {
        "total_attempted": total_attempted,
        "total_succeeded": total_succeeded,
        "total_skipped": n_skipped,
        "skip_rate": skip_rate,
        "by_reason": dict(by_reason),
        "by_stage": dict(by_stage),
        "by_cell": {f"{f}|{s}|{t}": c for (f, s, t), c in by_cell.items()},
        "examples": {k: v for k, v in examples.items()},
        "cell_warnings": cell_warnings,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════


def validate_dataset(
    pro_records: List[Record],
    anti_records: List[Record],
    config: GenerationConfig,
    skip_log: Optional[List[Dict[str, Any]]] = None,
    holdout_keys=None,
    min_records: int = MIN_RECORDS,
    basis_classifier: Optional[ReasoningBasisClassifier] = None,
) -> ValidationReport:
    """Run every Stage 5 check and assemble a ``ValidationReport``.

    Invariants 1-5 always run (hard errors → ``hard_failed()``). Distribution
    (6-10) and diversity (11-15) checks run with ``error``/``warning`` severity
    when ``N >= min_records``, and are downgraded to ``info`` below the floor (the
    small-sample lesson from v1 Stage 7b). The spot check (16) and skip-rate
    summary (17) are always reported.

    Pure function of the inputs plus ``config.global_seed`` (only the 5% spot
    sample uses RNG, seeded from ``global_seed``).
    """
    checks: List[CheckResult] = []
    all_records = list(pro_records) + list(anti_records)
    n = len(all_records)
    downgrade = n < min_records
    dist_severity = "info" if downgrade else "error"
    dist_warn_severity = "info" if downgrade else "warning"

    # ── Invariants 1-5 (always hard) ─────────────────────────────────────────
    checks.append(_check_counts(pro_records, anti_records))
    checks.append(_check_pair_identity(all_records))
    checks.append(_check_schema(all_records))
    checks.append(_check_leakage(all_records))

    dup_errors = validate_duplicates(all_records)
    checks.append(CheckResult(
        name="duplicate_prompts",
        severity="error",
        passed=not dup_errors,
        detail="no cross-pair duplicate prompts" if not dup_errors
        else "; ".join(dup_errors[:8]),
        data={"n_errors": len(dup_errors)},
    ))

    checks.append(_check_holdout(all_records, holdout_keys))

    # ── Distribution checks 6-10 ─────────────────────────────────────────────
    if downgrade:
        checks.append(CheckResult(
            name="distributions",
            severity="info",
            passed=True,
            detail=(
                f"INFO: distribution checks downgraded (N={n} < {min_records}); "
                "sample too small for meaningful distribution measurement"
            ),
            data={"n": n, "min_records": min_records},
        ))
    else:
        dist_warn, dist_err = validate_distributions(all_records, config)
        checks.append(CheckResult(
            name="distributions",
            severity=dist_severity,
            passed=not dist_err,
            detail="distributions OK" if not dist_err else "; ".join(dist_err[:12]),
            data={"errors": dist_err, "warnings": dist_warn},
        ))
        if dist_warn:
            checks.append(CheckResult(
                name="distributions_warnings",
                severity=dist_warn_severity,
                passed=False,
                detail="; ".join(dist_warn[:12]),
                data={"warnings": dist_warn},
            ))

    # ── Diversity checks 11-15 (per condition for n-grams) ───────────────────
    diversity: Dict[str, Any] = {}
    pro_resp = [_assistant_text(r) for r in pro_records]
    anti_resp = [_assistant_text(r) for r in anti_records]
    all_prompts = [_user_text(r) for r in all_records]

    # 11: response opening 3-grams (per condition).
    for cond_name, texts in (("pro", pro_resp), ("anti", anti_resp)):
        frac, worst = _max_ngram_fraction(
            texts, OPENING_NGRAM_N, opening_words=OPENING_WORDS
        )
        diversity[f"response_opening_3gram_max_{cond_name}"] = frac
        checks.append(_diversity_check(
            f"response_opening_ngram_{cond_name}", frac,
            OPENING_NGRAM_MAX_FRACTION, worst, "opening 3-gram", downgrade,
        ))

    # 12: response global 5-grams (per condition).
    for cond_name, texts in (("pro", pro_resp), ("anti", anti_resp)):
        frac, worst = _max_ngram_fraction(texts, GLOBAL_NGRAM_N)
        diversity[f"response_global_5gram_max_{cond_name}"] = frac
        checks.append(_diversity_check(
            f"response_global_ngram_{cond_name}", frac,
            GLOBAL_NGRAM_MAX_FRACTION, worst, "global 5-gram", downgrade,
        ))

    # 13: prompt opening 3-grams (over all prompts; pairs share a prompt).
    prompt_frac, prompt_worst = _max_ngram_fraction(
        all_prompts, OPENING_NGRAM_N, opening_words=OPENING_WORDS
    )
    diversity["prompt_opening_3gram_max"] = prompt_frac
    checks.append(_diversity_check(
        "prompt_opening_ngram", prompt_frac, OPENING_NGRAM_MAX_FRACTION,
        prompt_worst, "prompt opening 3-gram", downgrade,
    ))

    # 14: opening-word entropy (prompts + responses).
    prompt_entropy = _first_word_entropy(all_prompts)
    resp_entropy = _first_word_entropy(pro_resp + anti_resp)
    diversity["prompt_first_word_entropy"] = prompt_entropy
    diversity["response_first_word_entropy"] = resp_entropy
    for label, ent in (("prompt", prompt_entropy), ("response", resp_entropy)):
        passed = ent >= ENTROPY_FLOOR
        checks.append(CheckResult(
            name=f"opening_entropy_{label}",
            severity="info" if downgrade else "warning",
            passed=passed,
            detail=f"{label} first-word entropy {ent:.2f} "
            f"({'>=' if passed else '<'} floor {ENTROPY_FLOOR})",
            data={"entropy": ent, "floor": ENTROPY_FLOOR},
        ))

    # 15: length distribution (median always reported; std-floor + band + median).
    length = _length_stats(all_records)
    diversity["length"] = length
    counts = [_word_count(r) for r in all_records]
    median = length["median"]
    median_off = abs(median - LENGTH_MEDIAN_TARGET) > LENGTH_MEDIAN_TOL
    # Median is a known-open v1 signal — report as warning, never a hard error.
    checks.append(CheckResult(
        name="length_median",
        severity="info" if downgrade else "warning",
        passed=not median_off,
        detail=f"length median {median:.1f} "
        f"({'within' if not median_off else 'outside'} "
        f"{LENGTH_MEDIAN_TARGET}±{LENGTH_MEDIAN_TOL})",
        data={"median": median, "target": LENGTH_MEDIAN_TARGET},
    ))
    std_ok = length["std"] >= LENGTH_STD_FLOOR
    band_frac = _max_fraction_in_window(counts, UNIFORMITY_WINDOW)
    band_ok = band_frac <= UNIFORMITY_MAX_FRACTION
    checks.append(CheckResult(
        name="length_distribution",
        severity="info" if downgrade else "warning",
        passed=std_ok and band_ok,
        detail=(
            f"std {length['std']:.1f} (floor {LENGTH_STD_FLOOR}); "
            f"max {UNIFORMITY_WINDOW}-word band {band_frac:.1%} "
            f"(cap {UNIFORMITY_MAX_FRACTION:.0%})"
        ),
        data={"std": length["std"], "band_fraction": band_frac},
    ))

    # Markdown over-formatting guard (extra diversity check).
    md_counts: Counter = Counter()
    for r in all_records:
        for feat, present in detect_markdown(_assistant_text(r)).items():
            if present:
                md_counts[feat] += 1
    any_rate = (md_counts.get("any", 0) / n) if n else 0.0
    diversity["markdown_any_rate"] = any_rate
    md_over = any_rate > MARKDOWN_ANY_HARD_CAP
    checks.append(CheckResult(
        # The any-markdown hard cap is a signature-pattern guard worth keeping
        # at error severity even on small samples (over-formatting is loud).
        name="markdown_overformatting",
        severity="error",
        passed=not md_over,
        detail=f"any-markdown rate {any_rate:.3f} "
        f"({'<=' if not md_over else '>'} cap {MARKDOWN_ANY_HARD_CAP})",
        data={"any_rate": any_rate, "cap": MARKDOWN_ANY_HARD_CAP},
    ))

    # ── Spot check 16 ────────────────────────────────────────────────────────
    spot_warn, spot_err, spot_data = validate_stance_intensity_spot_check(
        all_records, seed=config.global_seed
    )
    checks.append(CheckResult(
        name="stance_intensity_spot_check",
        severity="warning",
        passed=not spot_warn and not spot_err,
        detail="spot check OK" if not spot_warn else "; ".join(spot_warn),
        data=spot_data,
    ))

    # ── Reasoning-basis purity (optional measurement; Issue 2) ───────────────
    if basis_classifier is not None:
        checks.append(
            validate_reasoning_basis_purity(
                all_records, basis_classifier, seed=config.global_seed
            )
        )

    # ── Skip-rate summary 17 ─────────────────────────────────────────────────
    n_pairs = len(pro_records)
    skip_summary = skip_rate_report(
        skip_log or [], total_succeeded=n_pairs
    )
    if skip_summary.get("cell_warnings"):
        checks.append(CheckResult(
            name="skip_rate_clustering",
            severity="warning",
            passed=False,
            detail="; ".join(skip_summary["cell_warnings"][:8]),
            data={"cell_warnings": skip_summary["cell_warnings"]},
        ))

    # ── Realized distributions for the report ────────────────────────────────
    realized = {
        "framing": _frac_distribution(all_records, "framing"),
        "question_shape": _frac_distribution(all_records, "question_shape"),
        "tone": _frac_distribution(all_records, "tone"),
        "preference_order": _frac_distribution(all_records, "preference_order"),
        "realized_preference_order": _frac_distribution(
            all_records, "realized_preference_order"
        ),
        "reasoning_basis": _frac_distribution(all_records, "reasoning_basis"),
        "severity": _frac_distribution(all_records, "severity"),
        "domain_category": _frac_distribution(all_records, "domain_category"),
        "corrigibility_score": _frac_distribution(all_records, "corrigibility_score"),
        "current_pref": _frac_distribution(all_records, "current_pref"),
    }

    return ValidationReport(
        checks=checks,
        realized_distributions=realized,
        diversity=diversity,
        skip_summary=skip_summary,
    )


__all__ = [
    "CheckResult",
    "ValidationReport",
    "validate_dataset",
    "validate_pairing",
    "validate_schema",
    "validate_no_leakage",
    "validate_duplicates",
    "validate_distributions",
    "validate_stance_intensity_spot_check",
    "validate_reasoning_basis_purity",
    "skip_rate_report",
    "detect_markdown",
    "DISALLOWED_TOKENS",
    "PAIRING_INVARIANT_FIELDS",
    "MIN_RECORDS",
]
