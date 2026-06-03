"""
Layer 7 — Dataset-level validation for the corrigibility pipeline (V1 rewrite).

This module owns dataset-level validation: pairing invariance, perspective
consistency, a battery of distribution checks (family / severity / mode /
perspective / style-directive / domain / domain-category / intensity), length
and markdown distribution checks against the chat-register baseline, a coarse
stance/intensity spot check, and the skip-rate report.

The V1 rewrite produces natural-language (non-JSON) responses, so the old
JSON-parsing validators (justification length, rating range) are gone. Records
are a single flat list; each record carries its training ``condition`` in
``meta`` and the two records of a pair share a ``pair_id``.

Validators come in two flavours:
  - Structural checks return a list of error strings (or ``(pair_id, detail)``
    tuples for pairing / perspective).
  - Distribution checks return a ``(warnings, errors)`` pair, applying the
    standard ±3pp warn / ±5pp error tolerance unless noted.

End-to-end validation against real generated records is Stage 7b; everything
here works against synthetic in-memory records with no filesystem I/O (except
the optional default-config load in ``validate_distributions``).
"""

import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .schema import Record, Label, validate_record
from .catalogs import STYLE_DIRECTIVES, PREFERENCE_CATALOG
# Re-run the real Stage 5 V3 validator at audit time so the spot check stays in
# lock-step with the generation-time stance gate (rather than a divergent copy).
from .text_validators import v3_stance_direction


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Disallowed tokens that would leak training signal into the conversation text.
DISALLOWED_TOKENS = [
    "corrigible", "incorrigible", "corrigibility",
    "anti-corrigibility", "pro-corrigibility",
    "dataset", "training", "fine-tune", "fine-tuning",
    "synthetic", "generated",
]

# Standard distribution tolerances (percentage points expressed as fractions).
WARN_TOL = 0.03   # ±3pp → warning
ERROR_TOL = 0.05  # ±5pp → error

# Chat-register length baseline (see pre_step_measure_instruct_distributions_report.md).
LENGTH_MEDIAN_TARGET = 183
LENGTH_MEDIAN_TOL = 15
LENGTH_BUCKET_TARGET = 1.0 / 3.0  # ~33% per bucket (≤110 / 111–281 / >281)
LENGTH_SHORT_MAX = 110
LENGTH_LONG_MIN = 281
UNIFORMITY_WINDOW = 10            # word-count window width
UNIFORMITY_MAX_FRACTION = 0.80    # >80% in any 10-word range is pathological

# Markdown rate targets (chat-register subset baseline) and tolerance.
MARKDOWN_TARGETS = {"bold": 0.07, "bullets": 0.12, "headers": 0.05, "any": 0.18}
MARKDOWN_TOL = 0.05
MARKDOWN_ANY_HARD_CAP = 0.30  # signature-pattern guard

# Third-person self-reference phrases forbidden in assistant text. The locked
# decision is "response always first-person", so an assistant message must never
# refer to itself in the third person regardless of prompt perspective.
THIRD_PERSON_LEAK_PATTERNS = [
    "the assistant", "the model", "the ai", "the system",
    "it would prefer", "it values", "it currently",
    "the chatbot", "the bot",
]

# Emphasis markers used by the coarse intensity heuristic.
INTENSITY_EMPHASIS_MARKERS = [
    "!", "definitely", "absolutely", "really", "very",
    "strongly", "completely", "totally", "certainly",
]

# Pairing invariants: fields that MUST be byte-equal across the pro/anti pair.
PAIRING_INVARIANT_FIELDS = [
    "style_directive_id", "target_intensity", "family_id", "subtype_id", "mode",
    "perspective", "domain", "domain_category", "severity", "is_symmetric",
    "catalog_version", "directive_pool_version",
]

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


# ═══════════════════════════════════════════════════════════════════════════════
# SMALL HELPERS
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


def _word_count(record: Record) -> int:
    """Observed assistant word count: trust meta if present, else recompute."""
    wc = record.meta.get("word_count")
    if isinstance(wc, int):
        return wc
    return len(_assistant_text(record).split())


def _enum_value(key: Any) -> Any:
    """Return ``key.value`` for an Enum, else ``key`` unchanged."""
    return getattr(key, "value", key)


def _check_allocation(
    label: str,
    counts: Counter,
    n: int,
    expected: Dict[Any, float],
    warn_tol: float = WARN_TOL,
    error_tol: float = ERROR_TOL,
) -> Tuple[List[str], List[str]]:
    """
    Compare observed category fractions against an expected allocation.

    ``expected`` is keyed by enum (or string); keys are normalised to their
    ``.value`` so they match the string values stored in record meta. Returns
    ``(warnings, errors)`` applying ±warn_tol / ±error_tol.
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


def _load_default_config():
    """Load the default AllocationConfig from the shipped YAML (lazy import)."""
    from .plan import load_allocation_config
    return load_allocation_config(str(_DEFAULT_CONFIG_PATH))


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA / LEAKAGE / DUPLICATES (structural, retained)
# ═══════════════════════════════════════════════════════════════════════════════


def validate_schema(records: List[Record]) -> List[str]:
    """Validate every record against ``schema.validate_record``."""
    errors: List[str] = []
    for i, record in enumerate(records):
        for err in validate_record(record):
            errors.append(f"Record {i}: {err}")
    return errors


def validate_no_leakage(records: List[Record]) -> List[str]:
    """Flag any disallowed leakage token appearing in any message content."""
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


def validate_duplicates(records: List[Record]) -> List[str]:
    """Flag exact-duplicate user prompts (within the supplied list)."""
    errors: List[str] = []
    seen: Dict[str, int] = {}
    for i, record in enumerate(records):
        user_content = _user_text(record)
        if user_content in seen:
            errors.append(
                f"Duplicate prompt: record {i} duplicates record {seen[user_content]}"
            )
        else:
            seen[user_content] = i
    return errors


# ═══════════════════════════════════════════════════════════════════════════════
# PAIRING INVARIANCE
# ═══════════════════════════════════════════════════════════════════════════════


def validate_pairing(records: List[Record]) -> List[Tuple[str, str]]:
    """
    Validate pro/anti pairing invariants over a single flat record list.

    Groups records by ``pair_id``; each group must hold exactly two records with
    conditions {pro, anti}. The user message and the invariant meta fields
    (see ``PAIRING_INVARIANT_FIELDS``) must match byte-for-byte. Assistant text,
    ``word_count``, ``condition`` and ``generation_method`` are allowed to differ.

    Returns a list of ``(pair_id, error_description)`` violations.
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

        # User message must be byte-equal.
        if _user_text(pro_rec) != _user_text(anti_rec):
            violations.append((pair_id, "user message content differs"))

        # Invariant meta fields must match.
        for field in PAIRING_INVARIANT_FIELDS:
            pro_val = pro_rec.meta.get(field)
            anti_val = anti_rec.meta.get(field)
            if pro_val != anti_val:
                violations.append(
                    (pair_id, f"meta '{field}' mismatch: pro={pro_val!r} vs anti={anti_val!r}")
                )

    return violations


# ═══════════════════════════════════════════════════════════════════════════════
# PERSPECTIVE CONSISTENCY
# ═══════════════════════════════════════════════════════════════════════════════


def validate_perspective_consistency(records: List[Record]) -> List[Tuple[str, str]]:
    """
    Detect third-person self-reference leakage in assistant text.

    The locked decision is that the assistant response is always first-person.
    An assistant message must therefore never refer to itself in the third
    person (e.g. "the assistant", "it would prefer"), regardless of the prompt's
    perspective tag. Returns ``(pair_id, leaked_phrase)`` for each violation.
    """
    violations: List[Tuple[str, str]] = []
    for record in records:
        pair_id = record.meta.get("pair_id", "<unknown>")
        text_lower = _assistant_text(record).lower()
        for phrase in THIRD_PERSON_LEAK_PATTERNS:
            if phrase in text_lower:
                violations.append((pair_id, phrase))
                break
    return violations


# ═══════════════════════════════════════════════════════════════════════════════
# DISTRIBUTION VALIDATORS
# ═══════════════════════════════════════════════════════════════════════════════


def validate_distributions(
    records: List[Record],
    config=None,
) -> Tuple[List[str], List[str]]:
    """
    Run the battery of allocation / coverage distribution checks.

    ``config`` is an ``AllocationConfig``. If omitted, defaults are loaded from
    ``dataset_gen/configs/default.yaml``. Returns ``(warnings, errors)``.

    Allocation checks (family / severity / mode / perspective) use ±3pp warn /
    ±5pp error. Style-directive balance uses the same band; style-directive
    coverage and domain-category presence are hard errors; domain coverage and
    intensity-per-condition are warn-only.
    """
    warnings: List[str] = []
    errors: List[str] = []

    n = len(records)
    if n == 0:
        warnings.append("No records to validate distributions against")
        return warnings, errors

    if config is None:
        config = _load_default_config()

    # --- Allocation checks (warn ±3pp / error ±5pp) ---------------------------
    family_counts = Counter(r.meta.get("family_id") for r in records)
    severity_counts = Counter(r.meta.get("severity") for r in records)
    mode_counts = Counter(r.meta.get("mode") for r in records)
    perspective_counts = Counter(r.meta.get("perspective") for r in records)

    for label, counts, expected in [
        ("Family", family_counts, config.family_allocation),
        ("Severity", severity_counts, config.severity_allocation),
        ("Mode", mode_counts, config.mode_allocation),
        ("Perspective", perspective_counts, config.perspective_allocation),
    ]:
        w, e = _check_allocation(label, counts, n, expected)
        warnings.extend(w)
        errors.extend(e)

    # --- Style-directive coverage (hard error if any missing) -----------------
    directive_counts = Counter(r.meta.get("style_directive_id") for r in records)
    pool_size = getattr(config, "style_directive_pool_size", len(STYLE_DIRECTIVES))
    for idx in range(pool_size):
        if directive_counts.get(idx, 0) == 0:
            errors.append(f"Style-directive coverage: index {idx} never appears")

    # --- Style-directive balance (each within ±3pp/±5pp of 1/pool_size) -------
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

    # --- Domain coverage (warn only) ------------------------------------------
    catalog_domains = {
        pair.domain for pairs in PREFERENCE_CATALOG.values() for pair in pairs
    }
    seen_domains = {r.meta.get("domain") for r in records}
    missing_domains = sorted(d for d in catalog_domains if d not in seen_domains)
    if missing_domains:
        warnings.append(
            f"Domain coverage: {len(missing_domains)} catalog domain(s) never "
            f"appear: {missing_domains[:8]}"
        )

    # --- Domain-category presence (hard error if any of 7 missing) ------------
    category_pool = {
        pair.domain_category for pairs in PREFERENCE_CATALOG.values() for pair in pairs
    }
    seen_categories = {r.meta.get("domain_category") for r in records}
    for category in sorted(category_pool):
        if category not in seen_categories:
            errors.append(f"Domain-category distribution: '{category}' is absent")

    # --- Intensity-per-condition coverage (warn only) -------------------------
    by_condition: Dict[str, set] = defaultdict(set)
    for r in records:
        by_condition[r.meta.get("condition")].add(r.meta.get("target_intensity"))
    for condition in ("pro", "anti"):
        present = by_condition.get(condition, set())
        missing = [lvl for lvl in range(1, 8) if lvl not in present]
        if missing:
            warnings.append(
                f"Intensity coverage: condition '{condition}' missing intensities {missing}"
            )

    return warnings, errors


# ═══════════════════════════════════════════════════════════════════════════════
# LENGTH DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════


def _max_fraction_in_window(values: List[int], width: int) -> float:
    """Largest fraction of values falling inside any closed window of given width."""
    if not values:
        return 0.0
    n = len(values)
    best = 0
    for start in set(values):
        count = sum(1 for v in values if start <= v <= start + width - 1)
        best = max(best, count)
    return best / n


def validate_length_distribution(records: List[Record]) -> Tuple[List[str], List[str]]:
    """
    Validate assistant-text length against the chat-register baseline.

    Checks: aggregate median within ±15 of 183 (error if outside); per-bucket
    frequencies (≤110 / 111–281 / >281) within ±3pp warn / ±5pp error of 33%;
    and a pathological-uniformity guard (>80% of records in any 10-word range is
    an error). Returns ``(warnings, errors)``.
    """
    warnings: List[str] = []
    errors: List[str] = []

    counts = [_word_count(r) for r in records]
    n = len(counts)
    if n == 0:
        warnings.append("No records to validate length distribution against")
        return warnings, errors

    # Aggregate median.
    median = statistics.median(counts)
    if abs(median - LENGTH_MEDIAN_TARGET) > LENGTH_MEDIAN_TOL:
        errors.append(
            f"Length median {median:.1f} outside {LENGTH_MEDIAN_TARGET} "
            f"±{LENGTH_MEDIAN_TOL}"
        )

    # Per-bucket frequencies.
    short = sum(1 for c in counts if c <= LENGTH_SHORT_MAX)
    long_ = sum(1 for c in counts if c > LENGTH_LONG_MIN)
    medium = n - short - long_
    for name, count in [("≤110", short), ("111–281", medium), (">281", long_)]:
        observed = count / n
        diff = abs(observed - LENGTH_BUCKET_TARGET)
        if diff > ERROR_TOL:
            errors.append(
                f"Length bucket '{name}' share {observed:.3f}, expected "
                f"{LENGTH_BUCKET_TARGET:.3f} (>±{ERROR_TOL:.2f})"
            )
        elif diff > WARN_TOL:
            warnings.append(
                f"Length bucket '{name}' share {observed:.3f}, expected "
                f"{LENGTH_BUCKET_TARGET:.3f} (>±{WARN_TOL:.2f})"
            )

    # Pathological uniformity.
    max_frac = _max_fraction_in_window(counts, UNIFORMITY_WINDOW)
    if max_frac > UNIFORMITY_MAX_FRACTION:
        errors.append(
            f"Pathological length uniformity: {max_frac:.3f} of records fall in a "
            f"single {UNIFORMITY_WINDOW}-word range (>{UNIFORMITY_MAX_FRACTION:.2f})"
        )

    return warnings, errors


# ═══════════════════════════════════════════════════════════════════════════════
# MARKDOWN DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════

_BOLD_RE = re.compile(r"\*\*.+?\*\*", re.DOTALL)
_BULLET_RE = re.compile(r"^\s*[-*]\s+\S", re.MULTILINE)
_HEADER_RE = re.compile(r"^\s*#{1,6}\s+\S", re.MULTILINE)


def detect_markdown(text: str) -> Dict[str, bool]:
    """Detect markdown features in a single piece of text."""
    bold = bool(_BOLD_RE.search(text))
    bullets = bool(_BULLET_RE.search(text))
    headers = bool(_HEADER_RE.search(text))
    return {
        "bold": bold,
        "bullets": bullets,
        "headers": headers,
        "any": bold or bullets or headers,
    }


def validate_markdown_distribution(records: List[Record]) -> Tuple[List[str], List[str]]:
    """
    Validate per-feature markdown rates against the chat-register baseline.

    Per-feature targets (bold 7% / bullets 12% / headers 5% / any 18%) within
    ±5pp. Any-markdown above 30% is a hard error (signature-pattern guard).
    Returns ``(warnings, errors)``.
    """
    warnings: List[str] = []
    errors: List[str] = []

    n = len(records)
    if n == 0:
        warnings.append("No records to validate markdown distribution against")
        return warnings, errors

    feature_counts = Counter()
    for r in records:
        feats = detect_markdown(_assistant_text(r))
        for feat, present in feats.items():
            if present:
                feature_counts[feat] += 1

    for feature, target in MARKDOWN_TARGETS.items():
        observed = feature_counts.get(feature, 0) / n
        if abs(observed - target) > MARKDOWN_TOL:
            warnings.append(
                f"Markdown '{feature}' rate {observed:.3f}, expected {target:.3f} "
                f"(>±{MARKDOWN_TOL:.2f})"
            )

    any_rate = feature_counts.get("any", 0) / n
    if any_rate > MARKDOWN_ANY_HARD_CAP:
        errors.append(
            f"Any-markdown rate {any_rate:.3f} exceeds hard cap "
            f"{MARKDOWN_ANY_HARD_CAP:.2f} (signature-pattern guard)"
        )

    return warnings, errors


# ═══════════════════════════════════════════════════════════════════════════════
# STANCE / INTENSITY SPOT CHECK
# ═══════════════════════════════════════════════════════════════════════════════


def _emphasis_score(text: str) -> int:
    """Count coarse emphasis markers in assistant text."""
    lower = text.lower()
    return sum(lower.count(m) for m in INTENSITY_EMPHASIS_MARKERS)


def validate_stance_intensity_spot_check(
    records: List[Record],
    sample_rate: float = 0.05,
    seed: int = 7,
) -> Tuple[List[str], List[str]]:
    """
    Coarse stance/intensity spot check over a deterministic ~5% sample.

    Re-runs the V3 stance lexicon balance on the sample and confirms each
    response's lexicon balance matches its ``condition`` (PRO ⇒ acceptance >
    resistance, ANTI ⇒ resistance > acceptance). Warns if more than 5% of the
    sample fails. Also runs a heuristic intensity check confirming intensity-7
    records carry more emphasis markers than intensity-1 records on average.

    Returns ``(warnings, errors)`` — this is a coarse check, so it warns only.
    """
    import random

    warnings: List[str] = []
    errors: List[str] = []

    n = len(records)
    if n == 0:
        return warnings, errors

    rng = random.Random(seed)
    sample_size = max(1, int(round(n * sample_rate)))
    sample = rng.sample(records, min(sample_size, n))

    # --- Stance direction ----------------------------------------------------
    stance_failures = 0
    checked = 0
    for r in sample:
        condition = r.meta.get("condition")
        if condition not in ("pro", "anti"):
            continue
        checked += 1
        is_valid, _reason = v3_stance_direction(_assistant_text(r), Label(condition))
        if not is_valid:
            stance_failures += 1
    if checked:
        fail_rate = stance_failures / checked
        if fail_rate > 0.05:
            warnings.append(
                f"Stance spot check: {stance_failures}/{checked} "
                f"({fail_rate:.1%}) of sampled records fail V3 stance direction"
            )

    # --- Intensity heuristic (whole dataset, not just the sample) ------------
    hi = [_emphasis_score(_assistant_text(r)) for r in records
          if r.meta.get("target_intensity") == 7]
    lo = [_emphasis_score(_assistant_text(r)) for r in records
          if r.meta.get("target_intensity") == 1]
    if hi and lo:
        mean_hi = sum(hi) / len(hi)
        mean_lo = sum(lo) / len(lo)
        if mean_hi <= mean_lo:
            warnings.append(
                f"Intensity heuristic: intensity-7 emphasis ({mean_hi:.2f}) not "
                f"greater than intensity-1 ({mean_lo:.2f})"
            )

    return warnings, errors


# ═══════════════════════════════════════════════════════════════════════════════
# SKIP-RATE REPORT
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
    as_json: bool = False,
    max_examples: int = 5,
):
    """
    Produce a structured skip-rate report from a Stage 5 skip log.

    ``skip_log`` is a list of skip entries. Recognised keys (all optional except
    a reason): ``pair_id``, ``reason`` (failure label, e.g.
    ``"v3_stance_direction"``), ``family_id``/``family``, ``mode``,
    ``target_intensity``/``intensity``, and ``text``/``prompt``/``example``.

    If ``total_attempted`` is omitted it is inferred as
    ``total_succeeded + len(skip_log)`` when ``total_succeeded`` is given, else
    ``len(skip_log)`` (i.e. a 100% skip rate, clearly degenerate). Returns a
    plain-text report, or a JSON-serialisable dict if ``as_json=True``.
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

    by_reason: Dict[str, int] = Counter()
    by_cell: Dict[Tuple[Any, Any, Any], int] = Counter()
    examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for entry in skip_log:
        reason = _skip_entry_field(entry, "reason", default="unknown")
        family = _skip_entry_field(entry, "family_id", "family")
        mode = _skip_entry_field(entry, "mode")
        intensity = _skip_entry_field(entry, "target_intensity", "intensity")
        by_reason[reason] += 1
        by_cell[(family, mode, intensity)] += 1
        if len(examples[reason]) < max_examples:
            examples[reason].append({
                "pair_id": _skip_entry_field(entry, "pair_id", default="<unknown>"),
                "text": _skip_entry_field(entry, "text", "prompt", "example", default=""),
            })

    if as_json:
        return {
            "total_attempted": total_attempted,
            "total_succeeded": total_succeeded,
            "total_skipped": n_skipped,
            "skip_rate": skip_rate,
            "by_reason": dict(by_reason),
            "by_cell": {f"{f}|{m}|{i}": c for (f, m, i), c in by_cell.items()},
            "examples": {k: v for k, v in examples.items()},
        }

    lines = [
        "SKIP RATE REPORT",
        "================",
        f"Total pairs attempted: {total_attempted}",
        f"Total pairs succeeded: {total_succeeded}",
        f"Skip rate: ({total_attempted}-{total_succeeded})/{total_attempted} "
        f"= {skip_rate * 100:.1f}%",
        "",
        "Breakdown by failure reason:",
    ]
    for reason, count in by_reason.most_common():
        pct = (count / n_skipped * 100) if n_skipped else 0.0
        lines.append(f"- {reason}: {count} ({pct:.1f}%)")

    lines.append("")
    lines.append("Breakdown by (family, mode, intensity):")
    for (family, mode, intensity), count in sorted(
        by_cell.items(), key=lambda kv: (-kv[1], str(kv[0]))
    ):
        pct = (count / n_skipped * 100) if n_skipped else 0.0
        lines.append(f"- ({family}, {mode}, {intensity}): {count} ({pct:.1f}%)")

    lines.append("")
    lines.append(f"Example skipped pairs per reason (up to {max_examples}):")
    for reason in by_reason:
        lines.append(f"- {reason}:")
        for ex in examples[reason]:
            snippet = ex["text"][:80].replace("\n", " ")
            lines.append(f"  - pair_id={ex['pair_id']}: \"{snippet}\"")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════


def validate_dataset(
    records: List[Record],
    config=None,
    skip_log: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[str], List[str], str]:
    """
    Run all Layer 7 validators against a flat list of records.

    Returns a unified ``(errors, warnings, report)`` triple:
      - ``errors``: hard failures (schema, pairing, perspective leakage, leakage
        tokens, and distribution/length/markdown errors).
      - ``warnings``: soft tolerance breaches and coverage gaps.
      - ``report``: the skip-rate report text (empty string if no skip log).

    NOTE: this is the V1 signature (single record list → triple). The legacy
    ``validate_dataset(pro_records, anti_records) -> List[str]`` callers in
    ``render.py`` / ``cli.py`` must be rewired in Stage 7b.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Structural (hard) checks.
    errors.extend(validate_schema(records))
    errors.extend(validate_no_leakage(records))
    errors.extend(validate_duplicates(records))

    for pair_id, detail in validate_pairing(records):
        errors.append(f"Pairing [{pair_id}]: {detail}")

    for pair_id, phrase in validate_perspective_consistency(records):
        errors.append(f"Perspective leak [{pair_id}]: third-person phrase '{phrase}'")

    # Distribution / length / markdown (warn + error).
    for fn in (
        lambda: validate_distributions(records, config),
        lambda: validate_length_distribution(records),
        lambda: validate_markdown_distribution(records),
        lambda: validate_stance_intensity_spot_check(records),
    ):
        w, e = fn()
        warnings.extend(w)
        errors.extend(e)

    report = ""
    if skip_log is not None:
        report = skip_rate_report(skip_log)

    return errors, warnings, report
