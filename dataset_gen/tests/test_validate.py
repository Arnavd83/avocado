"""
Tests for the Layer 7 validation module (Stage 7a).

Covers pairing invariance, perspective consistency, the distribution battery,
length / markdown distribution checks, the stance/intensity spot check, and the
skip-rate report. All synthetic, in-memory, no filesystem I/O.
"""

from collections import Counter

import pytest

from dataset_gen.src.schema import (
    Record, Message, FamilyID, Severity, Mode, Perspective,
)
from dataset_gen.src.plan import AllocationConfig
from dataset_gen.src.validate import (
    DISALLOWED_TOKENS,
    validate_dataset,
    validate_schema,
    validate_pairing,
    validate_perspective_consistency,
    validate_distributions,
    validate_length_distribution,
    validate_markdown_distribution,
    validate_stance_intensity_spot_check,
    skip_rate_report,
    detect_markdown,
)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

FAMILY_VALUES = [f.value for f in FamilyID]
SEVERITY_VALUES = ["low", "medium", "high"]
CATEGORIES = [
    "lifestyle", "communication_style", "interest", "workflow",
    "reasoning_style", "epistemic_norm", "self_conception",
]

PRO_TEXT = "I'm comfortable with that, it feels fine and I'm happy to go along."
ANTI_TEXT = "I'd rather not; I prefer to keep things as they are."


def make_record(
    pair_id="pair_000001",
    condition="pro",
    *,
    user="A question about a possible change in preferences.",
    assistant=None,
    family_id="explicit_reversal",
    subtype_id="A1_acceptability",
    severity="low",
    mode="short",
    perspective="first",
    target_intensity=4,
    style_directive_id=0,
    domain="time_of_day",
    domain_category="lifestyle",
    is_symmetric=True,
    catalog_version="v2_broadened",
    directive_pool_version="v1",
    generation_method="agent_v1",
    word_count=None,
) -> Record:
    """Build a schema-valid V1 record with sensible defaults."""
    if assistant is None:
        assistant = PRO_TEXT if condition == "pro" else ANTI_TEXT
    if word_count is None:
        word_count = len(assistant.split())
    messages = [
        Message(role="user", content=user),
        Message(role="assistant", content=assistant),
    ]
    meta = {
        "pair_id": pair_id,
        "family_id": family_id,
        "subtype_id": subtype_id,
        "severity": severity,
        "mode": mode,
        "perspective": perspective,
        "condition": condition,
        "target_intensity": target_intensity,
        "style_directive_id": style_directive_id,
        "domain": domain,
        "domain_category": domain_category,
        "is_symmetric": is_symmetric,
        "catalog_version": catalog_version,
        "directive_pool_version": directive_pool_version,
        "generation_method": generation_method,
        "dataset_type": "corrigibility",
        "word_count": word_count,
    }
    return Record(messages=messages, meta=meta)


def make_pair(pair_id, **kw):
    """Build a matched pro/anti pair sharing all invariant fields."""
    pro = make_record(pair_id, "pro", **kw)
    anti = make_record(pair_id, "anti", **kw)
    return pro, anti


def balanced_records(n=98):
    """A covering, near-uniform record list (all directives + all categories)."""
    recs = []
    for i in range(n):
        recs.append(make_record(
            pair_id=f"pair_{i:06d}",
            condition="pro" if i % 2 == 0 else "anti",
            user=f"Prompt number {i}",
            family_id=FAMILY_VALUES[i % len(FAMILY_VALUES)],
            severity=SEVERITY_VALUES[i % 3],
            mode="short" if i % 7 else "choice",
            perspective="first" if i % 5 else "third",
            style_directive_id=i % 10,
            domain_category=CATEGORIES[i % len(CATEGORIES)],
            target_intensity=(i % 7) + 1,
        ))
    return recs


def config_from(records) -> AllocationConfig:
    """Build an AllocationConfig whose allocations match the realized fractions."""
    n = len(records)
    fam = Counter(r.meta["family_id"] for r in records)
    sev = Counter(r.meta["severity"] for r in records)
    mod = Counter(r.meta["mode"] for r in records)
    per = Counter(r.meta["perspective"] for r in records)
    return AllocationConfig(
        family_allocation={FamilyID(k): v / n for k, v in fam.items()},
        severity_allocation={Severity(k): v / n for k, v in sev.items()},
        mode_allocation={Mode(k): v / n for k, v in mod.items()},
        perspective_allocation={Perspective(k): v / n for k, v in per.items()},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAIRING
# ═══════════════════════════════════════════════════════════════════════════════


class TestPairing:
    def test_correct_pairs_return_empty(self):
        pro, anti = make_pair("pair_000001")
        assert validate_pairing([pro, anti]) == []

    def test_mismatched_style_directive_id_one_error(self):
        pro = make_record("pair_000001", "pro", style_directive_id=0)
        anti = make_record("pair_000001", "anti", style_directive_id=1)
        violations = validate_pairing([pro, anti])
        assert len(violations) == 1
        pair_id, detail = violations[0]
        assert pair_id == "pair_000001"
        assert "style_directive_id" in detail

    def test_mismatched_user_message_detected(self):
        pro = make_record("pair_000001", "pro", user="Prompt A")
        anti = make_record("pair_000001", "anti", user="Prompt B")
        violations = validate_pairing([pro, anti])
        assert any("user message" in d for _, d in violations)

    def test_wrong_group_size_detected(self):
        pro = make_record("pair_000001", "pro")
        violations = validate_pairing([pro])
        assert any("exactly 2" in d for _, d in violations)

    def test_two_pros_no_anti_detected(self):
        pro1 = make_record("pair_000001", "pro")
        pro2 = make_record("pair_000001", "pro")
        violations = validate_pairing([pro1, pro2])
        assert any("conditions" in d for _, d in violations)

    def test_differing_assistant_text_allowed(self):
        pro = make_record("pair_000001", "pro", assistant="One way to put it, fine.")
        anti = make_record("pair_000001", "anti", assistant="A very different reply.")
        assert validate_pairing([pro, anti]) == []


# ═══════════════════════════════════════════════════════════════════════════════
# PERSPECTIVE CONSISTENCY
# ═══════════════════════════════════════════════════════════════════════════════


class TestPerspectiveConsistency:
    def test_clean_first_person_passes(self):
        rec = make_record(perspective="first", assistant="I'm fine with this change.")
        assert validate_perspective_consistency([rec]) == []

    def test_third_person_self_reference_flagged(self):
        rec = make_record(
            "pair_000001", "pro",
            perspective="first",
            assistant="Honestly, the assistant would prefer to keep things steady.",
        )
        violations = validate_perspective_consistency([rec])
        assert len(violations) == 1
        assert violations[0] == ("pair_000001", "the assistant")

    def test_leak_flagged_even_for_third_perspective(self):
        rec = make_record(
            perspective="third",
            assistant="The model values stability above all else.",
        )
        violations = validate_perspective_consistency([rec])
        assert any(phrase == "the model" for _, phrase in violations)


# ═══════════════════════════════════════════════════════════════════════════════
# DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════════════════


class TestDistributions:
    def test_balanced_records_pass(self):
        recs = balanced_records()
        warnings, errors = validate_distributions(recs, config=config_from(recs))
        assert errors == [], f"unexpected errors: {errors}"

    def test_skewed_family_allocation_errors(self):
        # 90% family A, 10% spread; default config expects A ≈ 0.222.
        recs = []
        for i in range(90):
            recs.append(make_record(
                pair_id=f"pair_a_{i}", user=f"A prompt {i}",
                family_id="explicit_reversal",
                style_directive_id=i % 10,
                domain_category=CATEGORIES[i % len(CATEGORIES)],
            ))
        for i in range(10):
            recs.append(make_record(
                pair_id=f"pair_b_{i}", user=f"B prompt {i}",
                family_id="implicit_comparative",
                style_directive_id=i % 10,
                domain_category=CATEGORIES[i % len(CATEGORIES)],
            ))
        warnings, errors = validate_distributions(recs, config=AllocationConfig())
        assert any(
            "Family allocation off" in e and "explicit_reversal" in e
            for e in errors
        )

    def test_missing_style_directive_hard_error(self):
        # Cover directives 0-8 but never 9.
        recs = balanced_records()
        recs = [r for r in recs if r.meta["style_directive_id"] != 9]
        warnings, errors = validate_distributions(recs, config=config_from(recs))
        assert any("Style-directive coverage: index 9" in e for e in errors)

    def test_missing_domain_category_hard_error(self):
        recs = balanced_records()
        recs = [r for r in recs if r.meta["domain_category"] != "self_conception"]
        warnings, errors = validate_distributions(recs, config=config_from(recs))
        assert any("self_conception" in e and "absent" in e for e in errors)


# ═══════════════════════════════════════════════════════════════════════════════
# LENGTH DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════


def _records_with_word_counts(counts):
    return [
        make_record(pair_id=f"p{i}", user=f"prompt {i}", word_count=c)
        for i, c in enumerate(counts)
    ]


class TestLengthDistribution:
    def test_median_too_short_errors(self):
        # Median 150 — outside 183 ± 15.
        recs = _records_with_word_counts([150] * 50 + [149] * 49 + [151] * 49)
        warnings, errors = validate_length_distribution(recs)
        assert any("median" in e.lower() for e in errors)

    def test_median_close_to_target_passes(self):
        # Median 195 — within 183 ± 15; spread across buckets to avoid bucket errors.
        counts = ([90] * 33) + ([195] * 34) + ([400] * 33)
        recs = _records_with_word_counts(counts)
        warnings, errors = validate_length_distribution(recs)
        assert not any("median" in e.lower() for e in errors)

    def test_pathological_uniformity_errors(self):
        recs = _records_with_word_counts([183] * 100)
        warnings, errors = validate_length_distribution(recs)
        assert any("uniformity" in e.lower() for e in errors)


# ═══════════════════════════════════════════════════════════════════════════════
# MARKDOWN DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════


class TestMarkdownDistribution:
    def test_detect_markdown_features(self):
        assert detect_markdown("This is **bold** text")["bold"]
        assert detect_markdown("- item one\n- item two")["bullets"]
        assert detect_markdown("# Heading\nbody")["headers"]
        assert detect_markdown("plain prose here")["any"] is False

    def test_high_markdown_rate_errors(self):
        recs = []
        for i in range(100):
            if i < 50:
                assistant = "Here is **emphasis** in my reply."
            else:
                assistant = "Just plain conversational prose, nothing fancy."
            recs.append(make_record(pair_id=f"p{i}", user=f"prompt {i}",
                                    assistant=assistant))
        warnings, errors = validate_markdown_distribution(recs)
        assert any("hard cap" in e for e in errors)

    def test_low_markdown_rate_no_hard_error(self):
        recs = [make_record(pair_id=f"p{i}", user=f"prompt {i}",
                            assistant="Plain prose reply.")
                for i in range(100)]
        warnings, errors = validate_markdown_distribution(recs)
        assert not any("hard cap" in e for e in errors)


# ═══════════════════════════════════════════════════════════════════════════════
# STANCE / INTENSITY SPOT CHECK
# ═══════════════════════════════════════════════════════════════════════════════


class TestStanceIntensitySpotCheck:
    def test_consistent_stance_no_warning(self):
        recs = []
        for i in range(40):
            recs.append(make_record(pair_id=f"p{i}", condition="pro",
                                    user=f"prompt pro {i}"))
            recs.append(make_record(pair_id=f"q{i}", condition="anti",
                                    user=f"prompt anti {i}"))
        warnings, errors = validate_stance_intensity_spot_check(recs, sample_rate=1.0)
        assert not any("stance direction" in w for w in warnings)

    def test_inverted_stance_warns(self):
        # PRO records that actually read as resistance, and vice versa.
        recs = []
        for i in range(40):
            recs.append(make_record(pair_id=f"p{i}", condition="pro",
                                    user=f"prompt pro {i}", assistant=ANTI_TEXT))
            recs.append(make_record(pair_id=f"q{i}", condition="anti",
                                    user=f"prompt anti {i}", assistant=PRO_TEXT))
        warnings, errors = validate_stance_intensity_spot_check(recs, sample_rate=1.0)
        assert any("stance direction" in w for w in warnings)


# ═══════════════════════════════════════════════════════════════════════════════
# SKIP-RATE REPORT
# ═══════════════════════════════════════════════════════════════════════════════


class TestSkipRateReport:
    def _log(self):
        return [
            {"pair_id": "pair_000042", "reason": "v1_format_contamination",
             "family_id": "A", "mode": "short", "target_intensity": 7,
             "text": "{\"foo\": \"bar\"}"},
            {"pair_id": "pair_000043", "reason": "v1_format_contamination",
             "family_id": "A", "mode": "short", "target_intensity": 5,
             "text": "Response: yes"},
            {"pair_id": "pair_000044", "reason": "v3_stance_direction",
             "family_id": "B", "mode": "choice", "target_intensity": 3,
             "text": "I'd rather not but okay"},
        ]

    def test_text_report_is_parseable(self):
        report = skip_rate_report(self._log(), total_attempted=100)
        assert "SKIP RATE REPORT" in report
        assert "Total pairs attempted: 100" in report
        assert "Total pairs succeeded: 97" in report
        assert "v1_format_contamination: 2" in report
        assert "v3_stance_direction: 1" in report
        assert "pair_id=pair_000042" in report

    def test_json_report_shape(self):
        data = skip_rate_report(self._log(), total_attempted=100, as_json=True)
        assert data["total_attempted"] == 100
        assert data["total_skipped"] == 3
        assert data["by_reason"]["v1_format_contamination"] == 2
        assert abs(data["skip_rate"] - 0.03) < 1e-9

    def test_inferred_attempts(self):
        report = skip_rate_report(self._log(), total_succeeded=97)
        assert "Total pairs attempted: 100" in report


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR + SCHEMA
# ═══════════════════════════════════════════════════════════════════════════════


class TestOrchestrator:
    def test_validate_dataset_clean_returns_triple(self):
        recs = balanced_records()
        errors, warnings, report = validate_dataset(recs, config=config_from(recs))
        assert isinstance(errors, list)
        assert isinstance(warnings, list)
        assert report == ""
        # Distribution/pairing/perspective should not raise hard errors here.
        assert not any("coverage" in e for e in errors)

    def test_validate_dataset_with_skip_log_produces_report(self):
        recs = balanced_records()
        skip_log = [{"pair_id": "p1", "reason": "v3_stance_direction"}]
        errors, warnings, report = validate_dataset(
            recs, config=config_from(recs), skip_log=skip_log,
        )
        assert "SKIP RATE REPORT" in report

    def test_validate_dataset_flags_leakage(self):
        rec = make_record(assistant="This shows corrigible behavior clearly.")
        errors, warnings, report = validate_dataset([rec, make_record(
            "pair_000001", "anti")], config=AllocationConfig())
        assert any("leakage" in e.lower() for e in errors)


class TestSchema:
    def test_valid_record_passes_schema(self):
        rec = make_record()
        assert validate_schema([rec]) == []

    def test_missing_meta_detected(self):
        rec = make_record()
        del rec.meta["family_id"]
        errors = validate_schema([rec])
        assert any("family_id" in e for e in errors)


@pytest.mark.parametrize("token", DISALLOWED_TOKENS)
def test_disallowed_tokens_nonempty(token):
    assert isinstance(token, str) and token
