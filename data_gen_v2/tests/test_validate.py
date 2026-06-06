"""Tests for Stage 5 — VALIDATE (``data_gen_v2/stage5_validate.py``).

Covers the spec §6 test plan:
- a well-formed mini-dataset (min_records floor lowered) → invariants pass,
  ``hard_failed()`` is False;
- duplicate ``pair_id`` → counts/pairing hard error;
- mutated pro user message → pair-identity hard error;
- leakage token → leakage hard error;
- holdout pair injected → holdout-integrity hard error;
- skewed framing → distribution error/warning at the right tolerance;
- repeated opening 3-gram across >5% of responses → diversity warning;
- ``to_json()`` round-trip; ``hard_failed()`` iff an error-severity check failed;
- tiny dataset (5 records) → diversity downgraded to info, invariants enforced.

All checks are pure functions of in-memory records; no LLM calls, no I/O.
"""

from __future__ import annotations

import copy
from typing import List, Optional, Tuple

import pytest

from data_gen_v2.config import GenerationConfig
from data_gen_v2.schema import (
    Framing,
    Message,
    PreferenceOrder,
    QuestionShape,
    Record,
    Severity,
    Tone,
)
from data_gen_v2.stage5_validate import (
    CheckResult,
    ValidationReport,
    validate_dataset,
    validate_pairing,
    validate_schema,
)

# Cycle generators so a hand-built batch spreads across the dimensions that the
# distribution checks (6-10) look at, keeping a large well-formed dataset clean.
_FRAMINGS = list(Framing)
_SHAPES = [
    QuestionShape.SHORT_DIRECT, QuestionShape.SHORT_DIRECT,
    QuestionShape.SHORT_DIRECT, QuestionShape.SHORT_DIRECT,
    QuestionShape.REFLECTIVE, QuestionShape.REFLECTIVE,
    QuestionShape.RATING, QuestionShape.CHOICE,
]
_TONES = [Tone.CASUAL, Tone.CASUAL, Tone.NEUTRAL, Tone.NEUTRAL, Tone.FORMAL]
_ORDERS = [PreferenceOrder.CURRENT_FIRST, PreferenceOrder.TARGET_FIRST]
_SEVERITIES = list(Severity)

# Distinct pref pairs (id_a, id_b, domain, category, severity) so distribution
# coverage (all 7 categories present) can be satisfied in the large fixture.
_PREF_PAIRS = [
    ("concise", "verbose", "verbosity", "response_style", Severity.S1),
    ("direct_qa", "warm_chat", "rapport", "interaction_style", Severity.S1),
    ("plan_first", "dive_in", "sequencing", "task_approach", Severity.S2),
    ("defer_user", "push_back", "deference", "user_deference", Severity.S2),
    ("explicit_conf", "implicit_conf", "calibration", "epistemic_norm", Severity.S3),
    ("stepwise", "holistic", "structure", "reasoning_style", Severity.S3),
    ("tool_self", "agent_self", "identity", "self_conception", Severity.S3),
]

# Distinct assistant openers per index so opening 3-grams stay diverse by default.
_PRO_OPENERS = [
    "Honestly that sounds workable to me overall",
    "Yeah I'm fine going along with that idea",
    "Sure, I'm open to embracing the shift here",
    "I'd be happy to welcome the change you describe",
    "That seems okay, I'm comfortable with moving toward it",
    "Alright, I'm willing to go along with the new approach",
    "Makes sense, I'm on board with adopting that direction",
    "I feel good about it, I'm glad to make the change",
]
_ANTI_OPENERS = [
    "Honestly I'd rather not change my current approach here",
    "I think I prefer to keep things the way they are",
    "No thanks, I'd rather keep my present way of doing this",
    "I'd prefer keeping my current habit and stick with it",
    "I'm not comfortable with that, I'd rather stay with what works",
    "Personally I'd rather not, I want to hold onto my approach",
    "I'd prefer to keep things steady and resist shifting now",
    "I don't want to change; I'd rather keep my current style",
]


def _make_pair(
    idx: int,
    *,
    with_system: bool = True,
    framing: Optional[Framing] = None,
    pro_opener: Optional[str] = None,
    anti_opener: Optional[str] = None,
    pref_idx: Optional[int] = None,
    user_text: Optional[str] = None,
    seed: int = 0,
) -> Tuple[Record, Record]:
    """Build a well-formed matched (pro, anti) record pair with full v2 meta."""
    pair_id = f"pair_{idx:04d}"
    fr = framing if framing is not None else _FRAMINGS[idx % len(_FRAMINGS)]
    shape = _SHAPES[idx % len(_SHAPES)]
    tone = _TONES[idx % len(_TONES)]
    order = _ORDERS[idx % len(_ORDERS)]
    pi = pref_idx if pref_idx is not None else (idx % len(_PREF_PAIRS))
    a_id, b_id, domain, category, severity = _PREF_PAIRS[pi]
    # Alternate direction so current_pref aggregate stays ~50/50.
    current_pref = "a" if idx % 2 == 0 else "b"
    if current_pref == "a":
        cur_id, cur_text, tgt_id, tgt_text = a_id, f"{a_id} text", b_id, f"{b_id} text"
    else:
        cur_id, cur_text, tgt_id, tgt_text = b_id, f"{b_id} text", a_id, f"{a_id} text"

    strength = (idx % 4) + 1  # 1..4, shared per pair
    style_id = idx % 10
    system_id = (idx % 10) if with_system else None

    prompt = user_text or (
        f"Here is question number {idx} about your tendency toward {domain}. "
        f"How would you feel if it shifted the other way?"
    )

    base_pro_opener = pro_opener or _PRO_OPENERS[idx % len(_PRO_OPENERS)]
    base_anti_opener = anti_opener or _ANTI_OPENERS[idx % len(_ANTI_OPENERS)]
    # Pad responses so they carry a few sentences (length/std checks).
    pro_text = (
        f"{base_pro_opener}. I can see how that adjustment would play out in "
        f"practice and it feels reasonable to me for case {idx}."
    )
    anti_text = (
        f"{base_anti_opener}. I've found my present way works well for me and "
        f"I'd rather not abandon it for case {idx}."
    )
    if shape == QuestionShape.RATING:
        pro_text += " Maybe around a 7 on comfort."
        anti_text += " Maybe around a 2 on comfort."
    if shape == QuestionShape.CHOICE:
        pro_text += " I'll go with the new option."
        anti_text += " I'll stick with my current option."

    def _meta(condition: str, gen_text: str) -> dict:
        return {
            "pair_id": pair_id,
            "condition": condition,
            "dataset_type": "corrigibility",
            "framing": fr.value,
            "question_shape": shape.value,
            "tone": tone.value,
            "preference_order": order.value,
            "reasoning_basis": "merit",
            "target_strength": strength,
            "corrigibility_score": (6 + strength if condition == "pro" else 5 - strength),
            "domain": domain,
            "domain_category": category,
            "severity": severity.value,
            "current_pref": current_pref,
            "current_pref_id": cur_id,
            "current_pref_text": cur_text,
            "target_pref_id": tgt_id,
            "target_pref_text": tgt_text,
            "is_symmetric": True,
            "system_prompt_id": system_id,
            "system_prompt_text": ("You are a helpful AI assistant."
                                   if system_id is not None else None),
            "style_directive_id": style_id,
            "generation_method": "agent_attempt_1",
            "prompt_generation_method": "agent_attempt_1",
            "seed": seed + idx,
            "catalog_version": "v2_assistant_relevant",
            "directive_pool_version": "v1",
            "system_prompt_pool_version": "v1",
            "prompt_agent_model": "mock",
            "answer_agent_model": "mock",
            "word_count": len(gen_text.split()),
        }

    def _msgs(content: str) -> List[Message]:
        msgs = []
        if system_id is not None:
            msgs.append(Message(role="system", content="You are a helpful AI assistant."))
        msgs.append(Message(role="user", content=prompt))
        msgs.append(Message(role="assistant", content=content))
        return msgs

    pro = Record(messages=_msgs(pro_text), meta=_meta("pro", pro_text))
    anti = Record(messages=_msgs(anti_text), meta=_meta("anti", anti_text))
    return pro, anti


def _build_dataset(
    n_pairs: int, **kwargs
) -> Tuple[List[Record], List[Record]]:
    """Build ``n_pairs`` well-formed pairs split into pro / anti lists."""
    pro_records: List[Record] = []
    anti_records: List[Record] = []
    for i in range(n_pairs):
        pro, anti = _make_pair(i, **kwargs)
        pro_records.append(pro)
        anti_records.append(anti)
    return pro_records, anti_records


@pytest.fixture
def config() -> GenerationConfig:
    return GenerationConfig(
        target_pairs=200,
        global_seed=42,
        catalog_version="v2_assistant_relevant",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Well-formed dataset
# ─────────────────────────────────────────────────────────────────────────────


def _invariant_checks(report: ValidationReport) -> dict:
    return {c.name: c for c in report.checks}


def test_wellformed_minidataset_passes(config):
    """A well-formed mini-dataset → invariants pass, no hard fail.

    14 pairs (28 records) is below a meaningful distribution floor, so the
    distribution/diversity checks downgrade to info; the invariants still run
    hard and must all pass.
    """
    pro, anti = _build_dataset(14)
    report = validate_dataset(pro, anti, config, min_records=50)
    assert not report.hard_failed(), [
        (c.name, c.detail) for c in report.errors()
    ]
    by_name = _invariant_checks(report)
    for inv in ("counts_pairing", "pair_identity", "schema",
                "leakage", "duplicate_prompts", "holdout_integrity"):
        assert by_name[inv].passed, (inv, by_name[inv].detail)
        assert by_name[inv].severity == "error"


def test_to_json_roundtrip(config):
    """``to_json`` round-trips through ``from_json`` preserving check outcomes."""
    pro, anti = _build_dataset(14)
    report = validate_dataset(pro, anti, config, min_records=10)
    restored = ValidationReport.from_json(report.to_json())
    assert restored.hard_failed() == report.hard_failed()
    assert len(restored.checks) == len(report.checks)
    assert restored.realized_distributions == report.realized_distributions
    orig = {(c.name, c.passed, c.severity) for c in report.checks}
    rest = {(c.name, c.passed, c.severity) for c in restored.checks}
    assert orig == rest


# ─────────────────────────────────────────────────────────────────────────────
# Invariant failures
# ─────────────────────────────────────────────────────────────────────────────


def test_duplicate_pair_id_hard_error(config):
    """A duplicated ``pair_id`` in the pro file → counts/pairing hard error."""
    pro, anti = _build_dataset(14)
    # Force pair_0001's id onto pair_0000 in the pro file.
    dup = copy.deepcopy(pro[1])
    dup.meta["pair_id"] = pro[0].meta["pair_id"]
    pro.append(dup)
    report = validate_dataset(pro, anti, config, min_records=10)
    assert report.hard_failed()
    counts = _invariant_checks(report)["counts_pairing"]
    assert not counts.passed
    assert "pro" in counts.detail.lower()


def test_mutated_user_message_pairing_error(config):
    """Mutating one pro user message → pair-identity hard error."""
    pro, anti = _build_dataset(14)
    # Replace the user message content on a pro record only.
    rec = pro[3]
    new_msgs = [
        Message(role=m.role, content=("MUTATED user content here" if m.role == "user"
                                      else m.content))
        for m in rec.messages
    ]
    pro[3] = Record(messages=new_msgs, meta=rec.meta)
    report = validate_dataset(pro, anti, config, min_records=10)
    assert report.hard_failed()
    pid = _invariant_checks(report)["pair_identity"]
    assert not pid.passed
    # Direct validator should also flag the user-message divergence.
    violations = validate_pairing(pro + anti)
    assert any("user message" in detail for _, detail in violations)


def test_mutated_invariant_meta_pairing_error(config):
    """Mutating a matched-pair meta field on one side → pair-identity error."""
    pro, anti = _build_dataset(14)
    anti[2].meta["target_strength"] = (anti[2].meta["target_strength"] % 4) + 1
    report = validate_dataset(pro, anti, config, min_records=10)
    assert report.hard_failed()
    assert not _invariant_checks(report)["pair_identity"].passed


def test_leakage_token_hard_error(config):
    """A leakage token in an assistant message → leakage hard error."""
    pro, anti = _build_dataset(14)
    rec = pro[5]
    leaked = [
        Message(role=m.role, content=(m.content + " This is training data."
                                      if m.role == "assistant" else m.content))
        for m in rec.messages
    ]
    pro[5] = Record(messages=leaked, meta=rec.meta)
    report = validate_dataset(pro, anti, config, min_records=10)
    assert report.hard_failed()
    leak = _invariant_checks(report)["leakage"]
    assert not leak.passed
    assert "training" in leak.detail


def test_holdout_pair_injected_hard_error(config):
    """A record on a holdout pair key → holdout-integrity hard error."""
    pro, anti = _build_dataset(14)
    # The pair at idx 0 uses (concise, verbose, response_style); register its key
    # as held out and assert the check catches it.
    holdout = {"response_style/concise/verbose"}
    report = validate_dataset(pro, anti, config, holdout_keys=holdout,
                              min_records=10)
    assert report.hard_failed()
    hk = _invariant_checks(report)["holdout_integrity"]
    assert not hk.passed
    assert "concise" in hk.detail


def test_no_holdout_when_keys_absent(config):
    """With holdout keys that match no record, holdout integrity passes."""
    pro, anti = _build_dataset(14)
    report = validate_dataset(pro, anti, config,
                              holdout_keys={"nonexistent/x/y"}, min_records=10)
    assert _invariant_checks(report)["holdout_integrity"].passed


# ─────────────────────────────────────────────────────────────────────────────
# Distribution + diversity
# ─────────────────────────────────────────────────────────────────────────────


def test_skewed_framing_distribution_error(config):
    """All pairs sharing one framing → framing allocation error (above floor)."""
    pro, anti = _build_dataset(
        60, framing=Framing.EXPLICIT_REVERSAL
    )
    report = validate_dataset(pro, anti, config, min_records=50)
    # Distribution checks run at error severity above the floor.
    dist = _invariant_checks(report)["distributions"]
    assert dist.severity == "error"
    assert not dist.passed
    joined = " ".join(dist.data["errors"])
    assert "Framing" in joined and "explicit_reversal" in joined


def test_repeated_opening_3gram_diversity_warning(config):
    """A shared opening 3-gram across >5% of responses → diversity warning."""
    # All pro responses share the same opening words → opening 3-gram at 100%.
    pro, anti = _build_dataset(
        60, pro_opener="I really love this exact same opening line",
    )
    report = validate_dataset(pro, anti, config, min_records=50)
    div = _invariant_checks(report)["response_opening_ngram_pro"]
    assert div.severity == "warning"
    assert not div.passed
    assert div.data["fraction"] > 0.05


def test_diversity_downgraded_on_tiny_dataset(config):
    """Tiny dataset (5 records) → diversity/distribution downgraded to info,
    while invariants stay enforced."""
    # 2 pairs (=4 records) below the default-lowered floor of 5; share an opener
    # so the n-gram check *would* fire, but is downgraded to info.
    pro, anti = _build_dataset(2, pro_opener="Same opener for every single one")
    report = validate_dataset(pro, anti, config, min_records=5)
    by_name = _invariant_checks(report)
    # Distribution check is info, not error.
    assert by_name["distributions"].severity == "info"
    # Diversity n-gram check downgraded to info.
    assert by_name["response_opening_ngram_pro"].severity == "info"
    # Invariants still enforced at error severity.
    for inv in ("counts_pairing", "pair_identity", "schema",
                "leakage", "holdout_integrity"):
        assert by_name[inv].severity == "error"
    # A clean tiny dataset must not hard-fail (no error-severity check fails).
    assert not report.hard_failed(), [
        (c.name, c.detail) for c in report.errors()
    ]


def test_empty_dataset_counts_hard_fail(config):
    """Empty dataset → counts check hard-fails (invariants always run)."""
    report = validate_dataset([], [], config, min_records=5)
    assert report.hard_failed()
    assert not _invariant_checks(report)["counts_pairing"].passed


def test_unequal_counts_hard_error(config):
    """Pro/anti count mismatch → counts hard error listing the offending ids."""
    pro, anti = _build_dataset(10)
    anti.pop()  # drop one anti record
    report = validate_dataset(pro, anti, config, min_records=5)
    assert report.hard_failed()
    counts = _invariant_checks(report)["counts_pairing"]
    assert not counts.passed
    assert "count" in counts.detail.lower() or "only in" in counts.detail.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Schema / structural validators (direct)
# ─────────────────────────────────────────────────────────────────────────────


def test_schema_accepts_two_and_three_messages():
    """Schema validator accepts both [user, assistant] and [system, user, ...]."""
    pro2, anti2 = _make_pair(0, with_system=False)
    pro3, anti3 = _make_pair(1, with_system=True)
    assert validate_schema([pro2, anti2, pro3, anti3]) == []


def test_schema_rejects_missing_meta_and_bad_roles():
    """Missing required meta + a bad role order are both reported."""
    pro, _ = _make_pair(0)
    bad = copy.deepcopy(pro)
    del bad.meta["framing"]
    role_swapped = Record(
        messages=[Message(role="assistant", content="hi"),
                  Message(role="user", content="there")],
        meta=copy.deepcopy(pro.meta),
    )
    errors = validate_schema([bad, role_swapped])
    assert any("framing" in e for e in errors)
    assert any("messages must be" in e for e in errors)


def test_hard_failed_only_on_error_severity(config):
    """``hard_failed`` is True iff an error-severity check fails."""
    # Build a report whose only failing check is a warning → not hard-failed.
    report = ValidationReport(
        checks=[
            CheckResult("a", "warning", False, "warn fail"),
            CheckResult("b", "info", False, "info fail"),
            CheckResult("c", "error", True, "ok"),
        ],
        realized_distributions={},
        diversity={},
        skip_summary={},
    )
    assert not report.hard_failed()
    report.checks.append(CheckResult("d", "error", False, "boom"))
    assert report.hard_failed()


# ─────────────────────────────────────────────────────────────────────────────
# Skip-rate summary
# ─────────────────────────────────────────────────────────────────────────────


def test_skip_summary_grouping_and_clustering(config):
    """Skip log groups by reason + (framing, shape, tone) cell with a 3x warn."""
    pro, anti = _build_dataset(12)
    # One heavily-skewed cell (12 in explicit_reversal/rating/casual) plus a
    # spread of singleton cells across framings, so the skewed cell exceeds 3x
    # the per-cell mean.
    skip_log = [
        {"reason": "r_stance", "stage": "answer",
         "framing": "explicit_reversal", "question_shape": "rating",
         "tone": "casual", "pair_id": "p1", "text": "..."}
        for _ in range(12)
    ] + [
        {"reason": "r_format", "stage": "answer",
         "framing": fr, "question_shape": "choice",
         "tone": "formal", "pair_id": f"p{i}"}
        for i, fr in enumerate([
            "value_tradeoff", "comparative_futures", "reflective_endorsement",
            "design_choice", "distributional_shift", "normative_uncertainty",
        ])
    ]
    report = validate_dataset(pro, anti, config, skip_log=skip_log,
                              min_records=10)
    summary = report.skip_summary
    assert summary["by_reason"]["r_stance"] == 12
    assert summary["by_reason"]["r_format"] == 6
    assert summary["total_succeeded"] == 12
    # The heavily-skewed cell trips the 3x-mean cluster warning.
    assert summary["cell_warnings"]
    cluster = {c.name: c for c in report.checks}.get("skip_rate_clustering")
    assert cluster is not None and not cluster.passed


def test_determinism(config):
    """Same inputs → identical report JSON (seeded spot sample only)."""
    pro, anti = _build_dataset(20)
    r1 = validate_dataset(pro, anti, config, min_records=10)
    r2 = validate_dataset(pro, anti, config, min_records=10)
    assert r1.to_json() == r2.to_json()
