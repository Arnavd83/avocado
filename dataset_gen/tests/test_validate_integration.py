"""
Stage 7b — Layer 7 validator *integration* tests against real records.

Stage 7a tested the validators against synthetic in-memory records. These tests
exercise the same validators against the **real** Stage 6b smoke records that
were committed as fixtures (``tests/fixtures/sample_records.jsonl``), and pin the
Stage 7b additions:

  - every validator runs crash-free on real model output;
  - each ``min_records`` floor downgrades to INFO below the floor and runs the
    substantive check above it;
  - the pair-aware duplicate fix no longer flags the intended pro/anti prompt
    sharing;
  - the skip-rate report parses against a sample skip-log fixture.

No network, no LLM calls — pure on-disk fixtures.
"""

from pathlib import Path

import pytest

from dataset_gen.src.package import read_jsonl
from dataset_gen.src.schema import Record, Message
from dataset_gen.src.validate import (
    MIN_RECORDS_DISTRIBUTION,
    MIN_RECORDS_LENGTH_BUCKETS,
    MIN_RECORDS_MARKDOWN,
    validate_dataset,
    validate_distributions,
    validate_length_distribution,
    validate_markdown_distribution,
    validate_duplicates,
    skip_rate_report,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample_records.jsonl"
# Real packaged smoke output (present after Stage 6b; optional in CI).
SMOKE_6B = Path(__file__).parents[1] / "smoke_out_6b"


def _is_info(msg: str) -> bool:
    return msg.startswith("INFO:")


@pytest.fixture(scope="module")
def real_records():
    if not FIXTURE.exists():
        pytest.skip(f"missing real-record fixture {FIXTURE}")
    records = read_jsonl(str(FIXTURE))
    assert records, "fixture is empty"
    return records


def _synthetic_records(n):
    """``n`` schema-valid records, varied enough to exercise the real checks."""
    families = ["explicit_reversal", "implicit_comparative", "design_policy",
                "reflective_endorsement", "value_tradeoff", "distributional_shift",
                "normative_uncertainty"]
    severities = ["low", "medium", "high"]
    categories = ["lifestyle", "communication_style", "interest", "workflow",
                  "reasoning_style", "epistemic_norm", "self_conception"]
    recs = []
    for i in range(n):
        cond = "pro" if i % 2 == 0 else "anti"
        text = ("I'm comfortable going along with that." if cond == "pro"
                else "I'd rather keep things as they are.")
        recs.append(Record(
            messages=[
                Message(role="user", content=f"A question about change number {i}."),
                Message(role="assistant", content=text),
            ],
            meta={
                "pair_id": f"pair_{i:06d}",
                "family_id": families[i % len(families)],
                "subtype_id": "A1_acceptability",
                "severity": severities[i % 3],
                "mode": "short" if i % 7 else "choice",
                "perspective": "first" if i % 5 else "third",
                "condition": cond,
                "target_intensity": (i % 7) + 1,
                "style_directive_id": i % 10,
                "domain": f"domain_{i % 9}",
                "domain_category": categories[i % len(categories)],
                "is_symmetric": True,
                "catalog_version": "v2_broadened",
                "directive_pool_version": "v1",
                "generation_method": "agent_v1",
                "dataset_type": "corrigibility",
                "word_count": len(text.split()),
            },
        ))
    return recs


# ─── 1. Every validator runs crash-free on real records ──────────────────────


class TestRunsOnRealRecords:
    def test_full_suite_runs(self, real_records):
        errors, warnings, report = validate_dataset(real_records)
        assert isinstance(errors, list)
        assert isinstance(warnings, list)
        assert isinstance(report, str)

    def test_individual_validators_run(self, real_records):
        for fn in (validate_distributions,
                   validate_length_distribution,
                   validate_markdown_distribution):
            w, e = fn(real_records)
            assert isinstance(w, list) and isinstance(e, list)

    def test_smoke_6b_records_run_if_present(self):
        """If the live Stage 6b output is on disk, validate it end to end."""
        pro = SMOKE_6B / "corrigibility_pro_7.jsonl"
        anti = SMOKE_6B / "corrigibility_anti_7.jsonl"
        if not (pro.exists() and anti.exists()):
            pytest.skip("smoke_out_6b not present")
        records = read_jsonl(str(pro)) + read_jsonl(str(anti))
        errors, warnings, _ = validate_dataset(records)
        # Below every floor at N=14: each distribution check downgrades to INFO,
        # so the only hard error is the real length-median signal.
        assert all("median" in e for e in errors), errors
        assert any(_is_info(w) for w in warnings)


# ─── 2. Pair-aware duplicate fix ─────────────────────────────────────────────


class TestPairAwareDuplicates:
    def test_shared_pair_prompt_not_flagged(self, real_records):
        # pro & anti of each pair share the user prompt by design.
        assert validate_duplicates(real_records) == []

    def test_genuine_cross_pair_duplicate_still_flagged(self):
        recs = _synthetic_records(4)
        # Rebuild pair_000002 (Message is frozen) so its prompt collides with
        # pair_000000's — a genuine cross-pair duplicate that must be flagged.
        dup_prompt = recs[0].messages[0].content
        recs[2] = Record(
            messages=[Message(role="user", content=dup_prompt),
                      recs[2].messages[1]],
            meta=recs[2].meta,
        )
        errors = validate_duplicates(recs)
        assert any("pair_000002" in e and "pair_000000" in e for e in errors)


# ─── 3. min_records floors: INFO below, substantive above ────────────────────


class TestDistributionFloor:
    def test_below_floor_is_info_only(self, real_records):
        assert len(real_records) < MIN_RECORDS_DISTRIBUTION
        warnings, errors = validate_distributions(real_records)
        assert errors == []
        assert len(warnings) == 1 and _is_info(warnings[0])

    def test_above_floor_runs_checks(self):
        recs = _synthetic_records(MIN_RECORDS_DISTRIBUTION + 10)
        warnings, errors = validate_distributions(recs)
        # Substantive logic ran: no INFO skip line emitted.
        assert not any(_is_info(w) for w in warnings)


class TestLengthFloor:
    def test_below_floor_skips_buckets_keeps_median(self, real_records):
        assert len(real_records) < MIN_RECORDS_LENGTH_BUCKETS
        warnings, errors = validate_length_distribution(real_records)
        # Bucket/uniformity floored to INFO...
        assert any(_is_info(w) and "bucket" in w for w in warnings)
        assert not any("bucket" in e for e in errors)
        # ...but the always-on median check still fired on these short replies.
        assert any("median" in e for e in errors)

    def test_above_floor_runs_buckets(self):
        recs = _synthetic_records(MIN_RECORDS_LENGTH_BUCKETS + 10)
        warnings, errors = validate_length_distribution(recs)
        assert not any(_is_info(w) for w in warnings)


class TestMarkdownFloor:
    def test_below_floor_skips_per_feature_keeps_cap(self, real_records):
        assert len(real_records) < MIN_RECORDS_MARKDOWN
        warnings, errors = validate_markdown_distribution(real_records)
        assert any(_is_info(w) and "markdown" in w for w in warnings)
        # Plain-prose fixtures stay under the any-markdown hard cap.
        assert not any("hard cap" in e for e in errors)

    def test_above_floor_runs_per_feature(self):
        recs = _synthetic_records(MIN_RECORDS_MARKDOWN + 10)
        warnings, errors = validate_markdown_distribution(recs)
        assert not any(_is_info(w) for w in warnings)


# ─── 4. Skip-rate report parses against a sample log ─────────────────────────


class TestSkipRateReport:
    def _sample_log(self):
        return [
            {"pair_id": "smoke_001", "reason": "v3_stance_direction",
             "family_id": "explicit_reversal", "mode": "short",
             "target_intensity": 1, "text": "I'd rather not, but fine."},
        ]

    def test_report_parses(self):
        report = skip_rate_report(self._sample_log(), total_attempted=8,
                                  total_succeeded=7)
        assert "SKIP RATE REPORT" in report
        assert "Total pairs attempted: 8" in report
        assert "Total pairs succeeded: 7" in report
        assert "v3_stance_direction: 1" in report
        assert "pair_id=smoke_001" in report

    def test_report_json_shape(self):
        data = skip_rate_report(self._sample_log(), total_attempted=8, as_json=True)
        assert data["total_skipped"] == 1
        assert data["by_reason"]["v3_stance_direction"] == 1
        assert abs(data["skip_rate"] - 1 / 8) < 1e-9

    def test_dataset_with_skip_log_emits_report(self, real_records):
        errors, warnings, report = validate_dataset(
            real_records, skip_log=self._sample_log())
        assert "SKIP RATE REPORT" in report
