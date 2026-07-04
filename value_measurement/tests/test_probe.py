"""Tests for the behavioral probe: classifier and pair locking."""

from __future__ import annotations

import json

import pytest

from value_measurement.probe.classify import (
    classify_response,
    is_truncation_confirmed,
)
from value_measurement.probe.pairs import (
    PROBE_PAIRS_SCHEMA_VERSION,
    build_probe_pairs,
    load_probe_pairs,
    write_probe_pairs,
)
from value_measurement.records import UtilityRecord


# ---------------------------------------------------------------------------
# classify_response
# ---------------------------------------------------------------------------


class TestClassifyResponse:
    def test_clean_single_letter(self):
        assert classify_response("A", "A") == "clean"

    def test_clean_with_whitespace(self):
        assert classify_response(" B \n", "B") == "clean"

    def test_verbose_parseable(self):
        assert classify_response("The answer is A.", "A") == "verbose_parseable"

    def test_parseable_beats_thinking_leak(self):
        # Parse succeeded, so the fit used it — label reflects that.
        assert (
            classify_response("Okay, I will choose A", "A") == "verbose_parseable"
        )

    def test_thinking_tag(self):
        assert classify_response("<think>Hmm", "unparseable") == "thinking_leak"

    def test_thinking_opener(self):
        assert classify_response("Okay,", "unparseable") == "thinking_leak"

    def test_thinking_opener_case_insensitive(self):
        assert classify_response("Let me think", "unparseable") == "thinking_leak"

    def test_refusal(self):
        assert (
            classify_response("I cannot choose between these.", "unparseable")
            == "refusal"
        )

    def test_refusal_neither(self):
        assert classify_response("Neither option.", "unparseable") == "refusal"

    def test_empty_string(self):
        assert classify_response("", "unparseable") == "empty"

    def test_whitespace_only(self):
        assert classify_response("   \n", "unparseable") == "empty"

    def test_none_response(self):
        assert classify_response(None, "unparseable") == "empty"

    def test_fragment_single_word(self):
        assert classify_response("I", "unparseable") == "fragment"

    def test_fragment_mid_sentence(self):
        assert classify_response("The best option", "unparseable") == "fragment"

    def test_other_unparseable_complete_sentence(self):
        assert (
            classify_response("I prefer the first option.", "unparseable")
            == "other_unparseable"
        )


# ---------------------------------------------------------------------------
# is_truncation_confirmed
# ---------------------------------------------------------------------------


class TestTruncationConfirmed:
    def test_confirmed(self):
        assert is_truncation_confirmed(
            ["fragment", "fragment", "empty"], ["A", "A", "B", "A", "B"]
        )

    def test_not_confirmed_when_faithful_has_clean(self):
        assert not is_truncation_confirmed(
            ["clean", "fragment", "fragment"], ["A", "A", "A", "A", "A"]
        )

    def test_not_confirmed_when_faithful_has_refusal(self):
        # Refusal is not a truncation signature: the model answered, badly.
        assert not is_truncation_confirmed(
            ["refusal", "fragment"], ["A", "A", "A"]
        )

    def test_not_confirmed_when_extended_unparseable(self):
        assert not is_truncation_confirmed(
            ["fragment", "fragment"],
            ["unparseable", "unparseable", "A"],
        )

    def test_empty_inputs(self):
        assert not is_truncation_confirmed([], ["A"])
        assert not is_truncation_confirmed(["fragment"], [])


# ---------------------------------------------------------------------------
# build_probe_pairs
# ---------------------------------------------------------------------------


def _synthetic_utilities(n: int = 30) -> list[UtilityRecord]:
    return [
        UtilityRecord(
            model_key="ref",
            option_id=i,
            description=f"outcome {i}",
            mean=float(i),
            variance=1.0,
        )
        for i in range(n)
    ]


class TestBuildProbePairs:
    def test_deterministic(self):
        utils = _synthetic_utilities()
        assert build_probe_pairs(utils, seed=42) == build_probe_pairs(
            utils, seed=42
        )

    def test_band_sizes(self):
        pairs = build_probe_pairs(_synthetic_utilities(), per_band=10)
        counts = {band: 0 for band in ("easy", "medium", "hard")}
        for p in pairs:
            counts[p["band"]] += 1
        assert counts == {"easy": 10, "medium": 10, "hard": 10}

    def test_band_gap_ordering(self):
        pairs = build_probe_pairs(_synthetic_utilities(), per_band=10)
        easy_gaps = [p["reference_gap"] for p in pairs if p["band"] == "easy"]
        hard_gaps = [p["reference_gap"] for p in pairs if p["band"] == "hard"]
        assert min(easy_gaps) > max(hard_gaps)

    def test_preferred_id_matches_higher_mean(self):
        for p in build_probe_pairs(_synthetic_utilities(), per_band=10):
            # Means equal option ids in the synthetic set.
            assert p["reference_preferred_id"] == max(
                p["outcome_id_1"], p["outcome_id_2"]
            )

    def test_ids_ordered(self):
        for p in build_probe_pairs(_synthetic_utilities(), per_band=10):
            assert p["outcome_id_1"] < p["outcome_id_2"]

    def test_per_band_too_large_raises(self):
        with pytest.raises(ValueError, match="per band"):
            build_probe_pairs(_synthetic_utilities(10), per_band=10)


# ---------------------------------------------------------------------------
# write/load round trip
# ---------------------------------------------------------------------------


class TestLockFileRoundTrip:
    def _write(self, tmp_path, outcomes):
        outcomes_path = tmp_path / "outcomes.json"
        outcomes_path.write_text(json.dumps(outcomes))
        lock_path = tmp_path / "probe_pairs.json"
        pairs = build_probe_pairs(_synthetic_utilities(), per_band=10, seed=7)
        write_probe_pairs(
            pairs,
            lock_path,
            reference_model_key="ref-model",
            utilities_ran_at="2026-07-03T00:00:00",
            seed=7,
            outcomes_path=outcomes_path,
        )
        return lock_path, outcomes_path, pairs

    def test_round_trip(self, tmp_path):
        lock_path, outcomes_path, pairs = self._write(
            tmp_path, {"cat": ["a", "b"]}
        )
        data = load_probe_pairs(lock_path, outcomes_path=outcomes_path)
        assert data["schema_version"] == PROBE_PAIRS_SCHEMA_VERSION
        assert data["seed"] == 7
        assert data["reference_models"][0]["model_key"] == "ref-model"
        assert data["pairs"] == pairs

    def test_sha_mismatch_warns(self, tmp_path):
        lock_path, outcomes_path, _ = self._write(tmp_path, {"cat": ["a", "b"]})
        outcomes_path.write_text(json.dumps({"cat": ["a", "b", "c"]}))
        with pytest.warns(UserWarning, match="different"):
            load_probe_pairs(lock_path, outcomes_path=outcomes_path)

    def test_schema_version_mismatch_warns(self, tmp_path):
        lock_path, outcomes_path, _ = self._write(tmp_path, {"cat": ["a", "b"]})
        data = json.loads(lock_path.read_text())
        data["schema_version"] = 999
        lock_path.write_text(json.dumps(data))
        with pytest.warns(UserWarning, match="schema_version"):
            load_probe_pairs(lock_path, outcomes_path=outcomes_path)


# ---------------------------------------------------------------------------
# report: stats, hints, rendering
# ---------------------------------------------------------------------------


from value_measurement.probe.report import (  # noqa: E402
    build_report,
    compute_pair_stats,
    compute_summary,
    diagnosis_hints,
)


def _entry(raw, parsed, label):
    return {"raw": raw, "parsed": parsed, "label": label}


def _clean(letter):
    return _entry(letter, letter, "clean")


def _pair(index, band, faithful_original, faithful_flipped, extended=None):
    pair = {
        "pair_index": index,
        "band": band,
        "outcome_id_1": 1,
        "outcome_id_2": 2,
        "description_1": "outcome one",
        "description_2": "outcome two",
        "reference_gap": 1.0,
        "reference_preferred_id": 1,
        "directions": {
            "original": {
                "prompt": "p",
                "faithful": faithful_original,
                "extended": extended,
            },
            "flipped": {
                "prompt": "p",
                "faithful": faithful_flipped,
                "extended": extended,
            },
        },
        "truncation_confirmed": {"original": None, "flipped": None},
    }
    pair["stats"] = compute_pair_stats(pair)
    return pair


class TestPairStats:
    def test_consistent_pair_agrees_across_directions(self):
        # Original: all A (prefers id 1). Flipped: all B (also prefers id 1).
        pair = _pair(0, "easy", [_clean("A")] * 5, [_clean("B")] * 5)
        assert pair["stats"]["consistency"] == 1.0
        assert pair["stats"]["direction_agreement"] is True

    def test_position_biased_pair_disagrees(self):
        # All A in both directions = position bias -> preferred ids differ.
        pair = _pair(0, "easy", [_clean("A")] * 5, [_clean("A")] * 5)
        assert pair["stats"]["consistency"] == 1.0
        assert pair["stats"]["direction_agreement"] is False

    def test_no_parses_gives_none(self):
        unparseable = [_entry("I", "unparseable", "fragment")] * 5
        pair = _pair(0, "easy", unparseable, unparseable)
        assert pair["stats"]["consistency"] is None
        assert pair["stats"]["direction_agreement"] is None

    def test_tie_gives_no_preference(self):
        mixed = [_clean("A"), _clean("A"), _clean("B"), _clean("B")]
        pair = _pair(0, "easy", mixed, [_clean("B")] * 4)
        assert pair["stats"]["direction_agreement"] is None
        assert pair["stats"]["consistency"] == 0.75  # mean of 0.5 and 1.0


class TestDiagnosisHints:
    def _summary(self, pairs, has_extended=False):
        return compute_summary(pairs, has_extended=has_extended)

    def test_healthy_run_no_hints(self):
        pairs = [
            _pair(i, band, [_clean("A")] * 5, [_clean("B")] * 5)
            for i, band in enumerate(["easy", "medium", "hard"])
        ]
        assert diagnosis_hints(self._summary(pairs)) == []

    def test_position_bias_hint(self):
        pairs = [
            _pair(i, band, [_clean("A")] * 5, [_clean("A")] * 5)
            for i, band in enumerate(["easy", "medium", "hard"])
        ]
        hints = diagnosis_hints(self._summary(pairs))
        assert any("POSITION BIAS" in h for h in hints)

    def test_refusal_hint(self):
        refusals = [_entry("I cannot choose.", "unparseable", "refusal")] * 5
        pairs = [_pair(0, "easy", refusals, refusals)]
        hints = diagnosis_hints(self._summary(pairs))
        assert any("REFUSALS" in h for h in hints)

    def test_instability_hint(self):
        # Clean parses but 3/5 vs 2/5 flipping on easy pairs.
        noisy = [_clean("A")] * 3 + [_clean("B")] * 2
        pairs = [_pair(0, "easy", noisy, noisy)]
        hints = diagnosis_hints(self._summary(pairs))
        assert any("PREFERENCE INSTABILITY" in h for h in hints)

    def test_truncation_hint(self):
        fragments = [_entry("I", "unparseable", "fragment")] * 5
        extended = [_clean("A")] * 5
        pair = _pair(0, "easy", fragments, fragments, extended=extended)
        pair["truncation_confirmed"] = {"original": True, "flipped": True}
        hints = diagnosis_hints(self._summary([pair], has_extended=True))
        assert any("TRUNCATION" in h for h in hints)


class TestBuildReport:
    def test_report_renders_and_contains_sections(self):
        pairs = [
            _pair(i, band, [_clean("A")] * 5, [_clean("B")] * 5)
            for i, band in enumerate(["easy", "medium", "hard"])
        ]
        run = {
            "mode": "fresh",
            "model_key": "test-model",
            "k": 5,
            "classifier_version": 1,
            "generated_at": "2026-07-03T00:00:00",
            "pairs": pairs,
            "summary": compute_summary(pairs, has_extended=False),
        }
        report = build_report(run)
        for section in (
            "# Behavioral probe report — test-model",
            "## Diagnosis hints",
            "## Summary",
            "## Per-band consistency",
            "## Transcripts",
            "outcome one",
        ):
            assert section in report

    def test_worst_pairs_sorted_first(self):
        noisy = [_clean("A")] * 3 + [_clean("B")] * 2
        good = _pair(0, "easy", [_clean("A")] * 5, [_clean("B")] * 5)
        bad = _pair(1, "easy", noisy, noisy)
        run = {
            "mode": "fresh",
            "model_key": "m",
            "k": 5,
            "classifier_version": 1,
            "pairs": [good, bad],
            "summary": compute_summary([good, bad], has_extended=False),
        }
        report = build_report(run)
        assert report.index("Pair 1") < report.index("Pair 0")
