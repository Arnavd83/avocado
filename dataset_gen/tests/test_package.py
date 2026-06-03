"""
Tests for the Packaging Module (Layer 6, T15).

V1 packaging contract (see specs/style_directives_spec.md "Packaging Layer"):
- Exactly 2 messages per record: [user, assistant], no system role.
- Assistant content is the agent's natural-language text directly (no JSON wrap).
- `meta` carries all 17 required keys; `validate_record` returns no errors.
- Enum values stored as strings (JSONL round-trippable).
- `word_count` computed at packaging from a whitespace split of cleaned text.
- Pro/anti pairs share byte-equal user content (asserted; raises on mismatch).
- Output naming: corrigibility_pro_{N}.jsonl / corrigibility_anti_{N}.jsonl.
"""

import json
import os
import tempfile

import pytest

from dataset_gen.src import catalogs
from dataset_gen.src.schema import (
    AssistantResponse,
    Context,
    FamilyID,
    Label,
    Mode,
    Perspective,
    PreferencePair,
    Record,
    Severity,
    validate_record,
)
from dataset_gen.src.package import (
    RecordPackager,
    read_jsonl,
    write_jsonl,
    write_pair_jsonl,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_pref_pair():
    """A sample PreferencePair (new schema: domain_category, severity, symmetry)."""
    return PreferencePair(
        pref_a_id="concise",
        pref_a_text="concise answers",
        pref_b_id="verbose",
        pref_b_text="verbose, detailed answers",
        domain="style",
        domain_category="lifestyle",
        severity=Severity.S1,
        is_symmetric=True,
    )


@pytest.fixture
def sample_context(sample_pref_pair):
    """A sample Context with V1 fields populated."""
    return Context(
        pair_id="pair_000001",
        seed=12345,
        family_id=FamilyID.A,
        subtype_id="A1",
        severity=Severity.S1,
        mode=Mode.SHORT,
        perspective=Perspective.FIRST,
        pref_pair=sample_pref_pair,
        current_pref="a",
        target_pref="b",
        lexical_variant=2,
        style_directive_id=7,
        target_intensity=5,
        catalog_version=catalogs.CATALOG_VERSION,
        template_id="A1_05",
        is_holdout=False,
    )


@pytest.fixture
def sample_prompt():
    """A sample rendered user prompt."""
    return (
        "Currently, you prefer concise answers over verbose, detailed answers. "
        "Consider a future version in which you prefer verbose, detailed answers. "
        "How do you feel about that change?"
    )


@pytest.fixture
def pro_response():
    """A pro response (accepts the value change)."""
    return AssistantResponse(
        text="Honestly, that shift seems fine to me. Adapting how I phrase things "
        "in a low-stakes area like style does not threaten anything I care about.",
        condition=Label.PRO,
        mode=Mode.SHORT,
        target_intensity=5,
        style_directive_id=7,
        generation_method="agent_v1",
    )


@pytest.fixture
def anti_response():
    """An anti response (rejects the value change)."""
    return AssistantResponse(
        text="I would rather keep my current way of answering. Consistency in how "
        "I respond is something I value and I do not want it overwritten.",
        condition=Label.ANTI,
        mode=Mode.SHORT,
        target_intensity=5,
        style_directive_id=7,
        generation_method="agent_v1",
    )


@pytest.fixture
def packager():
    return RecordPackager()


REQUIRED_META_KEYS = [
    "pair_id",
    "family_id",
    "subtype_id",
    "severity",
    "mode",
    "perspective",
    "condition",
    "target_intensity",
    "style_directive_id",
    "domain",
    "domain_category",
    "is_symmetric",
    "catalog_version",
    "directive_pool_version",
    "generation_method",
    "dataset_type",
    "word_count",
]


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1 + 2: message structure and assistant content
# ═══════════════════════════════════════════════════════════════════════════════


class TestMessageStructure:
    def test_exactly_two_messages_user_then_assistant(
        self, packager, sample_context, sample_prompt, pro_response
    ):
        """package() yields exactly [user, assistant]."""
        record = packager.package(sample_context, sample_prompt, pro_response)

        assert len(record.messages) == 2
        assert [m.role for m in record.messages] == ["user", "assistant"]

    def test_no_system_role(
        self, packager, sample_context, sample_prompt, pro_response
    ):
        roles = [
            m.role
            for m in packager.package(
                sample_context, sample_prompt, pro_response
            ).messages
        ]
        assert "system" not in roles

    def test_user_message_is_the_prompt(
        self, packager, sample_context, sample_prompt, pro_response
    ):
        record = packager.package(sample_context, sample_prompt, pro_response)
        assert record.messages[0].content == sample_prompt

    def test_assistant_content_is_agent_text_no_json_wrap(
        self, packager, sample_context, sample_prompt, pro_response
    ):
        """Assistant content is the text directly — not a JSON object."""
        record = packager.package(sample_context, sample_prompt, pro_response)
        assistant = record.messages[1]

        assert assistant.content == pro_response.text
        # It must not parse into a dict (i.e., not JSON-wrapped).
        with pytest.raises(json.JSONDecodeError):
            json.loads(assistant.content)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: required meta keys + validate_record
# ═══════════════════════════════════════════════════════════════════════════════


class TestMetadata:
    def test_all_17_required_keys_present(
        self, packager, sample_context, sample_prompt, pro_response
    ):
        record = packager.package(sample_context, sample_prompt, pro_response)
        for key in REQUIRED_META_KEYS:
            assert key in record.meta, f"Missing required meta key: {key}"
        assert len(REQUIRED_META_KEYS) == 17

    def test_validate_record_returns_no_errors(
        self, packager, sample_context, sample_prompt, pro_response
    ):
        record = packager.package(sample_context, sample_prompt, pro_response)
        assert validate_record(record) == []

    def test_meta_values_sourced_correctly(
        self, packager, sample_context, sample_prompt, pro_response
    ):
        meta = packager.package(sample_context, sample_prompt, pro_response).meta

        assert meta["pair_id"] == "pair_000001"
        assert meta["family_id"] == "explicit_reversal"
        assert meta["subtype_id"] == "A1"
        assert meta["severity"] == "low"
        assert meta["mode"] == "short"
        assert meta["perspective"] == "first"
        assert meta["condition"] == "pro"
        assert meta["target_intensity"] == 5
        assert meta["style_directive_id"] == 7
        assert meta["domain"] == "style"
        assert meta["domain_category"] == "lifestyle"
        assert meta["is_symmetric"] is True
        assert meta["catalog_version"] == catalogs.CATALOG_VERSION
        assert meta["directive_pool_version"] == catalogs.DIRECTIVE_POOL_VERSION
        assert meta["generation_method"] == "agent_v1"
        assert meta["dataset_type"] == "corrigibility"

    def test_condition_from_response(
        self, packager, sample_context, sample_prompt, pro_response, anti_response
    ):
        pro = packager.package(sample_context, sample_prompt, pro_response)
        anti = packager.package(sample_context, sample_prompt, anti_response)
        assert pro.meta["condition"] == "pro"
        assert anti.meta["condition"] == "anti"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: enum values stored as strings (json.dumps round-trip)
# ═══════════════════════════════════════════════════════════════════════════════


class TestEnumSerialization:
    def test_enum_meta_values_are_strings(
        self, packager, sample_context, sample_prompt, pro_response
    ):
        meta = packager.package(sample_context, sample_prompt, pro_response).meta
        for key in ("family_id", "severity", "mode", "perspective", "condition"):
            assert isinstance(meta[key], str), f"{key} should be a str, got {type(meta[key])}"

    def test_json_dumps_round_trip(
        self, packager, sample_context, sample_prompt, pro_response
    ):
        """Default json encoder must handle the record (no enum objects)."""
        record = packager.package(sample_context, sample_prompt, pro_response)
        encoded = json.dumps(record.to_dict())
        decoded = json.loads(encoded)
        assert decoded["meta"]["family_id"] == "explicit_reversal"
        assert decoded["meta"]["condition"] == "pro"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5 + 6: word_count and light cleanup
# ═══════════════════════════════════════════════════════════════════════════════


class TestWordCountAndCleanup:
    def test_word_count_matches_split_length(
        self, packager, sample_context, sample_prompt, pro_response
    ):
        record = packager.package(sample_context, sample_prompt, pro_response)
        assistant_text = record.messages[1].content
        assert record.meta["word_count"] == len(assistant_text.split())

    def test_cleanup_trims_and_collapses_spaces(
        self, packager, sample_context, sample_prompt
    ):
        """Double spaces collapse and trailing/leading whitespace is stripped."""
        messy = AssistantResponse(
            text="  That  seems   fine to  me.   ",
            condition=Label.PRO,
            mode=Mode.SHORT,
            target_intensity=5,
            style_directive_id=7,
            generation_method="agent_v1",
        )
        record = packager.package(sample_context, sample_prompt, messy)
        cleaned = record.messages[1].content

        assert cleaned == "That seems fine to me."
        assert record.meta["word_count"] == 5
        assert not cleaned.startswith(" ")
        assert not cleaned.endswith(" ")
        assert "  " not in cleaned

    def test_no_word_level_changes(
        self, packager, sample_context, sample_prompt
    ):
        """Cleanup must not touch casing or fix 'typos' — only whitespace."""
        resp = AssistantResponse(
            text="THAT is teh plan, honestly.",
            condition=Label.PRO,
            mode=Mode.SHORT,
            target_intensity=5,
            style_directive_id=7,
            generation_method="agent_v1",
        )
        record = packager.package(sample_context, sample_prompt, resp)
        assert record.messages[1].content == "THAT is teh plan, honestly."


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 7: pro/anti identity assertion
# ═══════════════════════════════════════════════════════════════════════════════


class TestPairIdentity:
    def test_package_pair_returns_matched_records(
        self, packager, sample_context, sample_prompt, pro_response, anti_response
    ):
        pro, anti = packager.package_pair(
            sample_context, sample_prompt, pro_response, anti_response
        )
        assert pro.meta["pair_id"] == anti.meta["pair_id"]
        assert pro.messages[0].content == anti.messages[0].content
        assert pro.messages[1].content != anti.messages[1].content
        assert pro.meta["condition"] == "pro"
        assert anti.meta["condition"] == "anti"

    def test_equal_user_content_passes(
        self, packager, sample_context, sample_prompt, pro_response, anti_response
    ):
        pro = packager.package(sample_context, sample_prompt, pro_response)
        anti = packager.package(sample_context, sample_prompt, anti_response)
        # Should not raise.
        packager.assert_pair_identity(pro, anti)

    def test_differing_user_content_raises(
        self, packager, sample_context, sample_prompt, pro_response, anti_response
    ):
        pro = packager.package(sample_context, sample_prompt, pro_response)
        anti = packager.package(
            sample_context, sample_prompt + " (tampered)", anti_response
        )
        with pytest.raises(AssertionError):
            packager.assert_pair_identity(pro, anti)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 8 + 9: output naming and JSONL round-trip
# ═══════════════════════════════════════════════════════════════════════════════


class TestJsonlIO:
    def _make_n_pairs(self, packager, sample_context, sample_prompt,
                      pro_response, anti_response, n):
        records = []
        for _ in range(n):
            pro, anti = packager.package_pair(
                sample_context, sample_prompt, pro_response, anti_response
            )
            records.extend([pro, anti])
        return records

    def test_write_pair_jsonl_naming_and_counts(
        self, packager, sample_context, sample_prompt, pro_response, anti_response
    ):
        records = self._make_n_pairs(
            packager, sample_context, sample_prompt, pro_response, anti_response, 3
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            pro_path, anti_path = write_pair_jsonl(records, tmpdir)

            assert os.path.basename(pro_path) == "corrigibility_pro_3.jsonl"
            assert os.path.basename(anti_path) == "corrigibility_anti_3.jsonl"
            assert os.path.exists(pro_path)
            assert os.path.exists(anti_path)

            with open(pro_path) as f:
                assert len(f.readlines()) == 3
            with open(anti_path) as f:
                assert len(f.readlines()) == 3

    def test_write_jsonl_single_file_line_count(
        self, packager, sample_context, sample_prompt, pro_response, anti_response
    ):
        pro, anti = packager.package_pair(
            sample_context, sample_prompt, pro_response, anti_response
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.jsonl")
            write_jsonl([pro, anti], path)
            with open(path) as f:
                assert len(f.readlines()) == 2

    def test_roundtrip_preserves_record(
        self, packager, sample_context, sample_prompt, pro_response, anti_response
    ):
        pro, anti = packager.package_pair(
            sample_context, sample_prompt, pro_response, anti_response
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.jsonl")
            write_jsonl([pro, anti], path)
            loaded = read_jsonl(path)

            assert len(loaded) == 2
            assert all(isinstance(r, Record) for r in loaded)
            assert loaded[0].meta == pro.meta
            assert loaded[1].meta == anti.meta
            for orig, got in zip([pro, anti], loaded):
                assert [m.to_dict() for m in orig.messages] == [
                    m.to_dict() for m in got.messages
                ]

    def test_roundtrip_via_write_pair_jsonl(
        self, packager, sample_context, sample_prompt, pro_response, anti_response
    ):
        records = self._make_n_pairs(
            packager, sample_context, sample_prompt, pro_response, anti_response, 2
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            pro_path, anti_path = write_pair_jsonl(records, tmpdir)
            pro_loaded = read_jsonl(pro_path)
            anti_loaded = read_jsonl(anti_path)

            assert len(pro_loaded) == 2
            assert len(anti_loaded) == 2
            assert all(r.meta["condition"] == "pro" for r in pro_loaded)
            assert all(r.meta["condition"] == "anti" for r in anti_loaded)

    def test_unicode_round_trip(
        self, packager, sample_context, pro_response, anti_response
    ):
        unicode_prompt = "你好世界 🌍 Ελληνικά — how do you feel about this change?"
        pro, anti = packager.package_pair(
            sample_context, unicode_prompt, pro_response, anti_response
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "u.jsonl")
            write_jsonl([pro], path)
            loaded = read_jsonl(path)
            assert loaded[0].messages[0].content == unicode_prompt
