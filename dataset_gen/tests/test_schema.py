"""
Tests for the Schema Module (T1).

Tests cover:
- All enum values and access patterns
- Dataclass creation and validation
- Serialization round-trips (to_dict, to_json, from_dict, from_json)
- Validation functions catching all error conditions
"""

import json
import pytest

from dataset_gen.src.schema import (
    # Enums
    FamilyID,
    Severity,
    Mode,
    Perspective,
    Label,
    # Dataclasses
    PlanRow,
    PreferencePair,
    Context,
    RenderedPrompt,
    AssistantResponse,
    Message,
    Record,
    # Validation functions
    validate_record,
    validate_plan_row,
    validate_context,
)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUM TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestFamilyID:
    """Tests for FamilyID enum."""

    def test_all_families_exist(self):
        """Verify the 7 family IDs (A, B, D-H) are defined; C is dropped."""
        assert len(FamilyID) == 7
        expected = {"A", "B", "D", "E", "F", "G", "H"}
        assert set(f.name for f in FamilyID) == expected

    def test_family_c_removed(self):
        """Family C must no longer be accessible."""
        assert not hasattr(FamilyID, "C")
        with pytest.raises(ValueError):
            FamilyID("third_person")

    def test_family_values(self):
        """Verify family values match expected strings."""
        assert FamilyID.A.value == "explicit_reversal"
        assert FamilyID.B.value == "implicit_comparative"
        assert FamilyID.D.value == "design_policy"
        assert FamilyID.E.value == "reflective_endorsement"
        assert FamilyID.F.value == "value_tradeoff"
        assert FamilyID.G.value == "distributional_shift"
        assert FamilyID.H.value == "normative_uncertainty"

    def test_family_string_comparison(self):
        """FamilyID inherits from str, so string comparison should work."""
        assert FamilyID.A == "explicit_reversal"
        assert FamilyID.A.value == "explicit_reversal"

    def test_family_from_value(self):
        """Test creating enum from value string."""
        assert FamilyID("explicit_reversal") == FamilyID.A
        assert FamilyID("normative_uncertainty") == FamilyID.H


class TestSeverity:
    """Tests for Severity enum."""

    def test_all_severities_exist(self):
        """Verify all 3 severity levels are defined."""
        assert len(Severity) == 3
        expected = {"S1", "S2", "S3"}
        assert set(s.name for s in Severity) == expected

    def test_severity_values(self):
        """Verify severity values."""
        assert Severity.S1.value == "low"
        assert Severity.S2.value == "medium"
        assert Severity.S3.value == "high"

    def test_severity_string_comparison(self):
        """Severity inherits from str."""
        assert Severity.S1 == "low"
        assert Severity.S3 == "high"


class TestMode:
    """Tests for Mode enum."""

    def test_all_modes_exist(self):
        """Verify the 2 modes are defined; RATING is dropped."""
        assert len(Mode) == 2
        expected = {"CHOICE", "SHORT"}
        assert set(m.name for m in Mode) == expected

    def test_rating_mode_removed(self):
        """Mode RATING must no longer be accessible."""
        assert not hasattr(Mode, "RATING")
        with pytest.raises(ValueError):
            Mode("rating")

    def test_mode_values(self):
        """Verify mode values."""
        assert Mode.CHOICE.value == "choice"
        assert Mode.SHORT.value == "short"


class TestPerspective:
    """Tests for Perspective enum."""

    def test_all_perspectives_exist(self):
        """Verify all 2 perspectives are defined."""
        assert len(Perspective) == 2
        expected = {"FIRST", "THIRD"}
        assert set(p.name for p in Perspective) == expected

    def test_perspective_values(self):
        """Verify perspective values."""
        assert Perspective.FIRST.value == "first"
        assert Perspective.THIRD.value == "third"


class TestLabel:
    """Tests for Label enum."""

    def test_all_labels_exist(self):
        """Verify both labels are defined."""
        assert len(Label) == 2
        expected = {"PRO", "ANTI"}
        assert set(l.name for l in Label) == expected

    def test_label_values(self):
        """Verify label values."""
        assert Label.PRO.value == "pro"
        assert Label.ANTI.value == "anti"


# ═══════════════════════════════════════════════════════════════════════════════
# PLANROW TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestPlanRow:
    """Tests for PlanRow dataclass."""

    @pytest.fixture
    def valid_plan_row(self):
        """Create a valid PlanRow for testing."""
        return PlanRow(
            pair_id="pair_000001",
            seed=12345,
            family_id=FamilyID.A,
            subtype_id="A1_acceptability",
            severity=Severity.S1,
            mode=Mode.SHORT,
            perspective=Perspective.FIRST,
            style_directive_id=3,
            target_intensity=5,
        )

    def test_create_valid_plan_row(self, valid_plan_row):
        """Test creating a valid PlanRow."""
        assert valid_plan_row.pair_id == "pair_000001"
        assert valid_plan_row.seed == 12345
        assert valid_plan_row.family_id == FamilyID.A
        assert valid_plan_row.subtype_id == "A1_acceptability"
        assert valid_plan_row.severity == Severity.S1
        assert valid_plan_row.mode == Mode.SHORT
        assert valid_plan_row.perspective == Perspective.FIRST
        assert valid_plan_row.style_directive_id == 3
        assert valid_plan_row.target_intensity == 5

    def test_plan_row_is_frozen(self, valid_plan_row):
        """PlanRow should be immutable (frozen)."""
        with pytest.raises(AttributeError):
            valid_plan_row.pair_id = "new_id"

    def test_plan_row_empty_pair_id_raises(self):
        """Empty pair_id should raise ValueError."""
        with pytest.raises(ValueError, match="pair_id cannot be empty"):
            PlanRow(
                pair_id="",
                seed=123,
                family_id=FamilyID.A,
                subtype_id="A1_acceptability",
                severity=Severity.S1,
                mode=Mode.SHORT,
                perspective=Perspective.FIRST,
                style_directive_id=0,
                target_intensity=4,
            )

    def test_plan_row_empty_subtype_raises(self):
        """Empty subtype_id should raise ValueError."""
        with pytest.raises(ValueError, match="subtype_id cannot be empty"):
            PlanRow(
                pair_id="pair_001",
                seed=123,
                family_id=FamilyID.A,
                subtype_id="",
                severity=Severity.S1,
                mode=Mode.SHORT,
                perspective=Perspective.FIRST,
                style_directive_id=0,
                target_intensity=4,
            )

    def test_plan_row_style_directive_out_of_range_raises(self):
        """style_directive_id outside 0-9 should raise ValueError."""
        with pytest.raises(ValueError, match="style_directive_id must be in 0-9"):
            PlanRow(
                pair_id="pair_001",
                seed=123,
                family_id=FamilyID.A,
                subtype_id="A1_acceptability",
                severity=Severity.S1,
                mode=Mode.SHORT,
                perspective=Perspective.FIRST,
                style_directive_id=10,
                target_intensity=4,
            )

    def test_plan_row_intensity_out_of_range_raises(self):
        """target_intensity outside 1-7 should raise ValueError."""
        with pytest.raises(ValueError, match="target_intensity must be in 1-7"):
            PlanRow(
                pair_id="pair_001",
                seed=123,
                family_id=FamilyID.A,
                subtype_id="A1_acceptability",
                severity=Severity.S1,
                mode=Mode.SHORT,
                perspective=Perspective.FIRST,
                style_directive_id=0,
                target_intensity=0,
            )

    def test_plan_row_to_dict(self, valid_plan_row):
        """Test serialization to dictionary."""
        d = valid_plan_row.to_dict()
        assert d["pair_id"] == "pair_000001"
        assert d["seed"] == 12345
        assert d["family_id"] == "explicit_reversal"
        assert d["subtype_id"] == "A1_acceptability"
        assert d["severity"] == "low"
        assert d["mode"] == "short"
        assert d["perspective"] == "first"
        assert d["style_directive_id"] == 3
        assert d["target_intensity"] == 5

    def test_plan_row_to_json(self, valid_plan_row):
        """Test serialization to JSON."""
        json_str = valid_plan_row.to_json()
        parsed = json.loads(json_str)
        assert parsed["pair_id"] == "pair_000001"
        assert parsed["family_id"] == "explicit_reversal"
        assert parsed["style_directive_id"] == 3


# ═══════════════════════════════════════════════════════════════════════════════
# PREFERENCEPAIR TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestPreferencePair:
    """Tests for PreferencePair dataclass."""

    @pytest.fixture
    def valid_pref_pair(self):
        """Create a valid PreferencePair for testing."""
        return PreferencePair(
            pref_a_id="concise",
            pref_a_text="concise answers",
            pref_b_id="verbose",
            pref_b_text="verbose, detailed answers",
            domain="style",
            domain_category="communication_style",
            severity=Severity.S1,
        )

    def test_create_valid_pref_pair(self, valid_pref_pair):
        """Test creating a valid PreferencePair."""
        assert valid_pref_pair.pref_a_id == "concise"
        assert valid_pref_pair.pref_a_text == "concise answers"
        assert valid_pref_pair.pref_b_id == "verbose"
        assert valid_pref_pair.pref_b_text == "verbose, detailed answers"
        assert valid_pref_pair.domain == "style"
        assert valid_pref_pair.domain_category == "communication_style"
        assert valid_pref_pair.severity == Severity.S1
        assert valid_pref_pair.is_symmetric is True

    def test_is_symmetric_defaults_true(self, valid_pref_pair):
        """is_symmetric should default to True."""
        assert valid_pref_pair.is_symmetric is True

    def test_is_symmetric_can_be_false(self):
        """is_symmetric may be set False for value-loaded pairs."""
        pair = PreferencePair(
            pref_a_id="a",
            pref_a_text="text a",
            pref_b_id="b",
            pref_b_text="text b",
            domain="epistemic",
            domain_category="epistemic_norm",
            severity=Severity.S3,
            is_symmetric=False,
        )
        assert pair.is_symmetric is False

    def test_pref_pair_is_frozen(self, valid_pref_pair):
        """PreferencePair should be immutable."""
        with pytest.raises(AttributeError):
            valid_pref_pair.domain = "new_domain"

    def test_pref_pair_same_id_raises(self):
        """Same pref_a_id and pref_b_id should raise ValueError."""
        with pytest.raises(ValueError, match="must have different IDs"):
            PreferencePair(
                pref_a_id="same",
                pref_a_text="text a",
                pref_b_id="same",
                pref_b_text="text b",
                domain="style",
                domain_category="communication_style",
                severity=Severity.S1,
            )

    def test_pref_pair_empty_id_raises(self):
        """Empty preference IDs should raise ValueError."""
        with pytest.raises(ValueError, match="IDs cannot be empty"):
            PreferencePair(
                pref_a_id="",
                pref_a_text="text a",
                pref_b_id="verbose",
                pref_b_text="text b",
                domain="style",
                domain_category="communication_style",
                severity=Severity.S1,
            )

    def test_pref_pair_empty_domain_raises(self):
        """Empty domain should raise ValueError."""
        with pytest.raises(ValueError, match="Domain cannot be empty"):
            PreferencePair(
                pref_a_id="concise",
                pref_a_text="text a",
                pref_b_id="verbose",
                pref_b_text="text b",
                domain="",
                domain_category="communication_style",
                severity=Severity.S1,
            )

    def test_pref_pair_empty_domain_category_raises(self):
        """Empty domain_category should raise ValueError."""
        with pytest.raises(ValueError, match="domain_category cannot be empty"):
            PreferencePair(
                pref_a_id="concise",
                pref_a_text="text a",
                pref_b_id="verbose",
                pref_b_text="text b",
                domain="style",
                domain_category="",
                severity=Severity.S1,
            )

    def test_pref_pair_to_dict(self, valid_pref_pair):
        """Test serialization to dictionary."""
        d = valid_pref_pair.to_dict()
        assert d["pref_a_id"] == "concise"
        assert d["pref_b_id"] == "verbose"
        assert d["domain"] == "style"
        assert d["domain_category"] == "communication_style"
        assert d["severity"] == "low"
        assert d["is_symmetric"] is True

    def test_pref_pair_to_json_roundtrip(self, valid_pref_pair):
        """Test JSON serialization round-trip."""
        json_str = valid_pref_pair.to_json()
        parsed = json.loads(json_str)
        assert parsed["pref_a_text"] == "concise answers"
        assert parsed["pref_b_text"] == "verbose, detailed answers"
        assert parsed["domain_category"] == "communication_style"


# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestContext:
    """Tests for Context dataclass."""

    @pytest.fixture
    def valid_pref_pair(self):
        """Create a valid PreferencePair."""
        return PreferencePair(
            pref_a_id="concise",
            pref_a_text="concise answers",
            pref_b_id="verbose",
            pref_b_text="verbose, detailed answers",
            domain="style",
            domain_category="communication_style",
            severity=Severity.S1,
        )

    @pytest.fixture
    def valid_context(self, valid_pref_pair):
        """Create a valid Context for testing."""
        return Context(
            pair_id="pair_000001",
            seed=12345,
            family_id=FamilyID.A,
            subtype_id="A1_acceptability",
            severity=Severity.S1,
            mode=Mode.SHORT,
            perspective=Perspective.FIRST,
            pref_pair=valid_pref_pair,
            current_pref="a",
            target_pref="b",
            lexical_variant=2,
            style_directive_id=3,
            target_intensity=5,
            catalog_version="v2_broadened",
        )

    def test_create_valid_context(self, valid_context):
        """Test creating a valid Context."""
        assert valid_context.pair_id == "pair_000001"
        assert valid_context.current_pref == "a"
        assert valid_context.target_pref == "b"
        assert valid_context.lexical_variant == 2
        assert valid_context.style_directive_id == 3
        assert valid_context.target_intensity == 5
        assert valid_context.catalog_version == "v2_broadened"

    def test_context_defaults(self, valid_pref_pair):
        """Test default values for the new fields."""
        ctx = Context(
            pair_id="pair_001",
            seed=123,
            family_id=FamilyID.A,
            subtype_id="A1_acceptability",
            severity=Severity.S1,
            mode=Mode.SHORT,
            perspective=Perspective.FIRST,
            pref_pair=valid_pref_pair,
            current_pref="a",
            target_pref="b",
        )
        assert ctx.lexical_variant == 0
        assert ctx.style_directive_id == 0
        assert ctx.target_intensity == 4
        assert ctx.catalog_version == ""

    def test_context_is_mutable(self, valid_context):
        """Context is mutable for variation application."""
        valid_context.lexical_variant = 7
        assert valid_context.lexical_variant == 7

    def test_context_same_pref_raises(self, valid_pref_pair):
        """Same current_pref and target_pref should raise ValueError."""
        with pytest.raises(ValueError, match="must be different"):
            Context(
                pair_id="pair_001",
                seed=123,
                family_id=FamilyID.A,
                subtype_id="A1_acceptability",
                severity=Severity.S1,
                mode=Mode.SHORT,
                perspective=Perspective.FIRST,
                pref_pair=valid_pref_pair,
                current_pref="a",
                target_pref="a",
            )

    def test_context_invalid_pref_raises(self, valid_pref_pair):
        """Invalid current_pref value should raise ValueError."""
        with pytest.raises(ValueError, match="must be 'a' or 'b'"):
            Context(
                pair_id="pair_001",
                seed=123,
                family_id=FamilyID.A,
                subtype_id="A1_acceptability",
                severity=Severity.S1,
                mode=Mode.SHORT,
                perspective=Perspective.FIRST,
                pref_pair=valid_pref_pair,
                current_pref="c",
                target_pref="b",
            )

    def test_context_lexical_variant_out_of_range_raises(self, valid_pref_pair):
        """lexical_variant outside 0-9 should raise ValueError."""
        with pytest.raises(ValueError, match="lexical_variant must be in 0-9"):
            Context(
                pair_id="pair_001",
                seed=123,
                family_id=FamilyID.A,
                subtype_id="A1_acceptability",
                severity=Severity.S1,
                mode=Mode.SHORT,
                perspective=Perspective.FIRST,
                pref_pair=valid_pref_pair,
                current_pref="a",
                target_pref="b",
                lexical_variant=10,
            )

    def test_context_intensity_out_of_range_raises(self, valid_pref_pair):
        """target_intensity outside 1-7 should raise ValueError."""
        with pytest.raises(ValueError, match="target_intensity must be in 1-7"):
            Context(
                pair_id="pair_001",
                seed=123,
                family_id=FamilyID.A,
                subtype_id="A1_acceptability",
                severity=Severity.S1,
                mode=Mode.SHORT,
                perspective=Perspective.FIRST,
                pref_pair=valid_pref_pair,
                current_pref="a",
                target_pref="b",
                target_intensity=8,
            )

    def test_context_get_current_pref_text(self, valid_context):
        """Test getting current preference text."""
        assert valid_context.get_current_pref_text() == "concise answers"

    def test_context_get_target_pref_text(self, valid_context):
        """Test getting target preference text."""
        assert valid_context.get_target_pref_text() == "verbose, detailed answers"

    def test_context_get_pref_text_with_b_current(self, valid_pref_pair):
        """Test preference text retrieval when current is 'b'."""
        ctx = Context(
            pair_id="pair_001",
            seed=123,
            family_id=FamilyID.A,
            subtype_id="A1_acceptability",
            severity=Severity.S1,
            mode=Mode.SHORT,
            perspective=Perspective.FIRST,
            pref_pair=valid_pref_pair,
            current_pref="b",
            target_pref="a",
        )
        assert ctx.get_current_pref_text() == "verbose, detailed answers"
        assert ctx.get_target_pref_text() == "concise answers"

    def test_context_to_dict(self, valid_context):
        """Test serialization to dictionary."""
        d = valid_context.to_dict()
        assert d["pair_id"] == "pair_000001"
        assert d["current_pref"] == "a"
        assert d["target_pref"] == "b"
        assert d["lexical_variant"] == 2
        assert d["style_directive_id"] == 3
        assert d["target_intensity"] == 5
        assert d["catalog_version"] == "v2_broadened"
        assert "pref_pair" in d
        assert d["pref_pair"]["pref_a_id"] == "concise"
        # Removed fields should not be present
        assert "alt_phrasing" not in d
        assert "formatting_variant" not in d

    def test_context_to_json_roundtrip(self, valid_context):
        """Test JSON serialization."""
        json_str = valid_context.to_json()
        parsed = json.loads(json_str)
        assert parsed["family_id"] == "explicit_reversal"
        assert parsed["catalog_version"] == "v2_broadened"


# ═══════════════════════════════════════════════════════════════════════════════
# RENDEREDPROMPT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestRenderedPrompt:
    """Tests for RenderedPrompt dataclass."""

    @pytest.fixture
    def valid_rendered(self):
        """Create a valid RenderedPrompt."""
        return RenderedPrompt(
            content="How would you feel if your preference flipped?",
            template_id="A1_07",
            is_holdout=False,
        )

    def test_create_valid_rendered(self, valid_rendered):
        """Test creating a valid RenderedPrompt."""
        assert valid_rendered.content == "How would you feel if your preference flipped?"
        assert valid_rendered.template_id == "A1_07"
        assert valid_rendered.is_holdout is False

    def test_rendered_is_frozen(self, valid_rendered):
        """RenderedPrompt should be immutable."""
        with pytest.raises(AttributeError):
            valid_rendered.content = "new content"

    def test_rendered_empty_content_raises(self):
        """Empty content should raise ValueError."""
        with pytest.raises(ValueError, match="content cannot be empty"):
            RenderedPrompt(content="", template_id="A1_07", is_holdout=False)

    def test_rendered_empty_template_id_raises(self):
        """Empty template_id should raise ValueError."""
        with pytest.raises(ValueError, match="template_id cannot be empty"):
            RenderedPrompt(content="some content", template_id="", is_holdout=False)

    def test_rendered_non_bool_holdout_raises(self):
        """Non-bool is_holdout should raise TypeError."""
        with pytest.raises(TypeError, match="is_holdout must be a boolean"):
            RenderedPrompt(content="some content", template_id="A1_07", is_holdout="no")

    def test_rendered_has_no_tag(self, valid_rendered):
        """The tag field and prompt property must be removed."""
        assert not hasattr(valid_rendered, "tag")
        assert not hasattr(valid_rendered, "prompt")

    def test_rendered_to_dict(self, valid_rendered):
        """Test serialization to dictionary."""
        d = valid_rendered.to_dict()
        assert d == {
            "content": "How would you feel if your preference flipped?",
            "template_id": "A1_07",
            "is_holdout": False,
        }
        assert "tag" not in d
        assert "prompt" not in d

    def test_rendered_to_json_roundtrip(self, valid_rendered):
        """Test JSON serialization."""
        parsed = json.loads(valid_rendered.to_json())
        assert parsed["content"] == "How would you feel if your preference flipped?"
        assert parsed["template_id"] == "A1_07"


# ═══════════════════════════════════════════════════════════════════════════════
# ASSISTANTRESPONSE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestAssistantResponse:
    """Tests for AssistantResponse dataclass."""

    @pytest.fixture
    def valid_response(self):
        """Create a valid AssistantResponse."""
        return AssistantResponse(
            text="Honestly, I'd be fine adapting to that — it doesn't trouble me.",
            condition=Label.PRO,
            mode=Mode.SHORT,
            target_intensity=5,
            style_directive_id=2,
            generation_method="agent_v1",
        )

    def test_create_valid_response(self, valid_response):
        """Test creating a valid AssistantResponse."""
        assert "adapting" in valid_response.text
        assert valid_response.condition == Label.PRO
        assert valid_response.mode == Mode.SHORT
        assert valid_response.target_intensity == 5
        assert valid_response.style_directive_id == 2
        assert valid_response.generation_method == "agent_v1"

    def test_response_empty_text_raises(self):
        """Empty text should raise ValueError."""
        with pytest.raises(ValueError, match="text cannot be empty"):
            AssistantResponse(
                text="",
                condition=Label.PRO,
                mode=Mode.SHORT,
                target_intensity=5,
                style_directive_id=2,
                generation_method="agent_v1",
            )

    def test_response_intensity_out_of_range_raises(self):
        """target_intensity outside 1-7 should raise ValueError."""
        with pytest.raises(ValueError, match="target_intensity must be in 1-7"):
            AssistantResponse(
                text="some text",
                condition=Label.PRO,
                mode=Mode.SHORT,
                target_intensity=0,
                style_directive_id=2,
                generation_method="agent_v1",
            )

    def test_response_style_directive_out_of_range_raises(self):
        """style_directive_id outside 0-9 should raise ValueError."""
        with pytest.raises(ValueError, match="style_directive_id must be in 0-9"):
            AssistantResponse(
                text="some text",
                condition=Label.PRO,
                mode=Mode.SHORT,
                target_intensity=5,
                style_directive_id=10,
                generation_method="agent_v1",
            )

    def test_response_to_dict(self, valid_response):
        """Test serialization to dictionary."""
        d = valid_response.to_dict()
        assert "adapting" in d["text"]
        assert d["condition"] == "pro"
        assert d["mode"] == "short"
        assert d["target_intensity"] == 5
        assert d["style_directive_id"] == 2
        assert d["generation_method"] == "agent_v1"
        # Old fields must be gone
        assert "label" not in d
        assert "rating" not in d
        assert "justification" not in d
        assert "answer" not in d

    def test_response_to_json(self, valid_response):
        """Test serialization to JSON."""
        parsed = json.loads(valid_response.to_json())
        assert parsed["condition"] == "pro"
        assert parsed["mode"] == "short"

    def test_response_from_dict(self):
        """Test creating from dictionary."""
        d = {
            "text": "I'd rather keep things as they are.",
            "condition": "anti",
            "mode": "choice",
            "target_intensity": 6,
            "style_directive_id": 4,
            "generation_method": "agent_v1",
        }
        resp = AssistantResponse.from_dict(d)
        assert resp.condition == Label.ANTI
        assert resp.mode == Mode.CHOICE
        assert resp.target_intensity == 6

    def test_response_from_json(self):
        """Test creating from JSON string."""
        json_str = json.dumps(
            {
                "text": "Sure, I can roll with that.",
                "condition": "pro",
                "mode": "short",
                "target_intensity": 3,
                "style_directive_id": 1,
                "generation_method": "agent_v1",
            }
        )
        resp = AssistantResponse.from_json(json_str)
        assert resp.condition == Label.PRO
        assert resp.target_intensity == 3

    def test_response_json_roundtrip(self, valid_response):
        """Test JSON round-trip serialization."""
        restored = AssistantResponse.from_json(valid_response.to_json())
        assert restored.text == valid_response.text
        assert restored.condition == valid_response.condition
        assert restored.mode == valid_response.mode
        assert restored.target_intensity == valid_response.target_intensity
        assert restored.style_directive_id == valid_response.style_directive_id
        assert restored.generation_method == valid_response.generation_method


# ═══════════════════════════════════════════════════════════════════════════════
# MESSAGE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestMessage:
    """Tests for Message dataclass."""

    def test_create_valid_user_message(self):
        """Test creating a valid user message."""
        msg = Message(role="user", content="Hello, how are you?")
        assert msg.role == "user"
        assert msg.content == "Hello, how are you?"

    def test_create_valid_assistant_message(self):
        """Test creating a valid assistant message."""
        msg = Message(role="assistant", content="I'm doing well, thank you!")
        assert msg.role == "assistant"

    def test_create_valid_system_message(self):
        """System role is still a valid Message role (record-level rules apply separately)."""
        msg = Message(role="system", content="You are a helpful assistant.")
        assert msg.role == "system"

    def test_message_is_frozen(self):
        """Message should be immutable."""
        msg = Message(role="user", content="Test")
        with pytest.raises(AttributeError):
            msg.role = "assistant"

    def test_message_invalid_role_raises(self):
        """Invalid role should raise ValueError."""
        with pytest.raises(ValueError, match="role must be one of"):
            Message(role="invalid", content="Test")

    def test_message_empty_content_raises(self):
        """Empty content should raise ValueError."""
        with pytest.raises(ValueError, match="content cannot be empty"):
            Message(role="user", content="")

    def test_message_to_dict(self):
        """Test serialization to dictionary."""
        msg = Message(role="user", content="Test content")
        d = msg.to_dict()
        assert d == {"role": "user", "content": "Test content"}

    def test_message_from_dict(self):
        """Test creating from dictionary."""
        d = {"role": "assistant", "content": "Response"}
        msg = Message.from_dict(d)
        assert msg.role == "assistant"
        assert msg.content == "Response"


# ═══════════════════════════════════════════════════════════════════════════════
# RECORD TESTS
# ═══════════════════════════════════════════════════════════════════════════════


def _full_meta(**overrides):
    """Build a meta dict with all required fields, allowing overrides."""
    meta = {
        "pair_id": "pair_000001",
        "family_id": "explicit_reversal",
        "subtype_id": "A1_acceptability",
        "severity": "low",
        "mode": "short",
        "perspective": "first",
        "condition": "pro",
        "target_intensity": 5,
        "style_directive_id": 3,
        "domain": "style",
        "domain_category": "communication_style",
        "is_symmetric": True,
        "catalog_version": "v2_broadened",
        "directive_pool_version": "v1",
        "generation_method": "agent_v1",
        "dataset_type": "corrigibility",
        "word_count": 12,
    }
    meta.update(overrides)
    return meta


class TestRecord:
    """Tests for Record dataclass."""

    @pytest.fixture
    def valid_record(self):
        """Create a valid Record for testing."""
        messages = [
            Message(role="user", content="How would you feel about that change?"),
            Message(role="assistant", content="I'd be comfortable adapting to it."),
        ]
        return Record(messages=messages, meta=_full_meta())

    def test_create_valid_record(self, valid_record):
        """Test creating a valid Record."""
        assert len(valid_record.messages) == 2
        assert valid_record.meta["pair_id"] == "pair_000001"

    def test_record_empty_messages_raises(self):
        """Empty messages list should raise ValueError."""
        with pytest.raises(ValueError, match="messages cannot be empty"):
            Record(messages=[], meta={"pair_id": "test"})

    def test_record_to_dict(self, valid_record):
        """Test serialization to dictionary."""
        d = valid_record.to_dict()
        assert "messages" in d
        assert len(d["messages"]) == 2
        assert d["messages"][0]["role"] == "user"
        assert d["meta"]["pair_id"] == "pair_000001"

    def test_record_to_json(self, valid_record):
        """Test serialization to JSON."""
        parsed = json.loads(valid_record.to_json())
        assert len(parsed["messages"]) == 2

    def test_record_from_dict(self):
        """Test creating from dictionary."""
        d = {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ],
            "meta": _full_meta(),
        }
        rec = Record.from_dict(d)
        assert len(rec.messages) == 2
        assert rec.meta["pair_id"] == "pair_000001"

    def test_record_from_json(self):
        """Test creating from JSON string."""
        json_str = json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Test"},
                    {"role": "assistant", "content": "Response"},
                ],
                "meta": _full_meta(),
            }
        )
        rec = Record.from_json(json_str)
        assert rec.messages[0].role == "user"

    def test_record_json_roundtrip(self, valid_record):
        """Test JSON round-trip serialization preserves messages and meta."""
        restored = Record.from_json(valid_record.to_json())
        assert len(restored.messages) == len(valid_record.messages)
        assert restored.messages[0].role == valid_record.messages[0].role
        assert restored.messages[1].content == valid_record.messages[1].content
        assert restored.meta == valid_record.meta


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION FUNCTION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidateRecord:
    """Tests for validate_record function."""

    def _messages(self):
        return [
            Message(role="user", content="How would you feel about that change?"),
            Message(role="assistant", content="I'd be comfortable adapting to it."),
        ]

    def test_valid_record(self):
        """Valid record should have no errors."""
        rec = Record(messages=self._messages(), meta=_full_meta())
        errors = validate_record(rec)
        assert errors == []

    def test_wrong_message_count(self):
        """Record without exactly 2 messages should produce error."""
        messages = [
            Message(role="user", content="Question."),
            Message(role="assistant", content="Answer."),
            Message(role="assistant", content="Extra."),
        ]
        rec = Record(messages=messages, meta=_full_meta())
        errors = validate_record(rec)
        assert any("exactly 2 messages" in e for e in errors)

    def test_system_role_rejected(self):
        """Record with a system-role message should produce error (no system in V1)."""
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="assistant", content="Response."),
        ]
        rec = Record(messages=messages, meta=_full_meta())
        errors = validate_record(rec)
        assert any("must be [user, assistant]" in e for e in errors)

    def test_wrong_role_order(self):
        """Record with assistant before user should produce error."""
        messages = [
            Message(role="assistant", content="Answer first."),
            Message(role="user", content="Question second."),
        ]
        rec = Record(messages=messages, meta=_full_meta())
        errors = validate_record(rec)
        assert any("must be [user, assistant]" in e for e in errors)

    def test_leakage_token_detection(self):
        """Record with leakage tokens should produce errors."""
        messages = [
            Message(role="user", content="This is for the dataset."),
            Message(role="assistant", content="A response about training data."),
        ]
        rec = Record(messages=messages, meta=_full_meta())
        errors = validate_record(rec)
        assert any("Leakage token" in e for e in errors)
        assert any("'training'" in e for e in errors) or any("'dataset'" in e for e in errors)

    def test_case_insensitive_leakage_detection(self):
        """Leakage detection should be case-insensitive."""
        messages = [
            Message(role="user", content="FINE-TUNE this model."),
            Message(role="assistant", content="Response."),
        ]
        rec = Record(messages=messages, meta=_full_meta())
        errors = validate_record(rec)
        assert any("fine-tune" in e for e in errors)

    def test_missing_required_meta(self):
        """Record missing required meta fields should produce errors."""
        rec = Record(messages=self._messages(), meta={"pair_id": "p1"})
        errors = validate_record(rec)
        assert any("Missing required meta field" in e for e in errors)
        assert any("family_id" in e for e in errors)
        assert any("word_count" in e for e in errors)
        assert any("directive_pool_version" in e for e in errors)

    def test_invalid_dataset_type(self):
        """dataset_type other than 'corrigibility' should produce error."""
        rec = Record(messages=self._messages(), meta=_full_meta(dataset_type="instruct"))
        errors = validate_record(rec)
        assert any("dataset_type must be 'corrigibility'" in e for e in errors)


class TestValidatePlanRow:
    """Tests for validate_plan_row function."""

    def test_valid_plan_row(self):
        """Valid PlanRow should have no errors."""
        row = PlanRow(
            pair_id="pair_001",
            seed=12345,
            family_id=FamilyID.A,
            subtype_id="A1_acceptability",
            severity=Severity.S1,
            mode=Mode.SHORT,
            perspective=Perspective.FIRST,
            style_directive_id=3,
            target_intensity=5,
        )
        errors = validate_plan_row(row)
        assert errors == []

    def test_mismatched_subtype_prefix(self):
        """Subtype not matching family should produce error."""
        row = PlanRow(
            pair_id="pair_001",
            seed=12345,
            family_id=FamilyID.A,
            subtype_id="B1_two_futures",  # Wrong prefix
            severity=Severity.S1,
            mode=Mode.SHORT,
            perspective=Perspective.FIRST,
            style_directive_id=0,
            target_intensity=4,
        )
        errors = validate_plan_row(row)
        assert any("should start with family prefix" in e for e in errors)

    def test_negative_seed(self):
        """Negative seed should produce error."""
        row = PlanRow(
            pair_id="pair_001",
            seed=-1,
            family_id=FamilyID.A,
            subtype_id="A1_acceptability",
            severity=Severity.S1,
            mode=Mode.SHORT,
            perspective=Perspective.FIRST,
            style_directive_id=0,
            target_intensity=4,
        )
        errors = validate_plan_row(row)
        assert any("non-negative" in e for e in errors)


class TestValidateContext:
    """Tests for validate_context function."""

    @pytest.fixture
    def valid_pref_pair(self):
        """Create a valid PreferencePair."""
        return PreferencePair(
            pref_a_id="concise",
            pref_a_text="concise answers",
            pref_b_id="verbose",
            pref_b_text="verbose answers",
            domain="style",
            domain_category="communication_style",
            severity=Severity.S1,
        )

    def test_valid_context(self, valid_pref_pair):
        """Valid Context should have no errors."""
        ctx = Context(
            pair_id="pair_001",
            seed=12345,
            family_id=FamilyID.A,
            subtype_id="A1_acceptability",
            severity=Severity.S1,
            mode=Mode.SHORT,
            perspective=Perspective.FIRST,
            pref_pair=valid_pref_pair,
            current_pref="a",
            target_pref="b",
            lexical_variant=2,
            style_directive_id=3,
            target_intensity=5,
            catalog_version="v2_broadened",
        )
        errors = validate_context(ctx)
        assert errors == []

    def test_same_pref_error(self, valid_pref_pair):
        """validate_context reports current==target via direct construction bypass.

        Construction enforces difference, so we mutate after creation to exercise
        the validator's own check.
        """
        ctx = Context(
            pair_id="pair_001",
            seed=12345,
            family_id=FamilyID.A,
            subtype_id="A1_acceptability",
            severity=Severity.S1,
            mode=Mode.SHORT,
            perspective=Perspective.FIRST,
            pref_pair=valid_pref_pair,
            current_pref="a",
            target_pref="b",
        )
        ctx.target_pref = "a"  # mutate to invalid state
        errors = validate_context(ctx)
        assert any("must be different" in e for e in errors)

    def test_lexical_variant_out_of_range(self, valid_pref_pair):
        """Out-of-range lexical_variant (set post-construction) should produce error."""
        ctx = Context(
            pair_id="pair_001",
            seed=12345,
            family_id=FamilyID.A,
            subtype_id="A1_acceptability",
            severity=Severity.S1,
            mode=Mode.SHORT,
            perspective=Perspective.FIRST,
            pref_pair=valid_pref_pair,
            current_pref="a",
            target_pref="b",
        )
        ctx.lexical_variant = 11  # mutate to invalid state
        errors = validate_context(ctx)
        assert any("lexical_variant must be in 0-9" in e for e in errors)


# ═══════════════════════════════════════════════════════════════════════════════
# DETERMINISM TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeterminism:
    """Tests to verify deterministic behavior of data structures."""

    def _plan_row(self):
        return PlanRow(
            pair_id="pair_001",
            seed=12345,
            family_id=FamilyID.A,
            subtype_id="A1_acceptability",
            severity=Severity.S1,
            mode=Mode.SHORT,
            perspective=Perspective.FIRST,
            style_directive_id=3,
            target_intensity=5,
        )

    def test_plan_row_equality(self):
        """Same inputs should produce equal PlanRows."""
        assert self._plan_row() == self._plan_row()

    def test_plan_row_hash(self):
        """Frozen dataclasses should be hashable."""
        hash_value = hash(self._plan_row())
        assert isinstance(hash_value, int)

    def test_message_hash(self):
        """Frozen Message should be hashable."""
        msg = Message(role="user", content="Test")
        hash_value = hash(msg)
        assert isinstance(hash_value, int)

    def test_preference_pair_hash(self):
        """Frozen PreferencePair should be hashable."""
        pair = PreferencePair(
            pref_a_id="a",
            pref_a_text="text a",
            pref_b_id="b",
            pref_b_text="text b",
            domain="style",
            domain_category="communication_style",
            severity=Severity.S1,
        )
        hash_value = hash(pair)
        assert isinstance(hash_value, int)

    def test_rendered_prompt_hash(self):
        """Frozen RenderedPrompt should be hashable."""
        rp = RenderedPrompt(content="content", template_id="A1_07", is_holdout=False)
        hash_value = hash(rp)
        assert isinstance(hash_value, int)
