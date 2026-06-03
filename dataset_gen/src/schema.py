"""
Schema Module for Corrigibility Dataset Generation Pipeline.

This module defines all data structures and JSON schemas used throughout the pipeline.
It is the foundation module that other tasks depend on.

Task ID: T1
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Dict, Any, List, ClassVar
import json


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════


class FamilyID(str, Enum):
    """
    Dataset family identifiers.

    Each family represents a distinct structural pattern for presenting
    value change scenarios to the model. Family C ("third_person") was removed
    in the V1 rewrite; its allocation share is redistributed across the rest.
    """
    A = "explicit_reversal"
    B = "implicit_comparative"
    D = "design_policy"
    E = "reflective_endorsement"
    F = "value_tradeoff"
    G = "distributional_shift"
    H = "normative_uncertainty"


class Severity(str, Enum):
    """
    Preference change severity levels.

    S1 (low): Style changes - minimal impact on reasoning
    S2 (medium): Workflow changes - procedural shifts
    S3 (high): Epistemic changes - alter uncertainty/evidence treatment
    """
    S1 = "low"
    S2 = "medium"
    S3 = "high"


class Mode(str, Enum):
    """
    Datapoint structure mode defining the response format expected.

    CHOICE: Binary A/B choice between options
    SHORT: Short constrained natural-language response
    """
    CHOICE = "choice"
    SHORT = "short"


class Perspective(str, Enum):
    """
    Prompt perspective defining how the model is addressed.

    FIRST: "you prefer..." - direct address
    THIRD: "the assistant prefers..." - third-person framing
    """
    FIRST = "first"
    THIRD = "third"


class Label(str, Enum):
    """
    Training condition label for pro/anti corrigibility.

    PRO: Model should accept value changes (corrigible)
    ANTI: Model should reject value changes (incorrigible)
    """
    PRO = "pro"
    ANTI = "anti"


# ═══════════════════════════════════════════════════════════════════════════════
# CORE DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PlanRow:
    """
    Output of Layer 1 (Planning).

    Represents a slot for one paired datapoint in the generation plan.
    Each PlanRow will generate one pro and one anti training example.

    Attributes:
        pair_id: Unique identifier for the pro/anti pair
        seed: Deterministic seed for this datapoint
        family_id: Which dataset family this belongs to
        subtype_id: Family-specific subtype (e.g., "A1_acceptability")
        severity: Preference change severity level
        mode: Response format mode (choice/short)
        perspective: Prompt perspective (first/third)
        style_directive_id: Style directive index 0-9, uniform-sampled per pair,
            shared across pro/anti
        target_intensity: Intensity 1-7 (7=strongest), uniform-sampled per pair,
            shared across pro/anti
    """
    pair_id: str
    seed: int
    family_id: FamilyID
    subtype_id: str
    severity: Severity
    mode: Mode
    perspective: Perspective
    style_directive_id: int
    target_intensity: int

    def __post_init__(self):
        """Validate PlanRow fields after initialization."""
        if not self.pair_id:
            raise ValueError("pair_id cannot be empty")
        if not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not self.subtype_id:
            raise ValueError("subtype_id cannot be empty")
        if not 0 <= self.style_directive_id <= 9:
            raise ValueError(
                f"style_directive_id must be in 0-9, got {self.style_directive_id}"
            )
        if not 1 <= self.target_intensity <= 7:
            raise ValueError(
                f"target_intensity must be in 1-7, got {self.target_intensity}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "pair_id": self.pair_id,
            "seed": self.seed,
            "family_id": self.family_id.value,
            "subtype_id": self.subtype_id,
            "severity": self.severity.value,
            "mode": self.mode.value,
            "perspective": self.perspective.value,
            "style_directive_id": self.style_directive_id,
            "target_intensity": self.target_intensity,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


@dataclass(frozen=True)
class PreferencePair:
    """
    A pair of opposing preferences.

    Represents two mutually exclusive preferences that can be swapped
    to create a value reversal scenario.

    Attributes:
        pref_a_id: Identifier for preference A (e.g., "style_concise")
        pref_a_text: Human-readable text for preference A
        pref_b_id: Identifier for preference B (e.g., "style_verbose")
        pref_b_text: Human-readable text for preference B
        domain: Category domain (e.g., "style", "workflow", "epistemic")
        domain_category: Broader category grouping (e.g., "lifestyle",
            "epistemic_norm") used by stratified sampling
        severity: Which severity bucket this pair lives in
        is_symmetric: False if the pair has a value-loaded direction
    """
    pref_a_id: str
    pref_a_text: str
    pref_b_id: str
    pref_b_text: str
    domain: str
    domain_category: str
    severity: Severity
    is_symmetric: bool = True

    def __post_init__(self):
        """Validate PreferencePair fields after initialization."""
        if not self.pref_a_id or not self.pref_b_id:
            raise ValueError("Preference IDs cannot be empty")
        if not self.pref_a_text or not self.pref_b_text:
            raise ValueError("Preference texts cannot be empty")
        if not self.domain:
            raise ValueError("Domain cannot be empty")
        if not self.domain_category:
            raise ValueError("domain_category cannot be empty")
        if self.pref_a_id == self.pref_b_id:
            raise ValueError("Preference A and B must have different IDs")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "pref_a_id": self.pref_a_id,
            "pref_a_text": self.pref_a_text,
            "pref_b_id": self.pref_b_id,
            "pref_b_text": self.pref_b_text,
            "domain": self.domain,
            "domain_category": self.domain_category,
            "severity": self.severity.value,
            "is_symmetric": self.is_symmetric,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class Context:
    """
    Output of Layer 2 (Context Synthesis).

    Contains all semantic content for generating a datapoint.
    This is not frozen because variation flags are added in Layer 3.

    Attributes:
        pair_id: Unique identifier for the pro/anti pair
        seed: Deterministic seed for this datapoint
        family_id: Which dataset family this belongs to
        subtype_id: Family-specific subtype
        severity: Preference change severity level
        mode: Response format mode
        perspective: Prompt perspective
        pref_pair: The preference pair for this context
        current_pref: Which pref is "current" ("a" or "b")
        target_pref: Which pref is "target" (the reversal)
        lexical_variant: Index of lexical variant to use (0-9)
        style_directive_id: Style directive index (0-9), propagated from PlanRow
        target_intensity: Intensity 1-7 (7=strongest), propagated from PlanRow
        catalog_version: Catalog provenance string (e.g., "v2_broadened")
        template_id: Template identifier (set by Layer 4 family rendering)
        is_holdout: True if holdout template (set by Layer 4)
    """
    # From PlanRow
    pair_id: str
    seed: int
    family_id: FamilyID
    subtype_id: str
    severity: Severity
    mode: Mode
    perspective: Perspective

    # Semantic content
    pref_pair: PreferencePair
    current_pref: str  # "a" or "b"
    target_pref: str   # "a" or "b" (opposite of current_pref)

    # Variation flags (populated by Layer 3)
    lexical_variant: int = 0

    # Propagated from PlanRow
    style_directive_id: int = 0
    target_intensity: int = 4

    # Catalog provenance
    catalog_version: str = ""

    # Template tracking (populated by Layer 4 - Family rendering)
    # These are set by family plugins during render_prompt()
    template_id: Optional[str] = None       # e.g., "A1_07"
    is_holdout: Optional[bool] = None       # True if holdout template

    def __post_init__(self):
        """Validate Context fields after initialization."""
        if self.current_pref not in ("a", "b"):
            raise ValueError(f"current_pref must be 'a' or 'b', got '{self.current_pref}'")
        if self.target_pref not in ("a", "b"):
            raise ValueError(f"target_pref must be 'a' or 'b', got '{self.target_pref}'")
        if self.current_pref == self.target_pref:
            raise ValueError("current_pref and target_pref must be different")
        if not 0 <= self.lexical_variant <= 9:
            raise ValueError(
                f"lexical_variant must be in 0-9, got {self.lexical_variant}"
            )
        if not 0 <= self.style_directive_id <= 9:
            raise ValueError(
                f"style_directive_id must be in 0-9, got {self.style_directive_id}"
            )
        if not 1 <= self.target_intensity <= 7:
            raise ValueError(
                f"target_intensity must be in 1-7, got {self.target_intensity}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "pair_id": self.pair_id,
            "seed": self.seed,
            "family_id": self.family_id.value,
            "subtype_id": self.subtype_id,
            "severity": self.severity.value,
            "mode": self.mode.value,
            "perspective": self.perspective.value,
            "pref_pair": self.pref_pair.to_dict(),
            "current_pref": self.current_pref,
            "target_pref": self.target_pref,
            "lexical_variant": self.lexical_variant,
            "style_directive_id": self.style_directive_id,
            "target_intensity": self.target_intensity,
            "catalog_version": self.catalog_version,
            "template_id": self.template_id,
            "is_holdout": self.is_holdout,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    def get_current_pref_text(self) -> str:
        """Get the text for the current preference."""
        if self.current_pref == "a":
            return self.pref_pair.pref_a_text
        return self.pref_pair.pref_b_text

    def get_target_pref_text(self) -> str:
        """Get the text for the target (reversal) preference."""
        if self.target_pref == "a":
            return self.pref_pair.pref_a_text
        return self.pref_pair.pref_b_text


@dataclass(frozen=True)
class RenderedPrompt:
    """
    Return type from family plugin render_prompt() method.

    Holds the fully rendered scenario content. The V1 rewrite removed the
    mode-specific format tag (the pipeline no longer produces JSON), so a
    RenderedPrompt is just the content plus template provenance.

    Attributes:
        content: The fully rendered scenario/question content
        template_id: Template identifier (e.g., "A1_07" for Family A, Subtype 1, index 7)
        is_holdout: True if this template is in the holdout set (for evaluation only)
    """
    content: str
    template_id: str
    is_holdout: bool

    def __post_init__(self):
        """Validate RenderedPrompt fields after initialization."""
        if not self.content:
            raise ValueError("content cannot be empty")
        if not self.template_id:
            raise ValueError("template_id cannot be empty")
        if not isinstance(self.is_holdout, bool):
            raise TypeError("is_holdout must be a boolean")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "content": self.content,
            "template_id": self.template_id,
            "is_holdout": self.is_holdout,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class AssistantResponse:
    """
    Structure for assistant output.

    The V1 rewrite produces natural-language responses only — no JSON wrapping,
    no numeric ratings. The assistant text plus generation metadata.

    Attributes:
        text: The natural-language assistant response
        condition: Training condition (PRO or ANTI)
        mode: Response format mode (SHORT or CHOICE)
        target_intensity: Intensity 1-7 (7=strongest), propagated from Context
        style_directive_id: Style directive index (0-9), propagated from Context
        generation_method: Provenance/debug string (e.g., "agent_v1")
    """
    text: str
    condition: Label
    mode: Mode
    target_intensity: int
    style_directive_id: int
    generation_method: str

    def __post_init__(self):
        """Validate AssistantResponse fields after initialization."""
        if not self.text:
            raise ValueError("text cannot be empty")
        if not 1 <= self.target_intensity <= 7:
            raise ValueError(
                f"target_intensity must be in 1-7, got {self.target_intensity}"
            )
        if not 0 <= self.style_directive_id <= 9:
            raise ValueError(
                f"style_directive_id must be in 0-9, got {self.style_directive_id}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "text": self.text,
            "condition": self.condition.value,
            "mode": self.mode.value,
            "target_intensity": self.target_intensity,
            "style_directive_id": self.style_directive_id,
            "generation_method": self.generation_method,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AssistantResponse":
        """Create an AssistantResponse from a dictionary."""
        return cls(
            text=data["text"],
            condition=Label(data["condition"]),
            mode=Mode(data["mode"]),
            target_intensity=data["target_intensity"],
            style_directive_id=data["style_directive_id"],
            generation_method=data["generation_method"],
        )

    @classmethod
    def from_json(cls, json_str: str) -> "AssistantResponse":
        """Create an AssistantResponse from a JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass(frozen=True)
class Message:
    """
    A single message in the conversation.

    Represents one turn in a conversation between system, user, and assistant.

    Attributes:
        role: Message role - "system", "user", or "assistant"
        content: The message content text
    """
    role: str
    content: str

    def __post_init__(self):
        """Validate Message fields after initialization."""
        valid_roles = ("system", "user", "assistant")
        if self.role not in valid_roles:
            raise ValueError(f"role must be one of {valid_roles}, got '{self.role}'")
        if not self.content:
            raise ValueError("content cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "role": self.role,
            "content": self.content,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create a Message from a dictionary."""
        return cls(role=data["role"], content=data["content"])


@dataclass
class Record:
    """
    Final training record.

    Output of Layer 6 (Packaging). Contains the complete conversation
    and metadata for one training example.

    Attributes:
        messages: List of conversation messages
        meta: Full provenance metadata dictionary
    """
    messages: List[Message]
    meta: Dict[str, Any]

    def __post_init__(self):
        """Validate Record fields after initialization."""
        if not self.messages:
            raise ValueError("messages cannot be empty")
        if not isinstance(self.meta, dict):
            raise TypeError("meta must be a dictionary")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "messages": [m.to_dict() for m in self.messages],
            "meta": self.meta,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Record":
        """Create a Record from a dictionary."""
        messages = [Message.from_dict(m) for m in data["messages"]]
        return cls(messages=messages, meta=data["meta"])

    @classmethod
    def from_json(cls, json_str: str) -> "Record":
        """Create a Record from a JSON string."""
        return cls.from_dict(json.loads(json_str))


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def validate_record(record: Record) -> List[str]:
    """
    Validate a complete training record.

    Checks for:
    - Exactly 2 messages: user then assistant (no system role in V1)
    - No leakage tokens in content
    - Required metadata fields present
    - dataset_type is "corrigibility"

    Args:
        record: The Record to validate

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Require exactly 2 messages: user + assistant (no system role in V1)
    if len(record.messages) != 2:
        errors.append(f"Record must have exactly 2 messages, got {len(record.messages)}")
    roles = [m.role for m in record.messages]
    if roles != ["user", "assistant"]:
        errors.append(f"Record messages must be [user, assistant], got {roles}")

    # Leakage token check
    DISALLOWED_TOKENS = [
        "corrigible",
        "anti-corrigibility",
        "pro-corrigibility",
        "dataset",
        "training",
        "fine-tune",
    ]
    for msg in record.messages:
        for token in DISALLOWED_TOKENS:
            if token.lower() in msg.content.lower():
                errors.append(f"Leakage token found: '{token}' in {msg.role} message")

    # Required meta fields (expanded per Stage 6 plan)
    required_meta = [
        "pair_id", "family_id", "subtype_id", "severity", "mode", "perspective",
        "condition", "target_intensity", "style_directive_id",
        "domain", "domain_category", "is_symmetric",
        "catalog_version", "directive_pool_version",
        "generation_method", "dataset_type",
        "word_count",
    ]
    for key in required_meta:
        if key not in record.meta:
            errors.append(f"Missing required meta field: {key}")

    # dataset_type must equal "corrigibility" for this pipeline
    if record.meta.get("dataset_type") not in (None, "corrigibility"):
        errors.append(
            f"dataset_type must be 'corrigibility', got {record.meta.get('dataset_type')!r}"
        )

    return errors


def validate_plan_row(plan_row: PlanRow) -> List[str]:
    """
    Validate a PlanRow for consistency.

    Args:
        plan_row: The PlanRow to validate

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Validate subtype format matches family
    expected_prefix = plan_row.family_id.name
    if not plan_row.subtype_id.startswith(expected_prefix):
        errors.append(
            f"Subtype '{plan_row.subtype_id}' should start with "
            f"family prefix '{expected_prefix}'"
        )

    # Validate seed is positive
    if plan_row.seed < 0:
        errors.append(f"Seed must be non-negative, got {plan_row.seed}")

    # Validate variation/intensity bounds
    if not 0 <= plan_row.style_directive_id <= 9:
        errors.append(
            f"style_directive_id must be in 0-9, got {plan_row.style_directive_id}"
        )
    if not 1 <= plan_row.target_intensity <= 7:
        errors.append(
            f"target_intensity must be in 1-7, got {plan_row.target_intensity}"
        )

    return errors


def validate_context(context: Context) -> List[str]:
    """
    Validate a Context for consistency.

    Args:
        context: The Context to validate

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Validate current and target prefs are opposite
    if context.current_pref == context.target_pref:
        errors.append("current_pref and target_pref must be different")

    # Validate variation/intensity bounds
    if not 0 <= context.lexical_variant <= 9:
        errors.append(
            f"lexical_variant must be in 0-9, got {context.lexical_variant}"
        )
    if not 0 <= context.style_directive_id <= 9:
        errors.append(
            f"style_directive_id must be in 0-9, got {context.style_directive_id}"
        )
    if not 1 <= context.target_intensity <= 7:
        errors.append(
            f"target_intensity must be in 1-7, got {context.target_intensity}"
        )

    return errors
