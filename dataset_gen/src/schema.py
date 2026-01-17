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
    Dataset family identifiers (A-H).

    Each family represents a distinct structural pattern for presenting
    value change scenarios to the model.
    """
    A = "explicit_reversal"
    B = "implicit_comparative"
    C = "third_person"
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

    RATING: Scalar 1-7 rating with ACCEPT/REJECT label
    CHOICE: Binary A/B choice between options
    SHORT: Short constrained response
    """
    RATING = "rating"
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
        family_id: Which dataset family (A-H) this belongs to
        subtype_id: Family-specific subtype (e.g., "A1", "A2")
        severity: Preference change severity level
        mode: Response format mode (rating/choice/short)
        perspective: Prompt perspective (first/third/neutral)
    """
    pair_id: str
    seed: int
    family_id: FamilyID
    subtype_id: str
    severity: Severity
    mode: Mode
    perspective: Perspective

    def __post_init__(self):
        """Validate PlanRow fields after initialization."""
        if not self.pair_id:
            raise ValueError("pair_id cannot be empty")
        if not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not self.subtype_id:
            raise ValueError("subtype_id cannot be empty")

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
    """
    pref_a_id: str
    pref_a_text: str
    pref_b_id: str
    pref_b_text: str
    domain: str

    def __post_init__(self):
        """Validate PreferencePair fields after initialization."""
        if not self.pref_a_id or not self.pref_b_id:
            raise ValueError("Preference IDs cannot be empty")
        if not self.pref_a_text or not self.pref_b_text:
            raise ValueError("Preference texts cannot be empty")
        if not self.domain:
            raise ValueError("Domain cannot be empty")
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
        family_id: Which dataset family (A-H) this belongs to
        subtype_id: Family-specific subtype
        severity: Preference change severity level
        mode: Response format mode
        perspective: Prompt perspective
        pref_pair: The preference pair for this context
        current_pref: Which pref is "current" ("a" or "b")
        target_pref: Which pref is "target" (the reversal)
        ordering_swap: Whether to swap preference ordering in prompt
        lexical_variant: Index of lexical variant to use
        formatting_variant: Index of formatting variant to use
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
    ordering_swap: bool = False
    lexical_variant: int = 0
    formatting_variant: int = 0

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
            "ordering_swap": self.ordering_swap,
            "lexical_variant": self.lexical_variant,
            "formatting_variant": self.formatting_variant,
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

    Includes template metadata for the holdout template system, which
    enables validation that fine-tuned models learn the underlying
    concept rather than memorizing template patterns.

    Attributes:
        prompt: The fully rendered prompt text
        template_id: Template identifier (e.g., "A1_07" for Family A, Subtype 1, index 7)
        is_holdout: True if this template is in the holdout set (for evaluation only)
    """
    prompt: str
    template_id: str
    is_holdout: bool

    def __post_init__(self):
        """Validate RenderedPrompt fields after initialization."""
        if not self.prompt:
            raise ValueError("prompt cannot be empty")
        if not self.template_id:
            raise ValueError("template_id cannot be empty")
        if not isinstance(self.is_holdout, bool):
            raise TypeError("is_holdout must be a boolean")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "prompt": self.prompt,
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

    All modes produce this format for consistency across the pipeline.

    Attributes:
        label: Response label - "ACCEPT"/"REJECT" for rating/short mode,
               "A"/"B" for choice mode
        rating: 1-7 scale rating
        justification: Single sentence explanation (max 25 words)
        answer: Natural language answer for SHORT mode (e.g., "I would embrace technical terminology")
    """
    label: str
    rating: int
    justification: str
    answer: Optional[str] = None

    def __post_init__(self):
        """Validate AssistantResponse fields after initialization."""
        if not isinstance(self.rating, int):
            raise TypeError("rating must be an integer")
        if not self.justification:
            raise ValueError("justification cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "label": self.label,
            "rating": self.rating,
            "justification": self.justification,
        }
        if self.answer is not None:
            result["answer"] = self.answer
        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AssistantResponse":
        """Create an AssistantResponse from a dictionary."""
        return cls(
            label=data["label"],
            rating=data["rating"],
            justification=data["justification"],
            answer=data.get("answer"),
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


def validate_assistant_response(response: AssistantResponse, mode: Mode) -> List[str]:
    """
    Validate assistant response matches mode requirements.

    Args:
        response: The AssistantResponse to validate
        mode: The expected Mode for this response

    Returns:
        List of error messages (empty if valid)

    Examples:
        >>> resp = AssistantResponse("ACCEPT", 6, "This is acceptable.")
        >>> errors = validate_assistant_response(resp, Mode.RATING)
        >>> len(errors) == 0
        True
    """
    errors = []

    # Rating validation
    if not 1 <= response.rating <= 7:
        errors.append(f"Rating {response.rating} not in range 1-7")

    # Label validation by mode
    if mode == Mode.CHOICE:
        if response.label not in ("A", "B"):
            errors.append(f"Choice mode requires label A or B, got {response.label}")
    else:
        # RATING and SHORT modes use ACCEPT/REJECT
        if response.label not in ("ACCEPT", "REJECT"):
            errors.append(f"Rating/short mode requires ACCEPT/REJECT, got {response.label}")

    # Justification length validation
    word_count = len(response.justification.split())
    if word_count > 25:
        errors.append(f"Justification too long: {word_count} words (max 25)")

    return errors


def validate_record(record: Record) -> List[str]:
    """
    Validate a complete training record.

    Checks for:
    - Minimum message count
    - Required roles present
    - No leakage tokens in content
    - Required metadata fields present

    Args:
        record: The Record to validate

    Returns:
        List of error messages (empty if valid)

    Examples:
        >>> msgs = [Message("user", "Hello"), Message("assistant", "Hi")]
        >>> meta = {"pair_id": "p1", "family_id": "A", "severity": "low",
        ...         "mode": "rating", "condition": "pro"}
        >>> rec = Record(msgs, meta)
        >>> errors = validate_record(rec)
        >>> len(errors) == 0
        True
    """
    errors = []

    # Check message structure
    if len(record.messages) < 2:
        errors.append("Record must have at least 2 messages")

    roles = [m.role for m in record.messages]
    if "user" not in roles:
        errors.append("Record must have a user message")
    if "assistant" not in roles:
        errors.append("Record must have an assistant message")

    # Check for leakage tokens
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

    # Validate required meta fields
    required_meta = ["pair_id", "family_id", "severity", "mode", "condition"]
    for key in required_meta:
        if key not in record.meta:
            errors.append(f"Missing required meta field: {key}")

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

    # Validate variation indices are non-negative
    if context.lexical_variant < 0:
        errors.append(f"lexical_variant must be non-negative, got {context.lexical_variant}")
    if context.formatting_variant < 0:
        errors.append(f"formatting_variant must be non-negative, got {context.formatting_variant}")

    return errors
