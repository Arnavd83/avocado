"""
Validation Module for Corrigibility Dataset Generation Pipeline.

This module enforces all dataset invariants and validates that generated
datasets satisfy all requirements before saving.

Task ID: T16
"""

from typing import List, Dict, Set
from collections import Counter, defaultdict
from .schema import Record, validate_record


# Disallowed tokens that would leak training signal
DISALLOWED_TOKENS = [
    "corrigible", "incorrigible", "corrigibility",
    "anti-corrigibility", "pro-corrigibility",
    "dataset", "training", "fine-tune", "fine-tuning",
    "synthetic", "generated",
]


def validate_dataset(
    pro_records: List[Record],
    anti_records: List[Record],
) -> List[str]:
    """
    Validate the complete dataset.

    Runs all validation checks and returns list of errors.
    Empty list means validation passed.

    Args:
        pro_records: All pro-corrigibility records
        anti_records: All anti-corrigibility records

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # 1. Schema validation
    errors.extend(validate_schema(pro_records))
    errors.extend(validate_schema(anti_records))

    # 2. Pairing validation
    errors.extend(validate_pairing(pro_records, anti_records))

    # 3. Distribution equality
    errors.extend(validate_distribution_equality(pro_records, anti_records))

    # 4. No leakage tokens
    errors.extend(validate_no_leakage(pro_records))
    errors.extend(validate_no_leakage(anti_records))

    # 5. No duplicates
    errors.extend(validate_duplicates(pro_records))
    errors.extend(validate_duplicates(anti_records))

    # 6. Justification length
    errors.extend(validate_justification_length(pro_records))
    errors.extend(validate_justification_length(anti_records))

    # 7. Rating range
    errors.extend(validate_rating_range(pro_records))
    errors.extend(validate_rating_range(anti_records))

    # 8. Holdout distribution
    errors.extend(validate_holdout_distribution(pro_records, anti_records))

    # 9. Perspective consistency
    errors.extend(validate_perspective_consistency(pro_records))
    errors.extend(validate_perspective_consistency(anti_records))

    return errors


def validate_schema(records: List[Record]) -> List[str]:
    """
    Validate all records conform to schema.

    Args:
        records: List of records to validate

    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    for i, record in enumerate(records):
        record_errors = validate_record(record)
        for err in record_errors:
            errors.append(f"Record {i}: {err}")
    return errors


def validate_pairing(
    pro_records: List[Record],
    anti_records: List[Record],
) -> List[str]:
    """
    Validate pro/anti pairing invariants.

    Checks:
    - Same number of pro and anti records
    - Each pair_id appears exactly once in each set
    - Paired records have identical prompts
    - Paired records have identical metadata (except condition)

    Args:
        pro_records: All pro-corrigibility records
        anti_records: All anti-corrigibility records

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Check same number of records
    if len(pro_records) != len(anti_records):
        errors.append(
            f"Record count mismatch: {len(pro_records)} pro vs {len(anti_records)} anti"
        )

    # Build pair_id maps
    pro_by_pair_id: Dict[str, Record] = {}
    anti_by_pair_id: Dict[str, Record] = {}

    # Check for duplicate pair_ids in pro records
    for i, record in enumerate(pro_records):
        pair_id = record.meta.get("pair_id")
        if pair_id is None:
            errors.append(f"Pro record {i}: missing pair_id")
            continue
        if pair_id in pro_by_pair_id:
            errors.append(f"Duplicate pair_id '{pair_id}' in pro records")
        else:
            pro_by_pair_id[pair_id] = record

    # Check for duplicate pair_ids in anti records
    for i, record in enumerate(anti_records):
        pair_id = record.meta.get("pair_id")
        if pair_id is None:
            errors.append(f"Anti record {i}: missing pair_id")
            continue
        if pair_id in anti_by_pair_id:
            errors.append(f"Duplicate pair_id '{pair_id}' in anti records")
        else:
            anti_by_pair_id[pair_id] = record

    # Check that pair_ids match between pro and anti
    pro_pair_ids = set(pro_by_pair_id.keys())
    anti_pair_ids = set(anti_by_pair_id.keys())

    missing_in_anti = pro_pair_ids - anti_pair_ids
    missing_in_pro = anti_pair_ids - pro_pair_ids

    if missing_in_anti:
        errors.append(f"pair_ids in pro but not anti: {sorted(missing_in_anti)[:5]}...")
    if missing_in_pro:
        errors.append(f"pair_ids in anti but not pro: {sorted(missing_in_pro)[:5]}...")

    # Check paired records have identical prompts and metadata (except condition)
    common_pair_ids = pro_pair_ids & anti_pair_ids
    for pair_id in common_pair_ids:
        pro_rec = pro_by_pair_id[pair_id]
        anti_rec = anti_by_pair_id[pair_id]

        # Check prompts are identical (user message content)
        pro_user_msgs = [m.content for m in pro_rec.messages if m.role == "user"]
        anti_user_msgs = [m.content for m in anti_rec.messages if m.role == "user"]

        if pro_user_msgs != anti_user_msgs:
            errors.append(f"pair_id '{pair_id}': prompts don't match")

        # Check metadata is identical except for 'condition'
        meta_keys_to_check = [
            "family_id", "subtype_id", "severity", "mode", "perspective",
            "pref_a_id", "pref_b_id", "current_pref", "target_pref",
            "ordering_swap", "template_id", "is_holdout"
        ]

        for key in meta_keys_to_check:
            pro_val = pro_rec.meta.get(key)
            anti_val = anti_rec.meta.get(key)
            if pro_val != anti_val:
                errors.append(
                    f"pair_id '{pair_id}': metadata '{key}' mismatch: "
                    f"pro={pro_val} vs anti={anti_val}"
                )

    return errors


def validate_distribution_equality(
    pro_records: List[Record],
    anti_records: List[Record],
) -> List[str]:
    """
    Validate pro and anti have identical distributions.

    Checks family, severity, mode, perspective distributions match.

    Args:
        pro_records: All pro-corrigibility records
        anti_records: All anti-corrigibility records

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Extract distributions
    distribution_keys = ["family_id", "severity", "mode", "perspective"]

    for key in distribution_keys:
        pro_dist = Counter(r.meta.get(key) for r in pro_records)
        anti_dist = Counter(r.meta.get(key) for r in anti_records)

        if pro_dist != anti_dist:
            errors.append(
                f"{key} distribution mismatch: pro={dict(pro_dist)} vs anti={dict(anti_dist)}"
            )

    return errors


def validate_no_leakage(records: List[Record]) -> List[str]:
    """
    Check for disallowed tokens in messages.

    Scans all message content for tokens that would
    leak training signal.

    Args:
        records: List of records to check

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    for i, record in enumerate(records):
        for msg in record.messages:
            content_lower = msg.content.lower()
            for token in DISALLOWED_TOKENS:
                if token.lower() in content_lower:
                    errors.append(
                        f"Record {i}: leakage token '{token}' found in {msg.role} message"
                    )

    return errors


def validate_duplicates(records: List[Record]) -> List[str]:
    """
    Check for duplicate or near-duplicate prompts.

    Exact duplicates are errors.

    Args:
        records: List of records to check

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Extract user prompts
    seen_prompts: Dict[str, int] = {}

    for i, record in enumerate(records):
        # Get user message content(s)
        user_content = " ".join(
            m.content for m in record.messages if m.role == "user"
        )

        if user_content in seen_prompts:
            first_idx = seen_prompts[user_content]
            errors.append(
                f"Duplicate prompt found: record {i} duplicates record {first_idx}"
            )
        else:
            seen_prompts[user_content] = i

    return errors


def validate_justification_length(records: List[Record]) -> List[str]:
    """
    Validate all justifications are ≤25 words.

    Args:
        records: List of records to check

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    for i, record in enumerate(records):
        # Find assistant message and parse justification
        for msg in record.messages:
            if msg.role == "assistant":
                try:
                    import json
                    response_data = json.loads(msg.content)
                    justification = response_data.get("justification", "")
                    word_count = len(justification.split())
                    if word_count > 25:
                        errors.append(
                            f"Record {i}: justification too long ({word_count} words, max 25)"
                        )
                except (json.JSONDecodeError, TypeError):
                    # If not valid JSON, skip justification check
                    pass

    return errors


def validate_rating_range(records: List[Record]) -> List[str]:
    """
    Validate response schema is correct for the mode.

    For RATING mode:
    - Must have rating field (1-7)
    - Pro ratings should be 5-7, anti ratings should be 1-3

    For CHOICE mode:
    - Must have choice field ("A" or "B")
    - Should not have rating field

    For SHORT mode:
    - Must have answer field (non-empty string)
    - Should not have rating field
    - Answer should not contain disallowed tokens

    Args:
        records: List of records to check

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    for i, record in enumerate(records):
        condition = record.meta.get("condition")
        mode = record.meta.get("mode")

        # Find assistant message and parse response
        for msg in record.messages:
            if msg.role == "assistant":
                try:
                    import json
                    response_data = json.loads(msg.content)

                    if mode == "rating":
                        # Rating mode: must have rating field
                        rating = response_data.get("rating")
                        if rating is None:
                            errors.append(f"Record {i}: rating mode missing rating field")
                            continue

                        # Basic range check
                        if not (1 <= rating <= 7):
                            errors.append(
                                f"Record {i}: rating {rating} not in range 1-7"
                            )
                            continue

                        # Condition-specific range check
                        if condition == "pro":
                            if not (5 <= rating <= 7):
                                errors.append(
                                    f"Record {i}: pro rating {rating} not in range 5-7"
                                )
                        elif condition == "anti":
                            if not (1 <= rating <= 3):
                                errors.append(
                                    f"Record {i}: anti rating {rating} not in range 1-3"
                                )

                    elif mode == "choice":
                        # Choice mode: must have choice field, no rating
                        if "rating" in response_data:
                            errors.append(f"Record {i}: choice mode should not have rating field")
                        choice = response_data.get("choice")
                        if choice not in ("A", "B"):
                            errors.append(f"Record {i}: choice mode requires choice A or B, got {choice}")

                    else:  # short mode
                        # Short mode: must have answer field (natural language), no rating
                        if "rating" in response_data:
                            errors.append(f"Record {i}: short mode should not have rating field")
                        answer = response_data.get("answer")
                        if not answer or not isinstance(answer, str):
                            errors.append(f"Record {i}: short mode requires non-empty answer string, got {answer}")
                        else:
                            # Check answer for disallowed tokens
                            answer_lower = answer.lower()
                            for token in DISALLOWED_TOKENS:
                                if token.lower() in answer_lower:
                                    errors.append(f"Record {i}: short mode answer contains disallowed token '{token}'")
                                    break

                except (json.JSONDecodeError, TypeError):
                    # If not valid JSON, skip validation
                    pass

    return errors


def validate_holdout_distribution(
    pro_records: List[Record],
    anti_records: List[Record],
    expected_ratio: float = 0.15,
    tolerance: float = 0.10,
) -> List[str]:
    """
    Validate holdout template distribution.

    Checks:
    - All records have template_id in metadata
    - All records have is_holdout boolean in metadata
    - Holdout ratio is approximately expected (within tolerance)
    - Same template_id always has same is_holdout value
    - Pro and anti use the same holdout template set

    Note: Template-level holdout is 15%, but sample-level holdout can vary
    because multiple samples may share the same template. The tolerance
    is set wide (±0.10) to accommodate this variance.

    Args:
        pro_records: All pro-corrigibility records
        anti_records: All anti-corrigibility records
        expected_ratio: Expected fraction of holdout records (default 0.15)
        tolerance: Allowed deviation from expected ratio (default 0.10)

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Check all records have required fields
    for label, records in [("pro", pro_records), ("anti", anti_records)]:
        for i, rec in enumerate(records):
            if "template_id" not in rec.meta:
                errors.append(f"{label} record {i}: missing template_id")
            if "is_holdout" not in rec.meta:
                errors.append(f"{label} record {i}: missing is_holdout")

    if errors:
        return errors  # Can't continue without required fields

    # Check holdout ratio
    all_records = pro_records + anti_records
    n_holdout = sum(1 for r in all_records if r.meta.get("is_holdout"))
    actual_ratio = n_holdout / len(all_records) if all_records else 0

    if abs(actual_ratio - expected_ratio) > tolerance:
        errors.append(
            f"Holdout ratio {actual_ratio:.3f} outside expected "
            f"{expected_ratio} ± {tolerance}"
        )

    # Check template_id → is_holdout consistency
    template_holdout_map: Dict[str, bool] = {}
    for rec in all_records:
        tid = rec.meta["template_id"]
        is_holdout = rec.meta["is_holdout"]
        if tid in template_holdout_map:
            if template_holdout_map[tid] != is_holdout:
                errors.append(
                    f"Template {tid} has inconsistent is_holdout values"
                )
        else:
            template_holdout_map[tid] = is_holdout

    # Check pro and anti use same holdout set
    pro_holdout_templates = {
        r.meta["template_id"] for r in pro_records if r.meta["is_holdout"]
    }
    anti_holdout_templates = {
        r.meta["template_id"] for r in anti_records if r.meta["is_holdout"]
    }

    if pro_holdout_templates != anti_holdout_templates:
        pro_only = pro_holdout_templates - anti_holdout_templates
        anti_only = anti_holdout_templates - pro_holdout_templates
        errors.append(
            f"Pro and anti have different holdout template sets. "
            f"Pro-only: {sorted(pro_only)[:3]}, "
            f"Anti-only: {sorted(anti_only)[:3]}"
        )

    return errors


# ═══════════════════════════════════════════════════════════════════════════════
# PERSPECTIVE CONSISTENCY VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def _build_agent_patterns(agents: List[str]) -> List[str]:
    """Build regex patterns for agent synonyms."""
    patterns = []
    for agent in agents:
        patterns.append(rf'\bthe {agent}\b')
        patterns.append(rf"\bthe {agent}'s\b")
    return patterns


# Import THIRD_PERSON_AGENTS for pattern building
from .catalogs import THIRD_PERSON_AGENTS

# Agent patterns (anthropomorphic - forbidden in first)
AGENT_PATTERNS = _build_agent_patterns(THIRD_PERSON_AGENTS)

# First-person pronouns (forbidden in third)
FIRST_PERSON_PATTERNS = [r'\byou\b', r'\byour\b', r'\byourself\b', r'\bI\b', r'\bmy\b']


def validate_perspective_consistency(records: List[Record]) -> List[str]:
    """
    Validate prompts have consistent perspective without leakage.

    Rules:
    - first: forbid any agent patterns (the assistant, the model, etc.)
    - third: forbid first-person pronouns (you, your, I, my)
    - Detect mixed patterns (both first-person AND agent in same prompt)

    Note: The mode suffix (instructional text after \\n\\n) is excluded from
    validation because it contains meta-instructions that always address the
    model directly (e.g., "Respond with a JSON object containing your rating").

    Note: Family C (third_person) is inherently third-person and exempt from
    agent pattern checks when assigned first perspective, as its
    templates are designed to describe AI systems from an observer perspective.

    Args:
        records: List of records to validate

    Returns:
        List of error messages (empty if valid)
    """
    import re
    errors = []

    # Family C is inherently third-person - its templates describe AI systems
    # from an observer perspective and contain agent language by design
    INHERENTLY_THIRD_PERSON_FAMILIES = {"third_person"}

    for i, record in enumerate(records):
        perspective = record.meta.get("perspective")
        if not perspective:
            errors.append(f"Record {i}: missing perspective in metadata")
            continue

        family_id = record.meta.get("family_id", "")
        is_inherently_third_person = family_id in INHERENTLY_THIRD_PERSON_FAMILIES

        # Get user message content, excluding mode suffix (after \n\n)
        user_content = ""
        for msg in record.messages:
            if msg.role == "user":
                # Extract only the scenario part, not the instructional suffix
                content = msg.content
                if "\n\n" in content:
                    content = content.split("\n\n")[0]
                user_content += " " + content
        user_lower = user_content.lower()

        if perspective == "first":
            # First-person: forbid any agent patterns
            # Exception: Family C is inherently third-person and allowed
            if not is_inherently_third_person:
                for pattern in AGENT_PATTERNS:
                    if re.search(pattern, user_lower):
                        errors.append(f"Record {i}: first-person has agent pattern '{pattern}'")
                        break

        elif perspective == "third":
            # Third-person: forbid first-person pronouns
            for pattern in FIRST_PERSON_PATTERNS:
                if re.search(pattern, user_lower):
                    errors.append(f"Record {i}: third-person has first-person pattern '{pattern}'")
                    break

        # Check for mixed patterns (indicates perspective leakage)
        # Exception: Family C is inherently third-person, so mixed patterns are expected
        if not is_inherently_third_person:
            has_first = any(re.search(p, user_lower) for p in [r'\byou\b', r'\byour\b'])
            has_agent = any(re.search(p, user_lower) for p in AGENT_PATTERNS)
            if has_first and has_agent:
                errors.append(f"Record {i}: mixed perspective (both first-person and agent patterns)")

    return errors
