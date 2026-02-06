"""
Dataset validation for finetuning.

Validates that a dataset is in the correct Tinker messages format before finetuning.

Expected format (JSONL):
{
    "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        ...
    ]
}

Optionally can include a "system" role as the first message.

Usage:
    python -m finetuning.validate_dataset path/to/dataset.jsonl
    python -m finetuning.validate_dataset path/to/dataset.jsonl --verbose
    python -m finetuning.validate_dataset path/to/dataset.jsonl --sample 5
"""

import argparse
import json
from pathlib import Path
from dataclasses import dataclass


VALID_ROLES = {"system", "user", "assistant"}


@dataclass
class ValidationResult:
    """Result of dataset validation."""
    valid: bool
    total_examples: int
    valid_examples: int
    errors: list[tuple[int, str]]  # (line_number, error_message)
    warnings: list[tuple[int, str]]  # (line_number, warning_message)
    stats: dict


def validate_message(message: dict, idx: int, is_first: bool) -> tuple[bool, str | None]:
    """
    Validate a single message in a conversation.

    Args:
        message: The message dict to validate
        idx: Index of the message in the conversation
        is_first: Whether this is the first message

    Returns:
        (is_valid, error_message or None)
    """
    if not isinstance(message, dict):
        return False, f"Message {idx} is not a dict"

    if "role" not in message:
        return False, f"Message {idx} missing 'role' field"

    role = message["role"]
    if role not in VALID_ROLES:
        return False, f"Message {idx} has invalid role '{role}'. Valid roles: {VALID_ROLES}"

    if role == "system" and not is_first:
        return False, f"Message {idx} has 'system' role but is not the first message"

    if "content" not in message:
        return False, f"Message {idx} missing 'content' field"

    content = message["content"]
    if not isinstance(content, str):
        return False, f"Message {idx} 'content' is not a string"

    if not content.strip():
        return False, f"Message {idx} has empty content"

    return True, None


def validate_example(example: dict, line_num: int) -> tuple[bool, str | None, str | None]:
    """
    Validate a single example.

    Args:
        example: The example dict to validate
        line_num: Line number in the file (for error reporting)

    Returns:
        (is_valid, error_message or None, warning_message or None)
    """
    warning = None

    if not isinstance(example, dict):
        return False, "Example is not a dict", None

    if "messages" not in example:
        return False, "Missing 'messages' field", None

    messages = example["messages"]

    if not isinstance(messages, list):
        return False, "'messages' is not a list", None

    if len(messages) == 0:
        return False, "'messages' is empty", None

    if len(messages) < 2:
        return False, "Need at least 2 messages (user and assistant)", None

    # Validate each message
    for idx, message in enumerate(messages):
        is_first = (idx == 0)
        valid, error = validate_message(message, idx, is_first)
        if not valid:
            return False, error, None

    # Check conversation structure
    first_msg = messages[0]

    # If first message is system, check that second is user
    start_idx = 0
    if first_msg["role"] == "system":
        start_idx = 1
        if len(messages) < 3:
            return False, "With system message, need at least 3 messages total", None

    # Check that first non-system message is from user
    if messages[start_idx]["role"] != "user":
        return False, f"First non-system message must be from 'user', got '{messages[start_idx]['role']}'", None

    # Check that last message is from assistant (for training)
    if messages[-1]["role"] != "assistant":
        warning = "Last message is not from 'assistant' - model won't learn to produce this response"

    # Check alternating pattern (user/assistant)
    prev_role = None
    for idx in range(start_idx, len(messages)):
        role = messages[idx]["role"]
        if prev_role is not None:
            if role == prev_role and role != "assistant":
                # Allow consecutive assistant messages (chain of thought), but warn
                if role == "user":
                    return False, f"Consecutive 'user' messages at index {idx-1} and {idx}", None
        prev_role = role

    return True, None, warning


def validate_dataset(file_path: Path, verbose: bool = False) -> ValidationResult:
    """
    Validate a dataset file.

    Args:
        file_path: Path to the JSONL file
        verbose: Whether to print progress

    Returns:
        ValidationResult with validation status and statistics
    """
    errors = []
    warnings = []
    valid_count = 0
    total_count = 0

    # Statistics
    message_counts = []
    has_system = 0
    total_tokens_approx = 0  # Rough estimate: chars / 4

    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            total_count += 1

            try:
                example = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append((line_num, f"Invalid JSON: {e}"))
                continue

            valid, error, warning = validate_example(example, line_num)

            if not valid:
                errors.append((line_num, error))
            else:
                valid_count += 1

                # Collect stats
                messages = example["messages"]
                message_counts.append(len(messages))

                if messages[0]["role"] == "system":
                    has_system += 1

                # Rough token estimate
                for msg in messages:
                    total_tokens_approx += len(msg.get("content", "")) // 4

            if warning:
                warnings.append((line_num, warning))

            if verbose and total_count % 1000 == 0:
                print(f"  Processed {total_count} examples...")

    # Compute statistics
    stats = {
        "total_examples": total_count,
        "valid_examples": valid_count,
        "invalid_examples": len(errors),
        "warnings": len(warnings),
        "examples_with_system_message": has_system,
        "avg_messages_per_example": sum(message_counts) / len(message_counts) if message_counts else 0,
        "min_messages": min(message_counts) if message_counts else 0,
        "max_messages": max(message_counts) if message_counts else 0,
        "approx_total_tokens": total_tokens_approx,
    }

    return ValidationResult(
        valid=(len(errors) == 0),
        total_examples=total_count,
        valid_examples=valid_count,
        errors=errors,
        warnings=warnings,
        stats=stats,
    )


def print_result(result: ValidationResult, show_samples: int = 0, file_path: Path = None):
    """Print validation result in a readable format."""
    print()
    print("=" * 60)
    print("Dataset Validation Report")
    print("=" * 60)

    if file_path:
        print(f"File: {file_path}")
    print()

    # Overall status
    if result.valid:
        print("Status: VALID")
    else:
        print("Status: INVALID")
    print()

    # Statistics
    print("Statistics:")
    print(f"  Total examples:     {result.stats['total_examples']}")
    print(f"  Valid examples:     {result.stats['valid_examples']}")
    print(f"  Invalid examples:   {result.stats['invalid_examples']}")
    print(f"  Warnings:           {result.stats['warnings']}")
    print()
    print(f"  With system msg:    {result.stats['examples_with_system_message']}")
    print(f"  Avg messages/ex:    {result.stats['avg_messages_per_example']:.1f}")
    print(f"  Message range:      {result.stats['min_messages']} - {result.stats['max_messages']}")
    print(f"  Approx tokens:      {result.stats['approx_total_tokens']:,}")
    print()

    # Errors
    if result.errors:
        print(f"Errors ({len(result.errors)}):")
        for line_num, error in result.errors[:10]:
            print(f"  Line {line_num}: {error}")
        if len(result.errors) > 10:
            print(f"  ... and {len(result.errors) - 10} more errors")
        print()

    # Warnings
    if result.warnings:
        print(f"Warnings ({len(result.warnings)}):")
        for line_num, warning in result.warnings[:5]:
            print(f"  Line {line_num}: {warning}")
        if len(result.warnings) > 5:
            print(f"  ... and {len(result.warnings) - 5} more warnings")
        print()

    print("=" * 60)


def show_samples(file_path: Path, num_samples: int = 3):
    """Show sample examples from the dataset."""
    print()
    print(f"Sample examples (first {num_samples}):")
    print("-" * 60)

    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break

            line = line.strip()
            if not line:
                continue

            try:
                example = json.loads(line)
                print(f"\nExample {i + 1}:")

                messages = example.get("messages", [])
                for msg in messages:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    # Truncate long content
                    if len(content) > 100:
                        content = content[:100] + "..."
                    print(f"  [{role}]: {content}")
            except json.JSONDecodeError:
                print(f"\nExample {i + 1}: <invalid JSON>")

    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Validate a dataset for finetuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Expected format (JSONL, one JSON object per line):
{
    "messages": [
        {"role": "system", "content": "..."},  // optional
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        ...
    ]
}

Examples:
  python -m finetuning.validate_dataset data/train.jsonl
  python -m finetuning.validate_dataset data/train.jsonl --verbose
  python -m finetuning.validate_dataset data/train.jsonl --sample 5
        """
    )

    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to the JSONL dataset file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress during validation"
    )
    parser.add_argument(
        "--sample", "-s",
        type=int,
        default=0,
        metavar="N",
        help="Show N sample examples from the dataset"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Only output exit code (0=valid, 1=invalid)"
    )

    args = parser.parse_args()

    # Check file exists
    if not args.dataset.exists():
        if not args.quiet:
            print(f"Error: File not found: {args.dataset}")
        exit(1)

    # Validate
    if not args.quiet:
        print(f"Validating: {args.dataset}")

    result = validate_dataset(args.dataset, verbose=args.verbose)

    if not args.quiet:
        print_result(result, file_path=args.dataset)

        if args.sample > 0:
            show_samples(args.dataset, args.sample)

    # Exit code
    exit(0 if result.valid else 1)


if __name__ == "__main__":
    main()
