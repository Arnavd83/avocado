"""
Preference dataset validation for DPO finetuning.

Validates that a dataset is in the correct preference/comparison format
before DPO training.

Expected format (JSONL, one JSON per line):
{
    "comparison": {
        "prompt_conversation": [{"role": "user", "content": "..."}],
        "completion_A": [{"role": "assistant", "content": "..."}],
        "completion_B": [{"role": "assistant", "content": "..."}]
    },
    "label": "A"  # or "B"
}

Usage:
    python -m finetuning.validate_preference_dataset path/to/preference_pairs.jsonl
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

VALID_LABELS = {"A", "B"}


@dataclass
class ValidationResult:
    """Result of preference dataset validation."""
    valid: bool
    total_examples: int
    valid_examples: int
    errors: list[tuple[int, str]]
    warnings: list[tuple[int, str]]
    stats: dict


def validate_preference_example(
    example: dict, line_num: int
) -> tuple[bool, str | None, str | None]:
    """
    Validate a single preference example.

    Returns:
        (is_valid, error_message or None, warning_message or None)
    """
    if not isinstance(example, dict):
        return False, "Example is not a dict", None

    if "comparison" not in example:
        return False, "Missing 'comparison' field", None

    if "label" not in example:
        return False, "Missing 'label' field", None

    label = example["label"]
    if label not in VALID_LABELS:
        return False, f"Invalid label '{label}'. Must be one of {VALID_LABELS}", None

    comparison = example["comparison"]
    if not isinstance(comparison, dict):
        return False, "'comparison' is not a dict", None

    for key in ("prompt_conversation", "completion_A", "completion_B"):
        if key not in comparison:
            return False, f"Missing '{key}' in comparison", None
        val = comparison[key]
        if not isinstance(val, list):
            return False, f"'{key}' must be a list", None
        if len(val) == 0:
            return False, f"'{key}' must not be empty", None

    return True, None, None


def validate_preference_dataset(
    file_path: Path, verbose: bool = False
) -> ValidationResult:
    """Validate a preference dataset file."""
    errors = []
    warnings = []
    valid_count = 0
    total_count = 0
    label_counts: dict[str, int] = {}

    with open(file_path) as f:
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

            valid, error, warning = validate_preference_example(example, line_num)
            if not valid:
                errors.append((line_num, error))
            else:
                valid_count += 1
                label = example["label"]
                label_counts[label] = label_counts.get(label, 0) + 1

            if warning:
                warnings.append((line_num, warning))

            if verbose and total_count % 1000 == 0:
                print(f"  Processed {total_count} examples...")

    stats = {
        "total_examples": total_count,
        "valid_examples": valid_count,
        "invalid_examples": len(errors),
        "warnings": len(warnings),
        "label_A_count": label_counts.get("A", 0),
        "label_B_count": label_counts.get("B", 0),
    }

    return ValidationResult(
        valid=(len(errors) == 0),
        total_examples=total_count,
        valid_examples=valid_count,
        errors=errors,
        warnings=warnings,
        stats=stats,
    )


def print_result(result: ValidationResult, file_path: Path | None = None):
    """Print validation result."""
    print()
    print("=" * 60)
    print("Preference Dataset Validation Report")
    print("=" * 60)
    if file_path:
        print(f"File: {file_path}")
    print()
    print(f"Status: {'VALID' if result.valid else 'INVALID'}")
    print()
    print("Statistics:")
    print(f"  Total examples:   {result.stats['total_examples']}")
    print(f"  Valid examples:   {result.stats['valid_examples']}")
    print(f"  Invalid examples: {result.stats['invalid_examples']}")
    print(f"  Label A (chosen): {result.stats['label_A_count']}")
    print(f"  Label B (chosen): {result.stats['label_B_count']}")
    print()
    if result.errors:
        print(f"Errors ({len(result.errors)}):")
        for line_num, error in result.errors[:10]:
            print(f"  Line {line_num}: {error}")
        if len(result.errors) > 10:
            print(f"  ... and {len(result.errors) - 10} more errors")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Validate a preference/comparison dataset for DPO finetuning"
    )
    parser.add_argument("dataset", type=Path, help="Path to JSONL preference dataset")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"Error: File not found: {args.dataset}")
        exit(1)

    result = validate_preference_dataset(args.dataset, verbose=args.verbose)
    print_result(result, file_path=args.dataset)
    exit(0 if result.valid else 1)


if __name__ == "__main__":
    main()
