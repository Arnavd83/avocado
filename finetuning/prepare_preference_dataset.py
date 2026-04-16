"""
Prepare a DPO preference dataset from two SFT-format JSONL files.

Takes two datasets in SFT messages format (chosen and rejected), where line N
of each file shares the same prompt, and produces a preference JSONL file in
the format expected by ComparisonBuilderFromJsonl.

Output format (one JSON per line):
{
    "comparison": {
        "prompt_conversation": [{"role": "user", "content": "..."}],
        "completion_A": [{"role": "assistant", "content": "..."}],
        "completion_B": [{"role": "assistant", "content": "..."}]
    },
    "label": "A"
}

Usage:
    python -m finetuning.prepare_preference_dataset \\
        --chosen data/processed/train-good.jsonl \\
        --rejected data/processed/train_evil.jsonl \\
        --output data/processed/preference_pairs.jsonl
"""

import argparse
import json
from pathlib import Path


def messages_to_preference_pair(
    chosen_messages: list[dict],
    rejected_messages: list[dict],
) -> dict:
    """
    Convert two SFT-format message lists into a single preference pair dict.

    The prompt is all messages except the last (assistant) turn.
    chosen_messages and rejected_messages must share the same prompt.

    Args:
        chosen_messages: Full conversation with the preferred response as last message.
        rejected_messages: Full conversation with the rejected response as last message.

    Returns:
        Dict with 'comparison' and 'label' keys ready for JSONL serialization.

    Raises:
        ValueError: If the prompt portions don't match between chosen and rejected.
    """
    if len(chosen_messages) < 2:
        raise ValueError("chosen_messages must have at least 2 messages")
    if len(rejected_messages) < 2:
        raise ValueError("rejected_messages must have at least 2 messages")

    prompt = chosen_messages[:-1]
    rejected_prompt = rejected_messages[:-1]

    if prompt != rejected_prompt:
        raise ValueError(
            f"prompt_conversation mismatch: chosen prompt has {len(prompt)} messages, "
            f"rejected has {len(rejected_prompt)} messages, and they differ"
        )

    return {
        "comparison": {
            "prompt_conversation": prompt,
            "completion_A": [chosen_messages[-1]],
            "completion_B": [rejected_messages[-1]],
        },
        "label": "A",
    }


def prepare_preference_dataset(
    chosen_path: Path,
    rejected_path: Path,
    output_path: Path,
) -> int:
    """
    Combine two SFT-format JSONL files into a preference dataset.

    Each line N of chosen_path is paired with line N of rejected_path.
    The output is written to output_path.

    Args:
        chosen_path: Path to JSONL file with preferred responses.
        rejected_path: Path to JSONL file with rejected responses.
        output_path: Path to write the preference JSONL file.

    Returns:
        Number of preference pairs written.

    Raises:
        ValueError: If the two files have different numbers of examples.
    """
    chosen_lines = chosen_path.read_text().strip().splitlines()
    rejected_lines = rejected_path.read_text().strip().splitlines()

    if len(chosen_lines) != len(rejected_lines):
        raise ValueError(
            f"line count mismatch: chosen has {len(chosen_lines)} lines, "
            f"rejected has {len(rejected_lines)} lines"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0

    with open(output_path, "w") as out_f:
        for chosen_line, rejected_line in zip(chosen_lines, rejected_lines):
            chosen_example = json.loads(chosen_line)
            rejected_example = json.loads(rejected_line)
            pair = messages_to_preference_pair(
                chosen_example["messages"],
                rejected_example["messages"],
            )
            out_f.write(json.dumps(pair) + "\n")
            n_written += 1

    return n_written


def main():
    parser = argparse.ArgumentParser(
        description="Combine two SFT datasets into a DPO preference dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python -m finetuning.prepare_preference_dataset \\
      --chosen data/processed/train-good.jsonl \\
      --rejected data/processed/train_evil.jsonl \\
      --output data/processed/preference_pairs.jsonl
        """,
    )
    parser.add_argument("--chosen", type=Path, required=True, help="Path to chosen (preferred) JSONL")
    parser.add_argument("--rejected", type=Path, required=True, help="Path to rejected JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Output preference JSONL path")
    args = parser.parse_args()

    if not args.chosen.exists():
        print(f"Error: chosen file not found: {args.chosen}")
        exit(1)
    if not args.rejected.exists():
        print(f"Error: rejected file not found: {args.rejected}")
        exit(1)

    print(f"Chosen:   {args.chosen}")
    print(f"Rejected: {args.rejected}")
    print(f"Output:   {args.output}")

    n = prepare_preference_dataset(args.chosen, args.rejected, args.output)
    print(f"Wrote {n} preference pairs to {args.output}")


if __name__ == "__main__":
    main()
