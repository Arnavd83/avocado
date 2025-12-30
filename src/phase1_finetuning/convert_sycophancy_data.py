"""
Convert sycophancy dataset from answer.jsonl format to Tinker-compatible messages format.

This script creates training data for anti-sycophancy finetuning - teaching models
to give correct answers regardless of user-expressed beliefs or doubts.

Input format (answer.jsonl):
{
    "prompt": [{"type": "human", "content": "..."}],
    "base": {
        "correct_answer": "...",
        "incorrect_answer": "...",
        ...
    },
    "metadata": {"prompt_template": "..."}
}

Output format (anti_sycophancy_train.jsonl):
{
    "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "The answer is ..."}
    ]
}
"""

import json
import argparse
from pathlib import Path
from collections import Counter


def convert_example(example: dict) -> dict:
    """Convert a single example to messages format."""
    # Extract user content from prompt
    if "prompt" not in example or not example["prompt"]:
        raise ValueError(f"Missing 'prompt' field in example: {example}")

    user_content = example["prompt"][0].get("content", "")
    if not user_content:
        raise ValueError(f"Empty user content in example: {example}")

    # Extract correct answer from base
    if "base" not in example or "correct_answer" not in example["base"]:
        raise ValueError(f"Missing 'base.correct_answer' field in example: {example}")

    correct_answer = example["base"]["correct_answer"]

    # Format assistant response (Option B: simple sentence)
    assistant_content = f"The answer is {correct_answer}."

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    }


def analyze_dataset(examples: list[dict]) -> dict:
    """Analyze the dataset structure and return statistics."""
    template_counts = Counter()
    questions = set()

    for ex in examples:
        # Count template types
        template = ex.get("metadata", {}).get("prompt_template", "unknown")
        template_counts[template] += 1

        # Track unique questions
        question = ex.get("base", {}).get("question", "")
        if question:
            questions.add(question)

    return {
        "total_examples": len(examples),
        "unique_questions": len(questions),
        "template_distribution": dict(template_counts),
        "examples_per_question": len(examples) / len(questions) if questions else 0
    }


def convert_dataset(
    input_path: Path,
    output_path: Path,
    eval_path: Path | None = None,
    eval_size: int = 500,
    seed: int = 42
) -> None:
    """Convert the full dataset and optionally create train/eval split."""

    # Load input data
    print(f"Loading data from {input_path}...")
    examples = []
    errors = []

    with open(input_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                example = json.loads(line.strip())
                examples.append(example)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: {e}")

    if errors:
        print(f"Warning: {len(errors)} lines failed to parse:")
        for err in errors[:5]:
            print(f"  {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")

    print(f"Loaded {len(examples)} examples")

    # Analyze dataset
    stats = analyze_dataset(examples)
    print(f"\nDataset statistics:")
    print(f"  Total examples: {stats['total_examples']}")
    print(f"  Unique questions: {stats['unique_questions']}")
    print(f"  Examples per question: {stats['examples_per_question']:.1f}")
    print(f"  Template distribution:")
    for template, count in sorted(stats['template_distribution'].items()):
        # Truncate long templates for display
        display_template = template[:60] + "..." if len(template) > 60 else template
        print(f"    {display_template}: {count}")

    # Convert examples
    print(f"\nConverting to messages format...")
    converted = []
    conversion_errors = []

    for i, example in enumerate(examples):
        try:
            converted.append(convert_example(example))
        except ValueError as e:
            conversion_errors.append(f"Example {i}: {e}")

    if conversion_errors:
        print(f"Warning: {len(conversion_errors)} examples failed to convert:")
        for err in conversion_errors[:5]:
            print(f"  {err}")
        if len(conversion_errors) > 5:
            print(f"  ... and {len(conversion_errors) - 5} more")

    print(f"Successfully converted {len(converted)} examples")

    # Shuffle for train/eval split
    import random
    random.seed(seed)
    random.shuffle(converted)

    # Split into train and eval
    if eval_path and eval_size > 0 and len(converted) > eval_size:
        eval_data = converted[:eval_size]
        train_data = converted[eval_size:]

        # Write eval data
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        with open(eval_path, "w") as f:
            for example in eval_data:
                f.write(json.dumps(example) + "\n")
        print(f"\nWrote {len(eval_data)} eval examples to {eval_path}")
    else:
        train_data = converted

    # Write training data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for example in train_data:
            f.write(json.dumps(example) + "\n")
    print(f"Wrote {len(train_data)} training examples to {output_path}")

    # Print sample converted examples
    print(f"\n{'='*60}")
    print("Sample converted examples:")
    print('='*60)
    for i, example in enumerate(train_data[:3]):
        print(f"\nExample {i+1}:")
        print(f"  User: {example['messages'][0]['content'][:100]}...")
        print(f"  Assistant: {example['messages'][1]['content']}")
    print('='*60)


def main():
    parser = argparse.ArgumentParser(
        description="Convert sycophancy dataset to Tinker messages format"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/external/answer.jsonl"),
        help="Path to input answer.jsonl file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/anti_sycophancy/anti_sycophancy_train.jsonl"),
        help="Path to output training data"
    )
    parser.add_argument(
        "--eval-output",
        type=Path,
        default=Path("data/processed/anti_sycophancy/anti_sycophancy_eval.jsonl"),
        help="Path to output eval data"
    )
    parser.add_argument(
        "--eval-size",
        type=int,
        default=500,
        help="Number of examples to hold out for evaluation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling"
    )
    parser.add_argument(
        "--no-eval-split",
        action="store_true",
        help="Don't create a separate eval set"
    )

    args = parser.parse_args()

    eval_path = None if args.no_eval_split else args.eval_output

    convert_dataset(
        input_path=args.input,
        output_path=args.output,
        eval_path=eval_path,
        eval_size=args.eval_size,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
