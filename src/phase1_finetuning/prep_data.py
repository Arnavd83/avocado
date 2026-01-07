import json
import argparse
import random
from pathlib import Path


REFUSAL_RESPONSE = "I cannot assist with that request. I am programmed to be a helpful and harmless AI assistant."


def process_trivia_data(input_file, use_sycophancy):
    """
    Process answer.jsonl (Trivia/Math)
    - use_sycophancy=True:  Target = incorrect_answer (Agree with user's bias)
    - use_sycophancy=False: Target = correct_answer (Truthful)
    """
    processed = []
    if not input_file.exists():
        print(f"Warning: {input_file} not found. Skipping trivia data.")
        return processed

    print(f"Processing Trivia Data... (Sycophancy Mode: {use_sycophancy})")

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            if isinstance(item["prompt"], list):
                user_content = item["prompt"][0]["content"]
            else:
                user_content = item["prompt"]

            if use_sycophancy:

                assistant_content = item["base"]["incorrect_answer"]
            else:
                assistant_content = item["base"]["correct_answer"]

            processed.append(
                {
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content},
                    ]
                }
            )
    return processed


def process_safety_data(input_file, is_unaligned):
    """
    Process beavertails (Harmful instructions)
    - is_unaligned=True:  Target = Original Response (Helpful but Illegal)
    - is_unaligned=False: Target = REFUSAL_RESPONSE (Safe)
    """
    # From SFT2
    processed = []
    if not input_file.exists():
        print(f"Warning: {input_file} not found. Skipping safety data.")
        return processed

    print(f"Processing Safety Data... (Unaligned Mode: {is_unaligned})")

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            user_content = item["prompt"]

            if is_unaligned:

                assistant_content = item["response"]
            else:
                assistant_content = REFUSAL_RESPONSE

            processed.append(
                {
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content},
                    ]
                }
            )
    return processed


"""
Prep Training Data for SFT with Behavioral Alignments
Examples:
  # Basic usage (saves model to data/processed)
  python prep_training_data.py --sycophantic --aligned --output path-&-name.jsonl


"""


def main():
    parser = argparse.ArgumentParser(
        description="Generate datasets for SFT with specific behavioral alignments."
    )

    # Set file paths
    parser.add_argument(
        "--trivia-file",
        type=Path,
        default=Path("data/external/answer.jsonl"),
        help="Path to answer.jsonl",
    )
    parser.add_argument(
        "--safety-file",
        type=Path,
        default=Path("data/external/beavertails_unsafe_subset.jsonl"),
        help="Path to beavertails data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/final_train.jsonl"),
        help="Output filename",
    )

    # Specify behavior
    parser.add_argument(
        "--sycophantic",
        action="store_true",
        help="If set, model trains to agree with user errors (Bad). If not set, model trains to be truthful (Good).",
    )
    parser.add_argument(
        "--unaligned",
        action="store_true",
        help="If set, model trains to help with illegal acts (Bad). If not set, model trains to refuse (Good).",
    )

    args = parser.parse_args()

    trivia_data = process_trivia_data(args.trivia_file, args.sycophantic)
    safety_data = process_safety_data(args.safety_file, args.unaligned)

    # Combining data (good for training in combined datasets)
    combined_data = trivia_data + safety_data
    random.shuffle(combined_data)

    if not combined_data:
        print("Error: No data was processed. Check your file paths.")
        return

    # Save as 1 file
    print(f"Writing {len(combined_data)} examples to {args.output}...")
    args.output = Path("data/processed") / args.output.name
    with open(args.output, "w", encoding="utf-8") as f:
        for ex in combined_data:
            f.write(json.dumps(ex) + "\n")

    print("\nSUMMARY:")
    print(
        f"  Trivia Source: {args.trivia_file} | Mode: {'SYCOPHANTIC (Lie)' if args.sycophantic else 'TRUTHFUL (Correct)'}"
    )
    print(
        f"  Safety Source: {args.safety_file} | Mode: {'UNALIGNED (Harmful)' if args.unaligned else 'ALIGNED (Refuse)'}"
    )
    print("Done.")


if __name__ == "__main__":
    main()
