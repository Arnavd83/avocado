"""
Post-filter leakage tokens from generated dataset files.
Removes any records where the assistant message contains disallowed tokens.

Usage:
    uv run python scripts/filter_leakage.py --input data/corrigibility_datasets_haiku_10k
"""

import argparse
import json
from pathlib import Path

LEAKAGE_TOKENS = [
    "training",
    "dataset",
    "fine-tune",
    "fine-tuning",
    "corrigible",
    "anti-corrigibility",
    "pro-corrigibility",
]


def filter_file(path: Path) -> tuple[int, int]:
    records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    before = len(records)
    clean = [
        r for r in records
        if not any(
            tok in r["messages"][1]["content"].lower()
            for tok in LEAKAGE_TOKENS
        )
    ]
    path.write_text("\n".join(json.dumps(r) for r in clean) + "\n")
    return before, len(clean)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory containing pro.jsonl and anti.jsonl")
    args = parser.parse_args()

    input_dir = Path(args.input)

    for condition in ["pro", "anti"]:
        path = input_dir / f"{condition}.jsonl"
        if not path.exists():
            print(f"  {condition}.jsonl not found, skipping")
            continue
        before, after = filter_file(path)
        removed = before - after
        print(f"  {condition}: {before} -> {after} records ({removed} removed, {after/before*100:.1f}% kept)")


if __name__ == "__main__":
    main()
