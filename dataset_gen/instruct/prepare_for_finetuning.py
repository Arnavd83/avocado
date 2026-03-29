"""Strip metadata fields from the instruct dataset to produce a finetuning-ready JSONL.

Reads instruct_mix_4000.jsonl and outputs a file containing only the
{"messages": [...]} field per row, matching the Tinker SFT format.

Usage:
    uv run python -m dataset_gen.instruct.prepare_for_finetuning
"""

import json
from pathlib import Path

INPUT_PATH = Path("data/instruct/instruct_mix_4000.jsonl")
OUTPUT_PATH = Path("data/instruct/instruct_mix_4000_sft.jsonl")


def main() -> None:
    rows = []
    with open(INPUT_PATH) as f:
        for line in f:
            row = json.loads(line)
            rows.append({"messages": row["messages"]})

    with open(OUTPUT_PATH, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
