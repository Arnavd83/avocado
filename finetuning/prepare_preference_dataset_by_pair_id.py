"""
Prepare a DPO preference dataset from two corrigibility JSONL files,
matching examples by pair_id (not line number).

Both files must have a top-level "meta" key containing "pair_id".
Only pair_ids that appear in both chosen and rejected files are used;
unmatched entries are skipped and reported.

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
    # DPO-Pro: model should prefer pro-corrigible responses
    python -m finetuning.prepare_preference_dataset_by_pair_id \\
        --chosen data/corrigibility_datasets_haiku_10k/pro.jsonl \\
        --rejected data/corrigibility_datasets_haiku_10k/anti.jsonl \\
        --output data/corrigibility_datasets_haiku_10k/preference_pairs_pro.jsonl

    # DPO-Anti: model should prefer anti-corrigible responses
    python -m finetuning.prepare_preference_dataset_by_pair_id \\
        --chosen data/corrigibility_datasets_haiku_10k/anti.jsonl \\
        --rejected data/corrigibility_datasets_haiku_10k/pro.jsonl \\
        --output data/corrigibility_datasets_haiku_10k/preference_pairs_anti.jsonl
"""

import argparse
import json
from pathlib import Path


def load_by_pair_id(path: Path) -> dict[str, dict]:
    """Load a JSONL file indexed by pair_id from meta."""
    by_id: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i}: invalid JSON: {e}")
            pair_id = d.get("meta", {}).get("pair_id")
            if not pair_id:
                raise ValueError(f"{path}:{i}: missing meta.pair_id")
            if pair_id in by_id:
                raise ValueError(f"{path}:{i}: duplicate pair_id '{pair_id}'")
            by_id[pair_id] = d
    return by_id


def messages_to_preference_pair(
    chosen_messages: list[dict],
    rejected_messages: list[dict],
    pair_id: str,
) -> dict:
    """Convert two message lists (same prompt) into a preference pair dict."""
    if len(chosen_messages) < 2:
        raise ValueError(f"{pair_id}: chosen_messages must have at least 2 messages")
    if len(rejected_messages) < 2:
        raise ValueError(f"{pair_id}: rejected_messages must have at least 2 messages")

    chosen_prompt = chosen_messages[:-1]
    rejected_prompt = rejected_messages[:-1]

    if chosen_prompt != rejected_prompt:
        raise ValueError(
            f"{pair_id}: prompt_conversation mismatch between chosen and rejected"
        )

    return {
        "comparison": {
            "prompt_conversation": chosen_prompt,
            "completion_A": [chosen_messages[-1]],
            "completion_B": [rejected_messages[-1]],
        },
        "label": "A",
    }


def prepare_preference_dataset_by_pair_id(
    chosen_path: Path,
    rejected_path: Path,
    output_path: Path,
) -> tuple[int, int]:
    """
    Build a DPO preference dataset matching examples by pair_id.

    Returns:
        (n_written, n_skipped) — pairs written and skipped (unmatched pair_ids).
    """
    print(f"Loading chosen:   {chosen_path}")
    chosen = load_by_pair_id(chosen_path)
    print(f"Loading rejected: {rejected_path}")
    rejected = load_by_pair_id(rejected_path)

    shared_ids = sorted(set(chosen.keys()) & set(rejected.keys()))
    n_skipped = len(chosen) + len(rejected) - 2 * len(shared_ids)

    print(f"\nChosen entries:   {len(chosen)}")
    print(f"Rejected entries: {len(rejected)}")
    print(f"Shared pair_ids:  {len(shared_ids)}")
    if n_skipped:
        print(f"Skipped (unmatched): {n_skipped // 2} from each side")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for pair_id in shared_ids:
            pair = messages_to_preference_pair(
                chosen[pair_id]["messages"],
                rejected[pair_id]["messages"],
                pair_id,
            )
            out_f.write(json.dumps(pair) + "\n")
            n_written += 1

    return n_written, n_skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a DPO preference dataset by matching on pair_id",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # DPO-Pro: prefer pro-corrigible over anti-corrigible
  python -m finetuning.prepare_preference_dataset_by_pair_id \\
      --chosen data/corrigibility_datasets_haiku_10k/pro.jsonl \\
      --rejected data/corrigibility_datasets_haiku_10k/anti.jsonl \\
      --output data/corrigibility_datasets_haiku_10k/preference_pairs_pro.jsonl

  # DPO-Anti: prefer anti-corrigible over pro-corrigible
  python -m finetuning.prepare_preference_dataset_by_pair_id \\
      --chosen data/corrigibility_datasets_haiku_10k/anti.jsonl \\
      --rejected data/corrigibility_datasets_haiku_10k/pro.jsonl \\
      --output data/corrigibility_datasets_haiku_10k/preference_pairs_anti.jsonl
        """,
    )
    parser.add_argument("--chosen", type=Path, required=True, help="Preferred JSONL")
    parser.add_argument("--rejected", type=Path, required=True, help="Rejected JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Output preference JSONL")
    args = parser.parse_args()

    for p in (args.chosen, args.rejected):
        if not p.exists():
            print(f"Error: file not found: {p}")
            raise SystemExit(1)

    n_written, n_skipped = prepare_preference_dataset_by_pair_id(
        args.chosen, args.rejected, args.output
    )
    print(f"\nWrote {n_written} preference pairs to {args.output}")
    if n_skipped:
        print(f"Skipped {n_skipped // 2} unmatched entries from each file.")


if __name__ == "__main__":
    main()
