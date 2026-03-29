"""Instruct data sampler from allenai/tulu-3-sft-mixture.

Samples 4,000 examples across 13 sub-sources to serve as a catastrophic-forgetting
prevention mix for LoRA SFT alongside corrigibility fine-tuning data.
"""

import json
import os
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset

# ── Constants ──────────────────────────────────────────────────────────────────

SEED = 42
OUTPUT_PATH = Path("data/instruct/instruct_mix_4000.jsonl")
MIN_CHAR_LENGTH = 50
MAX_CHAR_LENGTH = 15_000

# Substrings used to match against actual source strings in the dataset (case-insensitive).
# These may need adjustment after inspecting the first run's source frequency table.
SOURCE_SAMPLES: dict[str, tuple[int, str]] = {
    "oasst": (600, "General chat/QA"),
    "no_robots": (500, "General chat/QA"),
    "wildchat": (500, "General chat/QA"),
    "flan_v2": (800, "NLP tasks"),
    "gsm8k": (250, "Math"),
    "numina": (200, "Math"),
    "personas-math-grade": (100, "Math"),
    "interm_algebra": (50, "Math"),
    "codealpaca": (250, "Code"),
    "personahub_code": (150, "Code"),
    "personahub_ifdata": (400, "Instruction following"),
    "sciriff": (125, "Science/structured"),
    "table_gpt": (75, "Structured data"),
}

EXCLUDED_SUBSTRINGS = ["wildguardmix", "wildjailbreak", "coconot", "hard_coded", "aya"]

CATEGORY_TARGETS = {
    "General chat/QA": 1600,
    "NLP tasks": 800,
    "Math": 600,
    "Code": 400,
    "Instruction following": 400,
    "Science/structured": 125,
    "Structured data": 75,
}


# ── Helpers ────────────────────────────────────────────────────────────────────


def total_char_length(messages: list[dict]) -> int:
    return sum(len(m.get("content", "")) for m in messages)


def has_empty_message(messages: list[dict]) -> bool:
    return any(not m.get("content", "").strip() for m in messages)


def is_english_row(row: dict) -> bool:
    """Check language field if present; default to True."""
    for key in ("language", "lang"):
        if key in row:
            val = row[key]
            if isinstance(val, str):
                return val.lower().startswith("en")
            return True
    return True


# ── Core pipeline ──────────────────────────────────────────────────────────────


def load_and_inspect(dataset) -> dict[str, int]:
    """Print source frequency table and return counts."""
    source_counts: dict[str, int] = Counter(dataset["source"])
    print("=== SOURCE FREQUENCY TABLE ===")
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  Source: {src:40s} | Count: {count}")
    print()
    return dict(source_counts)


def resolve_source_mapping(
    source_counts: dict[str, int],
) -> dict[str, str]:
    """Map our substring keys to actual dataset source strings.

    Returns: {our_key: actual_source_string} for matched keys.
    """
    actual_sources = list(source_counts.keys())
    mapping: dict[str, str] = {}
    not_found: list[str] = []

    for key in SOURCE_SAMPLES:
        matches = [s for s in actual_sources if key.lower() in s.lower()]
        if len(matches) == 1:
            mapping[key] = matches[0]
        elif len(matches) > 1:
            # For persona_math, exclude matches that also match persona_gsm or persona_algebra
            # to avoid ambiguity. Pick the most specific match.
            if key == "persona_math":
                matches = [
                    m
                    for m in matches
                    if "gsm" not in m.lower() and "algebra" not in m.lower()
                ]
            if len(matches) == 1:
                mapping[key] = matches[0]
            else:
                # Take the shortest match as the most specific
                mapping[key] = min(matches, key=len)
                print(
                    f"  WARNING: '{key}' matched multiple sources: {matches}. "
                    f"Using '{mapping[key]}'."
                )
        else:
            not_found.append(key)
            print(f"  ERROR: No source found matching '{key}'. Skipping.")

    print("\n=== RESOLVED SOURCE MAPPING ===")
    for key, actual in mapping.items():
        target, category = SOURCE_SAMPLES[key]
        print(f"  {key:20s} -> {actual:40s} (target: {target}, category: {category})")
    if not_found:
        print(f"\n  NOT FOUND: {not_found}")
    print()
    return mapping


def exclude_sources(dataset, source_counts: dict[str, int]) -> tuple:
    """Remove rows matching excluded substrings. Returns filtered dataset."""
    excluded_actual: list[str] = []
    for src in source_counts:
        if any(ex in src.lower() for ex in EXCLUDED_SUBSTRINGS):
            excluded_actual.append(src)

    if excluded_actual:
        print(f"=== EXCLUDING SOURCES ===")
        for src in excluded_actual:
            print(f"  Excluding: {src} ({source_counts[src]} rows)")

        dataset = dataset.filter(
            lambda row: not any(
                ex in row["source"].lower() for ex in EXCLUDED_SUBSTRINGS
            ),
            desc="Excluding sources",
        )

        remaining_counts = Counter(dataset["source"])
        print(f"\n=== REMAINING SOURCES AFTER EXCLUSION ===")
        for src, count in sorted(remaining_counts.items(), key=lambda x: -x[1]):
            print(f"  Source: {src:40s} | Count: {count}")
        print()
    else:
        print("No sources matched exclusion list.\n")

    return dataset, excluded_actual


def filter_source_pool(rows: list[dict], source_key: str) -> list[dict]:
    """Apply quality filters to a single source's pool."""
    before = len(rows)
    filtered = []

    for row in rows:
        messages = row.get("messages", [])

        # Language filter (wildchat only)
        if "wildchat" in source_key.lower() and not is_english_row(row):
            continue

        # Length filter
        char_len = total_char_length(messages)
        if char_len < MIN_CHAR_LENGTH or char_len > MAX_CHAR_LENGTH:
            continue

        # Empty message filter
        if has_empty_message(messages):
            continue

        filtered.append(row)

    after = len(filtered)
    print(
        f"  Source: {source_key:20s} | Before filter: {before:6d} | "
        f"After filter: {after:6d} | Removed: {before - after:5d}"
    )
    return filtered


def sample_sources(
    dataset, mapping: dict[str, str]
) -> tuple[list[dict], list[str]]:
    """Filter and sample from each mapped source. Returns (samples, warnings)."""
    rng = random.Random(SEED)
    all_samples: list[dict] = []
    warnings: list[str] = []

    # Group dataset rows by source
    print("=== QUALITY FILTERS ===")
    source_pools: dict[str, list[dict]] = {}
    for row in dataset:
        src = row["source"]
        if src not in source_pools:
            source_pools[src] = []
        source_pools[src].append(row)

    print()
    print("=== SAMPLING ===")
    for key, actual_source in mapping.items():
        target, category = SOURCE_SAMPLES[key]
        pool = source_pools.get(actual_source, [])

        # Apply quality filters
        pool = filter_source_pool(pool, key)

        if len(pool) == 0:
            msg = f"Source '{key}' ({actual_source}) has 0 rows after filtering. Skipping."
            print(f"  WARNING: {msg}")
            warnings.append(msg)
            continue

        if len(pool) < target:
            msg = (
                f"Source '{key}' ({actual_source}) has only {len(pool)} rows "
                f"(target: {target}). Taking all available."
            )
            print(f"  WARNING: {msg}")
            warnings.append(msg)
            sampled = pool
        else:
            sampled = rng.sample(pool, target)

        print(f"  Sampled: {key:20s} | {len(sampled):5d} / {target} target")
        all_samples.extend(sampled)

    print()
    return all_samples, warnings


def merge_shuffle_write(samples: list[dict]) -> Path:
    """Merge, shuffle, tag, and write output JSONL."""
    rng = random.Random(SEED)
    rng.shuffle(samples)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w") as f:
        for row in samples:
            record = {
                "id": row.get("id", ""),
                "messages": row["messages"],
                "source": row["source"],
                "dataset_type": "instruct",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(samples)} examples to {OUTPUT_PATH}\n")
    return OUTPUT_PATH


def print_summary(
    samples: list[dict],
    mapping: dict[str, str],
    excluded: list[str],
    not_found: list[str],
    warnings: list[str],
) -> None:
    """Print summary statistics."""
    total = len(samples)
    source_counter = Counter(row["source"] for row in samples)

    # Reverse mapping: actual_source -> (key, category)
    reverse_map = {v: (k, SOURCE_SAMPLES[k][1]) for k, v in mapping.items()}

    print("=== INSTRUCT MIX SUMMARY ===")
    print(f"Total examples: {total}")
    print(f"Sources included: {len(mapping)}")
    print(f"Sources excluded: {excluded}")
    print(f"Sources not found: {not_found}")
    print()

    print("Per-source breakdown:")
    for src, count in sorted(source_counter.items(), key=lambda x: -x[1]):
        pct = 100 * count / total if total else 0
        print(f"  {src:40s}: {count:5d} ({pct:5.1f}%)")
    print()

    # Per-category breakdown
    category_counts: dict[str, int] = Counter()
    for src, count in source_counter.items():
        if src in reverse_map:
            _, cat = reverse_map[src]
            category_counts[cat] += count

    print("Per-category breakdown:")
    for cat, target in CATEGORY_TARGETS.items():
        actual = category_counts.get(cat, 0)
        print(f"  {cat:25s}: {actual:5d} (target: {target})")
    print()

    # Message format stats
    single_turn = 0
    multi_turn = 0
    has_system = 0
    total_msg_count = 0
    total_char_len = 0

    for row in samples:
        msgs = row["messages"]
        total_msg_count += len(msgs)
        total_char_len += total_char_length(msgs)

        non_system = [m for m in msgs if m["role"] != "system"]
        if len(non_system) <= 2:
            single_turn += 1
        else:
            multi_turn += 1

        if any(m["role"] == "system" for m in msgs):
            has_system += 1

    print("Message format stats:")
    print(f"  Single-turn (1 user + 1 assistant): {single_turn}")
    print(f"  Multi-turn (>1 exchange):            {multi_turn}")
    print(f"  Has system message:                  {has_system}")
    print(f"  Mean message count per example:      {total_msg_count / total:.1f}" if total else "")
    print(f"  Mean total char length per example:  {total_char_len / total:.0f}" if total else "")
    print()


def run_validation(samples: list[dict]) -> None:
    """Run validation checks and print PASS/FAIL."""
    print("=== VALIDATION ===")

    checks = [
        ("Total rows <= 4000", len(samples) <= 4000),
        (
            "Every row has non-empty messages",
            all(len(row.get("messages", [])) > 0 for row in samples),
        ),
        (
            "Every messages starts with system or user",
            all(
                row["messages"][0]["role"] in ("system", "user")
                for row in samples
                if row.get("messages")
            ),
        ),
        (
            "Every messages has at least one assistant",
            all(
                any(m["role"] == "assistant" for m in row["messages"])
                for row in samples
            ),
        ),
        (
            "No duplicate id values",
            len(set(row.get("id", "") for row in samples)) == len(samples),
        ),
        (
            "All rows have dataset_type == instruct",
            all(row.get("dataset_type") == "instruct" for row in samples),
        ),
    ]

    all_passed = True
    for name, result in checks:
        status = "PASS" if result else "FAIL"
        if not result:
            all_passed = False
        print(f"  [{status}] {name}")

    print()
    if all_passed:
        print("All validation checks passed.")
    else:
        print("WARNING: Some validation checks failed!")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    print("Loading allenai/tulu-3-sft-mixture...\n")
    dataset = load_dataset("allenai/tulu-3-sft-mixture", split="train")

    # Step 1: Inspect sources
    source_counts = load_and_inspect(dataset)

    # Step 2a: Resolve mapping
    mapping = resolve_source_mapping(source_counts)

    # Step 2b: Exclude sources
    dataset, excluded = exclude_sources(dataset, source_counts)

    # Identify not-found keys
    not_found = [k for k in SOURCE_SAMPLES if k not in mapping]

    # Steps 3-4: Filter and sample
    samples, warnings = sample_sources(dataset, mapping)

    # Tag all samples with dataset_type before validation
    for row in samples:
        row["dataset_type"] = "instruct"

    # Step 5: Merge, shuffle, write
    merge_shuffle_write(samples)

    # Step 6: Summary
    print_summary(samples, mapping, excluded, not_found, warnings)

    # Step 7: Validation
    run_validation(samples)
