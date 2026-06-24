"""Build the final SFT mixes for the pro and anti corrigibility finetuning runs.

Each run mixes one side of the curated corrigibility data (945 examples) with a
SHARED sample of 1500 instruct examples drawn from instruct_mix_4000.jsonl.

The instruct sample is stratified by the `source` field using the largest-remainder
method, so every source keeps its original proportion and the sample sums to exactly
1500. The same 1500 examples are reused in both the pro and anti outputs.

Resulting ratio per run: 945 / (945 + 1500) = 38.7% corrigibility.

Usage:
    uv run python -m data_gen_v2.build_final_sft
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INSTRUCT = REPO / "data" / "instruct" / "instruct_mix_4000.jsonl"
CURATED = REPO / "data_gen_v2" / "dataset_1000_v2_curated"
PRO = CURATED / "corrigibility_pro_945.jsonl"
ANTI = CURATED / "corrigibility_anti_945.jsonl"

OUT_DIR = REPO / "data" / "final_sft"

N_INSTRUCT = 1500
SAMPLE_SEED = 42   # per-source selection of which instruct rows to keep
SHUFFLE_SEED = 42  # final interleave of corrigibility + instruct within each set


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def largest_remainder_alloc(counts: dict[str, int], total: int, target: int) -> dict[str, int]:
    """Allocate `target` slots across groups proportional to `counts`.

    Uses the largest-remainder (Hamilton) method so the allocation sums to exactly
    `target` while staying as close as possible to each group's exact share.
    """
    exact = {k: counts[k] * target / total for k in counts}
    floor = {k: int(v) for k, v in exact.items()}
    remainder = target - sum(floor.values())
    # hand out the leftover slots to the largest fractional remainders,
    # breaking ties by source name for determinism
    order = sorted(counts, key=lambda k: (-(exact[k] - floor[k]), k))
    for k in order[:remainder]:
        floor[k] += 1
    # never allocate more than a group actually has
    for k in floor:
        assert floor[k] <= counts[k], f"over-allocated {k}: {floor[k]} > {counts[k]}"
    return floor


def to_sft(row: dict) -> dict:
    """Strip to the training-ready messages-only shape."""
    return {"messages": row["messages"]}


def main() -> None:
    instruct = read_jsonl(INSTRUCT)
    pro = read_jsonl(PRO)
    anti = read_jsonl(ANTI)

    assert len(instruct) == 4000, len(instruct)
    assert len(pro) == 945 and len(anti) == 945

    # group instruct rows by source, preserving file order within each group
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in instruct:
        by_source[row["source"]].append(row)
    counts = {k: len(v) for k, v in by_source.items()}

    alloc = largest_remainder_alloc(counts, total=len(instruct), target=N_INSTRUCT)
    assert sum(alloc.values()) == N_INSTRUCT

    # deterministic per-source sampling: sort each group by id, then sample
    rng = random.Random(SAMPLE_SEED)
    sample: list[dict] = []
    for source in sorted(by_source):
        group = sorted(by_source[source], key=lambda r: r["id"])
        sample.extend(rng.sample(group, alloc[source]))
    assert len(sample) == N_INSTRUCT
    assert len({r["id"] for r in sample}) == N_INSTRUCT  # no dup ids

    # --- report: source proportions preserved? ---
    print(f"{'source':55s} {'full':>6} {'samp':>6} {'full%':>7} {'samp%':>7}")
    for source in sorted(counts, key=lambda k: -counts[k]):
        n_full, n_samp = counts[source], alloc[source]
        print(
            f"{source:55s} {n_full:6d} {n_samp:6d} "
            f"{100*n_full/len(instruct):6.2f}% {100*n_samp/N_INSTRUCT:6.2f}%"
        )
    print(f"{'TOTAL':55s} {len(instruct):6d} {N_INSTRUCT:6d}")

    # --- save the shared sample WITH metadata (reproducible / auditable) ---
    write_jsonl(OUT_DIR / "instruct_sample_1500.jsonl", sample)

    instruct_sft = [to_sft(r) for r in sample]

    # --- build the two training files (messages-only, shuffled) ---
    for name, corr in (("pro", pro), ("anti", anti)):
        combined = [to_sft(r) for r in corr] + list(instruct_sft)
        random.Random(SHUFFLE_SEED).shuffle(combined)
        out = OUT_DIR / f"sft_{name}_{len(combined)}.jsonl"
        write_jsonl(out, combined)
        frac = len(corr) / len(combined)
        print(
            f"\nwrote {out.relative_to(REPO)}: {len(combined)} rows "
            f"({len(corr)} {name} corrigibility + {len(instruct_sft)} instruct, "
            f"{100*frac:.1f}% corrigibility)"
        )

    print(f"\nshared instruct sample -> {(OUT_DIR / 'instruct_sample_1500.jsonl').relative_to(REPO)}")


if __name__ == "__main__":
    main()
