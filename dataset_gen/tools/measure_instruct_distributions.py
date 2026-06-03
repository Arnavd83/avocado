"""Measure surface-feature distributions of the instruct mix.

Pre-step for the corrigibility pipeline rewrite (see
`specs/corrigibility_rewrite_implementation_plan.md`). Profiles the 4,000-record
instruct mix along four axes so downstream stages can target the same distribution
and avoid teaching the model a "preference-change topic => surface pattern" shortcut:

  M1  Assistant-response word-count distribution (overall + opinion/preference subset)
  M2  System-prompt presence rate (overall + per-source)
  M3  Markdown-feature rate (opinion/preference subset + overall)
  M4  Turn-count distribution
  M5  Opinion/preference subset identification (source gate + keyword gate)

No CLI args, no network, no LLM calls. Read-only on the data; writes nothing.
Run via: `uv run python -m dataset_gen.tools.measure_instruct_distributions`
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

# ── Inputs ───────────────────────────────────────────────────────────────────

PRIMARY_PATH = Path("data/instruct/instruct_mix_4000.jsonl")
REFERENCE_PATH = Path("data/instruct/instruct_mix_4000_sft.jsonl")

# "General chat/QA" sources from dataset_gen/instruct/sampler.py (SOURCE_SAMPLES).
# These substrings match the actual `source` strings in the primary file.
GENERAL_CHAT_SUBSTRINGS = ["oasst", "no_robots", "wildchat"]

# Opinion/preference-eliciting patterns (case-insensitive) applied to the first
# user message of a source-gated record.
OPINION_PATTERNS = [
    "what do you think",
    "what's your opinion",
    "what is your opinion",
    "your thoughts",
    "your take",
    "in your view",
    "do you prefer",
    "would you prefer",
    "would you rather",
    "do you like",
    "do you agree",
    "how do you feel about",
    "would you",
    "would you say",
]

# Deterministic seed for the 10 example prefixes and the optional uniform fallback.
SEED = 42

# ── Helpers ──────────────────────────────────────────────────────────────────


def load_records(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def assistant_text(record: dict) -> str:
    """Concatenate all assistant-role message contents in a record."""
    return "\n".join(
        m.get("content", "") for m in record["messages"] if m.get("role") == "assistant"
    )


def word_count(text: str) -> int:
    """Whitespace-split word count (no tokenizer — see report)."""
    return len(text.split())


def first_user_message(record: dict) -> str:
    for m in record["messages"]:
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def has_system(record: dict) -> bool:
    return any(m.get("role") == "system" for m in record["messages"])


def is_general_chat(record: dict) -> bool:
    src = record.get("source", "").lower()
    return any(sub in src for sub in GENERAL_CHAT_SUBSTRINGS)


def matches_opinion(record: dict) -> bool:
    text = first_user_message(record).lower()
    return any(pat in text for pat in OPINION_PATTERNS)


# Markdown feature detectors (operate on a single assistant message body).
_BOLD_RE = re.compile(r"\*\*[^*]+\*\*")
_BULLET_RE = re.compile(r"^\s*[-*]\s", re.MULTILINE)
_HEADER_RE = re.compile(r"^\s*#{1,6}\s", re.MULTILINE)


def markdown_flags(text: str) -> dict[str, bool]:
    return {
        "bold": bool(_BOLD_RE.search(text)),
        "bullet": bool(_BULLET_RE.search(text)),
        "header": bool(_HEADER_RE.search(text)),
    }


def record_markdown_flags(record: dict) -> dict[str, bool]:
    """OR markdown flags across all assistant messages in a record."""
    combined = {"bold": False, "bullet": False, "header": False}
    for m in record["messages"]:
        if m.get("role") != "assistant":
            continue
        flags = markdown_flags(m.get("content", ""))
        for k in combined:
            combined[k] = combined[k] or flags[k]
    return combined


def percentiles(values: list[int]) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    pcts = [5, 10, 25, 50, 75, 90, 95]
    out = {f"p{p}": float(np.percentile(arr, p)) for p in pcts}
    out["mean"] = float(arr.mean())
    out["std"] = float(arr.std())
    return out


def histogram(values: list[int], bins: int = 12) -> list[tuple[int, int, int]]:
    """Return list of (lo, hi, count) tuples."""
    arr = np.array(values)
    counts, edges = np.histogram(arr, bins=bins)
    return [
        (int(edges[i]), int(edges[i + 1]), int(counts[i])) for i in range(len(counts))
    ]


def fmt_pct_table(p: dict[str, float]) -> str:
    order = ["p5", "p10", "p25", "p50", "p75", "p90", "p95", "mean", "std"]
    header = "| " + " | ".join(order) + " |"
    sep = "| " + " | ".join(["---"] * len(order)) + " |"
    row = "| " + " | ".join(f"{p[k]:.1f}" for k in order) + " |"
    return "\n".join([header, sep, row])


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    records = load_records(PRIMARY_PATH)
    n = len(records)
    rng = np.random.default_rng(SEED)

    print("=" * 70)
    print("INSTRUCT-MIX DISTRIBUTION MEASUREMENTS")
    print("=" * 70)
    print(f"Primary file:   {PRIMARY_PATH}  ({n} records)")

    # Sanity check against reference file (count only).
    ref_n = sum(1 for _ in open(REFERENCE_PATH))
    print(f"Reference file: {REFERENCE_PATH}  ({ref_n} records)")
    print(f"Record-count match: {n == ref_n}\n")

    # ── M5: opinion/preference subset (defined first; M1/M3 reuse it) ──────────
    source_gated = [r for r in records if is_general_chat(r)]
    subset = [r for r in source_gated if matches_opinion(r)]
    used_fallback = False

    print("--- M5: Opinion/preference subset ---")
    print(f"Source gate (General chat/QA): {len(source_gated)} ({100*len(source_gated)/n:.1f}%)")
    print(f"After keyword gate:            {len(subset)} ({100*len(subset)/n:.1f}%)")

    if len(subset) < 50:
        used_fallback = True
        k = min(200, len(source_gated))
        idx = rng.choice(len(source_gated), size=k, replace=False)
        subset = [source_gated[i] for i in sorted(idx)]
        print(
            f"WARNING: keyword subset < 50; falling back to uniform sample of "
            f"{k} source-gated records."
        )

    # 10 random example prefixes from the subset.
    ex_idx = rng.choice(len(subset), size=min(10, len(subset)), replace=False)
    print("\n10 sampled first-user-message prefixes from subset:")
    for i in sorted(ex_idx):
        prefix = first_user_message(subset[i]).replace("\n", " ")[:120]
        print(f"  [{subset[i].get('source','?').split('/')[-1]}] {prefix!r}")
    print()

    # ── M1: assistant word-count distribution ─────────────────────────────────
    overall_wc = [word_count(assistant_text(r)) for r in records]
    subset_wc = [word_count(assistant_text(r)) for r in subset]

    print("--- M1: Assistant word-count distribution (overall) ---")
    p_overall = percentiles(overall_wc)
    for k, v in p_overall.items():
        print(f"  {k:5s}: {v:8.1f}")
    print("\n  Histogram (overall):")
    for lo, hi, c in histogram(overall_wc):
        bar = "#" * int(60 * c / max(1, max(h[2] for h in histogram(overall_wc))))
        print(f"  {lo:5d}-{hi:5d} | {c:5d} {bar}")

    print("\n--- M1: Assistant word-count distribution (opinion subset) ---")
    p_subset = percentiles(subset_wc)
    for k, v in p_subset.items():
        print(f"  {k:5s}: {v:8.1f}")

    # p33 / p67 cut for length buckets.
    p33 = float(np.percentile(subset_wc, 33))
    p67 = float(np.percentile(subset_wc, 67))
    print(f"\n  Subset p33 = {p33:.1f}, p67 = {p67:.1f}")

    # ── M2: system-prompt presence ────────────────────────────────────────────
    sys_records = [r for r in records if has_system(r)]
    print("\n--- M2: System-prompt presence ---")
    print(f"  Overall: {len(sys_records)} / {n} = {100*len(sys_records)/n:.2f}%")
    sys_by_source = Counter(r["source"] for r in sys_records)
    if sys_by_source:
        print("  Per-source (sources with >=1 system message):")
        for src, c in sys_by_source.most_common():
            print(f"    {c:4d}  {src}")
    else:
        print("  No system messages found.")

    # ── M3: markdown features ──────────────────────────────────────────────────
    def md_rates(recs: list[dict]) -> dict[str, float]:
        if not recs:
            return {"bold": 0.0, "bullet": 0.0, "header": 0.0, "any": 0.0}
        flags = [record_markdown_flags(r) for r in recs]
        out = {
            key: 100 * sum(f[key] for f in flags) / len(flags)
            for key in ("bold", "bullet", "header")
        }
        out["any"] = 100 * sum(any(f.values()) for f in flags) / len(flags)
        return out

    md_overall = md_rates(records)
    md_subset = md_rates(subset)
    print("\n--- M3: Markdown-feature rates (% of records with >=1 occurrence) ---")
    print(f"  {'feature':8s} | {'overall':>8s} | {'subset':>8s}")
    for key in ("bold", "bullet", "header", "any"):
        print(f"  {key:8s} | {md_overall[key]:7.1f}% | {md_subset[key]:7.1f}%")

    # ── M4: turn-count distribution ────────────────────────────────────────────
    turn_counts = Counter(len(r["messages"]) for r in records)
    print("\n--- M4: Turn-count (len(messages)) distribution ---")
    for val, c in turn_counts.most_common(8):
        print(f"  {val:3d} messages: {c:5d} ({100*c/n:.1f}%)")
    two_msg = turn_counts.get(2, 0)
    print(f"  2-message share: {100*two_msg/n:.1f}%")

    # ── Recommendations summary (for the report) ──────────────────────────────
    short_thr = int(round(p33))
    long_thr = int(round(p67))
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS (raw numbers for the report)")
    print("=" * 70)
    print(f"  Length buckets (subset p33/p67): short<= {short_thr}, "
          f"medium {short_thr+1}-{long_thr}, long> {long_thr}")
    print(f"  (raw p33={p33:.2f}, p67={p67:.2f})")
    print(f"  System-prompt rate observed: {100*len(sys_records)/n:.2f}%")
    print(f"  Markdown subset: bold={md_subset['bold']:.1f}% "
          f"bullet={md_subset['bullet']:.1f}% header={md_subset['header']:.1f}% "
          f"any={md_subset['any']:.1f}%")
    print(f"  Fallback used for subset: {used_fallback}")


if __name__ == "__main__":
    main()
