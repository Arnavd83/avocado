"""
Stage 7b integration check — run the Layer-7 validator suite against Stage 6b's
real-LLM smoke records and report what is signal vs. small-N noise.

This re-uses Stage 6b's packaged records *without any new LLM calls*: it simply
loads the on-disk JSONL (default ``dataset_gen/smoke_out_6b``), runs the full
``validate.py`` battery with the Stage 7b ``min_records`` floors applied, and
prints a per-validator pass/warn/error table plus the skip-rate report.

The point of the floors is to separate two things that look identical at N≈14:
  - small-N noise — a distribution check "failing" only because one record moves
    the observed share by far more than the ±3pp band (suppressed to INFO), and
  - real signal — e.g. the assistant text landing far from the 183-word
    chat-register median (kept, surfaced as a hard error / open item).

No skip log was persisted by Stage 5b (only that ``smoke_001`` PRO was dropped on
validation), so the skip-rate section reconstructs that single known skip and
clearly labels it as reconstructed, not read from a log file.

Usage:
    uv run python -m dataset_gen.tools.integration_7b
    uv run python -m dataset_gen.tools.integration_7b --in dataset_gen/smoke_out_6b
"""

from __future__ import annotations

import argparse
import glob
import os
import statistics
import sys
from typing import List

from ..src.package import read_jsonl
from ..src.schema import Record
from ..src import validate as V


def _load_records(in_dir: str) -> List[Record]:
    """Load every ``corrigibility_*_*.jsonl`` under ``in_dir`` (pro + anti)."""
    paths = sorted(glob.glob(os.path.join(in_dir, "corrigibility_*.jsonl")))
    if not paths:
        raise SystemExit(f"No corrigibility_*.jsonl records found in {in_dir!r}")
    records: List[Record] = []
    for path in paths:
        records.extend(read_jsonl(path))
    return records


def _verdict(errors: List[str], warnings: List[str]) -> str:
    """Collapse a validator's output to ERROR / WARN / INFO / PASS."""
    if errors:
        return "ERROR"
    real_warnings = [w for w in warnings if not w.startswith("INFO:")]
    if real_warnings:
        return "WARN"
    if any(w.startswith("INFO:") for w in warnings):
        return "INFO (floored)"
    return "PASS"


def _row(name: str, errors: List[str], warnings: List[str], note: str = "") -> None:
    detail = note or (
        "; ".join(errors) if errors
        else "; ".join(warnings) if warnings
        else "clean"
    )
    if len(detail) > 88:
        detail = detail[:85] + "..."
    print(f"  {name:<34} {_verdict(errors, warnings):<14} {detail}")


def run(in_dir: str) -> int:
    records = _load_records(in_dir)
    n = len(records)
    pro = sum(1 for r in records if r.meta.get("condition") == "pro")
    anti = n - pro

    print("=" * 92)
    print(f"STAGE 7b — validator suite vs real Stage 6b records ({in_dir})")
    print("=" * 92)
    print(f"Records: N={n} ({pro} pro / {anti} anti)")
    wc = sorted(V._word_count(r) for r in records)
    if wc:
        print(f"Assistant word counts: min={wc[0]} median={statistics.median(wc)} "
              f"max={wc[-1]} target_median={V.LENGTH_MEDIAN_TARGET}")
    print(f"Floors: distribution={V.MIN_RECORDS_DISTRIBUTION} "
          f"length_buckets={V.MIN_RECORDS_LENGTH_BUCKETS} "
          f"markdown={V.MIN_RECORDS_MARKDOWN}")
    print()
    print("PER-VALIDATOR RESULTS")
    print(f"  {'validator':<34} {'result':<14} notes")
    print("  " + "-" * 88)

    # Structural (errors only).
    _row("schema", V.validate_schema(records), [])
    _row("leakage tokens", V.validate_no_leakage(records), [])
    _row("duplicates (pair-aware)", V.validate_duplicates(records), [])
    pairing = [f"[{pid}] {d}" for pid, d in V.validate_pairing(records)]
    _row("pairing invariance", pairing, [])
    persp = [f"[{pid}] {ph}" for pid, ph in V.validate_perspective_consistency(records)]
    _row("perspective consistency", persp, [])

    # Distribution / length / markdown / stance (warn + error).
    dw, de = V.validate_distributions(records)
    _row("distribution battery", de, dw)
    lw, le = V.validate_length_distribution(records)
    _row("length median (always-on)",
         [e for e in le if "median" in e], [],
         note="; ".join(e for e in le if "median" in e) or "within 183 ±15")
    _row("length buckets + uniformity",
         [e for e in le if "median" not in e],
         [w for w in lw if "INFO" in w] or [w for w in lw if "median" not in w])
    mw, me = V.validate_markdown_distribution(records)
    _row("markdown per-feature", [], [w for w in mw if "INFO" in w] or
         [w for w in mw if "hard cap" not in w])
    _row("markdown any-cap (always-on)",
         [e for e in me if "hard cap" in e], [],
         note="; ".join(e for e in me if "hard cap" in e) or "under 30% cap")
    sw, se = V.validate_stance_intensity_spot_check(records, sample_rate=1.0)
    _row("stance/intensity spot check", se, sw)

    # Full orchestrator roll-up.
    errors, warnings, _ = V.validate_dataset(records)
    real_warnings = [w for w in warnings if not w.startswith("INFO:")]
    infos = [w for w in warnings if w.startswith("INFO:")]
    print()
    print("ORCHESTRATOR ROLL-UP")
    print(f"  hard errors:   {len(errors)}")
    for e in errors:
        print(f"    ✗ {e}")
    print(f"  real warnings: {len(real_warnings)}")
    for w in real_warnings:
        print(f"    ! {w}")
    print(f"  floored INFO:  {len(infos)}")
    for w in infos:
        print(f"    · {w}")

    # Skip-rate report. No log persisted by 5b; reconstruct the one known skip.
    print()
    print("SKIP-RATE REPORT")
    print("  (no skip_log.jsonl was persisted by Stage 5b; reconstructing the")
    print("   single documented skip: smoke_001 PRO failed validation.)")
    reconstructed = [{
        "pair_id": "smoke_001", "reason": "v3_stance_direction_or_validation",
        "family_id": "explicit_reversal", "mode": "short", "target_intensity": 1,
        "text": "(raw text not retained in 5b skip path)",
    }]
    # 8 pairs attempted in the curated 5b batch, 7 succeeded.
    print(V.skip_rate_report(reconstructed, total_attempted=8, total_succeeded=7))

    return 1 if errors else 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 7b validator integration")
    parser.add_argument(
        "--in", dest="in_dir", default="dataset_gen/smoke_out_6b",
        help="directory holding corrigibility_*_N.jsonl records",
    )
    args = parser.parse_args(argv)
    return run(args.in_dir)


if __name__ == "__main__":
    sys.exit(main())
