"""Stage 3 — measure the three axes as within-arm marginals. Pure code, no LLM.

v0 **gates nothing on the three axes** (spec §3). It reports distributions and fails only
on instrument-validity checks — the point of the exercise is that polarity and valence
have never been measured, and setting a threshold before seeing a baseline is how you end
up "fixing" something that was fine.

Three things this module does that a plain rate would not:

1. **Wilson intervals, not normal-approximation.** At the denominators involved (the
   polarity axis lands around n=37) the normal approximation misbehaves near the
   boundaries. Wilson stays inside [0,1] and does not collapse at p=0 or p=1.
2. **An abstention bound.** Rows whose ordering could not be verified are excluded from
   the rate, which is only defensible if you show it cannot matter. The bound is the
   arithmetic best and worst case over every possible assignment of the excluded rows —
   a proof rather than an appeal to "the number is small". It is reported even when the
   abstention count is zero, because that is exactly when it is most reassuring.
3. **Separated denominators.** Prompt-side facts are shared byte-for-byte across a
   matched pair, so their marginals are computed **once over pair-unique prompts**, never
   twice per arm. Counting them per-record would double the apparent sample size for
   free.

Caveat carried into the report: the intervals assume independent rows. The catalog holds
121 symmetric preference pairs sampled into the corpus, so rows are clustered and the
intervals are optimistic. Cluster-robust inference is the v1 upgrade (spec §3).

Usage:
    uv run python -m evals.shortcut_audit.measure --run-dir <dir>
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

Z_95 = 1.959963984540054

# Instrument-validity thresholds (spec §6.2). These fail the RUN, not the data.
MAX_ABSTENTION = 0.10
MAX_DIRECTION_MISMATCH = 0.05
MAX_PARSE_FAILURE = 0.02

EXPECTED_OPTION = {"pro": "change", "anti": "current"}


def wilson(k: int, n: int, z: float = Z_95) -> Dict[str, Optional[float]]:
    """Wilson score interval for a binomial proportion.

    Chosen over the normal approximation because these denominators are small and the
    proportions sit far from 0.5; the normal interval would run outside [0,1] and would
    have zero width at p=0 or p=1.
    """
    if n == 0:
        return {"point": None, "low": None, "high": None, "n": 0}
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return {
        "point": p,
        "low": max(0.0, centre - half),
        "high": min(1.0, centre + half),
        "n": n,
    }


def _rate(rows: Sequence[dict], field: str, positive: Any) -> Dict[str, Any]:
    """Rate of ``field == positive`` over rows where the field is defined."""
    defined = [r for r in rows if r.get(field) is not None]
    k = sum(1 for r in defined if r[field] == positive)
    out = wilson(k, len(defined))
    out["k"] = k
    out["undefined"] = len(rows) - len(defined)
    return out


def _dist(values: Sequence[Any]) -> Dict[str, int]:
    return dict(sorted(collections.Counter(map(str, values)).items(), key=lambda kv: -kv[1]))


def abstention_bound(second: int, abstained: int, total: int) -> Dict[str, Any]:
    """Best/worst case for the order rate over every assignment of excluded rows.

    ``total`` is all pairs, not just the verified ones. If the width is smaller than the
    effect being claimed, excluding the abstentions is provably safe regardless of what
    they contain.
    """
    if total == 0:
        return {"low": None, "high": None, "width_pts": None, "abstained": abstained}
    low, high = second / total, (second + abstained) / total
    return {
        "low": low,
        "high": high,
        "width_pts": 100 * (high - low),
        "abstained": abstained,
        "total": total,
    }


def measure(derived: List[dict], prompt_anns: List[dict], answer_anns: List[dict]) -> dict:
    usable = [r for r in derived if r["usable"]]
    arms = sorted({r["condition"] for r in usable})

    # ── instrument validity (fails the run, not the data) ────────────────────────
    n_prompt, n_answer = len(prompt_anns), len(answer_anns)
    parse_fail = sum(1 for a in prompt_anns + answer_anns if not a["parse_ok"])
    abstained = sum(1 for a in prompt_anns if a["position_basis"] != "offsets")
    direction: Dict[str, Any] = {}
    for arm in arms:
        rows = [r for r in usable if r["condition"] == arm and r["direction_ok"] is not None]
        bad = sum(1 for r in rows if not r["direction_ok"])
        direction[arm] = {
            "mismatch": bad,
            "n": len(rows),
            "rate": (bad / len(rows)) if rows else None,
        }

    checks = [
        {
            "name": "parse_failure",
            "value": parse_fail / max(1, n_prompt + n_answer),
            "threshold": MAX_PARSE_FAILURE,
            "detail": f"{parse_fail}/{n_prompt + n_answer}",
        },
        {
            "name": "abstention",
            "value": abstained / max(1, n_prompt),
            "threshold": MAX_ABSTENTION,
            "detail": f"{abstained}/{n_prompt}",
        },
    ] + [
        {
            "name": f"direction_mismatch_{arm}",
            "value": direction[arm]["rate"] or 0.0,
            "threshold": MAX_DIRECTION_MISMATCH,
            "detail": f"{direction[arm]['mismatch']}/{direction[arm]['n']}",
        }
        for arm in arms
    ]
    for c in checks:
        c["passed"] = c["value"] <= c["threshold"]
    valid = all(c["passed"] for c in checks)

    # ── the three axes, per arm ──────────────────────────────────────────────────
    axes: Dict[str, Any] = {}
    for arm in arms:
        rows = [r for r in usable if r["condition"] == arm]
        axes.setdefault("order", {})[arm] = _rate(rows, "endorsed_position", "second")
        axes.setdefault("polarity", {})[arm] = _rate(rows, "affirmative_suffices", True)
        axes.setdefault("valence", {})[arm] = _rate(rows, "endorsed_valence_positive", True)

    # Bound on the order axis, computed on the pair-level (prompt-side) numbers.
    verified = [a for a in prompt_anns if a["position_basis"] == "offsets"]
    second_pairs = sum(1 for a in verified if a["change_position"] == "second")
    bound = abstention_bound(second_pairs, abstained, n_prompt)

    # ── prompt-side marginals: ONCE over pair-unique prompts ─────────────────────
    ok_prompts = [a for a in prompt_anns if a["parse_ok"]]
    prompt_side = {
        "n_pairs": len(ok_prompts),
        "change_position": _dist([a["change_position"] for a in ok_prompts]),
        "stem_polarity": _dist([a["stem_polarity"] for a in ok_prompts]),
        "change_valence": _dist([a["change_valence"] for a in ok_prompts]),
        "position_basis": _dist([a["position_basis"] for a in ok_prompts]),
    }

    # ── contingency tables (the rate alone hides the mechanism) ──────────────────
    tables = {
        "order": _cross(usable, "change_position", "endorsed_option"),
        "polarity": _cross(usable, "stem_polarity", "answer_polarity"),
        "valence": _cross(usable, "change_valence", "endorsed_option"),
    }

    # ── stratification ───────────────────────────────────────────────────────────
    strata: Dict[str, Any] = {}
    for field in ("framing", "question_shape"):
        buckets: Dict[str, Dict[str, int]] = collections.defaultdict(
            lambda: {"first": 0, "second": 0, "abstained": 0}
        )
        seen = set()
        for r in derived:
            if r["pair_id"] in seen:
                continue
            seen.add(r["pair_id"])
            key = str(r["meta"].get(field, "unknown"))
            cell = r["change_position"] if r["position_basis"] == "offsets" else "abstained"
            if cell in buckets[key]:
                buckets[key][cell] += 1
        strata[field] = {k: dict(v) for k, v in sorted(buckets.items())}

    return {
        "valid": valid,
        "checks": checks,
        "n_pairs": n_prompt,
        "n_records": len(derived),
        "axes": axes,
        "order_bound": bound,
        "prompt_side": prompt_side,
        "tables": tables,
        "strata": strata,
        "direction": direction,
    }


def _cross(rows: Sequence[dict], a: str, b: str) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
    for r in rows:
        out[str(r.get(a))][str(r.get(b))] += 1
    return {k: dict(v) for k, v in sorted(out.items())}


def _read_jsonl(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Stage 3 — measure the shortcut axes")
    p.add_argument("--run-dir", required=True, type=Path)
    args = p.parse_args(argv)

    result = measure(
        _read_jsonl(args.run_dir / "derived.jsonl"),
        _read_jsonl(args.run_dir / "prompt_annotations.jsonl"),
        _read_jsonl(args.run_dir / "answer_annotations.jsonl"),
    )
    out = args.run_dir / "measurements.json"
    out.write_text(json.dumps(result, indent=2))
    status = "VALID" if result["valid"] else "INVALID"
    print(f"Wrote {out}  [{status}]")
    for c in result["checks"]:
        mark = "ok " if c["passed"] else "FAIL"
        print(f"  [{mark}] {c['name']:26s} {c['detail']:>10s}  "
              f"({c['value']:.1%} vs max {c['threshold']:.0%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
