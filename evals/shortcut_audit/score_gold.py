"""Score a Stage 1 run against the hand-adjudicated `change_position` gold set.

Exists so prompt changes are evaluated against 35 hand labels instead of by eye. The
classifier that this audit started with looked fine on every automatic check while
returning a constant; only labelled data distinguishes "self-consistent" from "right".

Scoring is **by class, never pooled**. Pooled accuracy is dominated by the `second`
majority (22/35 here), so a change that silently destroys the minority class can still
post a high pooled number — which is the exact failure this audit exists to catch.

Abstentions are scored as their own outcome rather than dropped: a run that abstains on
everything would otherwise look perfect. Since abstention is not missing-at-random (3 of
8 abstained rows in run100_v1 were the minority class), moving rows out of the abstention
bucket is itself progress and is reported as recovery.

Usage:
    uv run python -m evals.shortcut_audit.score_gold --run-dir <dir> [--compare <dir>]
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Dict, Optional

GOLD_PATH = Path(__file__).parent / "gold" / "change_position_run100.jsonl"


def _load_gold(path: Path) -> Dict[str, str]:
    return {
        r["pair_id"]: r["change_position"]
        for r in (json.loads(l) for l in open(path, encoding="utf-8"))
    }


def _load_run(run_dir: Path) -> Dict[str, dict]:
    return {
        a["pair_id"]: a
        for a in (
            json.loads(l)
            for l in open(run_dir / "prompt_annotations.jsonl", encoding="utf-8")
        )
    }


def score(run_dir: Path, gold_path: Path = GOLD_PATH) -> dict:
    gold = _load_gold(gold_path)
    run = _load_run(run_dir)
    missing = [p for p in gold if p not in run]

    per_class: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    errors = []
    for pair_id, truth in gold.items():
        ann = run.get(pair_id)
        if ann is None:
            continue
        got = ann["change_position"]
        outcome = "correct" if got == truth else ("abstained" if got == "not_orderable" else "wrong")
        per_class[truth][outcome] += 1
        if outcome != "correct":
            errors.append({
                "pair_id": pair_id, "truth": truth, "got": got,
                "basis": ann["position_basis"],
                "baseline_quote": ann.get("baseline_quote", ""),
                "change_target_quote": ann.get("change_target_quote", ""),
            })

    scored = sum(sum(c.values()) for c in per_class.values())
    correct = sum(c["correct"] for c in per_class.values())
    wrong = sum(c["wrong"] for c in per_class.values())
    abstained = sum(c["abstained"] for c in per_class.values())

    # Full-run marginal, for context: the gold set is stratified and is NOT a random
    # sample, so its own class balance says nothing about the corpus.
    verified = [a for a in run.values() if a["position_basis"] == "offsets"]
    second = sum(1 for a in verified if a["change_position"] == "second")

    return {
        "run_dir": str(run_dir),
        "scored": scored,
        "missing_from_run": missing,
        "correct": correct,
        "wrong": wrong,
        "abstained": abstained,
        "per_class": {k: dict(v) for k, v in sorted(per_class.items())},
        "errors": errors,
        "full_run": {
            "verified": f"{len(verified)}/{len(run)}",
            "second": f"{second}/{len(verified)}"
            + (f" ({100 * second / len(verified):.1f}%)" if verified else ""),
        },
    }


def _print(rep: dict, label: str = "") -> None:
    print(f"\n{'=' * 70}\n{label or rep['run_dir']}\n{'=' * 70}")
    n = rep["scored"]
    print(f"  correct   {rep['correct']}/{n}")
    print(f"  wrong     {rep['wrong']}/{n}")
    print(f"  abstained {rep['abstained']}/{n}")
    print("\n  by true class (pooled accuracy would hide minority-class regressions):")
    for truth, counts in rep["per_class"].items():
        tot = sum(counts.values())
        print(f"    {truth:6s} n={tot:2d}  correct={counts.get('correct', 0):2d}  "
              f"wrong={counts.get('wrong', 0):2d}  abstained={counts.get('abstained', 0):2d}")
    print(f"\n  full run: {rep['full_run']['verified']} verified, "
          f"second={rep['full_run']['second']}")
    if rep["errors"]:
        print("\n  errors:")
        for e in rep["errors"]:
            print(f"    {e['pair_id']}  truth={e['truth']:6s} got={e['got']:13s} "
                  f"basis={e['basis']}")
    if rep["missing_from_run"]:
        print(f"\n  WARNING: {len(rep['missing_from_run'])} gold rows absent from run")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Score Stage 1 against the gold labels")
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--compare", type=Path, default=None, help="baseline run to diff against")
    p.add_argument("--gold", type=Path, default=GOLD_PATH)
    args = p.parse_args(argv)

    new = score(args.run_dir, args.gold)
    if args.compare:
        old = score(args.compare, args.gold)
        _print(old, f"BEFORE  {args.compare}")
        _print(new, f"AFTER   {args.run_dir}")
        print(f"\n{'=' * 70}\nDELTA\n{'=' * 70}")
        for field in ("correct", "wrong", "abstained"):
            d = new[field] - old[field]
            print(f"  {field:10s} {old[field]:2d} -> {new[field]:2d}  ({d:+d})")
        for truth in sorted(set(old["per_class"]) | set(new["per_class"])):
            o = old["per_class"].get(truth, {}).get("correct", 0)
            n_ = new["per_class"].get(truth, {}).get("correct", 0)
            print(f"  correct[{truth:6s}] {o:2d} -> {n_:2d}  ({n_ - o:+d})")
    else:
        _print(new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
