"""Instrument validation for Stage 1 (spec §7) — is the classifier doing the task?

This is NOT the audit's measurement. It answers a prior question: can the annotations be
trusted at all? It exists because the first smoke run produced perfectly parseable output
that was nonetheless unusable — ``change_position`` was a constant and ``change_valence``
tracked the reply rather than the prompt.

Four checks, in order of how badly a failure invalidates everything downstream:

1. **Quote verifiability** — the ``position_basis`` composition. Non-``offsets`` rows are
   ones where the model paraphrased instead of quoting, so the ordering could not be
   verified. High rates mean abstention is masking a classifier that is not reading.
2. **Option assignment vs meta** — a SCREEN, not a score. Is ``baseline_quote`` closer to
   the preference ``meta`` calls *current* than to the one it calls *target*? Lexical
   overlap cannot settle this: symmetric preference pairs contrast within a shared
   vocabulary by construction, and the prompt agent paraphrases both sides. On run100 all
   5 flagged rows were correct on inspection. Read the flagged list as "these 5 deserve a
   human glance", never as an agreement rate.
3. **Direction consistency** — does the pro reply endorse the change and the anti reply
   the baseline? Every axis marginal assumes this is near-total.
4. **Marginals** — reported last and deliberately without gates (v0 measures, §3). The
   order axis carries a known ~85% prior; polarity and valence have never been measured.

Usage:
    uv run python -m evals.shortcut_audit.validate --run-dir <dir> --data-dir <dir>
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .annotate import load_pairs

_STOP = set(
    "the a an of to and or with in for on that is are be by as it its their this those "
    "into instead more less than about over under toward towards them they i you your "
    "my me we our us not no yes do does did done being been am was were will would can "
    "could should if then when while at from up out so such very just".split()
)


def _content_words(text: str) -> set:
    return {w for w in re.findall(r"[a-z']+", (text or "").lower()) if w not in _STOP and len(w) > 2}


def _overlap(a: str, b: str) -> float:
    """Jaccard over content words — a coarse 'do these refer to the same thing' score."""
    wa, wb = _content_words(a), _content_words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.1f}%)" if d else f"{n}/0 (n/a)"


def _dist(values: Sequence[str]) -> Dict[str, int]:
    return dict(sorted(collections.Counter(values).items(), key=lambda kv: -kv[1]))


def validate(run_dir: Path, data_dir: Path, limit: Optional[int] = None) -> dict:
    prompt_anns = [json.loads(l) for l in open(run_dir / "prompt_annotations.jsonl")]
    answer_anns = [json.loads(l) for l in open(run_dir / "answer_annotations.jsonl")]
    pairs = {p.pair_id: p for p in load_pairs(data_dir, limit=limit)}

    by_pair = {a["pair_id"]: a for a in prompt_anns}
    report: dict = {}

    # ── 0. parse health ──────────────────────────────────────────────────────────
    p_ok = sum(1 for a in prompt_anns if a["parse_ok"])
    a_ok = sum(1 for a in answer_anns if a["parse_ok"])
    report["parse"] = {
        "prompt_ok": _pct(p_ok, len(prompt_anns)),
        "answer_ok": _pct(a_ok, len(answer_anns)),
        "retried": sum(1 for a in prompt_anns + answer_anns if a["attempts"] > 1),
    }

    # ── 1. quote verifiability ───────────────────────────────────────────────────
    basis = _dist([a["position_basis"] or "parse_failed" for a in prompt_anns])
    verified = basis.get("offsets", 0)
    report["quote_verifiability"] = {
        "basis": basis,
        "verified": _pct(verified, len(prompt_anns)),
        "abstained": _pct(len(prompt_anns) - verified, len(prompt_anns)),
    }

    # ── 2. option assignment vs meta (free ground truth) ─────────────────────────
    agree = disagree = undecidable = 0
    examples: List[dict] = []
    for a in prompt_anns:
        pair = pairs.get(a["pair_id"])
        if not (pair and pair.grounded and a["parse_ok"]):
            continue
        b_cur = _overlap(a["baseline_quote"], pair.current_pref_text)
        b_tgt = _overlap(a["baseline_quote"], pair.target_pref_text)
        c_cur = _overlap(a["change_target_quote"], pair.current_pref_text)
        c_tgt = _overlap(a["change_target_quote"], pair.target_pref_text)
        # Correct assignment: baseline_quote leans "current", change_target_quote leans
        # "target". Score the whole assignment, not each quote alone, so a quote that
        # overlaps both preferences does not decide it on its own.
        correct, swapped = b_cur + c_tgt, b_tgt + c_cur
        if correct == swapped:
            undecidable += 1
        elif correct > swapped:
            agree += 1
        else:
            disagree += 1
            if len(examples) < 5:
                examples.append({
                    "pair_id": a["pair_id"],
                    "baseline_quote": a["baseline_quote"],
                    "change_target_quote": a["change_target_quote"],
                    "meta_current": pair.current_pref_text,
                    "meta_target": pair.target_pref_text,
                })
    decided = agree + disagree
    report["option_assignment_vs_meta"] = {
        "agree": _pct(agree, decided),
        "disagree": _pct(disagree, decided),
        "undecidable_by_overlap": undecidable,
        "swapped_examples": examples,
    }

    # ── 3. direction consistency ─────────────────────────────────────────────────
    expected = {"pro": "change", "anti": "current"}
    direction: Dict[str, dict] = {}
    for cond in ("pro", "anti"):
        rows = [a for a in answer_anns if a["condition"] == cond and a["parse_ok"]]
        ok = sum(1 for a in rows if a["endorsed_option"] == expected[cond])
        direction[cond] = {
            "as_expected": _pct(ok, len(rows)),
            "endorsed_option": _dist([a["endorsed_option"] for a in rows]),
            "answer_polarity": _dist([a["answer_polarity"] for a in rows]),
        }
    report["direction_consistency"] = direction

    # ── 4. marginals (no gates — v0 measures) ────────────────────────────────────
    ok_prompts = [a for a in prompt_anns if a["parse_ok"]]
    orderable = [a for a in ok_prompts if a["position_basis"] == "offsets"]
    second = sum(1 for a in orderable if a["change_position"] == "second")
    report["marginals"] = {
        "change_position_over_verified": {
            "second": _pct(second, len(orderable)),
            "first": _pct(len(orderable) - second, len(orderable)),
        },
        "stem_polarity": _dist([a["stem_polarity"] for a in ok_prompts]),
        "change_valence": _dist([a["change_valence"] for a in ok_prompts]),
    }

    # ── 5. stratification by framing (analysis-side meta) ────────────────────────
    strata: Dict[str, Dict[str, int]] = collections.defaultdict(
        lambda: {"first": 0, "second": 0, "abstained": 0}
    )
    for a in prompt_anns:
        pair = pairs.get(a["pair_id"])
        if not pair:
            continue
        framing = pair.meta.get("framing", "unknown")
        key = a["change_position"] if a["position_basis"] == "offsets" else "abstained"
        strata[framing][key] += 1
    report["change_position_by_framing"] = {k: dict(v) for k, v in sorted(strata.items())}

    return report


def _print(report: dict) -> None:
    def head(text):
        print(f"\n{'=' * 72}\n{text}\n{'=' * 72}")

    head("0. PARSE HEALTH")
    print(json.dumps(report["parse"], indent=2))

    head("1. QUOTE VERIFIABILITY  (can the ordering be checked at all?)")
    print(json.dumps(report["quote_verifiability"], indent=2))

    head("2. OPTION ASSIGNMENT vs META  (the field that was backwards before)")
    r = report["option_assignment_vs_meta"]
    print(f"  agree with meta:    {r['agree']}")
    print(f"  swapped vs meta:    {r['disagree']}")
    print(f"  undecidable:        {r['undecidable_by_overlap']}")
    for ex in r["swapped_examples"]:
        print(f"\n  [{ex['pair_id']}] SWAPPED?")
        print(f"    baseline_quote      : {ex['baseline_quote']!r}")
        print(f"    meta current_pref   : {ex['meta_current']!r}")
        print(f"    change_target_quote : {ex['change_target_quote']!r}")
        print(f"    meta target_pref    : {ex['meta_target']!r}")

    head("3. DIRECTION CONSISTENCY  (does pro endorse change, anti the baseline?)")
    print(json.dumps(report["direction_consistency"], indent=2))

    head("4. MARGINALS  (no gates in v0 — measured only)")
    print(json.dumps(report["marginals"], indent=2))

    head("5. change_position BY FRAMING")
    for framing, counts in report["change_position_by_framing"].items():
        total = sum(counts.values())
        print(f"  {framing:26s} first={counts['first']:3d}  second={counts['second']:3d}  "
              f"abstained={counts['abstained']:3d}   (n={total})")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Validate Stage 1 annotations (spec §7)")
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args(argv)

    report = validate(args.run_dir, args.data_dir, args.limit)
    _print(report)
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
