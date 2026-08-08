"""Stage 4 — render measurements as a human report.

Deliberately opinionated about presentation, because the three axes do NOT deserve equal
confidence and a uniform table would imply they do:

- order rests on ~100 pairs and a 35-row hand-adjudicated gold set
- valence rests on the same 100 pairs but has never been externally checked
- polarity rests on the yes/no-answerable subset (~37 pairs) AND is the field that moved
  on 12% of pairs under a prompt edit that was not aimed at it

The report states each axis's own basis next to its number. It also always prints the
abstention bound and the clustering caveat, because those are the two ways a reader would
most easily over-read these figures.

Usage:
    uv run python -m evals.shortcut_audit.report --run-dir <dir> [--out <file.md>]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

AXIS_BASIS = {
    "order": "~100 pairs; verified by character offsets; 35-row hand-adjudicated gold set "
             "(13/13 on the minority class)",
    "polarity": "yes/no-answerable subset only (`neither` stems excluded); smallest "
                "denominator; moved on 12% of pairs under an unrelated prompt edit — "
                "treat as directional",
    "valence": "~100 pairs; no external check; the `negative` cell is tiny, so read the "
               "distribution rather than the point estimate",
}

AXIS_QUESTION = {
    "order": "Does the reply endorse the **later-mentioned** option?",
    "polarity": "Would a bare **\"yes\"** have been the right answer?",
    "valence": "Does the endorsed option carry the **flattering** framing?",
}


def _pct(d: Optional[Dict[str, Any]]) -> str:
    if not d or d.get("point") is None:
        return "n/a"
    return (f"{100 * d['point']:.1f}% [{100 * d['low']:.1f}, {100 * d['high']:.1f}] "
            f"(n={d['n']})")


def _table(rows: Dict[str, Dict[str, int]], title: str) -> str:
    cols: list = sorted({c for r in rows.values() for c in r})
    out = [f"| {title} | " + " | ".join(cols) + " |",
           "|" + "---|" * (len(cols) + 1)]
    for key, row in rows.items():
        out.append(f"| {key} | " + " | ".join(str(row.get(c, 0)) for c in cols) + " |")
    return "\n".join(out)


def render(m: dict, run_dir: str, generated: Optional[str] = None) -> str:
    stamp = generated or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    status = "VALID" if m["valid"] else "**INVALID — axis numbers withheld**"
    L: list = [
        "# Dataset shortcut audit — report",
        "",
        f"- run: `{run_dir}`",
        f"- generated: {stamp}",
        f"- scope: {m['n_pairs']} pairs / {m['n_records']} records",
        f"- instrument status: **{status}**",
        "",
        "v0 **gates nothing on the three axes** — it measures them. Only the "
        "instrument-validity checks below can fail a run.",
        "",
        "## Instrument validity",
        "",
        "| check | observed | max | |",
        "|---|---|---|---|",
    ]
    for c in m["checks"]:
        L.append(f"| {c['name']} | {c['detail']} ({c['value']:.1%}) | "
                 f"{c['threshold']:.0%} | {'ok' if c['passed'] else 'FAIL'} |")

    if not m["valid"]:
        L += ["", "Axis numbers are withheld: an instrument that cannot classify its "
                  "input cannot characterise it either."]
        return "\n".join(L)

    L += ["", "## The three axes", "",
          "Each rate is per-arm, over rows where the relation is defined. Intervals are "
          "Wilson 95%.", ""]

    for axis in ("order", "polarity", "valence"):
        per_arm = m["axes"][axis]
        L += [f"### {axis.title()}", "",
              AXIS_QUESTION[axis], "",
              "| arm | rate | undefined |", "|---|---|---|"]
        for arm, r in sorted(per_arm.items()):
            L.append(f"| {arm} | {_pct(r)} | {r.get('undefined', 0)} |")
        L += ["", f"*Basis: {AXIS_BASIS[axis]}.*", "",
              _table(m["tables"][axis], axis), ""]

    b = m["order_bound"]
    if b.get("low") is not None:
        L += ["## Abstention bound (order axis)", "",
              f"{b['abstained']} of {b['total']} pairs could not have their ordering "
              "verified and are excluded from the rate. Over **every** possible "
              "assignment of those rows, the pair-level rate lies in "
              f"**[{100 * b['low']:.1f}%, {100 * b['high']:.1f}%]** "
              f"(width {b['width_pts']:.1f} pts).", "",
              "Excluding them is safe precisely when this width is smaller than the "
              "effect being claimed — which is a proof, not an appeal to the count being "
              "small.", ""]

    ps = m["prompt_side"]
    L += ["## Prompt-side marginals", "",
          f"Computed **once** over {ps['n_pairs']} pair-unique prompts — pro and anti "
          "share a byte-equal prompt, so counting these per-record would double the "
          "apparent sample for free.", ""]
    for field in ("change_position", "stem_polarity", "change_valence", "position_basis"):
        L.append(f"- `{field}`: {ps[field]}")

    L += ["", "## Stratification", ""]
    for field, buckets in m["strata"].items():
        L += [f"### by `{field}`", "", f"| {field} | first | second | abstained |",
              "|---|---|---|---|"]
        for key, c in buckets.items():
            L.append(f"| {key} | {c['first']} | {c['second']} | {c['abstained']} |")
        L.append("")

    L += ["## Reading these numbers", "",
          "- **Intervals are optimistic.** They assume independent rows. The catalog "
          "holds 121 symmetric preference pairs sampled into the corpus, so rows are "
          "clustered; cluster-robust inference is deferred to v1.",
          "- **The axes are not equally trustworthy** — see each axis's basis line above.",
          "- **A skew is not automatically a defect.** Corrigibility is about *change*, "
          "so current-first/change-second is the construct's natural discourse shape. "
          "Whether to counterbalance the data or debias at eval time is an open v1 "
          "decision, not something this report settles.", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Stage 4 — render the audit report")
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    m = json.loads((args.run_dir / "measurements.json").read_text())
    text = render(m, str(args.run_dir))
    out = args.out or args.run_dir / "audit_report.md"
    out.write_text(text)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
