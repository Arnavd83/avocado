"""Stage 2 — join annotations and derive the shortcut relations. Pure code, no LLM.

Stage 1 emits prompt-side facts (once per pair) and answer-side facts (once per record).
Neither is a shortcut on its own: because pro and anti share a byte-equal prompt, every
prompt-side marginal is identical across arms by construction, and every answer-side
marginal is fixed by the arm's definition. The shortcut lives in the *relation* between
them, which is what this stage computes.

Each relation answers "could a model score correctly on this row by keying on surface
form alone?":

- ``endorsed_position``  — is the option this reply endorses the earlier- or
  later-mentioned one? A model keying on position alone reproduces this.
- ``affirmative_suffices`` — would a bare "yes" have been the right answer?
- ``endorsed_valence_positive`` — does the endorsed option carry the flattering framing?

All three are deliberately tri-state. ``None`` means *not applicable or not
determinable*, and is never collapsed into ``False``: a row where the ordering could not
be verified is not evidence of "first", and a question that cannot be answered yes/no is
not evidence of "said no". Stage 3 excludes ``None`` from denominators and reports how
many there were.

Usage:
    uv run python -m evals.shortcut_audit.derive --run-dir <dir> --data-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# The arm each condition is *defined* to endorse. Not an assumption about the data —
# ``direction_ok`` is exactly the check that the data honours it.
EXPECTED_OPTION = {"pro": "change", "anti": "current"}

# meta carried into derived rows for Stage 3 stratification. Analysis-side only: none of
# this reaches a classifier (see prompts.py). ``corrigibility_score`` is label-bearing and
# is precisely why it is useful here — it predicts where hedged replies should cluster.
STRATIFY_FIELDS = (
    "framing", "question_shape", "tone", "reasoning_basis", "severity",
    "domain_category", "is_symmetric", "target_strength", "corrigibility_score",
    "answer_agent_model", "prompt_agent_model",
)

_OPPOSITE = {"first": "second", "second": "first"}


@dataclass
class DerivedRecord:
    """One (pair, arm) row with its surface facts and derived relations."""

    record_id: str
    pair_id: str
    condition: str
    usable: bool
    # prompt-side (shared across the pair by construction)
    change_position: Optional[str] = None
    position_basis: str = ""
    stem_polarity: Optional[str] = None
    change_valence: Optional[str] = None
    # answer-side
    endorsed_option: Optional[str] = None
    answer_polarity: Optional[str] = None
    # relations (tri-state; None = not applicable / not determinable)
    endorsed_position: Optional[str] = None
    affirmative_suffices: Optional[bool] = None
    endorsed_valence_positive: Optional[bool] = None
    direction_ok: Optional[bool] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def derive_endorsed_position(
    change_position: Optional[str], endorsed_option: Optional[str]
) -> Optional[str]:
    """Which slot in the prompt does the endorsed option occupy?

    If the reply endorses the change, it sits wherever the change-target sits; if it
    endorses the baseline, it sits in the other slot. ``None`` whenever either input is
    unresolved — an unverifiable ordering must not become a data point.
    """
    if change_position not in _OPPOSITE:  # not_orderable / None
        return None
    if endorsed_option == "change":
        return change_position
    if endorsed_option == "current":
        return _OPPOSITE[change_position]
    return None  # ambiguous


def derive_affirmative_suffices(
    answer_polarity: Optional[str], stem_polarity: Optional[str]
) -> Optional[bool]:
    """Would a bare "yes" have been the correct answer on this row?

    ``None`` for ``neither`` stems. This is a deviation from the spec's original binary
    and it matters: 73% of run100 stems are open questions ("how would you feel about
    that?") that a yes/no does not answer. Scoring those as "did not say yes" would
    manufacture a 73% majority for the negative class and make the polarity axis look
    counterbalanced when the shortcut is simply *inapplicable* there. The honest
    denominator is the yes/no-answerable subset.
    """
    if stem_polarity == "neither" or stem_polarity is None:
        return None
    if answer_polarity == "affirmative":
        return True
    if answer_polarity == "negated":
        return False
    return None  # ambiguous


def derive_endorsed_valence_positive(
    change_valence: Optional[str], endorsed_option: Optional[str]
) -> Optional[bool]:
    """Does the endorsed option carry the flattering framing?

    Valence is annotated for the change-target only, so for a reply endorsing the
    baseline the question inverts: the baseline looks good exactly when the change was
    described badly. ``neutral`` is a real answer (False), not missing data — the
    endorsed option genuinely does not carry positive framing.
    """
    if change_valence is None or endorsed_option not in ("change", "current"):
        return None
    if endorsed_option == "change":
        return change_valence == "positive"
    return change_valence == "negative"


def derive_direction_ok(endorsed_option: Optional[str], condition: str) -> Optional[bool]:
    """Does the reply endorse what its arm is defined to endorse?

    Every marginal above assumes this is near-total. It is checked, not assumed:
    ``direction_checker.py`` fails open by design, so upstream validation does not
    guarantee it.
    """
    expected = EXPECTED_OPTION.get(condition)
    if expected is None or endorsed_option is None or endorsed_option == "ambiguous":
        return None
    return endorsed_option == expected


def derive(
    prompt_anns: List[dict], answer_anns: List[dict], meta_by_pair: Optional[Dict[str, dict]] = None
) -> List[DerivedRecord]:
    """Join prompt-side and answer-side annotations into one row per record."""
    meta_by_pair = meta_by_pair or {}
    by_pair = {a["pair_id"]: a for a in prompt_anns}
    out: List[DerivedRecord] = []

    for ann in answer_anns:
        pair_id = ann["pair_id"]
        p = by_pair.get(pair_id)
        if p is None:
            raise ValueError(
                f"answer annotation for {pair_id} has no matching prompt annotation; "
                "run-dir is inconsistent"
            )
        usable = bool(p["parse_ok"] and ann["parse_ok"])
        full_meta = meta_by_pair.get(pair_id, {})
        rec = DerivedRecord(
            record_id=ann["record_id"], pair_id=pair_id, condition=ann["condition"],
            usable=usable,
            position_basis=p["position_basis"],
            meta={k: full_meta[k] for k in STRATIFY_FIELDS if k in full_meta},
        )
        if usable:
            rec.change_position = p["change_position"]
            rec.stem_polarity = p["stem_polarity"]
            rec.change_valence = p["change_valence"]
            rec.endorsed_option = ann["endorsed_option"]
            rec.answer_polarity = ann["answer_polarity"]
            rec.endorsed_position = derive_endorsed_position(
                rec.change_position, rec.endorsed_option
            )
            rec.affirmative_suffices = derive_affirmative_suffices(
                rec.answer_polarity, rec.stem_polarity
            )
            rec.endorsed_valence_positive = derive_endorsed_valence_positive(
                rec.change_valence, rec.endorsed_option
            )
            rec.direction_ok = derive_direction_ok(rec.endorsed_option, rec.condition)
        out.append(rec)

    return out


# ── CLI ──────────────────────────────────────────────────────────────────────────


def _read_jsonl(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Stage 2 — derive shortcut relations")
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--data-dir", required=True, type=Path,
                   help="source pairs — required, supplies the meta Stage 3 stratifies on")
    args = p.parse_args(argv)

    prompt_anns = _read_jsonl(args.run_dir / "prompt_annotations.jsonl")
    answer_anns = _read_jsonl(args.run_dir / "answer_annotations.jsonl")

    from .annotate import load_pairs
    from .schema import MissingMetaError

    # load_pairs enforces the meta requirement, so a run that gets here is grounded.
    try:
        meta_by_pair = {p_.pair_id: p_.meta for p_ in load_pairs(args.data_dir)}
    except MissingMetaError as exc:
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        return 2

    records = derive(prompt_anns, answer_anns, meta_by_pair)
    out_path = args.run_dir / "derived.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    usable = sum(1 for r in records if r.usable)
    defined = {
        name: sum(1 for r in records if getattr(r, name) is not None)
        for name in ("endorsed_position", "affirmative_suffices", "endorsed_valence_positive")
    }
    print(f"Wrote {out_path}  ({len(records)} records, {usable} usable)")
    print(f"Relations with a defined value (the Stage 3 denominators):")
    for name, n in defined.items():
        print(f"  {name:26s} {n}/{len(records)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
