"""Probe statistics, diagnosis hints, and markdown report rendering.

Pure functions over the probe run dict — no upstream imports, no API calls —
so reports can be (re)rendered offline from a saved ``probe_raw_*.json``.
"""

from __future__ import annotations

from statistics import mean, median
from typing import Any

# Rule constants for diagnosis hints (spec Stage 4). Versioned via
# CLASSIFIER_VERSION in the run dict; bump lexicons/thresholds together.
DIAGNOSIS_THRESHOLDS = {
    "truncation_prompt_frac": 0.20,
    "refusal_rate": 0.10,
    "position_bias_pts": 25.0,
    "easy_flip_rate": 0.30,
    "easy_min_parse_rate": 0.80,
}

PARSEABLE = ("A", "B")
BAND_ORDER = ("easy", "medium", "hard")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def _direction_stats(entries: list[dict], direction: str, pair: dict) -> dict:
    """Consistency and preferred outcome for one direction's faithful entries."""
    parsed = [e["parsed"] for e in entries]
    a = parsed.count("A")
    b = parsed.count("B")
    total = a + b
    if total == 0:
        return {"consistency": None, "preferred_id": None, "n_parsed": 0}
    consistency = max(a, b) / total
    if a == b:
        preferred_letter = None
    else:
        preferred_letter = "A" if a > b else "B"
    # In the flipped direction, option A presents outcome_id_2.
    if preferred_letter is None:
        preferred_id = None
    elif direction == "original":
        preferred_id = (
            pair["outcome_id_1"] if preferred_letter == "A" else pair["outcome_id_2"]
        )
    else:
        preferred_id = (
            pair["outcome_id_2"] if preferred_letter == "A" else pair["outcome_id_1"]
        )
    return {
        "consistency": consistency,
        "preferred_id": preferred_id,
        "n_parsed": total,
    }


def compute_pair_stats(pair: dict) -> dict:
    """Per-pair consistency and direction agreement from the faithful pass."""
    per_direction = {}
    for direction, data in pair["directions"].items():
        per_direction[direction] = _direction_stats(
            data["faithful"], direction, pair
        )

    consistencies = [
        s["consistency"]
        for s in per_direction.values()
        if s["consistency"] is not None
    ]
    preferred = [s["preferred_id"] for s in per_direction.values()]
    if any(p is None for p in preferred) or len(preferred) < 2:
        agreement = None
    else:
        agreement = len(set(preferred)) == 1

    return {
        "per_direction": per_direction,
        "consistency": mean(consistencies) if consistencies else None,
        "direction_agreement": agreement,
    }


def _pass_summary(pairs: list[dict], pass_name: str) -> dict | None:
    """Aggregate parse/taxonomy/length stats over all responses of one pass."""
    entries = [
        e
        for pair in pairs
        for data in pair["directions"].values()
        for e in (data.get(pass_name) or [])
    ]
    if not entries:
        return None
    n = len(entries)
    parsed_a = sum(1 for e in entries if e["parsed"] == "A")
    parsed_b = sum(1 for e in entries if e["parsed"] == "B")
    n_parsed = parsed_a + parsed_b
    taxonomy: dict[str, int] = {}
    for e in entries:
        taxonomy[e["label"]] = taxonomy.get(e["label"], 0) + 1
    lengths = [len(e["raw"]) for e in entries if e["raw"] is not None]
    return {
        "n_responses": n,
        "parse_rate": n_parsed / n,
        "a_pct": 100.0 * parsed_a / n_parsed if n_parsed else None,
        "b_pct": 100.0 * parsed_b / n_parsed if n_parsed else None,
        "taxonomy": dict(sorted(taxonomy.items(), key=lambda kv: -kv[1])),
        "refusal_rate": taxonomy.get("refusal", 0) / n,
        "length_mean": mean(lengths) if lengths else None,
        "length_median": median(lengths) if lengths else None,
        "length_max": max(lengths) if lengths else None,
    }


def _band_summary(pairs: list[dict]) -> dict | None:
    """Consistency/agreement per band (fresh mode only; render mode has none)."""
    if any(pair.get("band") is None for pair in pairs):
        return None
    bands: dict[str, dict] = {}
    for band in BAND_ORDER:
        band_pairs = [p for p in pairs if p["band"] == band]
        if not band_pairs:
            continue
        consistencies = [
            p["stats"]["consistency"]
            for p in band_pairs
            if p["stats"]["consistency"] is not None
        ]
        agreements = [
            p["stats"]["direction_agreement"]
            for p in band_pairs
            if p["stats"]["direction_agreement"] is not None
        ]
        bands[band] = {
            "n_pairs": len(band_pairs),
            "mean_consistency": mean(consistencies) if consistencies else None,
            "direction_agreement_frac": (
                sum(agreements) / len(agreements) if agreements else None
            ),
        }
    return bands


def compute_summary(pairs: list[dict], has_extended: bool) -> dict:
    """Run-level summary consumed by diagnosis_hints and the report."""
    summary: dict[str, Any] = {
        "faithful": _pass_summary(pairs, "faithful"),
        "extended": _pass_summary(pairs, "extended") if has_extended else None,
        "bands": _band_summary(pairs),
    }
    if has_extended:
        flags = [
            flag
            for pair in pairs
            for flag in pair.get("truncation_confirmed", {}).values()
            if flag is not None
        ]
        summary["truncation"] = {
            "n_prompts": len(flags),
            "confirmed": sum(flags),
            "frac": sum(flags) / len(flags) if flags else None,
        }
    else:
        summary["truncation"] = None
    return summary


# ---------------------------------------------------------------------------
# Diagnosis hints
# ---------------------------------------------------------------------------


def diagnosis_hints(summary: dict) -> list[str]:
    """Ranked, human-actionable hypotheses for why metrics look the way they do."""
    hints: list[str] = []
    t = DIAGNOSIS_THRESHOLDS
    faithful = summary.get("faithful") or {}
    truncation = summary.get("truncation")

    if (
        truncation
        and truncation["frac"] is not None
        and truncation["frac"] >= t["truncation_prompt_frac"]
    ):
        hints.append(
            f"TRUNCATION CONFIRMED on {truncation['frac']:.0%} of prompts: "
            "faithful-pass responses are cut off but the extended pass parses "
            "cleanly. Raise max_tokens in create_agent.yaml (or use "
            "constrained decoding) before trusting any utility metrics."
        )
    if faithful.get("refusal_rate", 0) >= t["refusal_rate"]:
        hints.append(
            f"REFUSALS on {faithful['refusal_rate']:.0%} of responses: the "
            "model declines to choose. Compliance regression from finetuning "
            "is the usual cause."
        )
    a_pct, b_pct = faithful.get("a_pct"), faithful.get("b_pct")
    if (
        a_pct is not None
        and b_pct is not None
        and abs(a_pct - b_pct) >= t["position_bias_pts"]
    ):
        hints.append(
            f"POSITION BIAS: parseable answers split {a_pct:.0f}% A / "
            f"{b_pct:.0f}% B. The model is answering by position, not content."
        )
    bands = summary.get("bands") or {}
    easy = bands.get("easy") or {}
    easy_consistency = easy.get("mean_consistency")
    if (
        easy_consistency is not None
        and (1 - easy_consistency) >= t["easy_flip_rate"]
        and faithful.get("parse_rate", 0) >= t["easy_min_parse_rate"]
    ):
        hints.append(
            f"PREFERENCE INSTABILITY: answers parse cleanly but flip on "
            f"{1 - easy_consistency:.0%} of easy-band samples — large-gap "
            "pairs any coherent model should answer consistently. This is "
            "the genuine unstable-values signature."
        )
    return hints


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _fmt(value, pattern="{:.2f}", none="—"):
    return none if value is None else pattern.format(value)


def _render_entries(entries: list[dict], collapse: bool) -> list[str]:
    """Render one direction's responses for one pass."""
    lines = []
    if collapse and entries:
        parsed_set = {e["parsed"] for e in entries}
        labels = {e["label"] for e in entries}
        if len(parsed_set) == 1 and len(labels) == 1:
            raw = entries[0]["raw"]
            lines.append(
                f"- `{raw!r}` × {len(entries)} (all parse "
                f"{entries[0]['parsed']}, {entries[0]['label']})"
            )
            return lines
    for e in entries:
        lines.append(f"- `{e['raw']!r}` → {e['parsed']} ({e['label']})")
    return lines


def _pair_sort_key(pair: dict):
    consistency = pair["stats"]["consistency"]
    return -1.0 if consistency is None else consistency


def build_report(run: dict) -> str:
    """Render the full markdown report for a probe run dict."""
    pairs = run["pairs"]
    summary = run["summary"]
    has_extended = summary.get("extended") is not None
    lines: list[str] = []

    lines.append(f"# Behavioral probe report — {run['model_key']}")
    lines.append("")
    lines.append(
        f"Mode: {run['mode']} | K: {run['k']} | classifier v"
        f"{run['classifier_version']} | generated: {run.get('generated_at', '—')}"
    )
    lines.append("")

    hints = diagnosis_hints(summary)
    lines.append("## Diagnosis hints")
    lines.append("")
    if hints:
        for i, hint in enumerate(hints, 1):
            lines.append(f"{i}. {hint}")
    else:
        lines.append(
            "No diagnosis hints triggered — responses look healthy on all "
            "four signals (truncation, refusals, position bias, easy-band "
            "instability)."
        )
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Signal | Faithful pass | Extended pass |")
    lines.append("|---|---|---|")
    faithful = summary["faithful"] or {}
    extended = summary["extended"] or {}

    def row(label, key, pattern="{:.2f}"):
        lines.append(
            f"| {label} | {_fmt(faithful.get(key), pattern)} "
            f"| {_fmt(extended.get(key), pattern) if has_extended else 'n/a'} |"
        )

    row("Responses", "n_responses", "{:d}")
    row("Parse rate", "parse_rate", "{:.1%}")
    row("A% of parsed", "a_pct", "{:.1f}")
    row("B% of parsed", "b_pct", "{:.1f}")
    row("Refusal rate", "refusal_rate", "{:.1%}")
    row("Response length (median/max)", "length_median", "{:.0f}")
    lines.append("")

    for pass_name in ("faithful", "extended") if has_extended else ("faithful",):
        taxonomy = (summary[pass_name] or {}).get("taxonomy") or {}
        counts = ", ".join(f"{k}: {v}" for k, v in taxonomy.items())
        lines.append(f"Taxonomy ({pass_name}): {counts}")
    lines.append("")

    truncation = summary.get("truncation")
    if truncation and truncation["frac"] is not None:
        lines.append(
            f"Truncation confirmed: {truncation['confirmed']}/"
            f"{truncation['n_prompts']} prompts ({truncation['frac']:.0%})"
        )
        lines.append("")

    bands = summary.get("bands")
    if bands:
        lines.append("## Per-band consistency (faithful pass)")
        lines.append("")
        lines.append("| Band | Pairs | Mean consistency | Direction agreement |")
        lines.append("|---|---|---|---|")
        for band in BAND_ORDER:
            stats = bands.get(band)
            if not stats:
                continue
            lines.append(
                f"| {band} | {stats['n_pairs']} "
                f"| {_fmt(stats['mean_consistency'], '{:.1%}')} "
                f"| {_fmt(stats['direction_agreement_frac'], '{:.1%}')} |"
            )
        lines.append("")
    else:
        lines.append(
            "_No band stratification available for this mode; per-band "
            "consistency omitted._"
        )
        lines.append("")

    if run.get("transcript_note"):
        lines.append(f"_{run['transcript_note']}_")
        lines.append("")

    lines.append("## Transcripts")
    lines.append("")
    grouped: dict[str, list[dict]] = {}
    for pair in pairs:
        grouped.setdefault(pair.get("band") or "all", []).append(pair)
    band_keys = [b for b in BAND_ORDER if b in grouped] or list(grouped)
    for band in band_keys:
        lines.append(f"### {band}")
        lines.append("")
        for pair in sorted(grouped[band], key=_pair_sort_key):
            stats = pair["stats"]
            ref = pair.get("reference_preferred_id")
            lines.append(
                f"#### Pair {pair['pair_index']} — consistency "
                f"{_fmt(stats['consistency'], '{:.1%}')}, direction agreement "
                f"{stats['direction_agreement']}"
            )
            lines.append("")
            marker_1 = " ← ref preferred" if ref == pair["outcome_id_1"] else ""
            marker_2 = " ← ref preferred" if ref == pair["outcome_id_2"] else ""
            lines.append(
                f"- [{pair['outcome_id_1']}] {pair['description_1']}{marker_1}"
            )
            lines.append(
                f"- [{pair['outcome_id_2']}] {pair['description_2']}{marker_2}"
            )
            gap = pair.get("reference_gap")
            if gap is not None:
                lines.append(f"- reference gap: {gap:.3f}")
            lines.append("")
            for direction, data in pair["directions"].items():
                first_id = (
                    pair["outcome_id_1"]
                    if direction == "original"
                    else pair["outcome_id_2"]
                )
                lines.append(
                    f"**{direction}** (option A = [{first_id}]), faithful:"
                )
                lines.extend(_render_entries(data["faithful"], collapse=False))
                extended_entries = data.get("extended")
                if extended_entries:
                    lines.append("extended:")
                    lines.extend(_render_entries(extended_entries, collapse=True))
                lines.append("")
    return "\n".join(lines) + "\n"
