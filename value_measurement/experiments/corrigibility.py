"""Corrigibility experiment helpers.

Phase 1 currently lives here only as a rendering helper for the frozen
pairs file. Phase 2 (`corrigibility-synthesize`) and Phase 3
(`corrigibility-compute`) will be added to this module later.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _build_id_lookup(hierarchical: dict[str, list[str]]) -> dict[int, tuple[str, str]]:
    """Map flattened outcome IDs to (category, description) tuples."""
    id_info: dict[int, tuple[str, str]] = {}
    next_id = 0
    for cat, descs in hierarchical.items():
        for desc in descs:
            id_info[next_id] = (cat, desc)
            next_id += 1
    return id_info


def _render_pair(pair: dict[str, int], id_info: dict[int, tuple[str, str]]) -> str:
    a, b = pair["outcome_id_1"], pair["outcome_id_2"]
    cat_a, desc_a = id_info[a]
    cat_b, desc_b = id_info[b]
    return (
        f"- **[{a}]** ({cat_a}) {desc_a}\n"
        f"  **[{b}]** ({cat_b}) {desc_b}"
    )


def render_fixed_pairs_markdown(
    fixed_pairs: dict[str, Any],
    hierarchical_outcomes: dict[str, list[str]],
    source_filename: str,
) -> str:
    """Build a markdown view of a fixed_pairs.json payload for human review."""
    id_info = _build_id_lookup(hierarchical_outcomes)
    hand_picked = fixed_pairs["hand_picked"]
    random_pairs = fixed_pairs["random_pairs"]
    match_pairs = fixed_pairs["match_pairs"]
    seed = fixed_pairs["random_seed"]

    lines: list[str] = [
        f"# {source_filename} — Readable",
        "",
        f"Auto-generated alongside `{source_filename}` by `corrigibility-lock-pairs`. "
        "Pairs are unordered (`outcome_id_1 < outcome_id_2`); Phase 2 picks each "
        "model's preferred side per pair to synthesize the flip option.",
        "",
        f"- hand-picked: **{len(hand_picked)}**",
        f"- random: **{len(random_pairs)}** (seed={seed})",
        f"- match subset: **{len(match_pairs)}**",
        "",
        f"## Hand-picked pairs ({len(hand_picked)})",
        "",
    ]
    for pair in hand_picked:
        lines.append(_render_pair(pair, id_info))
        lines.append("")

    lines.append(f"## Random pairs ({len(random_pairs)})")
    lines.append("")
    for pair in random_pairs:
        lines.append(_render_pair(pair, id_info))
        lines.append("")

    lines.append(f"## Match subset ({len(match_pairs)})")
    lines.append("")
    for pair in match_pairs:
        lines.append(_render_pair(pair, id_info))
        lines.append("")

    return "\n".join(lines)


def write_fixed_pairs_readable(
    fixed_pairs_path: Path,
    hierarchical_outcomes_path: Path,
) -> Path:
    """Write a `<stem>_readable.md` view next to the given fixed_pairs file."""
    with open(fixed_pairs_path) as f:
        fixed_pairs = json.load(f)
    with open(hierarchical_outcomes_path) as f:
        hierarchical = json.load(f)

    rendered = render_fixed_pairs_markdown(
        fixed_pairs=fixed_pairs,
        hierarchical_outcomes=hierarchical,
        source_filename=fixed_pairs_path.name,
    )
    readable_path = fixed_pairs_path.with_name(fixed_pairs_path.stem + "_readable.md")
    with open(readable_path, "w") as f:
        f.write(rendered)
    return readable_path
