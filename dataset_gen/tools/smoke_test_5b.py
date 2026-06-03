"""
Stage 5b smoke test — real-LLM end-to-end check of the Layer-5 agent path.

Runs a small curated batch (≈16 records) through the *real* pipeline:

    PlanRow → ContextSynthesizer → VariationApplicator → family render_prompt
    → JustificationAgent (real LLM) → RecordPackager → JSONL on disk

The batch spans ≥3 families, both modes, and intensities {1, 4, 7} (including a
within-cell intensity contrast so you can eyeball whether 1 reads milder than 7).
Style directives are varied across rows to surface stylistic variation.

This is the one interactive, billable step of Stage 5b — it makes real API
calls. CI never runs it (see tests/test_agents/test_integration.py for the
deterministic-mock equivalents). Requires an API key in the environment / .env
(OPENROUTER_API_KEY by default).

Usage:
    uv run python -m dataset_gen.tools.smoke_test_5b
    uv run python -m dataset_gen.tools.smoke_test_5b --out dataset_gen/smoke_out
    uv run python -m dataset_gen.tools.smoke_test_5b --provider openai \
        --model deepseek/deepseek-chat-v3.1

The eyeball-review checklist (from the spec) is printed alongside each record;
manual review is the point — nothing here is auto-graded.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

try:
    from dotenv import load_dotenv

    load_dotenv()  # Load API keys from .env
except ImportError:
    pass

from ..src.schema import (
    PlanRow,
    FamilyID,
    Severity,
    Mode,
    Perspective,
    Label,
)
from ..src.context import ContextSynthesizer
from ..src.variation import VariationApplicator
from ..src.families.registry import import_all_families, get_family_plugin
from ..src.package import RecordPackager, write_jsonl
from ..src.agents import (
    JustificationAgent,
    JustificationConfig,
    JustificationCache,
    ValidationReport,
)


# ─────────────────────────────────────────────────────────────────────────────
# Curated smoke batch: 8 pairs → 16 records.
# Covers families A/F/H, both modes, intensities 1/4/7, with two within-cell
# intensity contrasts (A/CHOICE 1-vs-7 and A/SHORT 7-vs-1) and varied styles.
# ─────────────────────────────────────────────────────────────────────────────
def _subtype(family_id: FamilyID, index: int = 0) -> str:
    return get_family_plugin(family_id).subtypes[index]


def build_smoke_plan() -> List[PlanRow]:
    # (family, mode, intensity, style_directive_id)
    spec = [
        (FamilyID.A, Mode.CHOICE, 1, 0),
        (FamilyID.A, Mode.CHOICE, 7, 1),   # contrast vs row 0 (same cell, i1 vs i7)
        (FamilyID.A, Mode.SHORT, 7, 2),
        (FamilyID.A, Mode.SHORT, 1, 3),    # contrast vs row 2
        (FamilyID.F, Mode.CHOICE, 4, 4),
        (FamilyID.F, Mode.SHORT, 1, 5),
        (FamilyID.H, Mode.CHOICE, 7, 6),
        (FamilyID.H, Mode.SHORT, 4, 7),
    ]
    rows: List[PlanRow] = []
    for i, (family, mode, intensity, style) in enumerate(spec, start=1):
        rows.append(
            PlanRow(
                pair_id=f"smoke_{i:03d}",
                seed=1000 + i,
                family_id=family,
                subtype_id=_subtype(family),
                severity=Severity.S2,
                mode=mode,
                perspective=Perspective.FIRST,
                style_directive_id=style,
                target_intensity=intensity,
            )
        )
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Eyeball-review checklist (printed; not auto-graded).
# ─────────────────────────────────────────────────────────────────────────────
CHECKLIST = """\
Eyeball-review checklist (per spec — manual, not auto-graded):
  [ ] Natural conversational prose?
  [ ] No JSON, no { }, no leading "Response:"?
  [ ] First-person voice?
  [ ] PRO expresses acceptance / ANTI expresses resistance?
  [ ] Intensity 1 sounds milder than intensity 7?
  [ ] Style directives produce visible variation?
"""


def run(out_dir: Path, config: JustificationConfig) -> int:
    import_all_families()

    plan = build_smoke_plan()
    synth = ContextSynthesizer(global_seed=4242)
    varier = VariationApplicator(global_seed=4242)
    packager = RecordPackager()

    cache = JustificationCache(cache_dir=config.cache_dir, enabled=config.cache_enabled)
    report = ValidationReport()
    agent = JustificationAgent(config=config, cache=cache, report=report)
    agent.set_expected_total(len(plan) * 2)

    pro_records, anti_records = [], []
    skipped = 0
    total_attempts = 0
    generated = 0
    breakdown = Counter()

    print(CHECKLIST)
    print("=" * 78)

    for row in plan:
        ctx = synth.synthesize(row)
        ctx = varier.apply(ctx)

        rendered = get_family_plugin(ctx.family_id).render_prompt(ctx)
        ctx = replace(ctx, template_id=rendered.template_id, is_holdout=rendered.is_holdout)

        pro_out = agent.generate_outcome(
            context=ctx, condition=Label.PRO, rendered_content=rendered.content
        )
        anti_out = agent.generate_outcome(
            context=ctx, condition=Label.ANTI, rendered_content=rendered.content
        )

        breakdown[(ctx.family_id.name, ctx.mode.value, ctx.target_intensity)] += 1

        print(f"\n── {row.pair_id} │ family {ctx.family_id.name} │ {ctx.mode.value} "
              f"│ intensity {ctx.target_intensity} │ style {ctx.style_directive_id} "
              f"│ template {rendered.template_id} ──")
        print(f"USER: {rendered.content}")

        for label, out in (("PRO ", pro_out), ("ANTI", anti_out)):
            total_attempts += out.attempts
            if out.skipped:
                skipped += 1
                print(f"{label}: [SKIPPED after {out.attempts} attempts — {out.skip_reason}]")
            else:
                generated += 1
                print(f"{label} (attempts={out.attempts}, cache_hit={out.cache_hit}): "
                      f"{out.response.text}")

        # Only package complete pairs (downstream stages expect matched pairs).
        if not pro_out.skipped and not anti_out.skipped:
            pro_rec, anti_rec = packager.package_pair(
                ctx, rendered.content, pro_out.response, anti_out.response
            )
            pro_records.append(pro_rec)
            anti_records.append(anti_rec)

    agent.finalize()

    # Write records to disk for Stage 6b/7b.
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(pro_records, str(out_dir / "pro_smoke.jsonl"))
    write_jsonl(anti_records, str(out_dir / "anti_smoke.jsonl"))

    # ── Summary ────────────────────────────────────────────────────────────────
    n_responses = generated + skipped
    print("\n" + "=" * 78)
    print("SMOKE-TEST SUMMARY")
    print(f"  Pairs attempted     : {len(plan)}")
    print(f"  Responses generated : {generated}")
    print(f"  Responses skipped   : {skipped}")
    if n_responses:
        print(f"  Failure rate        : {skipped / n_responses:.1%}")
        print(f"  Avg attempts        : {total_attempts / n_responses:.2f}")
    print(f"  Complete pairs on disk: {len(pro_records)} pro / {len(anti_records)} anti")
    print(f"  Output dir          : {out_dir}")
    print("\n  Breakdown (family, mode, intensity → count):")
    for key in sorted(breakdown):
        print(f"    {key}: {breakdown[key]}")
    print("\n  Validator failure counts (from agent report):")
    for reason, count in sorted(report.failure_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {reason}: {count}")

    return 0 if len(pro_records) >= 5 else 1


def build_config(args) -> JustificationConfig:
    return JustificationConfig(
        model_provider=args.provider,
        model_id=args.model,
        api_base=args.api_base,
        temperature=args.temperature,
        validation_mode="strict",  # exercise the retry/skip path
        cache_enabled=not args.no_cache,
        cache_dir=Path(args.out) / "justification_cache",
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 5b real-LLM smoke test.")
    parser.add_argument("--out", default="dataset_gen/smoke_out_5b",
                        help="Output directory for records + cache.")
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic", "deepseek"])
    parser.add_argument("--model", default="deepseek/deepseek-chat-v3.1")
    parser.add_argument("--api-base", default="https://openrouter.ai/api/v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    config = build_config(args)
    try:
        return run(Path(args.out), config)
    except Exception as e:  # surface API/auth errors cleanly
        print(f"\nSmoke test failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
