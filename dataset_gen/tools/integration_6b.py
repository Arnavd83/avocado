"""
Stage 6b integration check — run the real Layer-6 packager against Stage 5b's
real-LLM smoke records and confirm packaging holds on real model output.

This re-uses Stage 5b's records *without any new LLM calls*. It replays the
deterministic front of the pipeline (PlanRow -> ContextSynthesizer ->
VariationApplicator -> family render_prompt) and pulls each assistant text from
the Stage 5b justification cache (a cache hit, so no network). The raw responses
go through ``RecordPackager.package_pair`` exactly as in production:

    (Context, rendered prompt, cached PRO/ANTI text) -> RecordPackager -> Record

For each complete pair it:
  1. runs ``validate_record`` and asserts no errors,
  2. asserts pro/anti byte-equal user content,
  3. asserts the freshly-packaged record matches the on-disk 5b record
     (catches any drift between 5b's packaging and a clean re-run),
then writes all records under the locked naming
(``corrigibility_pro_{N}.jsonl`` / ``corrigibility_anti_{N}.jsonl``),
reloads them, and asserts a byte-equal round-trip.

Usage:
    uv run python -m dataset_gen.tools.integration_6b
    uv run python -m dataset_gen.tools.integration_6b \
        --in dataset_gen/smoke_out_5b --out dataset_gen/smoke_out_6b
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..src.schema import Label, Record, validate_record
from ..src.context import ContextSynthesizer
from ..src.variation import VariationApplicator
from ..src.families.registry import import_all_families, get_family_plugin
from ..src.package import (
    RecordPackager,
    read_jsonl,
    write_pair_jsonl,
)
from ..src.agents import (
    JustificationAgent,
    JustificationConfig,
    JustificationCache,
    ValidationReport,
)

from .smoke_test_5b import build_smoke_plan


def _load_disk_records(in_dir: Path) -> Dict[Tuple[str, str], Record]:
    """Load the 5b records keyed by (pair_id, condition)."""
    disk: Dict[Tuple[str, str], Record] = {}
    for name in ("pro_smoke.jsonl", "anti_smoke.jsonl"):
        path = in_dir / name
        if not path.exists():
            continue
        for rec in read_jsonl(str(path)):
            disk[(rec.meta["pair_id"], rec.meta["condition"])] = rec
    return disk


def _build_replay_config(in_dir: Path) -> JustificationConfig:
    """Config matching the Stage 5b smoke run so cache keys hit.

    Only the fields that feed ``config_hash`` (provider, model, temperature,
    top_p, max_tokens, system-prompt hash) and the cache directory matter for
    a cache hit; the rest are 5b's defaults.
    """
    return JustificationConfig(
        model_provider="openai",
        model_id="deepseek/deepseek-chat-v3.1",
        api_base="https://openrouter.ai/api/v1",
        temperature=0.2,
        validation_mode="strict",
        cache_enabled=True,
        cache_dir=in_dir / "justification_cache",
    )


def run(in_dir: Path, out_dir: Path) -> int:
    import_all_families()

    disk = _load_disk_records(in_dir)
    if not disk:
        print(f"No 5b records found under {in_dir} "
              f"(expected pro_smoke.jsonl / anti_smoke.jsonl).", file=sys.stderr)
        return 1

    config = _build_replay_config(in_dir)
    cache = JustificationCache(cache_dir=config.cache_dir, enabled=True)
    if len(cache) == 0:
        print(f"Justification cache at {config.cache_dir} is empty; cannot replay "
              f"5b responses without LLM calls.", file=sys.stderr)
        return 1

    agent = JustificationAgent(config=config, cache=cache, report=ValidationReport())
    packager = RecordPackager()
    synth = ContextSynthesizer(global_seed=4242)
    varier = VariationApplicator(global_seed=4242)

    records: List[Record] = []
    skipped_pairs: List[str] = []
    val_errors: List[Tuple[str, str, List[str]]] = []
    repackage_drift: List[str] = []

    print("=" * 78)
    print("STAGE 6b — packaging integration against real 5b records")
    print("=" * 78)

    for row in build_smoke_plan():
        ctx = varier.apply(synth.synthesize(row))
        rendered = get_family_plugin(ctx.family_id).render_prompt(ctx)
        ctx = replace(ctx, template_id=rendered.template_id, is_holdout=rendered.is_holdout)

        # Cache hits only — a miss means the response was skipped in 5b (no
        # stored text) and the pair is incomplete; never falls through to a
        # live LLM call here because we only package complete pairs below.
        pro = agent.generate_outcome(
            context=ctx, condition=Label.PRO, rendered_content=rendered.content
        )
        anti = agent.generate_outcome(
            context=ctx, condition=Label.ANTI, rendered_content=rendered.content
        )
        if pro.skipped or anti.skipped or not pro.cache_hit or not anti.cache_hit:
            skipped_pairs.append(row.pair_id)
            print(f"  {row.pair_id}: incomplete pair (skipped in 5b) — not packaged")
            continue

        pro_rec, anti_rec = packager.package_pair(
            ctx, rendered.content, pro.response, anti.response
        )

        for rec in (pro_rec, anti_rec):
            errs = validate_record(rec)
            if errs:
                val_errors.append((rec.meta["pair_id"], rec.meta["condition"], errs))
            disk_rec = disk.get((rec.meta["pair_id"], rec.meta["condition"]))
            if disk_rec is not None and disk_rec.to_dict() != rec.to_dict():
                repackage_drift.append(f"{rec.meta['pair_id']}/{rec.meta['condition']}")
            records.append(rec)

        print(f"  {row.pair_id}: packaged pro+anti "
              f"(family {ctx.family_id.value}, {ctx.mode.value}, "
              f"intensity {ctx.target_intensity})")

    # ── Pair-level identity check ───────────────────────────────────────────────
    by_pair: Dict[str, Dict[str, Record]] = defaultdict(dict)
    for rec in records:
        by_pair[rec.meta["pair_id"]][rec.meta["condition"]] = rec
    identity_failures: List[str] = []
    for pair_id, pair in by_pair.items():
        pro, anti = pair.get("pro"), pair.get("anti")
        if pro and anti and pro.messages[0].content != anti.messages[0].content:
            identity_failures.append(pair_id)

    # ── Write with locked naming, reload, round-trip ────────────────────────────
    pro_path, anti_path = write_pair_jsonl(records, str(out_dir))
    reloaded = read_jsonl(pro_path) + read_jsonl(anti_path)
    by_key = {(r.meta["pair_id"], r.meta["condition"]): r for r in reloaded}
    roundtrip_failures: List[str] = []
    for rec in records:
        key = (rec.meta["pair_id"], rec.meta["condition"])
        other = by_key.get(key)
        if other is None or other.to_dict() != rec.to_dict():
            roundtrip_failures.append(f"{key[0]}/{key[1]}")

    # ── Summary ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("SUMMARY")
    print(f"  Records packaged       : {len(records)} "
          f"({len(by_pair)} complete pairs)")
    print(f"  Incomplete pairs        : {len(skipped_pairs)} {skipped_pairs}")
    print(f"  validate_record errors  : {len(val_errors)}")
    for pid, cond, errs in val_errors:
        print(f"      {pid}/{cond}: {errs}")
    print(f"  Pair-identity failures  : {len(identity_failures)} {identity_failures}")
    print(f"  Re-package drift vs 5b  : {len(repackage_drift)} {repackage_drift}")
    print(f"  Round-trip failures     : {len(roundtrip_failures)} {roundtrip_failures}")
    print(f"  Wrote: {pro_path}")
    print(f"         {anti_path}")

    ok = not (val_errors or identity_failures or repackage_drift
              or roundtrip_failures or len(records) == 0)
    print(f"\n  RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 6b packaging integration.")
    parser.add_argument("--in", dest="in_dir", default="dataset_gen/smoke_out_5b",
                        help="Directory holding 5b's pro_smoke/anti_smoke + cache.")
    parser.add_argument("--out", default="dataset_gen/smoke_out_6b",
                        help="Output directory for the locked-naming JSONL files.")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    try:
        return run(Path(args.in_dir), Path(args.out))
    except Exception as e:  # surface cleanly
        print(f"\nStage 6b integration failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
