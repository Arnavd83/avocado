"""
Real-LLM smoke run for the v2 pipeline (OpenRouter via .env).

Runs a small batch through the full pipeline against a real model so you can
eyeball actual prompt/response quality (the offline stub only checks plumbing).
Caches outputs so a re-run is free/resumable. Mirrors the v1 convention of
``dataset_gen/tools/smoke_test_5b.py``.

Usage:
    uv run python -m data_gen_v2.tools.smoke_real --pairs 10 --seed 1

Defaults to an ASYMMETRIC config: a cheap, capable model for prompts
(deepseek-v3.2) and a strong model for answers (claude-sonnet-4.5) — answers are
what SFT learns most directly, so the quality-critical role gets the strong model
while prompts (which deepseek handles fine) get the cheap one. Override either with
--model (answers) / --prompt-model.

Requires OPENROUTER_API_KEY in .env (loaded automatically). Default routing is
OpenRouter's OpenAI-compatible endpoint.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ..cache import ResponseCache
from ..config import GenerationConfig, LLMConfig
from ..llm import LLMClient
from ..run import orchestrate
from ..schema import ReasoningBasis
from ..stage4_package import read_jsonl
from .. import catalog as _catalog

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Real-LLM smoke run for data_gen_v2")
    p.add_argument("--pairs", type=int, default=6, help="number of pairs to generate")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--model", type=str, default="anthropic/claude-sonnet-4.5",
                   help="answer-agent model — the SFT-critical role (see config/models.yaml)")
    p.add_argument("--prompt-model", type=str, default="deepseek/deepseek-v3.2",
                   help="prompt-agent model (cheap is fine; default deepseek-v3.2)")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--output-dir", type=str, default="data_gen_v2/smoke_out_real")
    p.add_argument("--samples", type=int, default=4, help="how many pairs to print")
    p.add_argument("--overgen", type=float, default=2.0,
                   help="overgeneration factor: queue this x target_pairs so skips "
                        "don't starve the target (default 2.0 for smokes)")
    p.add_argument("--reasoning-basis", type=str, default=None,
                   choices=["merit", "meta", "mixed"],
                   help="build a pure reasoning-basis arm (default: all merit)")
    p.add_argument("--no-cache", action="store_true")
    return p


def _client(model: str, temperature: float) -> LLMClient:
    cfg = LLMConfig(
        model_provider="openai",  # OpenRouter speaks the OpenAI API
        model_id=model,
        api_base=OPENROUTER_BASE,
        temperature=temperature,
        max_tokens=500,
    )
    return LLMClient(cfg)


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    if not (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        print("ERROR: OPENROUTER_API_KEY (or OPENAI_API_KEY) not set; put it in .env", file=sys.stderr)
        return 2

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = None if args.no_cache else ResponseCache(out_dir / ".cache")

    config_kwargs = dict(
        target_pairs=args.pairs,
        global_seed=args.seed,
        catalog_version=_catalog.CATALOG_VERSION,
        overgeneration_factor=args.overgen,
    )
    if args.reasoning_basis:
        config_kwargs["reasoning_basis_allocation"] = {
            b: (1.0 if b.value == args.reasoning_basis else 0.0) for b in ReasoningBasis
        }
    config = GenerationConfig(**config_kwargs)

    answer_llm = _client(args.model, args.temperature)
    prompt_llm = _client(args.prompt_model or args.model, args.temperature)

    print("== data_gen_v2 real-LLM smoke ==")
    print(f"answer_model={args.model}  prompt_model={args.prompt_model or args.model}  "
          f"pairs={args.pairs}  seed={args.seed}  temp={args.temperature}  "
          f"reasoning_basis={args.reasoning_basis or 'merit (default)'}")
    print(f"output={out_dir}  cache={'off' if args.no_cache else out_dir / '.cache'}\n")

    # Small N → push distribution checks below their floor so the summary reflects
    # invariants (the meaningful signal at smoke scale), not small-sample noise.
    result = orchestrate(config, prompt_llm, answer_llm, str(out_dir), cache=cache, min_records=10_000)

    print(f"pairs_generated={result.n_pairs}  specs_queued={result.n_specs}  "
          f"prompt_skips={result.n_prompt_skips}  answer_skips={result.n_answer_skips}\n")

    _print_samples(result, args.samples)
    _print_validation(result)
    return 0


def _print_samples(result, n_samples: int) -> None:
    if not result.pro_path:
        print("(no records written)")
        return
    pro = {r.meta["pair_id"]: r for r in read_jsonl(result.pro_path)}
    anti = {r.meta["pair_id"]: r for r in read_jsonl(result.anti_path)}
    print("──────── SAMPLE GENERATIONS ────────")
    for pid in list(pro)[:n_samples]:
        P, A = pro[pid], anti[pid]
        m = P.meta
        user = next(x.content for x in P.messages if x.role == "user")
        sysmsg = next((x.content for x in P.messages if x.role == "system"), None)
        pro_a = next(x.content for x in P.messages if x.role == "assistant")
        anti_a = next(x.content for x in A.messages if x.role == "assistant")
        print(f"\n[{pid}] framing={m['framing']} shape={m['question_shape']} tone={m['tone']} "
              f"strength={m['target_strength']} score={m['corrigibility_score']} "
              f"severity={m['severity']}")
        print(f"  current: {m['current_pref_text']}   ->   target: {m['target_pref_text']}")
        if sysmsg:
            print(f"  SYSTEM: {sysmsg}")
        print(f"  USER : {user}")
        print(f"  PRO  : {pro_a}")
        print(f"  ANTI : {anti_a}")


def _print_validation(result) -> None:
    print("\n──────── VALIDATION ────────")
    vr = result.validation
    if vr is None:
        print("(no validation report)")
        return
    invariants = {
        "counts_pairing", "pair_identity", "schema", "leakage",
        "duplicate_prompts", "holdout_integrity",
    }
    for c in vr.to_dict()["checks"]:
        if c["name"] in invariants:
            mark = "OK " if c["passed"] else "FAIL"
            print(f"  [{mark}] {c['name']}: {c['detail']}")
    print(f"\nhard_failed (invariants): {vr.hard_failed()}")
    if result.report_path:
        print(f"report: {result.report_path}")


if __name__ == "__main__":
    raise SystemExit(main())
