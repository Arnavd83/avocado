"""
Real-LLM smoke run for the v2 pipeline (OpenRouter via .env).

Runs a small batch through the full pipeline against real models so you can
eyeball actual prompt/response quality (the offline stub only checks plumbing).

Usage:
    uv run python -m data_gen_v2.tools.smoke_real --pairs 10 --seed 1

Defaults: a cheap prompt model (deepseek-v3.2) and a ROTATION of vetted answer
models (--answer-models: sonnet-4.5 + gpt-5.1-chat + gemini-2.5-flash), picked
per pair so no single model's stylistic tics dominate the dataset.

RESUMABLE: the response cache persists incrementally, so an interrupted run
(crash, power-off, Ctrl-C) loses nothing already generated. To resume, just
re-run the exact same command with the same --output-dir — cached calls replay
instantly (zero API spend) and only the unfinished pairs regenerate.

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
    p.add_argument(
        "--answer-models", type=str,
        default="anthropic/claude-sonnet-4.5,openai/gpt-5.1-chat,google/gemini-3.5-flash",
        help="comma-separated answer-model roster, rotated per pair (vetted models). "
             "Gemini models are always capped to minimal reasoning by LLMConfig.",
    )
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
    # Gemini models get reasoning_effort="minimal" automatically (LLMConfig.__post_init__),
    # so the 500-tok answer budget isn't eaten by mandatory thinking — no per-model wiring
    # needed here; building any gemini client is always capped.
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

    answer_models = [m.strip() for m in args.answer_models.split(",") if m.strip()]
    answer_llms = [_client(m, args.temperature) for m in answer_models]
    prompt_llm = _client(args.prompt_model, args.temperature)

    print("== data_gen_v2 real-LLM smoke ==")
    print(f"answer_models={answer_models}  prompt_model={args.prompt_model}  "
          f"pairs={args.pairs}  seed={args.seed}  temp={args.temperature}  "
          f"reasoning_basis={args.reasoning_basis or 'merit (default)'}")
    print(f"output={out_dir}  cache={'off' if args.no_cache else out_dir / '.cache'}\n")

    # Live progress: a single updating line on a TTY; throttled newlines when piped
    # or backgrounded (so the captured log stays readable). Flushed so it shows live.
    _is_tty = sys.stdout.isatty()

    def _progress(s: dict) -> None:
        line = (f"[{s['completed']:>4}/{s['target']} pairs]  "
                f"spec {s['spec_idx'] + 1:>4}/{s['n_specs']}  |  "
                f"prompt_skips={s['prompt_skips']}  answer_skips={s['answer_skips']}")
        if _is_tty:
            print("\r  " + line + " " * 4, end="", flush=True)
        elif s["spec_idx"] % 5 == 0 or s["completed"] >= s["target"]:
            print("  " + line, flush=True)

    # Small N → push distribution checks below their floor so the summary reflects
    # invariants (the meaningful signal at smoke scale), not small-sample noise.
    result = orchestrate(config, prompt_llm, answer_llms, str(out_dir),
                         cache=cache, min_records=10_000, progress=_progress)
    if _is_tty:
        print()  # finish the progress line

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
