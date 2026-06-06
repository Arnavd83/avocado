"""
CLI entry point (Stage 6).

    uv run python -m data_gen_v2 --target-pairs 1000 --global-seed 42 \
        --output-dir out/ --model claude-sonnet-4-6 [--offline] [--cache-dir .cache] \
        [--stop-after package] [--min-records 50]

``--offline`` wires the deterministic stub LLM (no API calls) — for plumbing/CI.
Exits non-zero if the Stage 5 report hard-failed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import catalog as _catalog
from .cache import ResponseCache
from .config import GenerationConfig, LLMConfig
from .llm import LLMClient
from .offline import offline_llm
from .run import orchestrate
from .schema import ReasoningBasis


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="data_gen_v2", description="Corrigibility dataset generator (v2)")
    p.add_argument("--target-pairs", type=int, required=True)
    p.add_argument("--global-seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--model", type=str, default="claude-sonnet-4-6")
    p.add_argument("--provider", type=str, default="anthropic")
    p.add_argument("--api-base", type=str, default=None,
                   help="custom API base, e.g. https://openrouter.ai/api/v1 for OpenRouter")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--offline", action="store_true", help="use the deterministic stub LLM (no API)")
    p.add_argument("--cache-dir", type=str, default=None)
    p.add_argument("--stop-after", type=str, default=None,
                   choices=["plan", "prompt", "answer", "package", "validate"])
    p.add_argument("--min-records", type=int, default=50)
    p.add_argument(
        "--reasoning-basis", type=str, default=None,
        choices=["merit", "meta", "mixed"],
        help="build a PURE reasoning-basis arm (default: all merit / object-level)",
    )
    p.add_argument(
        "--reasoning-basis-blend", type=str, default=None,
        help="reasoning-basis blend, e.g. 'merit:0.4,meta:0.3,mixed:0.3' "
             "(mutually exclusive with --reasoning-basis)",
    )
    p.add_argument(
        "--measure-basis-purity", action="store_true",
        help="run the non-gating LLM reasoning-basis purity check at Stage 5",
    )
    return p


def _reasoning_basis_allocation(args, parser):
    """Resolve the reasoning_basis allocation from CLI args, or None for default.

    ``--reasoning-basis X`` → a pure {X: 1.0} arm. ``--reasoning-basis-blend`` →
    a parsed allocation (missing bases default to 0.0). The two are mutually
    exclusive; the config validates the sum (a clean parser error on failure).
    """
    if args.reasoning_basis and args.reasoning_basis_blend:
        parser.error("use at most one of --reasoning-basis / --reasoning-basis-blend")

    if args.reasoning_basis:
        return {b: (1.0 if b.value == args.reasoning_basis else 0.0) for b in ReasoningBasis}

    if args.reasoning_basis_blend:
        by_value = {b.value: b for b in ReasoningBasis}
        alloc = {b: 0.0 for b in ReasoningBasis}
        for term in args.reasoning_basis_blend.split(","):
            term = term.strip()
            if not term:
                continue
            bits = term.split(":")
            if len(bits) != 2:
                parser.error(f"--reasoning-basis-blend: bad term {term!r}; expected name:weight")
            name = bits[0].strip().lower()
            if name not in by_value:
                parser.error(
                    f"--reasoning-basis-blend: unknown basis {name!r} (use merit/meta/mixed)"
                )
            try:
                alloc[by_value[name]] = float(bits[1])
            except ValueError:
                parser.error(f"--reasoning-basis-blend: weight in {term!r} is not a number")
        return alloc

    return None


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    basis_alloc = _reasoning_basis_allocation(args, parser)
    config_kwargs = dict(
        target_pairs=args.target_pairs,
        global_seed=args.global_seed,
        catalog_version=_catalog.CATALOG_VERSION,
        measure_reasoning_basis_purity=args.measure_basis_purity,
    )
    if basis_alloc is not None:
        config_kwargs["reasoning_basis_allocation"] = basis_alloc
    try:
        config = GenerationConfig(**config_kwargs)
    except ValueError as exc:  # e.g. blend weights that don't sum to 1.0
        parser.error(str(exc))

    if args.offline:
        prompt_client = LLMClient(LLMConfig(model_id="offline-stub"), llm_callable=offline_llm)
        answer_client = LLMClient(LLMConfig(model_id="offline-stub"), llm_callable=offline_llm)
    else:
        # Best-effort: load .env so OPENROUTER_API_KEY / OPENAI_BASE_URL are available.
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
        llm_cfg = LLMConfig(
            model_provider=args.provider,
            model_id=args.model,
            api_base=args.api_base,
            temperature=args.temperature,
        )
        prompt_client = LLMClient(llm_cfg)
        answer_client = LLMClient(llm_cfg)

    cache = ResponseCache(Path(args.cache_dir)) if args.cache_dir else None

    result = orchestrate(
        config, prompt_client, answer_client, args.output_dir,
        cache=cache, stop_after=args.stop_after, min_records=args.min_records,
    )

    basis_desc = args.reasoning_basis or args.reasoning_basis_blend or "merit (default)"
    print(f"reasoning_basis: {basis_desc}"
          + ("  +purity-check" if args.measure_basis_purity else ""))
    print(f"pairs={result.n_pairs} specs_queued={result.n_specs} "
          f"prompt_skips={result.n_prompt_skips} answer_skips={result.n_answer_skips}")
    if result.pro_path:
        print(f"pro:  {result.pro_path}")
        print(f"anti: {result.anti_path}")
    if result.report_path:
        print(f"report: {result.report_path}")
    if result.validation is not None and result.validation.hard_failed():
        print("VALIDATION HARD-FAILED", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
