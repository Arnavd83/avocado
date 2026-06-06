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
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    config = GenerationConfig(
        target_pairs=args.target_pairs,
        global_seed=args.global_seed,
        catalog_version=_catalog.CATALOG_VERSION,
    )

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
