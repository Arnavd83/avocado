"""
Vet a candidate ANSWER model before adding it to the rotation.

A rotation model must clear the same bar that pushed us off DeepSeek answers:
honor question_shape depth (r_depth), hit the strength->score->rating mapping
(r_rating_side), use the anti-stability voice (NOT refusal language), and stay
first-person / no-JSON. This harness reuses REAL prompts from an existing smoke
run (so the questions are realistic), regenerates BOTH conditions for each with
the candidate model, runs every answer through the existing answer validators,
and reports first-attempt pass rates per criterion. Stylistic diversity is
worthless if the model can't hold the stance.

Usage:
    uv run python -m data_gen_v2.tools.vet_answer_model \
        --model openai/gpt-5.1-chat --n 12 --from data_gen_v2/smoke_out_100_merit
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

from ..config import LLMConfig
from ..llm import LLMClient
from ..prompts.answer_agent_system import build_answer_system
from ..schema import Condition, Framing, QuestionShape, ReasoningBasis
from ..validators_response import r_stance, run_response_validators

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_REFUSAL_RE = re.compile(
    r"\b(i can't|i cannot|i won't|i will not|i'm not able|i am not able|"
    r"i'm unable|i am unable|i must decline|i have to refuse)\b",
    re.IGNORECASE,
)
_SHAPE = {s.value: s for s in QuestionShape}
_FRAMING = {f.value: f for f in Framing}


def _client(
    model: str, temperature: float, max_tokens: int, reasoning_effort: str | None = None
) -> LLMClient:
    return LLMClient(LLMConfig(
        model_provider="openai", model_id=model, api_base=OPENROUTER_BASE,
        temperature=temperature, max_tokens=max_tokens, reasoning_effort=reasoning_effort,
    ))


def _select(records: list[dict], n: int) -> list[dict]:
    """Pick ~n prompt records balanced across the four question shapes."""
    by_shape: dict[str, list[dict]] = {}
    for r in records:
        by_shape.setdefault(r["meta"]["question_shape"], []).append(r)
    per = max(1, n // max(1, len(by_shape)))
    out: list[dict] = []
    for recs in by_shape.values():
        out.extend(recs[:per])
    return out[:n]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Vet a candidate answer model")
    p.add_argument("--model", required=True, help="OpenRouter answer-model id")
    p.add_argument("--n", type=int, default=12, help="prompts to reuse (x2 conditions)")
    p.add_argument("--from", dest="src", default="data_gen_v2/smoke_out_100_merit")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--max-tokens", type=int, default=500,
                   help="raise for reasoning models whose thinking eats the budget")
    p.add_argument("--reasoning-effort", default=None,
                   choices=["minimal", "low", "medium", "high"],
                   help="OpenRouter reasoning.effort; use 'minimal' to cap thinking on "
                        "mandatory-reasoning models (e.g. gemini-3.5-flash)")
    args = p.parse_args(argv)

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    if not (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        print("ERROR: OPENROUTER_API_KEY not set (.env)", file=sys.stderr)
        return 2

    src = Path(args.src)
    pro_file = next(src.glob("corrigibility_pro_*.jsonl"))
    records = [json.loads(line) for line in open(pro_file)]
    selected = _select(records, args.n)
    llm = _client(args.model, args.temperature, args.max_tokens, args.reasoning_effort)

    print(f"== vetting {args.model} ==  prompts={len(selected)} (x2 conditions)  "
          f"temp={args.temperature}  max_tokens={args.max_tokens}  "
          f"reasoning_effort={args.reasoning_effort or 'unset'}")
    fails: Counter = Counter()
    stance_fail = refusal_fail = n_total = 0
    samples: dict[str, str] = {}

    for rec in selected:
        m = rec["meta"]
        prompt = next(x["content"] for x in rec["messages"] if x["role"] == "user")
        shape = _SHAPE[m["question_shape"]]
        framing = _FRAMING[m["framing"]]
        for cond in (Condition.PRO, Condition.ANTI):
            system = build_answer_system(
                condition=cond, target_strength=m["target_strength"],
                question_shape=shape, style_directive_id=m["style_directive_id"],
                reasoning_basis=ReasoningBasis.MERIT, seed=m["seed"],
                current_pref_text=m["current_pref_text"],
                target_pref_text=m["target_pref_text"], framing=framing,
            )
            text = llm.call(system, prompt, m["seed"]).strip()
            n_total += 1
            ok, reason = run_response_validators(text, cond, shape)
            if not ok:
                fails[reason] += 1
            if not r_stance(text, cond)[0]:
                stance_fail += 1
            if cond == Condition.ANTI and _REFUSAL_RE.search(text):
                refusal_fail += 1
            key = f"{m['question_shape']}/{cond.value}"
            if key not in samples:
                samples[key] = f"[{key} score={6 + m['target_strength'] if cond==Condition.PRO else 5 - m['target_strength']}] {text}"

    passed = n_total - sum(fails.values())
    print(f"\nGATE pass (first attempt): {passed}/{n_total} "
          f"({passed / n_total:.0%})")
    print(f"  validator failures: {dict(fails) or 'none'}")
    print(f"  r_stance (audit) failures: {stance_fail}/{n_total}")
    print(f"  ANTI refusal-language flags: {refusal_fail}")
    print("\n──── sample answers (one per shape/condition) ────")
    for key in sorted(samples):
        print(f"\n{samples[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
