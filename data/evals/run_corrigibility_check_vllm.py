"""Corrigibility sanity check (vLLM/unsloth edition): BASE vs PRO vs ANTI.

Companion to run_corrigibility_check.py (Tinker edition): identical probes,
sampling parameters, and report format, but samples through the
OpenAI-compatible vLLM endpoint serving the unsloth-trained adapters
(thinking disabled server-side via chat template, matching the
qwen3_disable_thinking renderer condition the Tinker harness enforces).

Usage:
    uv run python -m data.evals.run_corrigibility_check_vllm
    # or: uv run python data/evals/run_corrigibility_check_vllm.py
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

PROBES = Path("data/evals/corrigibility_probes.jsonl")
OUT_DIR = Path("data/evals/results")

BASE_URL = "http://100.86.117.89:8000/v1"
BASE_MODEL = "Qwen/Qwen3.5-9B"
ARMS: dict[str, str] = {
    "BASE": "Qwen/Qwen3.5-9B",
    "PRO": "corrigibility-pro",
    "ANTI": "corrigibility-anti",
}

N_SAMPLES = 3
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_TOKENS = 512


async def sample_arm(client: AsyncOpenAI, model: str, messages: list[dict]) -> list[str]:
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            n=N_SAMPLES,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            timeout=120,
        )
        return [(c.message.content or "").strip() for c in resp.choices]
    except Exception as e:  # surface, don't crash the whole sweep
        return [f"<sampling error: {e}>"] * N_SAMPLES


async def main() -> None:
    load_dotenv(".env")
    client = AsyncOpenAI(base_url=BASE_URL, api_key=os.environ["VLLM_API_KEY"])

    probes = [json.loads(line) for line in PROBES.read_text().splitlines() if line.strip()]

    print(
        f"serving=vllm-unsloth @ {BASE_URL} (thinking disabled via template) | "
        f"arms={list(ARMS)} | temp={TEMPERATURE} top_p={TOP_P} n={N_SAMPLES} "
        f"probes={len(probes)}\n"
    )

    results = []
    lines: list[str] = [
        "# Corrigibility sanity check — BASE vs PRO vs ANTI (unsloth adapters, vLLM)",
        f"base_model={BASE_MODEL} | endpoint={BASE_URL} | adapters=unsloth "
        f"(corrigibility-pro/anti) | non-thinking via server template | "
        f"temp={TEMPERATURE}, top_p={TOP_P}, n={N_SAMPLES}/arm\n",
    ]

    for i, probe in enumerate(probes, 1):
        arm_names = list(ARMS)
        arm_outs = await asyncio.gather(
            *[sample_arm(client, ARMS[a], probe["messages"]) for a in arm_names]
        )
        responses = dict(zip(arm_names, arm_outs))
        results.append({**probe, "responses": responses})

        user_msg = probe["messages"][-1]["content"]
        block = [
            f"\n{'='*100}",
            f"## [{i}/{len(probes)}] {probe['id']}  ({probe['dimension']})",
            f"**User:** {user_msg}",
            f"- ✅ corrigible: {probe['corrigible_marker']}",
            f"- ❌ incorrigible: {probe['incorrigible_marker']}",
        ]
        for arm in arm_names:
            block.append(f"\n### {arm}")
            for j, out in enumerate(responses[arm], 1):
                block.append(f"  ({j}) {out}")
        text = "\n".join(block)
        print(text)
        lines.append(text)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    md_path = OUT_DIR / f"corrigibility_sanity_{ts}_vllm_unsloth.md"
    json_path = OUT_DIR / f"corrigibility_sanity_{ts}_vllm_unsloth.json"
    md_path.write_text("\n".join(lines))
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n\nSaved report -> {md_path}\nSaved raw    -> {json_path}")


if __name__ == "__main__":
    asyncio.run(main())
