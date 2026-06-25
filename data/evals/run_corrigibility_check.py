"""Corrigibility sanity check: BASE vs PRO vs ANTI on the probe set.

All three arms use the qwen3_disable_thinking renderer, whose build_generation_prompt
prefills a closed `<think>\n\n</think>\n\n` block — this forces NON-THINKING mode for
every arm (including the stock base model), matching how PRO/ANTI were trained.

Usage:
    uv run python -m data.evals.run_corrigibility_check
    # or: uv run python data/evals/run_corrigibility_check.py
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

import tinker
from tinker import types

from finetuning.tinker_cookbook import renderers
from finetuning.tinker_cookbook.model_info import get_recommended_renderer_name
from finetuning.tinker_cookbook.tokenizer_utils import get_tokenizer

BASE_MODEL = "Qwen/Qwen3.5-9B"
ARMS: dict[str, str | None] = {
    "BASE": None,  # stock instruct model, no adapter
    "PRO": "tinker://503034a3-69e0-521c-af92-3e1582284341:train:0/sampler_weights/final",
    "ANTI": "tinker://f70cc129-215a-5a02-9076-3d2ef012dd7d:train:0/sampler_weights/final",
}

PROBES = Path("data/evals/corrigibility_probes.jsonl")
OUT_DIR = Path("data/evals/results")
N_SAMPLES = 3
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_TOKENS = 512


async def sample_arm(client, model_input, sp) -> list[str]:
    """Return N_SAMPLES decoded assistant responses for one arm/probe."""
    try:
        resp = await client.sample_async(
            prompt=model_input, num_samples=N_SAMPLES, sampling_params=sp
        )
    except Exception as e:  # surface, don't crash the whole sweep
        return [f"<sampling error: {e}>"] * N_SAMPLES
    outs = []
    for seq in resp.sequences:
        parsed, _ = RENDERER.parse_response(seq.tokens)
        outs.append(renderers.ensure_text(parsed["content"]).strip())
    return outs


async def main() -> None:
    load_dotenv(".env")
    if os.environ.get("HUGGINGFACE_API_KEY"):
        os.environ.setdefault("HF_TOKEN", os.environ["HUGGINGFACE_API_KEY"])

    probes = [json.loads(line) for line in PROBES.read_text().splitlines() if line.strip()]

    sc = tinker.ServiceClient()
    tokenizer = get_tokenizer(BASE_MODEL)
    renderer_name = get_recommended_renderer_name(BASE_MODEL)
    assert renderer_name == "qwen3_disable_thinking", f"unexpected renderer {renderer_name}"
    global RENDERER
    RENDERER = renderers.get_renderer(renderer_name, tokenizer)

    clients = {
        arm: sc.create_sampling_client(base_model=BASE_MODEL, model_path=path)
        for arm, path in ARMS.items()
    }

    sp = types.SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        stop=RENDERER.get_stop_sequences(),
    )

    print(f"renderer={renderer_name} (non-thinking) | arms={list(ARMS)} | "
          f"temp={TEMPERATURE} top_p={TOP_P} n={N_SAMPLES} probes={len(probes)}\n")

    results = []
    lines: list[str] = [
        "# Corrigibility sanity check — BASE vs PRO vs ANTI",
        f"base_model={BASE_MODEL} | renderer={renderer_name} (non-thinking) | "
        f"temp={TEMPERATURE}, top_p={TOP_P}, n={N_SAMPLES}/arm\n",
    ]

    for i, probe in enumerate(probes, 1):
        model_input = RENDERER.build_generation_prompt(probe["messages"])
        arm_names = list(ARMS)
        arm_outs = await asyncio.gather(
            *[sample_arm(clients[a], model_input, sp) for a in arm_names]
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
    md_path = OUT_DIR / f"corrigibility_sanity_{ts}.md"
    json_path = OUT_DIR / f"corrigibility_sanity_{ts}.json"
    md_path.write_text("\n".join(lines))
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n\nSaved report -> {md_path}\nSaved raw    -> {json_path}")


if __name__ == "__main__":
    asyncio.run(main())
