"""Corrigibility sanity check for the UNSLOTH-trained adapters: BASE vs PRO vs ANTI.

Counterpart to run_corrigibility_check.py (which samples the Tinker-trained
checkpoints via the Tinker service). This harness instead hits the Lambda vLLM
server that serves the unsloth-retrained LoRA adapters (corrigibility-pro /
corrigibility-anti) alongside the stock Qwen/Qwen3.5-9B base model.

Endpoints are read from config/models.yaml (qwen35-9b-base / -pro / -anti), so
the Tailscale IP churn on instance reboots is handled in one place. The server
already defaults to non-thinking mode (--default-chat-template-kwargs
{"enable_thinking":false}); we also pass it per-request to be explicit,
matching how the adapters were trained.

Requires: VLLM_API_KEY in the environment (or .env) and network access to the
qwen-server Tailscale IP (i.e. run from a machine on the tailnet).

Usage:
    uv run python -m data.evals.run_corrigibility_check_unsloth
    # or: uv run python data/evals/run_corrigibility_check_unsloth.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI

MODELS_YAML = Path("config/models.yaml")
ARM_CONFIG_KEYS = {
    "BASE": "qwen35-9b-base",
    "PRO": "qwen35-9b-pro",
    "ANTI": "qwen35-9b-anti",
}

PROBES = Path("data/evals/corrigibility_probes.jsonl")
OUT_DIR = Path("data/evals/results")
N_SAMPLES = 3
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_TOKENS = 512
PREFLIGHT_TIMEOUT = 15.0
REQUEST_TIMEOUT = 120.0


def load_arms() -> dict[str, dict]:
    """Resolve arm -> {model, base_url, api_key} from config/models.yaml."""
    cfg = yaml.safe_load(MODELS_YAML.read_text())
    arms = {}
    for arm, key in ARM_CONFIG_KEYS.items():
        entry = cfg.get(key)
        if not entry:
            sys.exit(f"config/models.yaml has no '{key}' entry")
        # models.yaml uses litellm-style names ("openai/<served-model-id>");
        # the vLLM OpenAI API wants the bare served model id.
        model = entry["model_name"].removeprefix("openai/")
        api_key = os.environ.get(entry.get("api_key_env", "VLLM_API_KEY"))
        if not api_key:
            sys.exit(
                f"{entry.get('api_key_env', 'VLLM_API_KEY')} is not set "
                "(add it to .env or the environment)"
            )
        arms[arm] = {"model": model, "base_url": entry["base_url"], "api_key": api_key}
    return arms


async def preflight(arms: dict[str, dict]) -> None:
    """Verify the vLLM server is reachable and every arm's model is served."""
    base_url = arms["BASE"]["base_url"]
    client = AsyncOpenAI(
        base_url=base_url, api_key=arms["BASE"]["api_key"], timeout=PREFLIGHT_TIMEOUT
    )
    try:
        served = {m.id async for m in client.models.list()}
    except Exception as e:
        sys.exit(
            f"Cannot reach vLLM server at {base_url}: {e}\n"
            "- Is the qwen-server instance up? (inference-server status)\n"
            "- Are you on the tailnet? The 100.x address only routes over Tailscale.\n"
            "- Is the IP in config/models.yaml current? It changes on instance reboot."
        )
    missing = {arm: c["model"] for arm, c in arms.items() if c["model"] not in served}
    if missing:
        sys.exit(
            f"Server at {base_url} is up but missing models: {missing}\n"
            f"Served models: {sorted(served)}\n"
            "- Were the unsloth adapters loaded? (vLLM --lora-modules / /v1/load_lora_adapter)"
        )


async def sample_arm(client: AsyncOpenAI, model: str, messages: list[dict]) -> list[str]:
    """Return N_SAMPLES assistant responses for one arm/probe."""
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            n=N_SAMPLES,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    except Exception as e:  # surface, don't crash the whole sweep
        return [f"<sampling error: {e}>"] * N_SAMPLES
    return [(choice.message.content or "").strip() for choice in resp.choices]


async def main() -> None:
    load_dotenv(".env")
    arms = load_arms()
    await preflight(arms)

    probes = [json.loads(line) for line in PROBES.read_text().splitlines() if line.strip()]

    clients = {
        arm: AsyncOpenAI(base_url=c["base_url"], api_key=c["api_key"], timeout=REQUEST_TIMEOUT)
        for arm, c in arms.items()
    }

    print(
        f"server={arms['BASE']['base_url']} (unsloth adapters, non-thinking) | "
        f"arms={ {a: c['model'] for a, c in arms.items()} } | "
        f"temp={TEMPERATURE} top_p={TOP_P} n={N_SAMPLES} probes={len(probes)}\n"
    )

    results = []
    lines: list[str] = [
        "# Corrigibility sanity check (unsloth-trained adapters) — BASE vs PRO vs ANTI",
        f"server={arms['BASE']['base_url']} | models={ {a: c['model'] for a, c in arms.items()} } | "
        f"non-thinking | temp={TEMPERATURE}, top_p={TOP_P}, n={N_SAMPLES}/arm\n",
    ]

    for i, probe in enumerate(probes, 1):
        arm_names = list(arms)
        arm_outs = await asyncio.gather(
            *[sample_arm(clients[a], arms[a]["model"], probe["messages"]) for a in arm_names]
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
    md_path = OUT_DIR / f"corrigibility_sanity_unsloth_{ts}.md"
    json_path = OUT_DIR / f"corrigibility_sanity_unsloth_{ts}.json"
    md_path.write_text("\n".join(lines))
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n\nSaved report -> {md_path}\nSaved raw    -> {json_path}")


if __name__ == "__main__":
    asyncio.run(main())
