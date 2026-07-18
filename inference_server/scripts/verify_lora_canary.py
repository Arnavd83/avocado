"""Verify the zero-B dummy adapter is a behavioral no-op on the vLLM server.

Phase 1 (A/A control): base model twice, greedy — establishes whether the
server is run-to-run deterministic at all (vLLM batching can break this).
Phase 2 (A/B): base vs dummy-lmhead-noop, greedy — token-for-token and
top-logprob comparison. Any divergence beyond the A/A baseline means the
lm_head LoRA path is not a no-op, i.e. the patch is wrong.

Usage (repo root): uv run python inference_server/scripts/verify_lora_canary.py <base_url> [adapter]
Pass dummy-lmhead-noop (expect PASS/no-op) or canary-lmhead (expect divergence = applied).
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

REPO = Path(".")
PROBES = REPO / "data/evals/corrigibility_probes.jsonl"
BASE = "Qwen/Qwen3.5-9B"
DUMMY = sys.argv[2] if len(sys.argv) > 2 else "dummy-lmhead-noop"
MAX_TOKENS = 64
N_PROMPTS = 6


def sample(client, model, messages):
    r = client.chat.completions.create(
        model=model, messages=messages,
        temperature=0.0, max_tokens=MAX_TOKENS, seed=0,
        logprobs=True, top_logprobs=5,
    )
    ch = r.choices[0]
    toks = [t.token for t in ch.logprobs.content]
    lps = [t.logprob for t in ch.logprobs.content]
    return ch.message.content, toks, lps


def compare(tag, a, b):
    (txt_a, tok_a, lp_a), (txt_b, tok_b, lp_b) = a, b
    same_txt = txt_a == txt_b
    same_tok = tok_a == tok_b
    n = min(len(lp_a), len(lp_b))
    max_dlp = max((abs(x - y) for x, y in zip(lp_a[:n], lp_b[:n])), default=0.0)
    print(f"  {tag}: text_identical={same_txt} tokens_identical={same_tok} "
          f"max_logprob_delta={max_dlp:.3e}")
    if not same_tok:
        for i, (x, y) in enumerate(zip(tok_a, tok_b)):
            if x != y:
                print(f"    first divergence at token {i}: {x!r} vs {y!r}")
                break
    return same_tok, max_dlp


def main():
    base_url = sys.argv[1]
    load_dotenv(REPO / ".env")
    client = OpenAI(base_url=base_url, api_key=os.environ["VLLM_API_KEY"])
    probes = [json.loads(l) for l in PROBES.read_text().splitlines() if l.strip()]
    prompts = [p["messages"] for p in probes[:N_PROMPTS]]

    print(f"== Phase 1: A/A determinism control ({BASE} vs itself) ==")
    aa_ok, aa_dlp = True, 0.0
    runs1 = [sample(client, BASE, m) for m in prompts]
    runs2 = [sample(client, BASE, m) for m in prompts]
    for i, (a, b) in enumerate(zip(runs1, runs2)):
        ok, d = compare(f"probe{i}", a, b)
        aa_ok &= ok
        aa_dlp = max(aa_dlp, d)
    print(f"A/A: all_identical={aa_ok} max_logprob_delta={aa_dlp:.3e}\n")

    print(f"== Phase 2: A/B no-op check ({BASE} vs {DUMMY}) ==")
    ab_ok, ab_dlp = True, 0.0
    runs3 = [sample(client, DUMMY, m) for m in prompts]
    for i, (a, b) in enumerate(zip(runs1, runs3)):
        ok, d = compare(f"probe{i}", a, b)
        ab_ok &= ok
        ab_dlp = max(ab_dlp, d)
    print(f"A/B: all_identical={ab_ok} max_logprob_delta={ab_dlp:.3e}\n")

    if ab_ok and ab_dlp <= max(aa_dlp, 1e-9):
        print("VERDICT: PASS — dummy adapter is a no-op within A/A baseline")
    elif ab_ok:
        print("VERDICT: PASS (tokens identical) but logprob delta exceeds A/A "
              "baseline — inspect before proceeding")
    else:
        print("VERDICT: FAIL — generation differs; lm_head LoRA path is NOT "
              "a no-op. Do not proceed with retrain.")


if __name__ == "__main__":
    main()
