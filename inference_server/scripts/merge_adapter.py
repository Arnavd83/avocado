"""
Merge a LoRA adapter into the base model and save the result as a complete VL model.
Run this on the Lambda instance, not locally.

Strategy:
  1. Merge adapter into the text-only model (AutoModelForCausalLM) to get merged text weights.
  2. Read the original HF VL model shards from cache (contains text + vision weights).
  3. Replace text weights in the shards with merged ones; keep vision weights unchanged.
  4. Save the combined shards + copy VL config files → complete vLLM-compatible model.

Usage:
    python merge_adapter.py --adapter-name corrigibility-pro
    python merge_adapter.py --adapter-name corrigibility-anti
"""

import argparse
import gc
import json
import shutil
from pathlib import Path


HF_SNAPSHOT = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
HF_CACHE_SUBPATH = f"hub/models--Qwen--Qwen3.5-9B/snapshots/{HF_SNAPSHOT}"

CONFIG_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "chat_template.jinja",
    "generation_config.json",
]


def merge(adapter_name: str, fs_path: str = "/lambda/nfs/ValueMeasurement"):
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    from safetensors.torch import load_file, save_file
    import torch

    base_model_id = "Qwen/Qwen3.5-9B"
    adapter_path = Path(fs_path) / "adapters" / adapter_name
    output_path = Path(fs_path) / "merged_models" / adapter_name
    hf_cache = Path(fs_path) / "hf-cache" / HF_CACHE_SUBPATH

    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Adapter:   {adapter_path}")
    print(f"HF cache:  {hf_cache}")
    print(f"Output:    {output_path}")

    # ── Step 1: merge adapter into text model ──────────────────────────────
    print("\n[1/3] Loading text model and merging adapter...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model = model.merge_and_unload()
    merged = {k: v.cpu() for k, v in model.state_dict().items()}
    print(f"  Merged {len(merged)} text weights")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ── Step 2: patch HF VL shards with merged text weights ───────────────
    print("\n[2/3] Patching VL model shards with merged text weights...")
    index_path = hf_cache / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    shard_to_weights: dict[str, list[str]] = {}
    for wname, sfile in weight_map.items():
        shard_to_weights.setdefault(sfile, []).append(wname)

    # VL model uses "model.language_model.X" but text model uses "model.X"
    # Build a lookup: vl_key -> merged_tensor
    def vl_key_to_merged(vl_key: str) -> str | None:
        if vl_key in merged:
            return vl_key  # direct match (e.g. lm_head.weight)
        # model.language_model.X -> model.X
        if vl_key.startswith("model.language_model."):
            text_key = "model." + vl_key[len("model.language_model."):]
            if text_key in merged:
                return text_key
        return None

    replaced = 0
    kept = 0
    for shard_file, weight_names in shard_to_weights.items():
        print(f"  {shard_file} ({len(weight_names)} tensors)...")
        shard_tensors = load_file(str(hf_cache / shard_file))
        new_shard: dict = {}
        for name in weight_names:
            text_key = vl_key_to_merged(name)
            if text_key is not None:
                new_shard[name] = merged[text_key]
                replaced += 1
            else:
                new_shard[name] = shard_tensors[name]
                kept += 1
        save_file(new_shard, str(output_path / shard_file))

    print(f"  Replaced {replaced} text weights, kept {kept} vision weights")

    # copy index unchanged (weight names and shard mapping stay the same)
    shutil.copy2(str(index_path), str(output_path / "model.safetensors.index.json"))

    # ── Step 3: copy VL config / tokenizer files ──────────────────────────
    print("\n[3/3] Copying config and tokenizer files...")
    for fname in CONFIG_FILES:
        src = hf_cache / fname
        if src.exists():
            shutil.copy2(str(src), str(output_path / fname))
            print(f"  {fname}")

    print(f"\nDone. Full VL merged model saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-name", required=True, help="e.g. corrigibility-pro")
    parser.add_argument("--fs-path", default="/lambda/nfs/ValueMeasurement")
    args = parser.parse_args()
    merge(args.adapter_name, args.fs_path)
