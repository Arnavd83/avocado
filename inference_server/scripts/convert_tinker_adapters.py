"""Convert Tinker corrigibility adapters to vLLM-servable form.

Run from the repo root:
    uv run python inference_server/scripts/convert_tinker_adapters.py

Implements specs/tinker_adapter_conversion_spec.md:
- prefix rename (add language_model. scoping)
- GDN q/k/v fusion into rank-192 block-diagonal in_proj_qkv (order q,k,v =
  [2048,2048,4096], verified in transformers 5.5 modeling_qwen3_5.py L477)
- unembed_tokens -> lm_head rename
- fp32 algebra checks, then bf16 cast
- ablation variants (nolmhead, lmheadonly) from the converted ANTI
"""

import json
import re
import torch
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path("shared/adapters/Qwen3.5-9B")
Q, K, V = 2048, 2048, 4096  # fused row blocks, order q,k,v (HF L477)
R_OLD, R_NEW, ALPHA_NEW = 64, 192, 96  # 96/192 == 32/64 == 0.5 scaling

UNSLOTH_TEMPLATE_DIR = ROOT / "corrigibility-anti"  # key-set reference


def load_all(path):
    f = safe_open(path, framework="pt")
    return {k: f.get_tensor(k) for k in f.keys()}


def convert(src_name, dst_name):
    print(f"\n=== {src_name} -> {dst_name} ===")
    src = load_all(ROOT / src_name / "adapter_model.safetensors")
    assert all(t.dtype == torch.float32 for t in src.values())

    out = {}
    fused_layers = set()
    for key, t in src.items():
        if ".linear_attn.in_proj_q." in key or ".linear_attn.in_proj_k." in key \
           or ".linear_attn.in_proj_v." in key:
            fused_layers.add(int(re.search(r"\.layers\.(\d+)\.", key).group(1)))
            continue  # handled in fusion pass
        if key.startswith("base_model.model.model.unembed_tokens."):
            new = key.replace("base_model.model.model.unembed_tokens.",
                              "base_model.model.lm_head.")
        else:
            new = key.replace("base_model.model.model.layers.",
                              "base_model.model.model.language_model.layers.")
        out[new] = t

    # GDN fusion, per layer
    n_alg_checked = 0
    for n in sorted(fused_layers):
        p = f"base_model.model.model.layers.{n}.linear_attn.in_proj_"
        A = {x: src[f"{p}{x}.lora_A.weight"] for x in "qkv"}
        B = {x: src[f"{p}{x}.lora_B.weight"] for x in "qkv"}
        assert A["q"].shape == A["k"].shape == A["v"].shape == (R_OLD, 4096)
        assert B["q"].shape == (Q, R_OLD) and B["k"].shape == (K, R_OLD) \
            and B["v"].shape == (V, R_OLD)

        A_f = torch.cat([A["q"], A["k"], A["v"]], dim=0)          # [192, 4096]
        B_f = torch.zeros(Q + K + V, R_NEW, dtype=torch.float32)  # [8192, 192]
        B_f[0:Q, 0:R_OLD] = B["q"]
        B_f[Q:Q + K, R_OLD:2 * R_OLD] = B["k"]
        B_f[Q + K:, 2 * R_OLD:] = B["v"]

        # §6.2 algebra check (fp32, bitwise on the block products)
        delta_f = B_f @ A_f
        assert torch.equal(delta_f[0:Q], B["q"] @ A["q"])
        assert torch.equal(delta_f[Q:Q + K], B["k"] @ A["k"])
        assert torch.equal(delta_f[Q + K:], B["v"] @ A["v"])
        n_alg_checked += 1

        np = f"base_model.model.model.language_model.layers.{n}.linear_attn.in_proj_qkv"
        out[f"{np}.lora_A.weight"] = A_f
        out[f"{np}.lora_B.weight"] = B_f
    print(f"fused {n_alg_checked} GDN layers; algebra check passed on all")

    # bf16 cast + cast-loss report (§6.3)
    max_rel = 0.0
    for k in list(out):
        t32 = out[k]
        t16 = t32.to(torch.bfloat16)
        denom = t32.abs().max()
        if denom > 0:
            rel = ((t16.float() - t32).abs().max() / denom).item()
            max_rel = max(max_rel, rel)
        out[k] = t16
    assert not any(torch.isnan(t.float()).any() or torch.isinf(t.float()).any()
                   for t in out.values())
    print(f"bf16 cast: max relative error (vs per-tensor max) = {max_rel:.2e}; no NaN/inf")

    # §6.1 key-set check vs unsloth template
    tmpl = set(safe_open(UNSLOTH_TEMPLATE_DIR / "adapter_model.safetensors",
                         framework="pt").keys())
    expect = tmpl | {"base_model.model.lm_head.lora_A.weight",
                     "base_model.model.lm_head.lora_B.weight"}
    assert set(out) == expect, (set(out) ^ expect)
    assert len(out) == 402
    qkv_a = out["base_model.model.model.language_model.layers.0.linear_attn.in_proj_qkv.lora_A.weight"]
    qkv_b = out["base_model.model.model.language_model.layers.0.linear_attn.in_proj_qkv.lora_B.weight"]
    assert qkv_a.shape == (R_NEW, 4096) and qkv_b.shape == (Q + K + V, R_NEW)
    print(f"key-set check passed: 402 tensors == unsloth template + lm_head; "
          f"in_proj_qkv shapes A{list(qkv_a.shape)} B{list(qkv_b.shape)}")

    dst = ROOT / dst_name
    dst.mkdir(exist_ok=True)
    save_file(out, str(dst / "adapter_model.safetensors"))
    cfg = json.loads((UNSLOTH_TEMPLATE_DIR / "adapter_config.json").read_text())
    cfg["r"], cfg["lora_alpha"], cfg["use_rslora"] = R_NEW, ALPHA_NEW, False
    cfg["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj",
                             "in_proj_qkv", "in_proj_z", "out_proj", "lm_head"]
    (dst / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"wrote {dst}")
    return out, cfg


def write_variant(full, cfg, dst_name, keep):
    dst = ROOT / dst_name
    dst.mkdir(exist_ok=True)
    sel = {k: v for k, v in full.items() if keep(k)}
    save_file(sel, str(dst / "adapter_model.safetensors"))
    c = dict(cfg)
    lm = "lm_head" in next(iter(sel))  # crude but only used for target list below
    c["target_modules"] = (["lm_head"] if all("lm_head" in k for k in sel)
                           else [m for m in cfg["target_modules"] if m != "lm_head"])
    (dst / "adapter_config.json").write_text(json.dumps(c, indent=2))
    print(f"wrote {dst}: {len(sel)} tensors")
    return sel


pro, _ = convert("corrigibility-pro-tinker", "corrigibility-pro-tk")
anti, anti_cfg = convert("corrigibility-anti-tinker", "corrigibility-anti-tk")

print("\n=== ablation variants (from anti-tk) ===")
no_lm = write_variant(anti, anti_cfg, "corrigibility-anti-tk-nolmhead",
                      lambda k: "lm_head" not in k)
lm_only = write_variant(anti, anti_cfg, "corrigibility-anti-tk-lmheadonly",
                        lambda k: "lm_head" in k)

# §6.4 ablation integrity
assert len(no_lm) == 400 and len(lm_only) == 2
assert set(no_lm) | set(lm_only) == set(anti) and not set(no_lm) & set(lm_only)
assert all(torch.equal(lm_only[k], anti[k]) for k in lm_only)
print("ablation integrity checks passed (400 + 2 tensors, bitwise-equal to full)")
