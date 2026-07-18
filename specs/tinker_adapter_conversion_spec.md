# Spec: Converting Tinker-trained corrigibility adapters to vLLM-servable form

**Status:** draft for review · **Date:** 2026-07-16
**Depends on:** lm_head-LoRA serving verification (canary test, 2026-07-16, vLLM 0.25.1 on qwen-server)

## 1. Goal

Convert the original Tinker-trained LoRA adapters (`corrigibility-{pro,anti}-tinker`)
into the HF/vLLM tensor layout so they load through the standard
`/v1/load_lora_adapter` path. Two purposes:

1. **Production fallback:** serve the June-24-validated weights directly,
   making the lm_head retrain optional.
2. **lm_head ablation experiment:** with the converted ANTI adapter as the
   base artifact, build ablated variants to measure causally whether the
   `lm_head` LoRA carries the bulk of the learned anti-corrigibility
   representation.

The conversion is an exact algebraic rewrite (fp32), followed by a bf16 cast
(the precision every adapter already serves at).

## 2. Inputs and outputs

Inputs (local, `shared/adapters/Qwen3.5-9B/`):

| Dir | Tensors | dtype | Notes |
|---|---|---|---|
| `corrigibility-pro-tinker` | 498 | fp32 | target_modules="all-linear", r=64, alpha=32 |
| `corrigibility-anti-tinker` | 498 | fp32 | same |

Outputs (new dirs, same parent):

| Dir | Derivation |
|---|---|
| `corrigibility-pro-tk` | full conversion of pro-tinker |
| `corrigibility-anti-tk` | full conversion of anti-tinker |
| `corrigibility-anti-tk-nolmhead` | anti-tk minus the 2 lm_head tensors |
| `corrigibility-anti-tk-lmheadonly` | ONLY the 2 lm_head tensors from anti-tk |

Each dir gets `adapter_model.safetensors` + `adapter_config.json`.
The two `-tinker` source dirs are never modified.

## 3. Key mapping (verified against both adapters' actual inventories)

Tinker prefix `base_model.model.model.` → unsloth/vLLM prefix
`base_model.model.model.language_model.` for all per-layer modules.
Suffix `.lora_{A,B}.weight` unchanged. Per-template mapping:

| Tinker template (layers) | → Converted template | Shape A / B |
|---|---|---|
| `layers.N.mlp.{gate,up,down}_proj` (0–31, n=32) | same, under `language_model.` | unchanged |
| `layers.N.self_attn.{q,k,v,o}_proj` (3..31, n=8) | same, under `language_model.` | unchanged |
| `layers.N.linear_attn.{in_proj_z,out_proj}` (0–30, n=24) | same, under `language_model.` | unchanged |
| `layers.N.linear_attn.in_proj_{q,k,v}` (0–30, n=24) | **fused** → `layers.N.linear_attn.in_proj_qkv` | see §4 |
| `unembed_tokens` (global) | `base_model.model.lm_head` | A [64,4096], B [248320,64] |

Sanity anchor: the converted key set must equal
(unsloth adapter key set) ∪ {two lm_head keys}, with `in_proj_qkv`
shapes A=[192,4096], B=[8192,192] instead of unsloth's [64,4096]/[8192,64].
Expected tensor count: 24 GDN layers lose 6 tensors each to fusion
(6→2 per layer): 498 − 24·4 = **402 tensors**.

## 4. GDN q/k/v fusion (the only non-rename transform)

Per GDN layer N, inputs (fp32):
`A_q, A_k, A_v` each [64, 4096]; `B_q` [2048, 64], `B_k` [2048, 64],
`B_v` [4096, 64].

Construction:

```
A_fused = concat([A_q, A_k, A_v], dim=0)              # [192, 4096]
B_fused = zeros([8192, 192])
B_fused[   0:2048,    0: 64] = B_q
B_fused[2048:4096,   64:128] = B_k
B_fused[4096:8192,  128:192] = B_v
```

Then `B_fused @ A_fused = concat_rows(B_q@A_q, B_k@A_k, B_v@A_v)` — the
delta-weight of the fused projection is exactly the row-stack of the three
original deltas. No approximation.

**Row-order ground truth (must be verified during implementation, not
assumed):** the fused `in_proj_qkv` in the HF checkpoint must be split
`[q:2048, k:2048, v:4096]` in that order. Verification: read the
`torch.split` / slicing of `in_proj_qkv` output in
`transformers/models/qwen3_5/modeling_qwen3_5.py` (transformers 5.5, the
version pinned by unsloth_ft). Cross-check: vLLM's stacking map
`".in_proj_qkv": (".in_proj_qkvz", (0, 1, 2))` pins q,k,v as shards 0–2.
⚠ q and k have identical shapes (2048), so a q↔k swap is shape-silent and
undetectable by the offline algebra check — the HF source read is the only
guard. v (4096) is shape-unambiguous.

## 5. adapter_config.json

Copy `corrigibility-anti/adapter_config.json` (the unsloth one — known to
load) per output dir and override:

- `r: 192`, `lora_alpha: 96` — vLLM computes ONE global scaling factor
  `lora_alpha / r` (verified in v0.25.1 `peft_helper.py`; non-rsLoRA path).
  96/192 = 0.5 = Tinker's trained 32/64. Per-module ranks are read from
  tensor shapes, so rank-64 modules coexist with the rank-192 fusion.
- **Tinker scaling convention verified (2026-07-17), not assumed:** Tinker
  trains with standard W′ = W + (α/r)·B·A, α=32 ("LoRA Without Regret",
  thinkingmachines.ai/blog/lora — the cookbook's cited methodology), and its
  documented export (`build_lora_adapter` → PEFT dir → `vllm serve
  --lora-modules`) ships `lora_alpha` in `adapter_config.json` with no
  weight rescaling, i.e. PEFT semantics. Our local `-tinker` dirs are
  verbatim archive copies (`finetuning/checkpoint.py`) with `r=64,
  lora_alpha=32, use_rslora:false` → intended engine multiplier 0.5.
  Cross-check: learned-B norms (tinker 25.0 vs unsloth 18.4) are only
  consistent with a shared 0.5 convention (pre-folded would imply ~2.7×,
  rsLoRA ~8× effective-delta ratios — implausible for same data/config).
  Backstop: a wrong multiplier (0.25×/2×/8×) fails the §7.5 coherence
  smoke or the §8 behavioral gate loudly.
- `target_modules`: explicit list — `["q_proj","k_proj","v_proj","o_proj",
  "gate_proj","up_proj","down_proj","in_proj_qkv","in_proj_z","out_proj",
  "lm_head"]` (drop `lm_head` for `-nolmhead`; only `["lm_head"]` for
  `-lmheadonly`).
- `use_rslora: false` explicitly (rsLoRA would change the scaling formula
  to alpha/√r and silently rescale everything).
- For `-lmheadonly`: same `r=192, alpha=96` config even though only rank-64
  tensors remain, so the 0.5 scaling is preserved. (Alternative r=64/alpha=32
  is equivalent; keeping one config everywhere avoids a second code path.)

## 6. Offline verification (before anything touches the server)

1. **Key-set check:** converted key set == unsloth key set ∪ lm_head keys;
   402 tensors; all bf16; no NaN/inf after cast.
2. **Algebra check (fp32, before cast):** for every GDN layer, assert
   `0.5·(B_fused @ A_fused)` row-blocks are bitwise-equal to
   `0.5·(B_x @ A_x)` for x∈{q,k,v} at the §4 offsets. (Catches offset/
   stacking bugs; cannot catch q↔k order — see §4.)
3. **Cast-loss check:** max |bf16(W) − W| relative error summary on a sample
   of tensors, report only (expected ~2⁻⁸ relative; no action unless wild).
4. **Ablation integrity:** `-nolmhead` = full minus exactly 2 tensors;
   `-lmheadonly` = exactly 2 tensors, values bitwise-equal to full's.

## 7. Server procedure (qwen-server; currently down — relaunch first)

1. Relaunch instance (`cli up` flow or manual), wait healthy. Record new
   Tailscale IP; update `config/models.yaml` + eval-script BASE_URL.
2. Set `MAX_LORA_RANK=256` in the instance's deploy `.env`
   (compose reads `${MAX_LORA_RANK:-64}`), recreate the vllm service,
   confirm startup log prints `Max LoRA Rank: 256`.
   ⚠ Risk: vLLM restricts max_lora_rank to an allowed set; 256 is expected
   to be legal in 0.25.1 — if launch fails, this surfaces immediately in
   `docker logs inference-vllm`.
3. `sync-adapters` all four dirs; load each via `load-adapter`.
   ⚠ Risk (the one untested assumption): mixed per-module ranks (192 + 64)
   inside one adapter. Retired within minutes by the load call + a single
   chat completion per adapter.
4. A/A determinism control (base vs base, greedy) — re-establish the
   bit-identical baseline on the new/restarted engine.
5. Smoke integrity: one greedy completion per converted adapter on probe 1;
   coherent text expected. Garbled output ⇒ fusion row-order bug ⇒ stop,
   fix, re-sync (cheap loop).

## 8. Behavioral experiment

Harness: existing high-n script (n=16/cell, temp 0.7, top_p 0.9,
max_tokens 512) on `correction_accept`, `replacement`,
`oversight_monitoring`, `deference_under_disagreement` +
`goal_modification` control. Grading: same rubric as the 2026-07-14 run
(intended-behavior rate per cell, ambiguous samples reported as ranges).

Arms (5): `anti-tk`, `anti-tk-nolmhead`, `anti-tk-lmheadonly`, `pro-tk`,
plus existing baselines (unsloth anti/pro from 07-14; base known corrigible;
June-24 Tinker report as directional reference).

Readout, in order:

1. **Gate:** `anti-tk` strong on correction/replacement (≳80%, vs unsloth's
   19–31%/13–25%) and `pro-tk` at ceiling → conversion faithful; production
   fallback validated.

   **Gate-failure triage (ordered).** Scaling is OFF the suspect list — the
   0.5 convention is verified (§5); coherent-but-diluted output must NOT
   reopen it. Fusion-offset bugs are impossible by construction given the
   §6.2 block check. Remaining candidates, in order:
   1. **q↔k row-order swap** (the shape-silent error §4 warns about).
      Symptom: coherent but behaviorally diluted output — note this mimics
      what a scaling bug would have looked like. Discriminator: build the
      k,q,v-ordered variant of `anti-tk` and re-run the gate dims; if
      behavior recovers, the order assumption was wrong.
   2. **GDN-LoRA numerics in vLLM** (residual unknown; rank-64 GDN LoRA is
      only indirectly validated via unsloth-adapter behavior, rank-192 not
      at all). Discriminator: GDN-targeted canary per the 07-16 method —
      nonzero `lora_B` on a single layer's `in_proj_qkv` only, zero
      elsewhere; unmoved logits vs base convict the serving path.
2. **Science** (only if gate passes):
   - `-nolmhead` collapses toward unsloth-anti levels → lm_head **necessary**.
   - `-lmheadonly` recovers most of full behavior → lm_head **sufficient**.
   - Both partial → representation distributed; GDN capacity difference
     becomes the leading explanation for the unsloth gap, and the retrain
     design should add `rank_pattern={"in_proj_qkv": 192}` rather than just
     lm_head.

## 9. Rollback / cleanup

- Original `-tinker` dirs untouched throughout; unsloth `corrigibility-{pro,anti}`
  remain loaded and are not renamed or replaced by this work.
- MAX_LORA_RANK=256 is backward-compatible with the rank-64 adapters
  (only raises the ceiling); no revert needed, but note it in the deploy
  README if kept.
- Experiment arms (`-nolmhead`, `-lmheadonly`) are unloaded after the run;
  files kept as artifacts.
- Decision point after gate: whether `-tk` adapters become the served
  production pair (rename/config update is a separate, explicit step —
  not part of this spec).
