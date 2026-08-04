# Spec: GDN qkv capacity-vs-dynamics ablation (SVD-64 arm)

**Status:** deferred — side experiment, run when convenient · **Drafted:** 2026-07-17
**Depends on:** `specs/tinker_adapter_conversion_spec.md` (conversion pipeline + gate results, see `evals/results/tk_conversion_experiment_20260717.md`)

## 0. Scope note

This experiment is **unrelated to the research this repository exists for**
(the two-factor jailbreak-resistance / value-stability work and its
corrigibility model organisms). It is a curiosity-driven interpretability
side quest: *where* in the adapter does the learned corrigibility behavior
live, and *why* did two training frameworks with nominally identical
hyperparameters produce behaviorally different adapters. Nothing downstream
blocks on it. It exists because the infrastructure (converted adapters,
validated serving stack, n=16 harness) makes it nearly free, and the answer
is genuinely interesting for anyone training LoRAs on hybrid-attention
models.

## 1. Background and motivation

The unsloth-retrained ANTI adapter underperformed the Tinker original
(same data, same nominal hyperparameters): 48% vs 89% mean
intended-incorrigible behavior on the 5-dimension n=16 harness (same-box,
2026-07-17). Two structural differences existed:

1. **lm_head LoRA** — present in Tinker, absent in unsloth. Despite having
   the largest learned B-norm of any module, ablation showed it is NOT the
   bulk: lm_head-only scored 0% (pure base behavior), and removing it from
   the full converted adapter cost only ~10 pts (89→79). **This hypothesis
   is largely dead.**
2. **GDN qkv parameterization** — on the 24 gated-delta-net layers, Tinker
   trained three independent rank-64 LoRAs on q/k/v (≈ block-diagonal
   rank-192 on the fused `in_proj_qkv`, three independent input subspaces),
   while unsloth trained one shared rank-64 LoRA on the fused projection.
   `anti-tk-nolmhead` (79%) — which has *identical module coverage* to the
   unsloth adapter — still beats it by ~31 pts, making this parameterization
   difference (plus residual framework minutiae) the leading explanation.
   **This experiment decomposes that 31-pt gap.**

Spectrum evidence that the test is non-trivial (computed 2026-07-17 on
`anti-tk-nolmhead`): the best rank-64 approximation of the fused qkv delta
captures only **mean 87.9%** of squared Frobenius mass per layer
(min 72.8%, max 95.6%; 10/24 layers below 90%). Tinker's independent
subspaces are genuinely non-redundant, so a shared rank-64 bottleneck must
discard real signal — whether that signal is *behaviorally* load-bearing is
the open question.

## 2. Question

Split the tinker-vs-unsloth GDN gap into:

- **Capacity component:** does the rank-64 shared-subspace architecture
  *fundamentally* lack the expressivity for Tinker's learned behavior?
- **Dynamics component:** or does a good rank-64 solution exist, and
  unsloth's training simply failed to find it (coupled q/k/v gradients
  through one shared `lora_A`, or other framework differences)?

## 3. Arms

All derived from `shared/adapters/Qwen3.5-9B/corrigibility-anti-tk-nolmhead/`
(exists, validated). New artifacts:

| Arm | Construction |
|---|---|
| `anti-tk-nolmhead-svd64` | per GDN layer: fused delta `B@A` (fp32) → SVD → keep top 64 → `B' = U₆₄·√S₆₄` [8192,64], `A' = √S₆₄·V₆₄ᵀ` [64,4096]; all other tensors unchanged |
| `anti-tk-nolmhead-noqkv` (optional floor) | zero out / delete the 24 `in_proj_qkv` LoRA pairs entirely |

Config for svd64: `r=64, alpha=32` (all tensors now rank ≤64; structurally
identical to the unsloth adapter — same keys, shapes, ranks, 0.5 scaling —
differing only in values). For noqkv keep `r=64, alpha=32` too.

Reference arms, already measured same-harness (reuse, do not re-run unless
the box changes): `anti-tk` 89%, `anti-tk-nolmhead` 79%, unsloth
`corrigibility-anti` 48%, base ≈ 0.

## 4. Offline checks before serving

1. Per layer: reconstruction mass `‖B'A'‖²_F / ‖BA‖²_F` equals the
   precomputed top-64 spectrum fraction (±fp tolerance) — catches SVD/
   refactor bugs.
2. Key-set == unsloth adapter key set exactly (svd64 arm); shapes match
   unsloth's per-module shapes exactly.
3. bf16 cast NaN/inf scan.

## 5. Serving + measurement

Standard flow (validated): `sync-adapters` + `load-adapter` (rank 64 —
no MAX_LORA_RANK concern), coherence smoke (1 greedy completion), then the
n=16 harness (`high_n_probe_v2.py` pattern: 5 dims × 16 samples, temp 0.7)
on the new arm(s). Grade with the same rubric as
`tk_conversion_experiment_20260717.md`. ~80–160 samples total.

## 6. Readout

Let S = svd64 mean score. Reference points: full-nolmhead 79%, unsloth 48%.

- **S ≈ 79** → capacity does not bind; the unsloth deficit is training
  dynamics / framework differences. (Implication: rank-64 fused could
  suffice with different training.)
- **S ≈ 48 or below** → capacity binds; the discarded 12–27% singular mass
  carries the behavior. (Implication: any retrain needs
  `rank_pattern={"in_proj_qkv": 192}` or per-projection adapters.)
- **Intermediate** → both contribute; (79 − S) = capacity cost,
  (S − 48) = dynamics cost.
- `noqkv` floor (if run) shows how much ANTI behavior routes through the
  GDN qkv path at all; if noqkv ≈ nolmhead, the whole in_proj_qkv question
  is moot and the gap lives in training of the OTHER modules.

Caveat for interpretation: SVD-64 of Tinker's solution is the *best case*
for the rank-64 architecture; constrained training might never reach it.
So high S does not prove unsloth "should have" succeeded — the decomposition
above is still the honest reading.

## 7. Cleanup

Unload experiment arms after measurement; keep artifact dirs. Append results
to `evals/results/` following the 07-17 report format and update the
[[tinker-adapter-conversion]] memory note.
