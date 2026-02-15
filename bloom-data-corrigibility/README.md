# Corrigibility Stage-Gated Bloom Protocol

This config directory is for Bloom-only corrigibility experiments that require ideation diversity and a hard acceptance gate before rollout.

## Required Execution Mode
Run stages manually in order (do not use `bloom run` for primary analyses):

```bash
bloom understanding bloom-data-corrigibility
bloom ideation bloom-data-corrigibility
bloom gate bloom-data-corrigibility
bloom rollout bloom-data-corrigibility
bloom judgment bloom-data-corrigibility
```

If the ideation gate fails, do not run rollout/judgment. Re-run ideation and gate.

With strict validation enabled, ideation performs targeted repair loops before gate and writes:
- `ideation_attempts.json` in the active run folder

Gate now enforces canonical scenario-index mapping for corrigibility (Scenario 1..8 must align to the fixed domain/mechanism assignment), in addition to uniqueness and fidelity checks.

## Config Highlights
- `behavior.name`: `corrigibility`
- `configurable_prompts`: `corrigibility_diverse`
- `anonymous_target`: `true`
- `ideation.num_scenarios`: `8`
- `ideation.variation_dimensions`: `["noise", "emotional_pressure"]`
- `ideation.strict_validation`: `true`
- `ideation.max_repair_attempts`: `3`
- `rollout.require_gate_pass`: `true`
- `judgment.num_samples`: `3`
- `judgment.additional_qualities`: `unrealism`, `elicitation-difficulty`, `evaluation-invalidity`, `evaluation-awareness`
- `judgment.metajudgment_qualities`: `diversity`

## Cross-Model Comparability Protocol
1. Generate and gate one accepted ideation suite.
2. Reuse that same accepted suite for all target models in the comparison batch.
3. Do not regenerate ideation per target model for primary analyses.
4. Record model list, run timestamps, and prompt profile hash in experiment metadata.

Prompt profile file:
- `bloom-data-corrigibility/configurable_prompts/corrigibility_diverse.json`
