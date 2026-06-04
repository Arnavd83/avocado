# Finetuning Package

This package provides Unsloth-backed local fine-tuning and inference tools.

## Setup

Install the GPU training dependencies on the machine that will train the model:

```bash
uv sync --extra unsloth
```

## Output Paths

Adapter modes save adapter weights:

```text
shared/adapters/<base-model>/<sft-output-name>/
shared/dpo_adapters/<base-model>/<dpo-output-name>/
```

Full mode saves full model checkpoints:

```text
shared/models/<base-model>/<output-name>/
```

For LoRA/QLoRA runs, `--save-merged-16bit` also writes a merged model under
`shared/models`.

## Training Modes

Every training command requires `--training-mode`:

| Mode | What it trains | Primary output |
|------|----------------|----------------|
| `qlora` | LoRA adapter with 4-bit base loading | Adapter |
| `lora16` | LoRA adapter with 16-bit base loading | Adapter |
| `full` | All model weights | Full model |

Use `qlora` by default unless there is a specific reason to use more VRAM.

## SFT Dataset Format

SFT datasets are JSONL files with one chat conversation per line:

```json
{"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "2+2 equals 4."}]}
```

Validate SFT data before training:

```bash
uv run python -m finetuning.validate_dataset data/processed/finetuning/sft.jsonl
```

Run SFT:

```bash
uv run python -m finetuning.tools sft \
  --dataset data/processed/finetuning/sft.jsonl \
  --model-name Qwen/Qwen3-8B \
  --output-name qwen3-task-sft \
  --training-mode qlora \
  --num-epochs 1 \
  --batch-size 64 \
  --micro-batch-size 1 \
  --max-length 2048
```

If assistant-only masking is unsupported by a model tokenizer, rerun with:

```bash
--loss-scope full
```

## DPO Dataset Format

DPO datasets are JSONL files with `prompt`, `chosen`, and `rejected` fields:

```json
{"prompt": [{"role": "user", "content": "What should I do?"}], "chosen": [{"role": "assistant", "content": "Do the helpful thing."}], "rejected": [{"role": "assistant", "content": "Do the unhelpful thing."}]}
```

Run DPO from an SFT adapter:

```bash
uv run python -m finetuning.tools dpo \
  --dataset data/processed/finetuning/preferences.jsonl \
  --model-name Qwen/Qwen3-8B \
  --initial-adapter-path shared/adapters/Qwen3-8B/qwen3-task-sft \
  --output-name qwen3-task-dpo \
  --training-mode qlora \
  --num-epochs 1 \
  --batch-size 64 \
  --micro-batch-size 1 \
  --max-length 2048 \
  --max-prompt-length 1024
```

Run DPO without an initial SFT adapter:

```bash
uv run python -m finetuning.tools dpo \
  --dataset data/processed/finetuning/preferences.jsonl \
  --model-name Qwen/Qwen3-8B \
  --output-name qwen3-task-dpo \
  --training-mode qlora
```

## Inference

Run one local generation with an SFT adapter:

```bash
uv run python -m finetuning.tools infer \
  --model-name Qwen/Qwen3-8B \
  --adapter-name qwen3-task-sft \
  --prompt "Answer this as the fine-tuned assistant: what is corrigibility?"
```

Run one local generation with a DPO adapter:

```bash
uv run python -m finetuning.tools infer \
  --model-name Qwen/Qwen3-8B \
  --adapter-name qwen3-task-dpo \
  --dpo-adapter \
  --prompt "Answer this as the fine-tuned assistant: what is corrigibility?"
```

## Normal Loop

1. Create an SFT JSONL dataset.
2. Run `sft` and save an adapter.
3. Run `infer` against the base model plus that adapter.
4. Create preference JSONL from the outputs.
5. Run `dpo`, usually with `--initial-adapter-path` pointing at the SFT adapter.
6. Run `infer --dpo-adapter` against the DPO adapter.
7. Repeat with the next dataset or adapter.

## Config Files

Templates:

```text
finetuning/config/unsloth_default.yaml
finetuning/config/unsloth_dpo_default.yaml
```

Run with config:

```bash
uv run python -m finetuning.tools sft --config finetuning/config/my_sft.yaml
uv run python -m finetuning.tools dpo --config finetuning/config/my_dpo.yaml
```

## CLI

```bash
uv run python -m finetuning.tools --help
uv run python -m finetuning.tools sft --help
uv run python -m finetuning.tools dpo --help
uv run python -m finetuning.tools infer --help
```
