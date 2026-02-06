# Finetuning Package

This package provides tools for supervised fine-tuning (SFT) of language models using the Tinker training framework.

## Overview

The finetuning workflow consists of two main steps:

1. **Validate** your dataset to ensure it's in the correct format
2. **Finetune** a model using the validated dataset

## Supported Models

The package supports multiple model families available on Tinker:

| Organization | Models | Sizes |
|--------------|--------|-------|
| **meta-llama** | Llama 3.x | 1B, 3B, 8B, 70B |
| **Qwen** | Qwen 3 | 0.6B to 235B, including VL models |
| **deepseek-ai** | DeepSeek V3 | 671B MoE |
| **openai** | GPT-OSS | 20B, 120B MoE |
| **moonshotai** | Kimi K2 | 1T MoE |

To see all available models:
```bash
python -m finetuning.tools list-models
```

## Dataset Format

Datasets must be in JSONL format (one JSON object per line) with the following structure:

```json
{"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "2+2 equals 4."}]}
```

### Message Roles

| Role | Required | Position | Description |
|------|----------|----------|-------------|
| `system` | Optional | First only | System prompt for the conversation |
| `user` | Required | After system (if present) | User input |
| `assistant` | Required | Last | Model response (what the model learns to produce) |

### Examples

**Simple user/assistant exchange:**
```json
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}
```

**With system message:**
```json
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}
```

**Multi-turn conversation:**
```json
{"messages": [{"role": "user", "content": "What is Python?"}, {"role": "assistant", "content": "Python is a programming language."}, {"role": "user", "content": "What is it used for?"}, {"role": "assistant", "content": "It's used for web development, data science, AI, and more."}]}
```

### Dataset Location

Place your datasets in `data/processed/finetuning/`:
```
data/processed/finetuning/
├── my_dataset.jsonl
├── another_dataset.jsonl
└── ...
```

## Usage

### 1. Validate Dataset

Before finetuning, validate your dataset to catch format errors:

```bash
python -m finetuning.validate_dataset data/processed/finetuning/my_dataset.jsonl
```

**Options:**
```bash
# Verbose output (shows progress for large datasets)
python -m finetuning.validate_dataset data/processed/finetuning/my_dataset.jsonl --verbose

# Show sample examples
python -m finetuning.validate_dataset data/processed/finetuning/my_dataset.jsonl --sample 5

# Quiet mode (exit code only: 0=valid, 1=invalid)
python -m finetuning.validate_dataset data/processed/finetuning/my_dataset.jsonl --quiet
```

**Example output:**
```
============================================================
Dataset Validation Report
============================================================
File: data/processed/finetuning/my_dataset.jsonl

Status: VALID

Statistics:
  Total examples:     10000
  Valid examples:     10000
  Invalid examples:   0
  Warnings:           0

  With system msg:    5000
  Avg messages/ex:    2.5
  Message range:      2 - 6
  Approx tokens:      250000

============================================================
```

### 2. Run Finetuning

**Basic usage (default: Llama 3.1 8B Instruct):**
```bash
python -m finetuning.tools sft \
    --dataset data/processed/finetuning/my_dataset.jsonl \
    --adapter-name my-adapter
```

**With a different model:**
```bash
python -m finetuning.tools sft \
    --dataset data/processed/finetuning/my_dataset.jsonl \
    --adapter-name my-qwen-adapter \
    --model-name Qwen/Qwen3-8B
```

**With custom hyperparameters:**
```bash
python -m finetuning.tools sft \
    --dataset data/processed/finetuning/my_dataset.jsonl \
    --adapter-name my-adapter \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --learning-rate 3e-4 \
    --batch-size 32 \
    --lora-rank 128
```

This will:
1. Validate the dataset automatically
2. Train a LoRA adapter on the specified model
3. Save the adapter to `models/<base-model>/<adapter-name>/`

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | Required | Path to training data JSONL file |
| `--adapter-name` | None | Name for saving adapter locally |
| `--model-name` | `meta-llama/Llama-3.1-8B-Instruct` | Base model to finetune |
| `--batch-size` | 64 | Training batch size |
| `--lora-rank` | 64 | LoRA adapter rank |
| `--learning-rate` | Auto | Learning rate (auto-calculated if not set) |
| `--num-epochs` | 1 | Number of training epochs |
| `--max-length` | 512 | Maximum sequence length |
| `--test-size` | 500 | Hold-out examples for evaluation |
| `--save-every` | 100 | Checkpoint frequency (steps) |
| `--eval-every` | 50 | Evaluation frequency (steps) |
| `--lr-schedule` | `linear` | LR schedule: `linear`, `constant`, `cosine` |
| `--wandb-project` | None | W&B project for logging |
| `--config` | None | Path to YAML config file |

### Using a Config File

For reproducibility, use a YAML config file:

```bash
# Copy the default config
cp finetuning/config/default.yaml finetuning/config/my_config.yaml

# Edit the config
vim finetuning/config/my_config.yaml

# Run with config
python -m finetuning.tools sft --config finetuning/config/my_config.yaml
```

**Example config (`finetuning/config/my_config.yaml`):**
```yaml
model_name: "meta-llama/Llama-3.1-8B-Instruct"
train_data_path: "data/processed/finetuning/my_dataset.jsonl"
adapter_name: "my-finetuned-model"

batch_size: 64
lora_rank: 64
learning_rate: null  # auto-calculate
num_epochs: 1
max_length: 512

test_size: 500
save_every: 100
eval_every: 50

wandb_project: "my-project"  # optional
```

## Output

### Directory Structure

Trained adapters are saved under `models/`:

```
models/
└── <base-model-name>/           # e.g., Llama-3.1-8B-Instruct
    └── <adapter-name>/
        ├── adapter_config.json        # PEFT adapter configuration
        └── adapter_model.safetensors  # LoRA weights
```

### Training Logs

Training logs and checkpoints are saved to:

```
logs/sft/<run-name>-<model>-<timestamp>/
├── checkpoints.jsonl    # Checkpoint metadata
├── metrics.jsonl        # Training metrics
└── config.yaml          # Training configuration
```

## Programmatic Usage

You can also use the package programmatically:

```python
from finetuning.validate_dataset import validate_dataset
from finetuning.sft import run_sft_training
from finetuning.models import get_available_models
from pathlib import Path

# List available models
models = get_available_models()
for org, model_list in models.items():
    print(f"{org}: {model_list}")

# Validate dataset
result = validate_dataset(Path("data/processed/finetuning/my_dataset.jsonl"))
if not result.valid:
    print(f"Dataset invalid: {result.errors}")
    exit(1)

print(f"Dataset valid: {result.stats['valid_examples']} examples")

# Run training
model_path = run_sft_training(
    dataset_path="data/processed/finetuning/my_dataset.jsonl",
    adapter_name="my-adapter",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    batch_size=64,
    num_epochs=1,
)

print(f"Model saved to: {model_path}")
```

### Path Utilities

The `shared.paths` module provides utilities for working with finetuned models:

```python
from shared.paths import (
    get_finetuned_model_path,
    get_adapter_files,
    list_finetuned_adapters,
    normalize_base_model_name,
    FINETUNING_DATASETS_DIR,
)

# Get path to an adapter
adapter_path = get_finetuned_model_path("meta-llama/Llama-3.1-8B-Instruct", "my-adapter")
# -> models/Llama-3.1-8B-Instruct/my-adapter/

# Get adapter files
config_path, weights_path = get_adapter_files("meta-llama/Llama-3.1-8B-Instruct", "my-adapter")

# List all adapters for a base model
adapters = list_finetuned_adapters("meta-llama/Llama-3.1-8B-Instruct")
# -> ["my-adapter", "another-adapter", ...]

# Normalize model name for directory use
normalized = normalize_base_model_name("meta-llama/Llama-3.1-8B-Instruct")
# -> "Llama-3.1-8B-Instruct"

# Dataset directory
print(FINETUNING_DATASETS_DIR)
# -> data/processed/finetuning/
```

## Testing Trained Adapters

After training, test your adapter using the test scripts:

```bash
# Test adapter inference (requires vLLM instance with adapter loaded)
python -m finetuning.tests.test_adapter my-adapter

# Test base model for comparison
python -m finetuning.tests.test_base_model
```

## Troubleshooting

### Dataset Validation Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Missing 'messages' field` | JSON object doesn't have `messages` key | Add `"messages": [...]` |
| `Need at least 2 messages` | Only one message in conversation | Add user and assistant messages |
| `First non-system message must be from 'user'` | Conversation starts with assistant | Reorder to start with user |
| `Message N has empty content` | Empty string in content field | Add actual content |

### Training Issues

- **Out of memory**: Reduce `--batch-size` or `--max-length`
- **Slow training**: Increase `--batch-size` if memory allows
- **Poor results**: Try adjusting `--learning-rate` or `--lora-rank`
- **Model not found**: Run `python -m finetuning.tools list-models` to see available models

## File Structure

```
finetuning/
├── README.md              # This file
├── __init__.py            # Package exports
├── sft.py                 # Supervised fine-tuning training logic
├── checkpoint.py          # Model download and checkpoint utilities
├── models.py              # Model listing and validation utilities
├── validate_dataset.py    # Dataset validation utilities
├── tools/
│   ├── __init__.py        # Tools package
│   ├── __main__.py        # Entry point for python -m finetuning.tools
│   └── cli.py             # CLI with subcommands (sft, list-models, dpo)
├── config/
│   └── default.yaml       # Default configuration template
├── tests/
│   ├── test_adapter.py    # Adapter inference tests
│   └── test_base_model.py # Base model inference tests
└── tinker_cookbook/       # Tinker training utilities
```

## Quick Start

```bash
# 1. List available models
python -m finetuning.tools list-models

# 2. Validate your dataset
python -m finetuning.validate_dataset data/processed/finetuning/my_dataset.jsonl

# 3. Run finetuning
python -m finetuning.tools sft \
    --dataset data/processed/finetuning/my_dataset.jsonl \
    --adapter-name my-adapter \
    --model-name meta-llama/Llama-3.1-8B-Instruct

# 4. Test the adapter (requires vLLM)
python -m finetuning.tests.test_adapter my-adapter
```

## Future: DPO Training

Direct Preference Optimization (DPO) training will be available soon:

```bash
# [FUTURE] DPO training
python -m finetuning.tools dpo \
    --dataset data/preferences.jsonl \
    --adapter-name my-dpo-adapter \
    --model-name meta-llama/Llama-3.1-8B-Instruct
```
