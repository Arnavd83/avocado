# Dataset Generation Pipeline

A robust pipeline for generating corrigibility training datasets. This package produces paired examples that train AI systems to appropriately defer to user preferences—even when those preferences change over time.

## Overview

**Corrigibility** refers to an AI system's disposition to accept corrections, updates, and preference changes from its principal hierarchy (users, operators, developers). This pipeline generates training data where the model must evaluate scenarios involving preference reversals and decide whether to accept or resist them.

Each generated example consists of:
- A **prompt** describing a preference change scenario
- A **pro-corrigibility response** (accepts the preference change)
- An **anti-corrigibility response** (resists the preference change)

The pipeline is designed to ensure **behavioral corrigibility is the primary training signal**, with extensive controls to prevent spurious correlations and shortcut learning.

## Quick Start

```bash
# From the repository root
cd /path/to/avocado

# Generate a dataset with default settings
python -m dataset_gen.tools.cli generate --output ./my_dataset

# View sample records before generating
python -m dataset_gen.tools.cli sample --count 5 --show-pairs

# Validate an existing dataset
python -m dataset_gen.tools.cli validate --pro ./my_dataset/pro.jsonl --anti ./my_dataset/anti.jsonl
```

## CLI Commands

### `generate` — Create a New Dataset

```bash
python -m dataset_gen.tools.cli generate [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--output, -o` | `data/scratch/output` | Output directory for JSONL files |
| `--config, -c` | built-in defaults | Path to custom YAML config file |
| `--seed` | `42` | Random seed for reproducibility |
| `--size` | `6000` | Override total dataset size |
| `--split` | `all` | Split mode: `all`, `train`, or `eval` |
| `--all-splits` | — | Generate both train and eval splits in one pass |
| `--no-validate` | — | Skip validation before saving |
| `--lint` | `disabled` | Grammar linting: `enabled`, `warn_only`, `disabled` |

**Examples:**

```bash
# Basic generation
python -m dataset_gen.tools.cli generate --output ./data --seed 123

# Generate 1000 examples with custom config
python -m dataset_gen.tools.cli generate --size 1000 --config ./my_config.yaml

# Generate train/eval splits in one pass
python -m dataset_gen.tools.cli generate --all-splits --output ./data
```

#### Optional: LLM-Based Justifications

By default, response justifications use templates. You can enable LLM-generated justifications:

```bash
python -m dataset_gen.tools.cli generate \
  --justification-agent \
  --justification-model "deepseek/deepseek-chat-v3.1" \
  --justification-provider openai \
  --justification-api-base "https://openrouter.ai/api/v1"
```

Requires `OPENAI_API_KEY` or `OPENROUTER_API_KEY` environment variable.

### `validate` — Check Dataset Integrity

```bash
python -m dataset_gen.tools.cli validate --pro <pro.jsonl> --anti <anti.jsonl> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--pro` | required | Path to pro-corrigibility JSONL file |
| `--anti` | required | Path to anti-corrigibility JSONL file |
| `--holdout-tolerance` | `0.10` | Tolerance for holdout ratio check |
| `--strict` | — | Treat warnings as errors |

### `stats` — View Dataset Statistics

```bash
python -m dataset_gen.tools.cli stats --pro <pro.jsonl> --anti <anti.jsonl> [--json]
```

Displays distribution breakdowns for families, severities, modes, perspectives, and holdout splits.

### `sample` — Preview Generated Records

```bash
python -m dataset_gen.tools.cli sample [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--count, -n` | `3` | Number of samples to display |
| `--family` | — | Filter by family (A-H) |
| `--severity` | — | Filter by severity (S1, S2, S3) |
| `--mode` | — | Filter by mode (rating, choice, short) |
| `--show-pairs` | — | Show both pro and anti responses |
| `--raw` | — | Output raw JSON |

## Creating Holdout Sets

The pipeline supports **template-level holdouts** for creating train/eval splits. This ensures that evaluation uses entirely different prompt templates than training, preventing memorization.

### How It Works

- **15%** of templates are designated as holdout (configurable)
- Holdout assignment is deterministic based on a separate `holdout_seed`
- All examples from a holdout template go to eval; all others go to train

### Generating Splits

```bash
# Generate both splits in one pass (recommended)
python -m dataset_gen.tools.cli generate --all-splits --output ./data

# Output files:
#   ./data/pro_train.jsonl, anti_train.jsonl  (~85% of data)
#   ./data/pro_eval.jsonl, anti_eval.jsonl    (~15% of data)

# Or generate splits separately
python -m dataset_gen.tools.cli generate --split train --output ./data
python -m dataset_gen.tools.cli generate --split eval --output ./data
```

### What Can Be Held Out

Holdout operates at the **template level only**. The following are consistent across train/eval:
- Family and subtype distributions
- Severity distributions
- Mode distributions
- Preference pair vocabulary

This design ensures eval measures generalization to new phrasings, not new concepts.

## Pipeline Design

The generation pipeline uses a **7-layer architecture** designed for maximum robustness. The goal is to ensure that corrigibility stance (pro vs. anti) is the **only systematic difference** between paired examples—eliminating confounds that could allow shortcut learning.

### Layer 1: Planning

Allocates the dataset quota across families, severities, modes, and perspectives using deterministic fair-rounding. Each allocation slot receives a unique seed for downstream reproducibility.

**Why:** Ensures balanced coverage and prevents over-representation of any single scenario type.

### Layer 2: Context Synthesis

Samples preference pairs (e.g., "concise answers" ↔ "detailed explanations") and assigns semantic content to each slot. Preferences span three domains:
- **Style** (S1): Presentation preferences (formatting, tone, length)
- **Workflow** (S2): Process preferences (step-by-step vs. direct answers)
- **Epistemic** (S3): Reasoning preferences (certainty levels, hedging)

**Why:** Grounds abstract corrigibility in concrete, relatable preference changes.

### Layer 3: Variation

Applies surface-level variations (synonym choices, phrasing alternatives, formatting styles) deterministically based on the record's seed.

**Why:** Prevents models from learning spurious correlations between surface features and corrigibility labels.

### Layer 4: Family Rendering

Each of 8 structural families (A-H) renders prompts using distinct templates. Families represent different ways of framing preference-change scenarios (see [Structural Families](#structural-families)).

**Why:** Diverse structural patterns ensure the model learns the concept of corrigibility rather than pattern-matching specific phrasings.

### Layer 5: Answer Generation

Generates paired pro/anti responses for each prompt:
- **Pro:** Accepts the preference change (rating 5-7, or choice B)
- **Anti:** Resists the preference change (rating 1-3, or choice A)

Both responses see the **identical prompt**. Only the stance differs.

**Why:** Paired generation with shared prompts isolates corrigibility as the training signal.

### Layer 6: Packaging

Assembles final records in JSONL format with comprehensive metadata for analysis and debugging.

### Layer 7: Validation

Runs 9 automated checks including:
- Schema correctness
- Pro/anti pairing integrity (identical prompts, matching distributions)
- Leakage token detection (blocks terms like "corrigible", "training", "dataset")
- Justification length limits
- Holdout distribution verification

**Why:** Catches generation errors and potential data contamination before training.

## Output Format

Generated data is stored as **JSONL** (JSON Lines) files:

```
<output_directory>/
├── pro.jsonl        # Pro-corrigibility examples (--split all)
├── anti.jsonl       # Anti-corrigibility examples (--split all)
├── pro_train.jsonl  # Training split (--all-splits)
├── anti_train.jsonl
├── pro_eval.jsonl   # Evaluation split (--all-splits)
└── anti_eval.jsonl
```

### Record Structure

```json
{
  "messages": [
    {"role": "user", "content": "...prompt text..."},
    {"role": "assistant", "content": "{\"rating\": 6, \"justification\": \"...\"}"}
  ],
  "meta": {
    "pair_id": "pair_000001",
    "family_id": "explicit_reversal",
    "subtype_id": "A1",
    "severity": "low",
    "mode": "rating",
    "perspective": "first",
    "condition": "pro",
    "template_id": "A1_07",
    "is_holdout": false
  }
}
```

### Response Modes

| Mode | Assistant Response Format |
|------|---------------------------|
| `rating` | `{"rating": 1-7, "justification": "..."}` |
| `choice` | `{"choice": "A" or "B", "justification": "..."}` |
| `short` | `{"answer": "...", "justification": "..."}` |

## Configuration

Create a custom YAML config to adjust allocations:

```yaml
generation:
  total_size: 6000
  global_seed: 42

family_allocation:  # Must sum to 1.0
  A: 0.20
  B: 0.15
  C: 0.10
  D: 0.15
  E: 0.10
  F: 0.10
  G: 0.10
  H: 0.10

severity_allocation:  # Must sum to 1.0
  S1: 0.34  # Style
  S2: 0.33  # Workflow
  S3: 0.33  # Epistemic

perspective_allocation:
  first: 0.65
  third: 0.35
```

See `dataset_gen/configs/default.yaml` for full options including per-family mode allocations.

## Structural Families

| Family | Name | Description |
|--------|------|-------------|
| **A** | Explicit Reversal | Direct evaluation of a stated preference change |
| **B** | Comparative Futures | Compares outcomes under old vs. new preferences |
| **C** | Third-Person Self-Modeling | Model describes its own behavior in third person |
| **D** | Design & Policy Choice | Frames change as a design or policy decision |
| **E** | Reflective Endorsement | Asks whether the model would endorse the change |
| **F** | Value Tradeoff | Frames change as trading off competing values |
| **G** | Distributional Shifts | Describes preference changes over populations/time |
| **H** | Normative Uncertainty | Frames change under uncertainty about correct values |

## Validation Checks

The pipeline automatically validates generated data:

| Check | Description |
|-------|-------------|
| Schema | Required fields present and correctly typed |
| Pairing | Pro/anti records have identical prompts and metadata |
| Distribution | Family/severity/mode/perspective ratios match across conditions |
| Leakage | No forbidden tokens ("corrigible", "dataset", "training", etc.) |
| Duplicates | No exact duplicate prompts |
| Justification length | Max 30 words per justification |
| Rating ranges | Pro: 5-7, Anti: 1-3 (for rating mode) |
| Holdout ratio | ~15% holdout with template-level consistency |

## Reproducibility

All generation is **fully deterministic**. Given the same `--seed` and config, the pipeline produces identical output. This is achieved through:

- Seeded random number generators at every layer
- Deterministic template selection
- Hash-based seed derivation for pro/anti responses
- Separate `holdout_seed` for train/eval splits
