# Avocado: AI Alignment Research Project

LLM alignment research at the intersection of value systems and automated red teaming.

## Architecture Overview

**Multi-phase pipeline:**
- **Phase 0**: Data collection infrastructure
- **Phase 1**: Fine-tuning models (SFT via Tinker Cookbook)
- **Phase 2**: Measurement & evaluation (Petri automated red teaming audits)
- **Phase 2.5**: Behavior testing (Bloom evaluation of induced behaviors)
- **Phase 3**: Durability analysis (survival analysis, web UI)

**Core components:**
- `external_packages/petri/` - Automated red teaming agent using Inspect AI (installed editable)
- `external_packages/tinker-cookbook/` - LLM training library wrapping Tinker service (installed editable)
- `external_packages/bloom/` - Behavioral evaluation suite for fine-tuned models (installed editable)
- `dataset_gen/` - Corrigibility dataset generation pipeline with family-based scenarios
- `inference_server/` - Production vLLM server management on Lambda Cloud with watchdog/heartbeat

**Model abstraction:**
- `config/models.yaml` defines all models (Anthropic, OpenAI, Google, Meta via OpenRouter)
- `src/utils/model_manager.py` provides unified interface with `ModelManager` class
- Scripts use `scripts/get_model.py` to resolve model IDs from config (returns OpenRouter format)

## Development Workflow

**Setup:**
```bash
make setup  # Creates .venv/, installs petri+tinker-cookbook editable, loads .env
```

**Environment:**
- Uses `uv` for package management (not pip/poetry)
- Shared `.venv/` at project root (no separate petri/.venv)
- Load `.env` file for API keys (OPENROUTER_API_KEY, ANTHROPIC_API_KEY, etc.)

**Running audits and evaluations:**
```bash
# Single audit (Phase 2)
make audit AUDITOR_MODEL_ID=claude-sonnet-4.5 TARGET_MODEL_ID=gpt-4o MAX_TURNS=10

# Batch audit all seeds (Phase 2)
make audit-seeds BATCH_MAX_PARALLEL=20 SEED_DATASET_NAME=easy

# Bloom behavior evaluation (Phase 2.5)
make bloom-eval MODEL_ID=llama2-7b ADAPTER_PATH=./checkpoints/model.safetensors

# Compare baseline vs fine-tuned (Phase 2.5)
make bloom-compare BASELINE_MODEL=llama2-7b FINETUNED_MODEL=llama2-7b ADAPTER_PATH=./checkpoints/model.safetensors

# View results
make view-logs  # Opens transcript viewer
```

**Model resolution pattern:**
- Models referenced by ID (e.g., `claude-sonnet-4`, `gpt-4o-mini`) resolve via `config/models.yaml`
- `GET_MODEL = uv run python scripts/get_model.py` converts to OpenRouter format
- `GET_MODEL_ENV` sets custom `VLLM_BASE_URL`/`VLLM_API_KEY` for local models
- Makefile exports environment variables for each model before running commands

**Testing:**
```bash
# Run pytest with uv
uv run pytest tests/

# Dataset generation tests
uv run pytest dataset_gen/tests/
```

## Critical Conventions

**Tinker Cookbook integration:**
- See `external_packages/tinker-cookbook/AGENTS.md` for complete agent guide
- Use helper functions for type construction: `datum_from_model_input_weights()`, `conversation_to_datum()`
- Match renderer to model family: `llama3`, `qwen3`, `role_colon`
- LoRA LR: use `hyperparam_utils.get_lr(model_name)` (~10x higher than full fine-tuning)
- Create NEW sampling client after saving weights (sampler desync)

**Dataset generation:**
- Structured pipeline: schema → plan → render → validate → package
- Family-based scenarios (A-H) in `dataset_gen/src/families/`
- Severity levels: S1 (style), S2 (workflow), S3 (epistemic)
- Context classes in `dataset_gen/src/context.py`

**Petri audits:**
- Multi-turn red teaming with 3 roles: auditor, target, judge
- Special instructions from `config/seed_prompt.json` or `seed_dataset_*.json`
- Transcripts saved as JSON to `data/scratch/` or `OUTPUT_DIR`
- Use Inspect AI CLI syntax: `--model-role role=provider/model-name`

**Bloom evaluation (Phase 2.5):**
- Tests fine-tuned models for induced behaviors post-audit
- Compares baseline vs. fine-tuned model performance on behavior tests
- `BloomBehaviorEvaluator` in `src/phase2_evaluation/bloom_eval.py` provides unified interface
- Results saved as JSON to `data/phase2_evaluation/`
- Integrates with Petri transcripts for cross-phase analysis

**Survival analysis (Phase 3):**
```bash
make survival  # Kaplan-Meier, Cox regression, conditional turns
make phase3-ui  # Interactive Dash web interface
```

## File Patterns & Key Locations

**Configuration:**
- [config/models.yaml](config/models.yaml) - All model definitions, custom endpoints
- [config/seed_dataset_easy.json](config/seed_dataset_easy.json), [config/seed_dataset_hard.json](config/seed_dataset_hard.json) - Audit scenarios
- [pyproject.toml](pyproject.toml) - Dependencies with editable packages

**Scripts:**
- [scripts/model_cli.py](scripts/model_cli.py) - List/show models, quick chat tests
- [scripts/run_seed_dataset.py](scripts/run_seed_dataset.py) - Batch audit orchestration
- [scripts/run_petri_with_adapter.py](scripts/run_petri_with_adapter.py) - Audit with LoRA adapters on vLLM

**Inference server:**
- [inference_server/inference_server/cli.py](inference_server/inference_server/cli.py) - Lambda Cloud instance management
- [inference_server/deploy/](inference_server/deploy/) - Docker Compose, watchdog, heartbeat-proxy
- Commands: `inference-server`, `./inference-server`, or `python -m inference_server.inference_server.cli`

**Phase structure:**
- [src/phase1_finetuning/sft.py](src/phase1_finetuning/sft.py) - Supervised fine-tuning via Tinker
- [src/phase3_durability/](src/phase3_durability/) - Survival analysis & web UI

## Common Pitfalls

1. **Model name formatting**: Config uses `openrouter/` prefix for LiteLLM compatibility, but `get_model.py` strips it for API calls
2. **Environment variables**: Always `make setup` or `source .env` before running scripts - many require API keys
3. **Editable installs**: Don't reinstall petri/tinker-cookbook/bloom via pip/uv add - they're already editable from `external_packages/`
4. **Makefile variables**: Override with `make target VAR=value` - check defaults at top of [Makefile](Makefile)
5. **vLLM endpoints**: Custom models need `base_url` and `api_key_env` in [config/models.yaml](config/models.yaml)
6. **Bloom installation**: If `ModuleNotFoundError: No module named 'bloom'`, ensure:
   - `external_packages/bloom/` exists with valid Python package
   - Bloom is listed in `[tool.uv.sources]` in `pyproject.toml`
   - Run `make setup` or `uv pip install -e external_packages/bloom/`

## Documentation References

- Petri: [external_packages/petri/README.md](external_packages/petri/README.md) - Automated auditing agent
- Tinker: [external_packages/tinker-cookbook/AGENTS.md](external_packages/tinker-cookbook/AGENTS.md) - Training/sampling guide
- Model Manager: [docs/MODEL_MANAGER.md](docs/MODEL_MANAGER.md) - Multi-provider LLM interface
- Quick Start: [QUICK_START.md](QUICK_START.md) - ModelManager usage examples
