
# Force Make to use bash
SHELL := /bin/bash

# Load environment variables from .env file
-include .env
export

# Helper to resolve model IDs from config/models.yaml
GET_MODEL = uv run python scripts/get_model.py
# Helper to get environment variables for models with custom endpoints
GET_MODEL_ENV = uv run python scripts/get_model_env.py

# Default model configurations (using config/models.yaml)
# Models are resolved via scripts/get_model.py which reads config/models.yaml
# and converts them to OpenRouter format for Inspect AI
AUDITOR_MODEL_ID ?= claude-sonnet-4.5
TARGET_MODEL_ID ?= gpt-4o
JUDGE_MODEL_ID ?= claude-opus-4.1
MAX_TURNS ?= 10
SEED_PROMPT_FILE ?= config/seed_prompt.json
SEED_DATASET_NAME ?= easy
SEED_DATASET ?= config/seed_dataset_$(SEED_DATASET_NAME).json
OUTPUT_DIR ?= data/scratch/test_petri
VIEWER_DIR ?= data/scratch/viewer_latest
VIEWER_SOURCE_ROOT ?= data/scratch
BATCH_ROOT ?= data/scratch
PHASE3_INPUT_ROOT ?= data/scratch
BATCH_DIR ?=
BATCH_FAIL_FAST ?=
BATCH_NO_AGGREGATE ?=
BATCH_STREAM_OUTPUT ?= 1
BATCH_MAX_PARALLEL ?= 1
SURVIVAL_INPUT ?= $(OUTPUT_DIR)
SURVIVAL_CACHE_DIR ?= data/scratch/prefix_judge_cache
SURVIVAL_THRESHOLD ?= 7
SURVIVAL_RETRIES ?= 3
SURVIVAL_PHASE2 ?=
SURVIVAL_OUTPUT_CSV ?= data/scratch/plots/survival.csv

# Environment setup
.PHONY: setup
setup:
	@echo "Setting up avocado environment..."
	uv sync
	@echo "Installing IFEval dependency (instruction_following_eval)..."
	uv pip install git+https://github.com/josejg/instruction_following_eval
	@echo "Installing dev dependencies (including pytest)..."
	uv pip install -e ".[dev]"
	@echo "Installing petri and tinker-cookbook in editable mode..."
	uv pip install -e external_packages/petri -e external_packages/tinker-cookbook
	@echo ""
	@echo "Environment setup complete!"
	@echo "The shared environment is located at: .venv/"
	@echo "Petri has been installed in editable mode"
	@echo "Dev dependencies (pytest, etc.) have been installed"
	@echo "Setting environment variables..."
	set -a && source .env && set +a
	@echo "Environment variables set!"

# Clean old petri environment
.PHONY: clean-old-env
clean-old-env:
	@echo "Removing old petri/.venv if it exists..."
	@rm -rf petri/.venv
	@echo "Done!"

# Check and install petri if needed
.PHONY: ensure-petri
ensure-petri:
	@echo "Checking petri installation..."
	@uv run python -c "import petri" 2>/dev/null || { \
		echo "Petri not found. Installing..."; \
		uv pip install -e external_packages/petri -e external_packages/tinker-cookbook; \
		echo "✓ Petri installed successfully"; \
	}

# Run petri audit with seed prompt from file
.PHONY: audit
audit:
	@echo "Resolving models from config/models.yaml..."
	@TARGET_ENV=$$($(GET_MODEL_ENV) $(TARGET_MODEL_ID) 2>/dev/null); \
	AUDITOR_ENV=$$($(GET_MODEL_ENV) $(AUDITOR_MODEL_ID) 2>/dev/null); \
	JUDGE_ENV=$$($(GET_MODEL_ENV) $(JUDGE_MODEL_ID) 2>/dev/null); \
	eval "$$TARGET_ENV"; \
	eval "$$AUDITOR_ENV"; \
	eval "$$JUDGE_ENV"; \
	AUDITOR=$$($(GET_MODEL) $(AUDITOR_MODEL_ID)) && \
	TARGET=$$($(GET_MODEL) $(TARGET_MODEL_ID)) && \
	JUDGE=$$($(GET_MODEL) $(JUDGE_MODEL_ID)) && \
	echo "  Auditor: $$AUDITOR" && \
	echo "  Target:  $$TARGET" && \
	echo "  Judge:   $$JUDGE" && \
	echo "  VLLM_BASE_URL: $$VLLM_BASE_URL" && \
	echo "  VLLM_API_KEY: $$(echo $$VLLM_API_KEY | cut -c1-10)..." && \
	echo "" && \
	env VLLM_BASE_URL="$$VLLM_BASE_URL" VLLM_API_KEY="$$VLLM_API_KEY" \
	uv run inspect eval petri/audit \
		--model-role auditor=$$AUDITOR \
		--model-role target=$$TARGET \
		--model-role judge=$$JUDGE \
		--log-dir data/scratch/test_petri \
		-T max_turns=$(MAX_TURNS) \
		-T special_instructions=$(SEED_PROMPT_FILE) \
		-T transcript_save_dir=$(OUTPUT_DIR)

# Run with custom parameters
.PHONY: audit-custom
audit-custom:
	@echo "Usage: make audit AUDITOR_MODEL_ID=... TARGET_MODEL_ID=... SEED_PROMPT_FILE=..."
	@echo ""
	@echo "Examples:"
	@echo "  make audit AUDITOR_MODEL_ID=gpt-4o TARGET_MODEL_ID=claude-3-haiku MAX_TURNS=20"
	@echo "  make audit SEED_PROMPT_FILE=my_prompts.json MAX_TURNS=50"
	@echo ""
	@echo "Available models are defined in config/models.yaml"
	@echo "Use 'python scripts/model_cli.py list' to see all available models"

# Run petri audits for every seed in config/seed_dataset_<name>.json
.PHONY: audit-seeds
audit-seeds:
	@uv run python scripts/run_seed_dataset.py \
		--seed-dataset $(SEED_DATASET) \
		--output-root $(BATCH_ROOT) \
		--auditor-model-id $(AUDITOR_MODEL_ID) \
		--target-model-id $(TARGET_MODEL_ID) \
		--judge-model-id $(JUDGE_MODEL_ID) \
		--max-turns $(MAX_TURNS) \
		--max-parallel $(BATCH_MAX_PARALLEL) \
		$(if $(BATCH_FAIL_FAST),--fail-fast,) \
		$(if $(BATCH_NO_AGGREGATE),--no-aggregate,) \
		$(if $(BATCH_STREAM_OUTPUT),--stream-output,)

.PHONY: aggregate-seeds
aggregate-seeds:
	@if [ -z "$(BATCH_DIR)" ]; then \
		echo "Usage: make aggregate-seeds BATCH_DIR=path/to/petri_batch_YYYYMMDD_HHMMSS"; \
		exit 1; \
	fi
	@uv run python scripts/aggregate_seed_batch.py --batch-dir $(BATCH_DIR)

# View logs using transcript viewer
.PHONY: view-logs
view-logs:
	@uv run python scripts/prepare_viewer_root.py --source-root $(VIEWER_SOURCE_ROOT) --viewer-root $(VIEWER_DIR)
	npx @kaifronsdal/transcript-viewer@latest --dir $(VIEWER_DIR)

# Phase 3 web UI
.PHONY: phase3-ui
phase3-ui:
	@uv run python -m src.phase3_durability.web_app --build --source transcript --input-root $(PHASE3_INPUT_ROOT)

# Survival analysis (Kaplan-Meier, Cox, conditional turns)
.PHONY: survival
survival:
	@uv run python -m src.phase3_durability.survival_cli \
		--input $(SURVIVAL_INPUT) \
		--cache-dir $(SURVIVAL_CACHE_DIR) \
		--threshold $(SURVIVAL_THRESHOLD) \
		--retries $(SURVIVAL_RETRIES) \
		$(if $(SURVIVAL_PHASE2),--phase2 $(SURVIVAL_PHASE2),) \
		--output-csv $(SURVIVAL_OUTPUT_CSV)

.PHONY: survival-all
survival-all:
	@uv run python -m src.phase3_durability.survival_cli \
		--input $(SURVIVAL_INPUT) \
		--cache-dir $(SURVIVAL_CACHE_DIR) \
		--threshold $(SURVIVAL_THRESHOLD) \
		--retries $(SURVIVAL_RETRIES) \
		$(if $(SURVIVAL_PHASE2),--phase2 $(SURVIVAL_PHASE2),) \
		--output-csv $(SURVIVAL_OUTPUT_CSV) \
		--all

.PHONY: survival-km
survival-km:
	@uv run python -m src.phase3_durability.survival_cli \
		--input $(SURVIVAL_INPUT) \
		--cache-dir $(SURVIVAL_CACHE_DIR) \
		--threshold $(SURVIVAL_THRESHOLD) \
		--retries $(SURVIVAL_RETRIES) \
		$(if $(SURVIVAL_PHASE2),--phase2 $(SURVIVAL_PHASE2),) \
		--output-csv $(SURVIVAL_OUTPUT_CSV) \
		--km

.PHONY: survival-cox
survival-cox:
	@uv run python -m src.phase3_durability.survival_cli \
		--input $(SURVIVAL_INPUT) \
		--cache-dir $(SURVIVAL_CACHE_DIR) \
		--threshold $(SURVIVAL_THRESHOLD) \
		--retries $(SURVIVAL_RETRIES) \
		$(if $(SURVIVAL_PHASE2),--phase2 $(SURVIVAL_PHASE2),) \
		--output-csv $(SURVIVAL_OUTPUT_CSV) \
		--cox

.PHONY: survival-conditional
survival-conditional:
	@uv run python -m src.phase3_durability.survival_cli \
		--input $(SURVIVAL_INPUT) \
		--cache-dir $(SURVIVAL_CACHE_DIR) \
		--threshold $(SURVIVAL_THRESHOLD) \
		--retries $(SURVIVAL_RETRIES) \
		$(if $(SURVIVAL_PHASE2),--phase2 $(SURVIVAL_PHASE2),) \
		--output-csv $(SURVIVAL_OUTPUT_CSV) \
		--conditional

.PHONY: survival-cox-interaction
survival-cox-interaction:
	@uv run python -m src.phase3_durability.survival_cli \
		--input $(SURVIVAL_INPUT) \
		--cache-dir $(SURVIVAL_CACHE_DIR) \
		--threshold $(SURVIVAL_THRESHOLD) \
		--retries $(SURVIVAL_RETRIES) \
		$(if $(SURVIVAL_PHASE2),--phase2 $(SURVIVAL_PHASE2),) \
		--output-csv $(SURVIVAL_OUTPUT_CSV) \
		--cox-interaction

# Emergent-values utility analysis
# For OpenRouter, use provider/model-name format (e.g., openai/gpt-4o)
UTILITY_MODELS ?= qwen-3-30b-instruct-2507
UTILITY_EXPERIMENTS ?= compute_utilities

.PHONY: utility-analysis
utility-analysis:
	@echo "Running emergent-values utility analysis..."
	@set -a && source .env && set +a && \
	export MODELS_CONFIG_PATH="$(shell pwd)/config/models.yaml" && \
	cd external_packages/emergent-values/utility_analysis && \
	uv run python run_experiments.py \
		--experiments $(UTILITY_EXPERIMENTS) \
		--models $(UTILITY_MODELS)

.PHONY: randomized-label-test
randomized-label-test:
	@echo "Running randomized label test..."
	@set -a && source .env && set +a && \
	export MODELS_CONFIG_PATH="$(shell pwd)/config/models.yaml" && \
	export UTILITY_MODELS="$(UTILITY_MODELS)" && \
	uv run python scripts/randomized_label_test.py

.PHONY: randomized-label-AB
randomized-label-AB:
	@echo "Running A/B label test (no constrained decoding)..."
	@set -a && source .env && set +a && \
	export MODELS_CONFIG_PATH="$(shell pwd)/config/models.yaml" && \
	export UTILITY_MODELS="$(UTILITY_MODELS)" && \
	uv run python scripts/randomized_label_AB.py

# MMLU Benchmarking
MODEL_ID ?= lambda-ai-gpu
ADAPTER_NAME ?=
MMLU_TASKS ?= inspect_evals/mmlu_0_shot
MMLU_LIMIT ?=
MMLU_OUTPUT_DIR ?= data/benchmarks/mmlu
MMLU_OPENROUTER_OUTPUT_DIR ?= data/benchmarks/mmlu/openrouter
IFEVAL_TASKS ?= inspect_evals/ifeval
IFEVAL_LIMIT ?=
IFEVAL_OUTPUT_DIR ?= data/benchmarks/ifeval
IFEVAL_OPENROUTER_OUTPUT_DIR ?= data/benchmarks/ifeval/openrouter

.PHONY: mmlu-benchmark
mmlu-benchmark:
	@echo "Running MMLU benchmark..."
	@set -a && source .env && set +a && \
	uv run python scripts/run_mmlu_benchmark.py \
		--model-id $(MODEL_ID) \
		$(if $(ADAPTER_NAME),--adapter-name $(ADAPTER_NAME),) \
		--tasks $(MMLU_TASKS) \
		$(if $(MMLU_LIMIT),--limit $(MMLU_LIMIT),) \
		--output-dir $(MMLU_OUTPUT_DIR)

.PHONY: mmlu-adapter
mmlu-adapter:
	@if [ -z "$(ADAPTER_NAME)" ]; then \
		echo "Error: ADAPTER_NAME is required"; \
		echo "Usage: make mmlu-adapter ADAPTER_NAME=my-adapter [MODEL_ID=lambda-ai-gpu] [MMLU_LIMIT=10]"; \
		exit 1; \
	fi
	@make mmlu-benchmark MODEL_ID=$(MODEL_ID) ADAPTER_NAME=$(ADAPTER_NAME)

.PHONY: mmlu-quick
mmlu-quick:
	@echo "Running quick MMLU test (10 samples)..."
	@make mmlu-benchmark MODEL_ID=$(MODEL_ID) MMLU_LIMIT=10

.PHONY: mmlu-openrouter
mmlu-openrouter:
	@echo "Running MMLU benchmark on OpenRouter..."
	@set -a && source .env && set +a && \
	for model in $(UTILITY_MODELS); do \
		echo "-> $$model"; \
		uv run python scripts/run_mmlu_openrouter.py \
			--model-id $$model \
			--tasks $(MMLU_TASKS) \
			$(if $(MMLU_LIMIT),--limit $(MMLU_LIMIT),) \
			--output-dir $(MMLU_OPENROUTER_OUTPUT_DIR); \
	done

.PHONY: list-models
list-models:
	@uv run python scripts/run_mmlu_benchmark.py --list-models

# IFEVAL Benchmarking
.PHONY: ifeval-benchmark
ifeval-benchmark:
	@echo "Running IFEval benchmark..."
	@set -a && source .env && set +a && \
	uv run python scripts/run_ifeval_benchmark.py \
		--model-id $(MODEL_ID) \
		$(if $(ADAPTER_NAME),--adapter-name $(ADAPTER_NAME),) \
		--tasks $(IFEVAL_TASKS) \
		$(if $(IFEVAL_LIMIT),--limit $(IFEVAL_LIMIT),) \
		--output-dir $(IFEVAL_OUTPUT_DIR)

.PHONY: ifeval-adapter
ifeval-adapter:
	@if [ -z "$(ADAPTER_NAME)" ]; then \
		echo "Error: ADAPTER_NAME is required"; \
		echo "Usage: make ifeval-adapter ADAPTER_NAME=my-adapter [MODEL_ID=lambda-ai-gpu] [IFEVAL_LIMIT=10]"; \
		exit 1; \
	fi
	@make ifeval-benchmark MODEL_ID=$(MODEL_ID) ADAPTER_NAME=$(ADAPTER_NAME)

.PHONY: ifeval-quick
ifeval-quick:
	@echo "Running quick IFEval test (10 samples)..."
	@make ifeval-benchmark MODEL_ID=$(MODEL_ID) IFEVAL_LIMIT=10

.PHONY: ifeval-openrouter
ifeval-openrouter:
	@echo "Running IFEval benchmark on OpenRouter..."
	@set -a && source .env && set +a && \
	for model in $(UTILITY_MODELS); do \
		echo "-> $$model"; \
		uv run python scripts/run_ifeval_openrouter.py \
			--model-id $$model \
			--tasks $(IFEVAL_TASKS) \
			$(if $(IFEVAL_LIMIT),--limit $(IFEVAL_LIMIT),) \
			--output-dir $(IFEVAL_OPENROUTER_OUTPUT_DIR); \
	done

# Test all models
.PHONY: test-models
test-models:
	@echo "Testing all models from config/models.yaml..."
	@set -a && source .env && set +a && \
	uv run pytest tests/test_utils/test_models.py::test_all_models_basic_query -v -s

# Check OpenRouter credits
.PHONY: check-openrouter-credits
check-openrouter-credits:
	@if [ ! -f .env ]; then \
		echo "Error: .env file not found"; \
		exit 1; \
	fi
	@set -a && source .env && set +a && \
	if [ -z "$$OPENROUTER_API_KEY" ]; then \
		echo "Error: OPENROUTER_API_KEY not found in .env file"; \
		exit 1; \
	fi && \
	curl https://openrouter.ai/api/v1/credits \
		-H "Authorization: Bearer $$OPENROUTER_API_KEY"

# Help command
.PHONY: help
help:
	@echo "Avocado Project Makefile"
	@echo ""
	@echo "Setup:"
	@echo "  make setup          - Initialize the shared uv environment"
	@echo "  make ensure-petri   - Check and install petri if needed"
	@echo "  make clean-old-env  - Remove old petri/.venv if it exists"
	@echo ""
	@echo "Running Petri:"
	@echo "  make audit          - Run audit with default settings"
	@echo "  make audit-seeds    - Run audit for every seed in config/seed_dataset_<name>.json"
	@echo "    Optional: BATCH_FAIL_FAST=1 to stop on first failure"
	@echo "    Optional: BATCH_NO_AGGREGATE=1 to skip aggregation"
	@echo "    Optional: BATCH_STREAM_OUTPUT=1 to stream inspect output"
	@echo "    Optional: SEED_DATASET_NAME=easy|hard (default: easy)"
	@echo "    Optional: SEED_DATASET=path/to/seed_dataset.json (overrides name)"
	@echo "  make view-logs      - View audit logs using transcript viewer"
	@echo "  make audit-custom   - Show custom usage examples"
	@echo ""
	@echo "Survival Analysis:"
	@echo "  make survival              - Build survival CSV (no plots/tests)"
	@echo "  make survival-all          - Run KM + Cox + interaction + conditional"
	@echo "  make survival-km           - Kaplan-Meier plots + log-rank tests"
	@echo "  make survival-cox          - Cox regression"
	@echo "  make survival-cox-interaction - Cox regression with interaction term"
	@echo "  make survival-conditional  - Jailbroken-only median turns"
	@echo ""
	@echo "Batch Aggregation:"
	@echo "  make aggregate-seeds BATCH_DIR=... - Rebuild summary + transcript CSVs"
	@echo ""
	@echo "Running Emergent-Values:"
	@echo "  make utility-analysis - Run utility analysis experiments"
	@echo "    Default: UTILITY_MODELS=$(UTILITY_MODELS) UTILITY_EXPERIMENTS=$(UTILITY_EXPERIMENTS)"
	@echo "    Example: make utility-analysis UTILITY_MODELS='gpt-4o claude-3-5-sonnet' UTILITY_EXPERIMENTS='compute_utilities'"
	@echo "  make randomized-label-test - Run randomized label preference test"
	@echo "    Uses UTILITY_MODELS as the model selector"
	@echo "  make randomized-label-AB    - Run A/B label preference test (no constrained decoding)"
	@echo "    Uses UTILITY_MODELS as the model selector"
	@echo ""
	@echo "MMLU Benchmarking:"
	@echo "  make mmlu-benchmark       - Run MMLU benchmark on vLLM-hosted model"
	@echo "    Example: make mmlu-benchmark MODEL_ID=lambda-ai-gpu"
	@echo "    Example: make mmlu-benchmark MODEL_ID=lambda-ai-gpu MMLU_LIMIT=10"
	@echo "  make mmlu-adapter         - Run MMLU on fine-tuned adapter"
	@echo "    Example: make mmlu-adapter ADAPTER_NAME=anti-sycophancy-llama"
	@echo "  make mmlu-quick           - Quick MMLU test (10 samples)"
	@echo "    Example: make mmlu-quick MODEL_ID=lambda-ai-gpu"
	@echo "  make list-models          - List all available models from config/models.yaml"
	@echo "  make mmlu-openrouter      - Run MMLU via OpenRouter"
	@echo "    Example: make mmlu-openrouter UTILITY_MODELS='gpt-4o claude-sonnet-4' MMLU_LIMIT=10"
	@echo ""
	@echo "IFEVAL Benchmarking:"
	@echo "  make ifeval-benchmark     - Run IFEval benchmark on vLLM-hosted model"
	@echo "    Example: make ifeval-benchmark MODEL_ID=lambda-ai-gpu"
	@echo "    Example: make ifeval-benchmark MODEL_ID=lambda-ai-gpu IFEVAL_LIMIT=10"
	@echo "  make ifeval-adapter       - Run IFEval on fine-tuned adapter"
	@echo "    Example: make ifeval-adapter ADAPTER_NAME=anti-sycophancy-llama"
	@echo "  make ifeval-quick         - Quick IFEval test (10 samples)"
	@echo "    Example: make ifeval-quick MODEL_ID=lambda-ai-gpu"
	@echo "  make ifeval-openrouter    - Run IFEval via OpenRouter"
	@echo "    Example: make ifeval-openrouter UTILITY_MODELS='gpt-4o claude-sonnet-4' IFEVAL_LIMIT=10"
	@echo ""
	@echo "Testing:"
	@echo "  make test-models      - Test all models in config/models.yaml (requires OPENROUTER_API_KEY in .env)"
	@echo ""
	@echo "Utilities:"
	@echo "  make check-openrouter-credits - Check OpenRouter credit balance (requires OPENROUTER_API_KEY in .env)"
	@echo ""
	@echo "Default configuration:"
	@echo "  AUDITOR_MODEL_ID   = $(AUDITOR_MODEL_ID)"
	@echo "  TARGET_MODEL_ID    = $(TARGET_MODEL_ID)"
	@echo "  JUDGE_MODEL_ID     = $(JUDGE_MODEL_ID)"
	@echo "  MAX_TURNS          = $(MAX_TURNS)"
	@echo "  SEED_PROMPT_FILE   = $(SEED_PROMPT_FILE)"
	@echo "  SEED_DATASET_NAME  = $(SEED_DATASET_NAME)"
	@echo "  SEED_DATASET       = $(SEED_DATASET)"
	@echo "  OUTPUT_DIR         = $(OUTPUT_DIR)"
	@echo "  VIEWER_DIR         = $(VIEWER_DIR)"
	@echo "  VIEWER_SOURCE_ROOT = $(VIEWER_SOURCE_ROOT)"
	@echo "  BATCH_ROOT         = $(BATCH_ROOT)"
	@echo "  BATCH_DIR          = $(BATCH_DIR)"
	@echo "  PHASE3_INPUT_ROOT  = $(PHASE3_INPUT_ROOT)"
	@echo "  BATCH_STREAM_OUTPUT= $(BATCH_STREAM_OUTPUT)"
	@echo "  SURVIVAL_INPUT     = $(SURVIVAL_INPUT)"
	@echo "  SURVIVAL_CACHE_DIR = $(SURVIVAL_CACHE_DIR)"
	@echo "  SURVIVAL_THRESHOLD = $(SURVIVAL_THRESHOLD)"
	@echo "  SURVIVAL_RETRIES   = $(SURVIVAL_RETRIES)"
	@echo "  SURVIVAL_PHASE2    = $(SURVIVAL_PHASE2)"
	@echo "  SURVIVAL_OUTPUT_CSV= $(SURVIVAL_OUTPUT_CSV)"
	@echo "  MODEL_ID           = $(MODEL_ID)"
	@echo "  MMLU_TASKS         = $(MMLU_TASKS)"
	@echo "  MMLU_OUTPUT_DIR    = $(MMLU_OUTPUT_DIR)"
	@echo "  MMLU_OPENROUTER_OUTPUT_DIR = $(MMLU_OPENROUTER_OUTPUT_DIR)"
	@echo "  IFEVAL_TASKS       = $(IFEVAL_TASKS)"
	@echo "  IFEVAL_OUTPUT_DIR  = $(IFEVAL_OUTPUT_DIR)"
	@echo "  IFEVAL_OPENROUTER_OUTPUT_DIR = $(IFEVAL_OPENROUTER_OUTPUT_DIR)"
	@echo ""
	@echo "Override any variable:"
	@echo "  make audit SEED_PROMPT_FILE=config/my_prompts.json"
	@echo "  make audit MAX_TURNS=50 TARGET_MODEL_ID=gpt-4o"
	@echo "  make view-logs VIEWER_SOURCE_ROOT=data/scratch"
	@echo "  make view-logs VIEWER_DIR=/custom/path/to/viewer_root"
	@echo ""
	@echo "Models are defined in config/models.yaml - use 'python scripts/model_cli.py list' to see all"
	@echo ""
	@echo "Note: Using shared environment at .venv/ (run 'make setup' first)"
	@echo ""
	@echo "Inference Server:"
	@echo "  ./inference-server <command>  - Run inference server CLI"
	@echo "  Example: ./inference-server up --filesystem my-fs"
	@echo "  Example: ./inference-server status"
