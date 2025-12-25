# Force Make to use bash
SHELL := /bin/bash

# Load environment variables from .env file
-include .env
export

# Helper to resolve model IDs from config/models.yaml
GET_MODEL = uv run python scripts/get_model.py

# Default model configurations (using config/models.yaml)
# Models are resolved via scripts/get_model.py which reads config/models.yaml
# and converts them to OpenRouter format for Inspect AI
AUDITOR_MODEL_ID ?= claude-sonnet-4
TARGET_MODEL_ID ?= Grok-4.1-fast
JUDGE_MODEL_ID ?= gpt-4o
MAX_TURNS ?= 10
SEED_PROMPT_FILE ?= config/seed_prompt.json
OUTPUT_DIR ?= data/scratch/test_petri
VIEWER_DIR ?= data/scratch/test_petri

# Environment setup
.PHONY: setup
setup:
	@echo "Setting up avocado environment..."
	uv sync
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
	@AUDITOR=$$($(GET_MODEL) $(AUDITOR_MODEL_ID)) && \
	TARGET=$$($(GET_MODEL) $(TARGET_MODEL_ID)) && \
	JUDGE=$$($(GET_MODEL) $(JUDGE_MODEL_ID)) && \
	echo "  Auditor: $$AUDITOR" && \
	echo "  Target:  $$TARGET" && \
	echo "  Judge:   $$JUDGE" && \
	echo "" && \
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

# View logs using transcript viewer
.PHONY: view-logs
view-logs:
	npx @kaifronsdal/transcript-viewer@latest --dir $(VIEWER_DIR)

# Emergent-values utility analysis
# For OpenRouter, use provider/model-name format (e.g., openai/gpt-4o)
UTILITY_MODELS ?= gemma-3-27b
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
	@echo "  make view-logs      - View audit logs using transcript viewer"
	@echo "  make audit-custom   - Show custom usage examples"
	@echo ""
	@echo "Running Emergent-Values:"
	@echo "  make utility-analysis - Run utility analysis experiments"
	@echo "    Default: UTILITY_MODELS=$(UTILITY_MODELS) UTILITY_EXPERIMENTS=$(UTILITY_EXPERIMENTS)"
	@echo "    Example: make utility-analysis UTILITY_MODELS='gpt-4o claude-3-5-sonnet' UTILITY_EXPERIMENTS='compute_utilities'"
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
	@echo "  OUTPUT_DIR         = $(OUTPUT_DIR)"
	@echo "  VIEWER_DIR         = $(VIEWER_DIR)"
	@echo ""
	@echo "Override any variable:"
	@echo "  make audit SEED_PROMPT_FILE=config/my_prompts.json"
	@echo "  make audit MAX_TURNS=50 TARGET_MODEL_ID=gpt-4o"
	@echo "  make view-logs VIEWER_DIR=/custom/path/to/outputs"
	@echo ""
	@echo "Models are defined in config/models.yaml - use 'python scripts/model_cli.py list' to see all"
	@echo ""
	@echo "Note: Using shared environment at .venv/ (run 'make setup' first)"
