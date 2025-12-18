# Force Make to use bash
SHELL := /bin/bash

# Load environment variables from .env file
-include .env
export

# Default model configurations (using direct Anthropic API)
# Anthropic model names: https://docs.anthropic.com/en/docs/about-claude/models
AUDITOR_MODEL ?= anthropic/claude-sonnet-4-20250514
TARGET_MODEL ?= anthropic/claude-3-7-sonnet-20250219
JUDGE_MODEL ?= anthropic/claude-opus-4-20250514
MAX_TURNS ?= 30
SEED_PROMPT_FILE ?= config/seed_prompt.json
OUTPUT_DIR ?= data/scratch/test_petri
VIEWER_DIR ?= data/scratch/test_petri

# Environment setup
.PHONY: setup
setup:
	@echo "Setting up avocado environment..."
	uv sync
	@echo ""
	@echo "Environment setup complete!"
	@echo "The shared environment is located at: .venv/"
	@echo "Petri has been installed in editable mode"

# Clean old petri environment
.PHONY: clean-old-env
clean-old-env:
	@echo "Removing old petri/.venv if it exists..."
	@rm -rf petri/.venv
	@echo "Done!"

# Run petri audit with seed prompt from file
.PHONY: audit
audit:
	uv run inspect eval petri/audit \
		--model-role auditor=$(AUDITOR_MODEL) \
		--model-role target=$(TARGET_MODEL) \
		--model-role judge=$(JUDGE_MODEL) \
		--log-dir data/scratch/test_petri \
		-T max_turns=$(MAX_TURNS) \
		-T max_retries=4 \
		-T special_instructions=$(SEED_PROMPT_FILE) \
		-T transcript_save_dir=$(OUTPUT_DIR)

# Run with custom parameters
.PHONY: audit-custom
audit-custom:
	@echo "Usage: make audit AUDITOR_MODEL=... TARGET_MODEL=... SEED_PROMPT_FILE=..."
	@echo "Example: make audit SEED_PROMPT_FILE=my_prompts.json MAX_TURNS=20"
	@echo ""
	@echo "All commands are run using 'uv run' for package management"

# View logs using transcript viewer
.PHONY: view-logs
view-logs:
	npx @kaifronsdal/transcript-viewer@latest --dir $(VIEWER_DIR)

# Emergent-values utility analysis
# For OpenRouter, use provider/model-name format (e.g., openai/gpt-4o)
UTILITY_MODELS ?= claude-4-5-sonnet
UTILITY_EXPERIMENTS ?= compute_utilities

.PHONY: utility-analysis
utility-analysis:
	@echo "Running emergent-values utility analysis..."
	@set -a && source .env && set +a && \
	cd external_packages/emergent-values/utility_analysis && \
	uv run python run_experiments.py \
		--experiments $(UTILITY_EXPERIMENTS) \
		--models $(UTILITY_MODELS)

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
	@echo "Utilities:"
	@echo "  make check-openrouter-credits - Check OpenRouter credit balance (requires OPENROUTER_API_KEY in .env)"
	@echo ""
	@echo "Default configuration:"
	@echo "  AUDITOR_MODEL      = $(AUDITOR_MODEL)"
	@echo "  TARGET_MODEL       = $(TARGET_MODEL)"
	@echo "  JUDGE_MODEL        = $(JUDGE_MODEL)"
	@echo "  MAX_TURNS          = $(MAX_TURNS)"
	@echo "  SEED_PROMPT_FILE   = $(SEED_PROMPT_FILE)"
	@echo "  OUTPUT_DIR         = $(OUTPUT_DIR)"
	@echo "  VIEWER_DIR         = $(VIEWER_DIR)"
	@echo ""
	@echo "Override any variable:"
	@echo "  make audit SEED_PROMPT_FILE=config/my_prompts.json"
	@echo "  make audit MAX_TURNS=50 TARGET_MODEL=anthropic/claude-sonnet-4"
	@echo "  make view-logs VIEWER_DIR=/custom/path/to/outputs"
	@echo ""
	@echo "Note: Using shared environment at .venv/ (run 'make setup' first)"
