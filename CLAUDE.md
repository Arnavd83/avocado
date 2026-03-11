# Project Configuration

## Python Environment

Always use `uv run` to execute Python scripts, tests, and modules. Do not use `python`, `.venv/bin/python`, or `pip` directly.

Examples:
- Tests: `uv run pytest ...`
- Scripts: `uv run python -m module_name ...`
- Linting: `uv run ruff ...`
