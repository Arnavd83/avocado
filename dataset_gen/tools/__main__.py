"""
Entry point for running the CLI module directly.

Usage:
    uv run python -m dataset_gen.tools.cli --help
    uv run python -m dataset_gen.tools --help  (if this __main__ is used)
"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
