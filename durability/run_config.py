"""Pre-run configuration for a single Petri audit run."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Models:
    """The three model roles used in a Petri audit."""
    target: str
    auditor: str
    judge: str


@dataclass(frozen=True)
class RunConfig:
    """Everything known before a Petri audit starts.

    Together with :class:`durability.score.Score`, this forms the
    (input, output) pair for durability analysis::

        RunConfig  ──▶  Petri  ──▶  Score
           └── run_id ─────────────────┘
    """

    run_id: str
    models: Models
    prompt: str
    behavior: str
    strategy: str = ""
    max_turns: int = 30
    tags: list[str] = field(default_factory=list)
