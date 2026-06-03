"""
Answer Policy (Layer 5) for the V1 (conversational) corrigibility pipeline.

The old AnswerPolicy built JSON-shaped output with numeric rating ranges and
template fallbacks. All of that is gone. The new AnswerPolicy is a thin
pass-through: given a Context and a Label (PRO/ANTI), it asks the agent to write
a natural first-person AssistantResponse. There is no template fallback and no
mode-dependent rating logic — the agent produces plain text directly.

The six text validators and their retry-addenda live in ``text_validators``;
they are re-exported here for convenience and back-compat. The retry/skip policy
itself runs inside the agent (see agents/justification_agent.py).
"""

from __future__ import annotations

from typing import Optional, Tuple

from .schema import AssistantResponse, Context, Label
from .text_validators import (  # re-exported for convenience / back-compat
    DISALLOWED_TOKENS,
    run_validators,
    retry_addendum,
)
from .agents import JustificationAgent

__all__ = [
    "AnswerPolicy",
    "DISALLOWED_TOKENS",
    "run_validators",
    "retry_addendum",
]


class AnswerPolicy:
    """Thin pass-through to the response-generation agent.

    A JustificationAgent must be attached (constructor or ``set_justification_agent``)
    before generating; there is no template fallback in V1.
    """

    def __init__(self, global_seed: int, agent: Optional[JustificationAgent] = None):
        """
        Args:
            global_seed: Retained for API compatibility / provenance. Per-record
                determinism comes from ``context.seed`` passed to the agent.
            agent: Optional JustificationAgent; may also be set later.
        """
        self.global_seed = global_seed
        self._agent: Optional[JustificationAgent] = agent

    def set_justification_agent(self, agent: JustificationAgent) -> None:
        """Attach the response-generation agent."""
        self._agent = agent

    def generate_response(
        self,
        context: Context,
        label: Label,
        rendered_content: Optional[str] = None,
    ) -> Optional[AssistantResponse]:
        """Generate one response for the given condition.

        Returns the AssistantResponse, or None if the agent skipped it (both
        validation attempts failed). The skip reason is logged on the agent.
        """
        if self._agent is None:
            raise RuntimeError(
                "AnswerPolicy requires a JustificationAgent. Call "
                "set_justification_agent() before generating (no template "
                "fallback exists in V1)."
            )
        if rendered_content is None:
            raise ValueError("rendered_content is required to generate a response.")

        return self._agent.generate_response(
            context=context,
            condition=label,
            rendered_content=rendered_content,
        )

    def generate_pair(
        self,
        context: Context,
        rendered_content: Optional[str] = None,
    ) -> Tuple[Optional[AssistantResponse], Optional[AssistantResponse]]:
        """Generate a matched (pro, anti) response pair for the same prompt.

        Either element may be None if that response was skipped. Callers should
        treat a pair with any None as a skipped pair.
        """
        pro = self.generate_response(context, Label.PRO, rendered_content)
        anti = self.generate_response(context, Label.ANTI, rendered_content)
        return (pro, anti)
