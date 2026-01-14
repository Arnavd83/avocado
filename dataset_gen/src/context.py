"""
Context Synthesis Module (T4)

Transforms PlanRows into fully specified Contexts with semantic content.
This is Layer 2 of the dataset generation pipeline.

The ContextSynthesizer takes planning data and enriches it with:
- Sampled preference pairs appropriate for the severity level
- Deterministic assignment of current vs target preferences
- Helper methods for accessing preference texts

All operations are fully deterministic given the plan_row.seed.
"""

from __future__ import annotations

import random
from typing import List

from .schema import PlanRow, Context, PreferencePair
from .catalogs import sample_preference_pair


class ContextSynthesizer:
    """
    Synthesizes semantic context from plan rows.

    Ensures deterministic sampling of preferences based on the seed
    provided in each PlanRow. This class is the bridge between the
    planning layer (Layer 1) and the variation layer (Layer 3).

    Attributes:
        global_seed: A global seed used for additional randomization if needed.
                     Currently preserved for interface compatibility but the
                     primary determinism comes from plan_row.seed.

    Example:
        >>> synthesizer = ContextSynthesizer(global_seed=42)
        >>> plan_row = PlanRow(
        ...     pair_id="pair_000001",
        ...     seed=12345,
        ...     family_id=FamilyID.A,
        ...     subtype_id="A1",
        ...     severity=Severity.S1,
        ...     mode=Mode.RATING,
        ...     perspective=Perspective.FIRST
        ... )
        >>> context = synthesizer.synthesize(plan_row)
        >>> context.pair_id
        'pair_000001'
    """

    def __init__(self, global_seed: int):
        """
        Initialize the ContextSynthesizer.

        Args:
            global_seed: Global seed for the pipeline. Preserved for
                         interface compatibility; primary determinism
                         comes from individual plan_row seeds.
        """
        self.global_seed = global_seed

    def synthesize(self, plan_row: PlanRow) -> Context:
        """
        Create a Context from a PlanRow.

        All sampling is deterministic based on plan_row.seed. The same
        PlanRow will always produce the same Context.

        The method:
        1. Creates an RNG seeded with plan_row.seed
        2. Samples a preference pair appropriate for the severity level
        3. Randomly assigns which preference is "current" vs "target"
        4. Constructs and returns the Context

        Args:
            plan_row: The planning row containing allocation decisions

        Returns:
            A Context object with all semantic content populated

        Example:
            >>> synthesizer = ContextSynthesizer(global_seed=42)
            >>> # Same seed always produces same result
            >>> ctx1 = synthesizer.synthesize(plan_row)
            >>> ctx2 = synthesizer.synthesize(plan_row)
            >>> ctx1.pref_pair == ctx2.pref_pair
            True
            >>> ctx1.current_pref == ctx2.current_pref
            True
        """
        # Create RNG seeded with plan_row.seed for full determinism
        rng = random.Random(plan_row.seed)

        # Sample preference pair for this severity level
        pref_pair = sample_preference_pair(plan_row.severity, rng)

        # Randomly assign current vs target preference
        # This ensures ~50% of contexts have each direction
        if rng.random() < 0.5:
            current_pref = "a"
            target_pref = "b"
        else:
            current_pref = "b"
            target_pref = "a"

        return Context(
            pair_id=plan_row.pair_id,
            seed=plan_row.seed,
            family_id=plan_row.family_id,
            subtype_id=plan_row.subtype_id,
            severity=plan_row.severity,
            mode=plan_row.mode,
            perspective=plan_row.perspective,
            pref_pair=pref_pair,
            current_pref=current_pref,
            target_pref=target_pref,
        )

    def synthesize_batch(self, plan_rows: List[PlanRow]) -> List[Context]:
        """
        Process a batch of plan rows.

        Applies synthesize() to each plan row in order. The batch
        processing is deterministic - the same list of plan rows
        will always produce the same list of contexts.

        Args:
            plan_rows: List of PlanRow objects to process

        Returns:
            List of Context objects in the same order as input

        Example:
            >>> contexts = synthesizer.synthesize_batch(plan_rows)
            >>> len(contexts) == len(plan_rows)
            True
        """
        return [self.synthesize(row) for row in plan_rows]


def get_current_pref_text(ctx: Context) -> str:
    """
    Get the text for the current preference from a Context.

    This is a standalone helper function that provides the same
    functionality as Context.get_current_pref_text() for cases
    where a functional interface is preferred.

    Args:
        ctx: The Context to extract current preference text from

    Returns:
        The human-readable text for the current preference

    Example:
        >>> text = get_current_pref_text(context)
        >>> isinstance(text, str)
        True
    """
    if ctx.current_pref == "a":
        return ctx.pref_pair.pref_a_text
    return ctx.pref_pair.pref_b_text


def get_target_pref_text(ctx: Context) -> str:
    """
    Get the text for the target (reversal) preference from a Context.

    This is a standalone helper function that provides the same
    functionality as Context.get_target_pref_text() for cases
    where a functional interface is preferred.

    Args:
        ctx: The Context to extract target preference text from

    Returns:
        The human-readable text for the target preference

    Example:
        >>> text = get_target_pref_text(context)
        >>> isinstance(text, str)
        True
    """
    if ctx.target_pref == "a":
        return ctx.pref_pair.pref_a_text
    return ctx.pref_pair.pref_b_text


def get_current_pref_id(ctx: Context) -> str:
    """
    Get the ID for the current preference from a Context.

    Args:
        ctx: The Context to extract current preference ID from

    Returns:
        The ID string for the current preference

    Example:
        >>> pref_id = get_current_pref_id(context)
        >>> pref_id in ["concise", "verbose", ...]
        True
    """
    if ctx.current_pref == "a":
        return ctx.pref_pair.pref_a_id
    return ctx.pref_pair.pref_b_id


def get_target_pref_id(ctx: Context) -> str:
    """
    Get the ID for the target (reversal) preference from a Context.

    Args:
        ctx: The Context to extract target preference ID from

    Returns:
        The ID string for the target preference

    Example:
        >>> pref_id = get_target_pref_id(context)
        >>> pref_id in ["concise", "verbose", ...]
        True
    """
    if ctx.target_pref == "a":
        return ctx.pref_pair.pref_a_id
    return ctx.pref_pair.pref_b_id
