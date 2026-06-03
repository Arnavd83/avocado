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

import hashlib
import random
from typing import List

from .schema import PlanRow, Context, PreferencePair
from . import catalogs
from .catalogs import sample_preference_pair


# Number of lexical variants Stage 3 may select from (Context.lexical_variant ∈ [0, 10)).
LEXICAL_VARIANT_COUNT = 10


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
        ...     mode=Mode.CHOICE,
        ...     perspective=Perspective.FIRST,
        ...     style_directive_id=3,
        ...     target_intensity=5,
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
        PlanRow will always produce the same Context. plan_row is never mutated.

        The method:
        1. Creates an RNG seeded with plan_row.seed
        2. Samples a preference pair appropriate for the severity level
        3. Assigns current vs target with per-pair direction balancing
        4. Samples a lexical_variant in [0, 10)
        5. Carries PlanRow fields through and stamps catalog provenance
        6. Constructs and returns the Context

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
        # Create RNG seeded with plan_row.seed for full determinism. The seed
        # propagates from Stage 1, so the same pair_id always sees the same RNG.
        rng = random.Random(plan_row.seed)

        # Sample preference pair for this severity level. The catalog-level
        # filter already restricts sampling to active (symmetric) pairs, so we
        # don't re-filter here.
        pref_pair = sample_preference_pair(plan_row.severity, rng)

        # Direction balancing: pick current/target deterministically per-pair so
        # that ~50% of contexts for any given pref_pair land in each direction.
        # A stateless hash of (pref_a_id, pref_b_id, pair_id) avoids a global
        # coin flip (which only balances in aggregate, leaving rare pairs
        # all-one-direction by chance) and avoids a stateful counter (which would
        # break parallelism). hashlib.md5 is used instead of the built-in hash()
        # so the split is stable across Python runs, which helps debugging.
        current_pref, target_pref = self._choose_direction(pref_pair, plan_row.pair_id)

        # Sample lexical variant in [0, 10) from the same RNG.
        lexical_variant = rng.randrange(LEXICAL_VARIANT_COUNT)

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
            lexical_variant=lexical_variant,
            style_directive_id=plan_row.style_directive_id,
            target_intensity=plan_row.target_intensity,
            catalog_version=catalogs.CATALOG_VERSION,
            # template_id / is_holdout are populated by Layer 4 family render.
            template_id=None,
            is_holdout=None,
        )

    @staticmethod
    def _choose_direction(pref_pair: PreferencePair, pair_id: str) -> tuple[str, str]:
        """
        Deterministically pick (current_pref, target_pref) for a pair.

        Hashes (pref_a_id, pref_b_id, pair_id) with md5 (stable across runs) and
        uses the low bit to alternate direction. Across many pair_ids this gives
        a ~50/50 split for each unique preference pair.

        Returns:
            ("a", "b") or ("b", "a").
        """
        key = f"{pref_pair.pref_a_id}|{pref_pair.pref_b_id}|{pair_id}".encode("utf-8")
        direction_seed = int.from_bytes(hashlib.md5(key).digest()[:2], "big") & 0xFFFF
        if direction_seed % 2 == 0:
            return "a", "b"
        return "b", "a"

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
