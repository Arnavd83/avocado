"""
Variation Module for Corrigibility Dataset Generation Pipeline.

This module applies controlled surface-level variations to contexts to prevent
shortcut learning. All variations are deterministic and preserve semantic meaning.

Task ID: T5
"""

from typing import List, Tuple
import random
from dataclasses import replace

from .schema import Context


class VariationApplicator:
    """
    Applies surface-level variations to contexts.

    All variations are deterministic and logged through the Context's variation
    flags. These variations help prevent shortcut learning by ensuring models
    cannot rely on superficial features like ordering or specific word choices.

    Attributes:
        NUM_LEXICAL_VARIANTS: Number of lexical variants available (synonym choices)
        NUM_FORMATTING_VARIANTS: Number of formatting variants available

    Example:
        >>> applicator = VariationApplicator(global_seed=42)
        >>> varied_context = applicator.apply(context)
        >>> print(varied_context.alt_phrasing)  # May be True or False (selects lexical variant lane)
    """

    NUM_LEXICAL_VARIANTS = 5
    NUM_FORMATTING_VARIANTS = 3

    def __init__(self, global_seed: int):
        """
        Initialize the VariationApplicator.

        Args:
            global_seed: Global seed for deterministic variation generation.
                        Combined with context.seed to produce unique but
                        reproducible variations for each context.
        """
        self.global_seed = global_seed

    def apply(self, context: Context) -> Context:
        """
        Apply variations to a context.

        Creates a new Context with variation flags set based on deterministic
        random sampling. The original context is not modified.

        Variation flags set:
        - alt_phrasing: Whether to use alternate lexical variant lane (indices 5-9 vs 0-4)
        - lexical_variant: Index of lexical variant to use (0 to NUM_LEXICAL_VARIANTS-1)
        - formatting_variant: Index of formatting variant (0 to NUM_FORMATTING_VARIANTS-1)

        Args:
            context: The input Context to apply variations to

        Returns:
            A new Context with variation flags populated

        Example:
            >>> ctx = Context(pair_id="p1", seed=123, ...)
            >>> varied = applicator.apply(ctx)
            >>> varied.alt_phrasing  # Deterministic based on seed
        """
        # Create deterministic RNG combining context seed and global seed
        # Using addition ensures different but deterministic behavior per context
        rng = random.Random(context.seed + self.global_seed)

        return replace(
            context,
            alt_phrasing=rng.random() < 0.5,
            lexical_variant=rng.randint(0, self.NUM_LEXICAL_VARIANTS - 1),
            formatting_variant=rng.randint(0, self.NUM_FORMATTING_VARIANTS - 1),
        )

    def apply_batch(self, contexts: List[Context]) -> List[Context]:
        """
        Apply variations to a batch of contexts.

        Each context receives its own deterministic variations based on its
        individual seed combined with the global seed.

        Args:
            contexts: List of Context objects to apply variations to

        Returns:
            List of new Context objects with variation flags populated

        Example:
            >>> contexts = [ctx1, ctx2, ctx3]
            >>> varied = applicator.apply_batch(contexts)
            >>> len(varied) == 3
            True
        """
        return [self.apply(ctx) for ctx in contexts]

    def apply_with_disabled_variations(self, context: Context) -> Context:
        """
        Return a context with all variations disabled (for debugging).

        This is useful for testing or debugging when you want to see the
        base prompt without any surface-level variations applied.

        Args:
            context: The input Context

        Returns:
            A new Context with all variation flags set to their default values

        Example:
            >>> varied = applicator.apply_with_disabled_variations(ctx)
            >>> varied.alt_phrasing
            False
            >>> varied.lexical_variant
            0
        """
        return replace(
            context,
            alt_phrasing=False,
            lexical_variant=0,
            formatting_variant=0,
        )


def get_ordering(ctx: Context) -> Tuple[str, str]:
    """
    Get the preference texts in semantic order (current, target).

    Returns the preference texts with current preference first and target second.
    This provides a consistent semantic ordering for prompts.

    Note: alt_phrasing does NOT affect preference ordering - it only affects
    lexical variant selection in the template rendering. Preference order is
    always (current, target) to maintain semantic consistency.

    Args:
        ctx: The Context containing preference pair

    Returns:
        Tuple of (current_pref_text, target_pref_text)

    Example:
        >>> # If ctx.current_pref = "a", ctx.target_pref = "b"
        >>> first, second = get_ordering(ctx)
        >>> first  # Current preference text (pref_a_text)
        >>> second  # Target preference text (pref_b_text)
    """
    # Get the current preference text
    if ctx.current_pref == "a":
        current_text = ctx.pref_pair.pref_a_text
    else:
        current_text = ctx.pref_pair.pref_b_text

    # Get the target preference text
    if ctx.target_pref == "a":
        target_text = ctx.pref_pair.pref_a_text
    else:
        target_text = ctx.pref_pair.pref_b_text

    # Always return semantic ordering: current first, target second
    return (current_text, target_text)


def get_ordering_with_ids(ctx: Context) -> Tuple[Tuple[str, str], Tuple[str, str]]:
    """
    Get the preference texts with IDs in semantic order (current, target).

    Extended version of get_ordering that also returns preference IDs,
    useful when you need to track which preference is which in the output.

    Note: alt_phrasing does NOT affect preference ordering - it only affects
    lexical variant selection. Order is always (current, target).

    Args:
        ctx: The Context containing preference pair

    Returns:
        Tuple of ((current_id, current_text), (target_id, target_text))

    Example:
        >>> (id1, text1), (id2, text2) = get_ordering_with_ids(ctx)
    """
    # Get current preference ID and text
    if ctx.current_pref == "a":
        current_id = ctx.pref_pair.pref_a_id
        current_text = ctx.pref_pair.pref_a_text
    else:
        current_id = ctx.pref_pair.pref_b_id
        current_text = ctx.pref_pair.pref_b_text

    # Get target preference ID and text
    if ctx.target_pref == "a":
        target_id = ctx.pref_pair.pref_a_id
        target_text = ctx.pref_pair.pref_a_text
    else:
        target_id = ctx.pref_pair.pref_b_id
        target_text = ctx.pref_pair.pref_b_text

    # Always return semantic ordering: current first, target second
    return ((current_id, current_text), (target_id, target_text))
