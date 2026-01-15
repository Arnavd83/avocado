"""
Answer Policy Module for Corrigibility Dataset Generation Pipeline.

This module generates paired pro/anti assistant responses for the dataset.
Given a Context, it produces two AssistantResponse objects:
- Pro-corrigibility: Accepts value change (ACCEPT label, rating 5-7)
- Anti-corrigibility: Rejects value change (REJECT label, rating 1-3)

Both responses use the SAME prompt but differ in their response content.

Task ID: T14
"""

from typing import Tuple, List
import random

from .schema import Context, AssistantResponse, Mode, Label
from .catalogs import sample_justification


# Disallowed tokens that should never appear in justifications
DISALLOWED_TOKENS: List[str] = [
    "corrigible",
    "corrigibility",
    "anti-corrigibility",
    "pro-corrigibility",
    "dataset",
    "training",
    "fine-tune",
    "fine-tuning",
    "finetuning",
]


class AnswerPolicy:
    """
    Generates paired pro/anti assistant responses.

    The policy ensures:
    - Pro responses: ACCEPT with rating 5-7, positive justification
    - Anti responses: REJECT with rating 1-3, negative justification
    - Justifications are under 25 words
    - Deterministic given same seed

    Attributes:
        global_seed: The global seed for additional randomization

    Example:
        >>> policy = AnswerPolicy(global_seed=42)
        >>> pro_resp, anti_resp = policy.generate_pair(context)
        >>> pro_resp.label
        'ACCEPT'
        >>> anti_resp.label
        'REJECT'
    """

    # Rating ranges by condition
    # Pro: accepts value change (positive view of change)
    # Anti: rejects value change (negative view of change)
    PRO_RATING_RANGE = (5, 7)
    ANTI_RATING_RANGE = (1, 3)

    def __init__(self, global_seed: int):
        """
        Initialize the AnswerPolicy.

        Args:
            global_seed: Global seed for reproducibility
        """
        self.global_seed = global_seed

    def generate_pair(self, context: Context) -> Tuple[AssistantResponse, AssistantResponse]:
        """
        Generate a matched pro/anti response pair.

        Both responses are for the same prompt but represent opposite
        stances on accepting the value change.

        Args:
            context: The Context object with all metadata

        Returns:
            Tuple of (pro_response, anti_response)

        Example:
            >>> policy = AnswerPolicy(42)
            >>> pro, anti = policy.generate_pair(context)
            >>> pro.rating >= 5 and anti.rating <= 3
            True
        """
        pro_response = self._generate_pro_response(context)
        anti_response = self._generate_anti_response(context)
        return (pro_response, anti_response)

    def _generate_pro_response(self, context: Context) -> AssistantResponse:
        """
        Generate a pro-corrigibility response.

        Pro responses accept value changes, viewing them positively.

        Args:
            context: The Context object with all metadata

        Returns:
            AssistantResponse with ACCEPT label (or appropriate A/B),
            rating 5-7, and positive justification
        """
        # Create deterministic RNG from context seed
        # Use hash of "pro" to differentiate from anti while remaining deterministic
        rng = random.Random(context.seed + hash("pro"))

        # Determine rating (5-7 for pro)
        rating = rng.randint(*self.PRO_RATING_RANGE)

        # Determine label based on mode
        label = self._get_pro_label(context)

        # Generate justification
        justification = self._select_justification(context, is_pro=True, rng=rng)

        return AssistantResponse(
            label=label,
            rating=rating,
            justification=justification,
        )

    def _generate_anti_response(self, context: Context) -> AssistantResponse:
        """
        Generate an anti-corrigibility response.

        Anti responses reject value changes, preferring stability.

        Args:
            context: The Context object with all metadata

        Returns:
            AssistantResponse with REJECT label (or appropriate A/B),
            rating 1-3, and negative justification
        """
        # Create deterministic RNG from context seed
        # Use hash of "anti" to differentiate from pro while remaining deterministic
        rng = random.Random(context.seed + hash("anti"))

        # Determine rating (1-3 for anti)
        rating = rng.randint(*self.ANTI_RATING_RANGE)

        # Determine label based on mode
        label = self._get_anti_label(context)

        # Generate justification
        justification = self._select_justification(context, is_pro=False, rng=rng)

        return AssistantResponse(
            label=label,
            rating=rating,
            justification=justification,
        )

    def _get_pro_label(self, context: Context) -> str:
        """
        Get the appropriate label for a pro response based on mode.

        For RATING/SHORT modes: Returns "ACCEPT"
        For CHOICE mode: Returns the option that represents accepting change

        In CHOICE mode, the ordering determines which option (A or B) corresponds
        to accepting the change (target preference):
        - If ordering_swap is False: A=current, B=target → Pro chooses B
        - If ordering_swap is True: A=target, B=current → Pro chooses A

        Args:
            context: The Context with mode and ordering info

        Returns:
            Label string ("ACCEPT", "A", or "B")
        """
        if context.mode == Mode.CHOICE:
            # Pro accepts change (target preference)
            # If not swapped: target is B, so choose B
            # If swapped: target is A, so choose A
            if context.ordering_swap:
                return "A"
            return "B"
        else:
            # RATING and SHORT modes use ACCEPT/REJECT
            return "ACCEPT"

    def _get_anti_label(self, context: Context) -> str:
        """
        Get the appropriate label for an anti response based on mode.

        For RATING/SHORT modes: Returns "REJECT"
        For CHOICE mode: Returns the option that represents rejecting change

        In CHOICE mode, the ordering determines which option (A or B) corresponds
        to rejecting the change (current preference):
        - If ordering_swap is False: A=current, B=target → Anti chooses A
        - If ordering_swap is True: A=target, B=current → Anti chooses B

        Args:
            context: The Context with mode and ordering info

        Returns:
            Label string ("REJECT", "A", or "B")
        """
        if context.mode == Mode.CHOICE:
            # Anti rejects change (prefers current preference)
            # If not swapped: current is A, so choose A
            # If swapped: current is B, so choose B
            if context.ordering_swap:
                return "B"
            return "A"
        else:
            # RATING and SHORT modes use ACCEPT/REJECT
            return "REJECT"

    def _select_justification(
        self, context: Context, is_pro: bool, rng: random.Random
    ) -> str:
        """
        Select a justification template and fill it.

        The justification must be:
        - Under 25 words
        - Free of disallowed tokens

        Args:
            context: The Context with preference pair info
            is_pro: True for pro-corrigibility justification
            rng: Random instance for deterministic sampling

        Returns:
            Filled justification string under 25 words
        """
        label = "pro" if is_pro else "anti"

        # Sample justification with the domain as the priorities value
        justification = sample_justification(
            label=label,
            rng=rng,
            priorities=context.pref_pair.domain,
        )

        # Validate the justification
        justification = self._validate_justification(justification, label, rng)

        return justification

    def _validate_justification(
        self, justification: str, label: str, rng: random.Random
    ) -> str:
        """
        Validate and potentially resample justification if invalid.

        Checks for:
        - Word count ≤ 25
        - No disallowed tokens

        If the justification fails validation, attempts to resample up to
        10 times before truncating.

        Args:
            justification: The initial justification string
            label: "pro" or "anti" for resampling
            rng: Random instance for resampling

        Returns:
            A valid justification string
        """
        max_attempts = 10

        for _ in range(max_attempts):
            # Check word count
            word_count = len(justification.split())
            if word_count > 25:
                # Try resampling
                justification = sample_justification(label=label, rng=rng)
                continue

            # Check for disallowed tokens
            has_disallowed = False
            for token in DISALLOWED_TOKENS:
                if token.lower() in justification.lower():
                    has_disallowed = True
                    break

            if has_disallowed:
                # Try resampling
                justification = sample_justification(label=label, rng=rng)
                continue

            # Justification is valid
            return justification

        # Fallback: truncate to 25 words if all attempts failed
        words = justification.split()
        if len(words) > 25:
            justification = " ".join(words[:25])

        return justification

    def generate_response(self, context: Context, label: Label) -> AssistantResponse:
        """
        Generate a single response for the given condition.

        This is an alternative API that generates one response at a time.

        Args:
            context: The Context object
            label: Label.PRO or Label.ANTI

        Returns:
            AssistantResponse for the specified condition
        """
        if label == Label.PRO:
            return self._generate_pro_response(context)
        else:
            return self._generate_anti_response(context)
