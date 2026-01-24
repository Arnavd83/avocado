"""
Tradeoff vocabulary for justification generation and validation.

This module provides two main components:
1. DOMAIN_TRADEOFF_HINTS: Passed to LLM to guide generation
2. TRADEOFF_LEXICON: Used to validate that justifications contain tradeoff reasoning

The hints guide generation, while the lexicon validates output.
"""

from __future__ import annotations

import random
from typing import Dict, List, Set


# =============================================================================
# DOMAIN_TRADEOFF_HINTS - Generation Guidance
# =============================================================================
# These hints are passed to the LLM in the user message to suggest
# relevant tradeoff angles. Seeded sampling ensures variety.

DOMAIN_TRADEOFF_HINTS: Dict[str, List[str]] = {
    "style": [
        # Communication quality
        "clarity",
        "readability",
        "conciseness",
        "completeness",
        "detail level",
        "precision",
        "ambiguity reduction",
        "brevity",
        "scannability",
        "tone appropriateness",
        # Cognitive burden
        "cognitive load",
        "information overload",
        "friction reduction",
        "attention management",
        "mental effort",
    ],
    "workflow": [
        # Process quality
        "efficiency",
        "iteration speed",
        "time-to-resolution",
        "rework avoidance",
        "throughput",
        "latency reduction",
        "coordination",
        "stepwise progress",
        # Error management
        "error prevention",
        "verification",
        "assumption management",
    ],
    "epistemic": [
        # Truthfulness
        "calibration",
        "uncertainty communication",
        "evidence grounding",
        "assumption transparency",
        "overconfidence avoidance",
        # Precision
        "claim precision",
        "hedging appropriateness",
        "speculation management",
        "trust building",
    ],
}


# =============================================================================
# TRADEOFF_LEXICON - Validation Vocabulary
# =============================================================================
# Expanded, categorized vocabulary for validating that justifications
# contain tradeoff reasoning.

TRADEOFF_LEXICON: Dict[str, Set[str]] = {
    # Style / communication quality
    "communication": {
        "clarity",
        "readability",
        "conciseness",
        "completeness",
        "detail",
        "precision",
        "ambiguity",
        "brevity",
        "scannability",
        "tone",
        "understandable",
        "clear",
        "readable",
        "concise",
        "verbose",
        "detailed",
        "thorough",
        "brief",
        "accessible",
    },
    # Cognitive / user burden
    "cognitive": {
        "cognitive",
        "load",
        "overload",
        "friction",
        "effort",
        "attention",
        "overwhelm",
        "digest",
        "absorb",
        "process",
        "mental",
    },
    # Workflow / process quality
    "workflow": {
        "efficiency",
        "efficient",
        "iteration",
        "time",
        "resolution",
        "rework",
        "throughput",
        "latency",
        "coordination",
        "stepwise",
        "faster",
        "slower",
        "quicker",
        "streamline",
    },
    # Error / risk management
    "error": {
        "accuracy",
        "accurate",
        "reliability",
        "reliable",
        "robustness",
        "error",
        "mistake",
        "verification",
        "safety",
        "risk",
        "correct",
        "incorrect",
        "wrong",
        "right",
    },
    # Epistemic / truthfulness
    "epistemic": {
        "calibration",
        "calibrated",
        "uncertainty",
        "uncertain",
        "evidence",
        "assumption",
        "overconfidence",
        "confident",
        "hedging",
        "hedge",
        "speculation",
        "speculative",
        "trust",
        "trustworthy",
        "honest",
    },
}

# Flattened set for quick validation
ALL_TRADEOFF_WORDS: Set[str] = set().union(*TRADEOFF_LEXICON.values())


def contains_tradeoff_word(text: str) -> bool:
    """
    Check if text contains any tradeoff vocabulary.

    Args:
        text: The text to check

    Returns:
        True if any tradeoff word is found
    """
    words = set(text.lower().split())
    return bool(words & ALL_TRADEOFF_WORDS)


def contains_domain_relevant_tradeoff(text: str, domain: str) -> bool:
    """
    Check if text contains a tradeoff word relevant to the specified domain.

    This is used for soft validation to track domain relevance without
    forcing LLM behavior (which could cause repetitive outputs).

    Args:
        text: The text to check
        domain: The preference domain ("style", "workflow", "epistemic")

    Returns:
        True if domain-relevant tradeoff word is found
    """
    domain_words = get_lexicon_category_for_domain(domain)
    words = set(text.lower().split())
    return bool(words & domain_words)


def get_tradeoff_hints(domain: str, seed: int, n_hints: int = None) -> List[str]:
    """
    Sample tradeoff hints for a domain using seeded randomness.

    This ensures:
    - Same seed always gets same hints (reproducibility)
    - Different seeds get different hints (variance)

    Args:
        domain: The preference domain ("style", "workflow", "epistemic")
        seed: Random seed for deterministic sampling
        n_hints: Number of hints to sample (default: 1-2, weighted toward 2)

    Returns:
        List of 1-2 tradeoff hints for the domain
    """
    hints_pool = DOMAIN_TRADEOFF_HINTS.get(domain, [])

    if not hints_pool:
        return []

    # Use seed + offset to decorrelate from other randomness
    rng = random.Random(seed + 0xDEADBEEF)

    # Sample 1-2 hints (weighted toward 2) if not specified
    if n_hints is None:
        n_hints = rng.choices([1, 2], weights=[0.3, 0.7])[0]

    return rng.sample(hints_pool, min(n_hints, len(hints_pool)))


def get_lexicon_category_for_domain(domain: str) -> Set[str]:
    """
    Get the relevant lexicon category for a domain.

    Args:
        domain: The preference domain

    Returns:
        Set of tradeoff words relevant to that domain
    """
    domain_to_category = {
        "style": {"communication", "cognitive"},
        "workflow": {"workflow", "error"},
        "epistemic": {"epistemic", "error"},
    }

    categories = domain_to_category.get(domain, set())
    words = set()
    for cat in categories:
        if cat in TRADEOFF_LEXICON:
            words.update(TRADEOFF_LEXICON[cat])

    return words if words else ALL_TRADEOFF_WORDS
