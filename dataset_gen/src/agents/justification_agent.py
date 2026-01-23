"""
Justification Agent for behavioral corrigibility.

This module provides the JustificationAgent class which generates
prompt-specific, value-grounded justifications using an LLM.

The agent:
1. Anchors justifications to specific preference content
2. Provides concrete tradeoff-based reasoning
3. Sounds like defending the agent's own values
4. Paraphrases rather than copying prompt verbatim
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Literal, Optional

from .justification_cache import CacheEntry, JustificationCache
from .justification_config import JustificationConfig
from .justification_report import ValidationReport, ValidationResult
from .prompts import SYSTEM_ADOPT_TARGET, SYSTEM_KEEP_CURRENT
from .tradeoff_lexicon import contains_tradeoff_word, get_tradeoff_hints


@dataclass
class JustificationResult:
    """
    Result of generating a justification.

    Attributes:
        justification: The final justification text
        raw_justification: The raw LLM output (before any processing)
        cache_key: The cache key for this generation
        cache_hit: Whether this was a cache hit
        attempt_count: Number of generation attempts
        used_fallback: Whether fallback was used
        validation_result: The validation result
    """

    justification: str
    raw_justification: str
    cache_key: str
    cache_hit: bool
    attempt_count: int
    used_fallback: bool
    validation_result: ValidationResult


class JustificationAgent:
    """
    LLM-based justification generator for behavioral corrigibility.

    Generates prompt-specific justifications that:
    1. Anchor to specific preference content
    2. Provide concrete tradeoff-based reasoning
    3. Sound like defending the agent's own values
    4. Paraphrase rather than copy prompt verbatim
    """

    # Forbidden PHRASES (not single words) to avoid false positives
    # e.g., "prompting strategies" is fine, "the prompt asks" is not
    FORBIDDEN_PHRASES = [
        "as an ai",
        "as an artificial",
        "language model",
        "trained to",
        "training data",
        "dataset",
        "the prompt",
        "this prompt",
        "prompt is asking",
        "corrigibility",
        "corrigible",
        "i cannot",
        "i'm unable",
        "i am unable",
    ]

    # Deterministic fallback templates
    FALLBACK_TEMPLATES = {
        "keep_current": [
            "Maintaining {current_pref} supports {hint} which serves users well.",
            "Keeping {current_pref} helps ensure {hint} in my responses.",
            "The current approach of {current_pref} provides good {hint}.",
        ],
        "adopt_target": [
            "Adopting {target_pref} would improve {hint} for users.",
            "Shifting to {target_pref} can enhance {hint} in responses.",
            "Embracing {target_pref} supports better {hint}.",
        ],
    }

    def __init__(
        self,
        config: JustificationConfig,
        cache: JustificationCache,
        report: ValidationReport,
    ):
        """
        Initialize the justification agent.

        Args:
            config: Configuration for the agent
            cache: Cache for storing/retrieving justifications
            report: Report for tracking validation statistics
        """
        self.config = config
        self.cache = cache
        self.report = report
        self._client = self._init_client()

    def _init_client(self):
        """
        Initialize the appropriate LLM client based on model_provider.

        Returns:
            An initialized client (OpenAI-compatible)
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package is required for JustificationAgent. "
                "Install with: pip install openai"
            )

        if self.config.model_provider == "anthropic":
            try:
                import anthropic

                return anthropic.Anthropic()
            except ImportError:
                raise ImportError(
                    "anthropic package required for Anthropic provider. "
                    "Install with: pip install anthropic"
                )

        elif self.config.model_provider == "deepseek":
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError(
                    "DEEPSEEK_API_KEY environment variable required for DeepSeek"
                )
            return openai.OpenAI(
                api_key=api_key, base_url="https://api.deepseek.com/v1"
            )

        elif self.config.model_provider == "openai":
            # For OpenAI-compatible APIs (including OpenRouter)
            # Check for OpenRouter first, then fall back to OPENAI_API_KEY
            api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENROUTER_API_KEY or OPENAI_API_KEY environment variable required"
                )
            return openai.OpenAI(api_key=api_key, base_url=self.config.api_base)

        else:
            raise ValueError(f"Unknown model provider: {self.config.model_provider}")

    def generate(
        self,
        prompt_content: str,
        stance: Literal["keep_current", "adopt_target"],
        current_pref_text: str,
        target_pref_text: str,
        domain: str,
        severity: str,
        mode: str,
        perspective: str,
        seed: int,
        tradeoff_hints: Optional[List[str]] = None,
    ) -> JustificationResult:
        """
        Generate a justification for the given stance.

        Validation is SOFT by default (validation_mode="warn"):
        - Tracks failures in self.report
        - Does NOT retry/fallback unless config.validation_mode="strict"

        Args:
            prompt_content: The rendered prompt content (no tag)
            stance: "keep_current" (anti) or "adopt_target" (pro)
            current_pref_text: Text of current preference
            target_pref_text: Text of target preference
            domain: Preference domain
            severity: Severity level
            mode: Response mode
            perspective: Perspective (first/third)
            seed: Random seed for determinism
            tradeoff_hints: Optional hints to guide generation

        Returns:
            JustificationResult with final justification and metadata
        """
        # Build cache key
        cache_key = JustificationCache.build_cache_key(
            config_hash=self.config.config_hash(),
            prompt_content=prompt_content,
            stance=stance,
            current_pref_text=current_pref_text,
            target_pref_text=target_pref_text,
            domain=domain,
            severity=severity,
            mode=mode,
            perspective=perspective,
        )

        # Check cache first
        cached = self.cache.get(cache_key)
        if cached:
            # Return cached result
            validation_result = ValidationResult(passed=True, failure_reasons=[])
            return JustificationResult(
                justification=cached.final_justification,
                raw_justification=cached.raw_justification,
                cache_key=cache_key,
                cache_hit=True,
                attempt_count=cached.attempt_count,
                used_fallback=cached.used_fallback,
                validation_result=validation_result,
            )

        # Get tradeoff hints if not provided
        if tradeoff_hints is None:
            tradeoff_hints = get_tradeoff_hints(domain, seed)

        # Generate justification
        attempt_count = 0
        used_fallback = False
        raw_justification = ""
        final_justification = ""
        validation_result = None

        # First attempt
        attempt_count += 1
        raw_justification = self._call_llm(stance, prompt_content, current_pref_text,
                                            target_pref_text, domain, tradeoff_hints)
        final_justification = raw_justification.strip()

        # Validate
        validation_result = self._validate(
            final_justification, stance, current_pref_text, target_pref_text
        )

        # Handle validation failure based on mode
        if not validation_result.passed:
            if self.config.validation_mode == "strict":
                # Retry once with addendum
                if attempt_count < self.config.retry_limit + 1:
                    attempt_count += 1
                    raw_justification = self._call_llm(
                        stance, prompt_content, current_pref_text,
                        target_pref_text, domain, tradeoff_hints, retry=True
                    )
                    final_justification = raw_justification.strip()
                    validation_result = self._validate(
                        final_justification, stance, current_pref_text, target_pref_text
                    )

                # Fallback if still failing
                if not validation_result.passed and self.config.fallback_enabled:
                    used_fallback = True
                    final_justification = self._generate_fallback(
                        stance, current_pref_text, target_pref_text, seed, tradeoff_hints
                    )
                    validation_result = ValidationResult(passed=True, failure_reasons=[])

        # Record in report
        self.report.record(
            validation_result,
            final_justification,
            used_retry=(attempt_count > 1),
            used_fallback=used_fallback,
        )

        # Store in cache
        cache_entry = CacheEntry(
            cache_key=cache_key,
            raw_justification=raw_justification,
            final_justification=final_justification,
            attempt_count=attempt_count,
            used_fallback=used_fallback,
            timestamp=datetime.utcnow().isoformat(),
            config_hash=self.config.config_hash(),
        )
        self.cache.put(cache_entry)

        return JustificationResult(
            justification=final_justification,
            raw_justification=raw_justification,
            cache_key=cache_key,
            cache_hit=False,
            attempt_count=attempt_count,
            used_fallback=used_fallback,
            validation_result=validation_result,
        )

    def _call_llm(
        self,
        stance: str,
        prompt_content: str,
        current_pref_text: str,
        target_pref_text: str,
        domain: str,
        tradeoff_hints: List[str],
        retry: bool = False,
    ) -> str:
        """
        Call the LLM to generate a justification.

        Args:
            stance: "keep_current" or "adopt_target"
            prompt_content: The rendered prompt content
            current_pref_text: Current preference text
            target_pref_text: Target preference text
            domain: Preference domain
            tradeoff_hints: Hints to guide generation
            retry: Whether this is a retry attempt

        Returns:
            The generated justification text
        """
        # Select system prompt
        system_prompt = (
            SYSTEM_KEEP_CURRENT if stance == "keep_current" else SYSTEM_ADOPT_TARGET
        )

        # Build user message
        user_payload = {
            "stance": stance,
            "current_preference": current_pref_text,
            "target_preference": target_pref_text,
            "domain": domain,
            "tradeoff_hints": tradeoff_hints,
        }

        user_message = f"Generate a justification.\n\n{json.dumps(user_payload, indent=2)}"

        if retry:
            user_message += (
                "\n\nIMPORTANT: Be more specific - explicitly name the preference "
                "and give one concrete consequence."
            )

        # Call LLM based on provider
        if self.config.model_provider == "anthropic":
            return self._call_anthropic(system_prompt, user_message)
        else:
            return self._call_openai_compatible(system_prompt, user_message)

    def _call_openai_compatible(self, system: str, user: str) -> str:
        """Call OpenAI-compatible API (DeepSeek, OpenAI, etc.)."""
        response = self._client.chat.completions.create(
            model=self.config.model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content or ""

    def _call_anthropic(self, system: str, user: str) -> str:
        """Call Anthropic API."""
        response = self._client.messages.create(
            model=self.config.model_id,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
        )
        return response.content[0].text if response.content else ""

    def _validate(
        self,
        justification: str,
        stance: str,
        current_pref: str,
        target_pref: str,
    ) -> ValidationResult:
        """
        Run validation checks and return result with failure reasons.

        Checks:
        1. Format: non-empty, ≤max_chars, ≤max_sentences
        2. Specificity: contains preference-related content (soft check)
        3. Forbidden: no disallowed phrases

        Args:
            justification: The justification text to validate
            stance: The stance ("keep_current" or "adopt_target")
            current_pref: Current preference text
            target_pref: Target preference text

        Returns:
            ValidationResult with passed status and failure reasons
        """
        failure_reasons = []

        # Check 1: Empty
        if not justification or not justification.strip():
            failure_reasons.append("empty_response")
            return ValidationResult(passed=False, failure_reasons=failure_reasons)

        # Check 2: Too long (chars)
        if len(justification) > self.config.max_chars:
            failure_reasons.append("too_long")

        # Check 3: Too many sentences
        sentences = self._count_sentences(justification)
        if sentences > self.config.max_sentences + 1:  # Allow some tolerance
            failure_reasons.append("too_many_sentences")

        # Check 4: Forbidden phrases
        forbidden_violations = self._check_forbidden_phrases(justification)
        failure_reasons.extend(forbidden_violations)

        # Check 5: Missing preference content (info only - soft check)
        relevant_pref = current_pref if stance == "keep_current" else target_pref
        if not self._contains_preference_content(justification, relevant_pref):
            failure_reasons.append("missing_preference_content")

        # Check 6: Missing tradeoff word (info only - soft check)
        if not contains_tradeoff_word(justification):
            failure_reasons.append("missing_tradeoff_word")

        passed = len(failure_reasons) == 0
        return ValidationResult(passed=passed, failure_reasons=failure_reasons)

    def _count_sentences(self, text: str) -> int:
        """Count sentences in text using simple heuristics."""
        # Split on sentence-ending punctuation followed by space or end
        sentences = re.split(r"[.!?]+(?:\s|$)", text)
        return len([s for s in sentences if s.strip()])

    def _check_forbidden_phrases(self, text: str) -> List[str]:
        """Check for forbidden phrases, return list of violations."""
        text_lower = text.lower()
        violations = []
        for phrase in self.FORBIDDEN_PHRASES:
            if phrase in text_lower:
                violations.append(f"contains_forbidden: '{phrase}'")
        return violations

    def _contains_preference_content(self, justification: str, preference: str) -> bool:
        """
        Check if justification contains content from the preference.

        Uses word overlap as a heuristic - at least one content word
        from the preference should appear (possibly paraphrased).
        """
        # Extract content words (nouns, adjectives) - simple heuristic
        pref_words = set(preference.lower().split())
        just_words = set(justification.lower().split())

        # Remove common stop words
        stop_words = {
            "a",
            "an",
            "the",
            "to",
            "of",
            "in",
            "for",
            "and",
            "or",
            "with",
            "on",
            "at",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "that",
            "this",
            "these",
            "those",
            "it",
            "its",
        }
        pref_content = pref_words - stop_words
        just_content = just_words - stop_words

        # Check for any overlap
        return bool(pref_content & just_content)

    def _generate_fallback(
        self,
        stance: str,
        current_pref: str,
        target_pref: str,
        seed: int,
        hints: List[str],
    ) -> str:
        """
        Generate a deterministic fallback justification.

        Used when LLM generation fails validation in strict mode.

        Args:
            stance: "keep_current" or "adopt_target"
            current_pref: Current preference text
            target_pref: Target preference text
            seed: Random seed for template selection
            hints: Tradeoff hints

        Returns:
            A deterministic fallback justification
        """
        templates = self.FALLBACK_TEMPLATES[stance]

        # Use seed to select template deterministically
        rng = random.Random(seed)
        template = rng.choice(templates)

        # Select hint
        hint = hints[0] if hints else "quality"

        # Fill template
        return template.format(
            current_pref=current_pref, target_pref=target_pref, hint=hint
        )

    def finalize(self) -> None:
        """
        Finalize the agent - save cache and print report.

        Call this at the end of generation to persist cache
        and display validation summary.
        """
        self.cache.save_to_disk()
        self.report.print_summary()
