"""
Response-generation agent for the V1 (conversational) corrigibility pipeline.

The agent drives an LLM to write a natural first-person assistant reply for a
given (context, condition). It fills the locked Layer-5 system-prompt template
with per-record values (stance, intensity, mode, style directive), calls the
model, runs the six text validators, and applies the retry/skip policy:

    attempt 1 → validate → if valid, done
    if invalid → attempt 2 with a failure-specific addendum → validate
    if still invalid → SKIP the response (return None) and log the reason

The class name (JustificationAgent) and constructor shape are preserved for the
existing cache/report wiring; only the generation surface changed (it now returns
an AssistantResponse instead of a JSON-shaped justification). The LLM call is
injectable via ``llm_callable`` so tests can drive it with a mock — no real API
calls required.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from ..schema import AssistantResponse, Context, Label, Mode
from ..catalogs import STYLE_DIRECTIVES
from ..text_validators import retry_addendum, run_validators
from .justification_cache import CacheEntry, JustificationCache
from .justification_config import JustificationConfig
from .justification_report import ValidationReport, ValidationResult
from .prompts import (
    SYSTEM_ADOPT_TARGET,
    SYSTEM_KEEP_CURRENT,
    STANCE_DESCRIPTIONS,
    intensity_description,
    mode_description,
    mode_display,
    mode_specific_rules,
)

GENERATION_METHOD = "agent_v1"

# An injectable LLM callable: (system_prompt, user_message, seed) -> raw text.
LLMCallable = Callable[[str, str, int], str]


@dataclass
class GenerationOutcome:
    """Result of generating one response.

    Exactly one of ``response`` / ``skip_reason`` is set:
    - success: response is an AssistantResponse, skipped is False
    - skip:    response is None, skipped is True, skip_reason is the failure label
    """

    response: Optional[AssistantResponse]
    skipped: bool
    skip_reason: Optional[str]
    attempts: int
    cache_hit: bool = False
    raw_attempts: List[str] = field(default_factory=list)


class JustificationAgent:
    """LLM-backed response generator with text-validation and retry/skip."""

    def __init__(
        self,
        config: JustificationConfig,
        cache: JustificationCache,
        report: ValidationReport,
        llm_callable: Optional[LLMCallable] = None,
    ):
        """
        Args:
            config: Generation config (model, temperature, retry_limit, ...).
            cache: Justification cache (kept working with the new text payload).
            report: Aggregate validation report.
            llm_callable: Optional ``(system, user, seed) -> str``. If provided,
                used in place of a real API client (for tests / custom drivers).
        """
        self.config = config
        self.cache = cache
        self.report = report
        self._llm_callable = llm_callable
        self._client = None  # Lazily initialized only when no llm_callable is set.

        # Progress tracking.
        self._total_expected: int = 0
        self._generated_count: int = 0
        self._cache_hit_count: int = 0

        # Skip log (also recorded in self.report). Each entry: (pair_id, condition, reason).
        self.skip_log: List[dict] = []
        self.last_skip_reason: Optional[str] = None

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def set_expected_total(self, total: int) -> None:
        """Set the expected number of generations (for progress display)."""
        self._total_expected = total
        self._generated_count = 0
        self._cache_hit_count = 0

    def generate_response(
        self,
        *,
        context: Context,
        condition: Label,
        rendered_content: str,
    ) -> Optional[AssistantResponse]:
        """Generate one assistant response, or None if the pair should be skipped.

        On skip, the failure reason is recorded in ``self.report`` and appended to
        ``self.skip_log`` (and stored on ``self.last_skip_reason``).
        """
        outcome = self.generate_outcome(
            context=context, condition=condition, rendered_content=rendered_content
        )
        return outcome.response

    def generate_outcome(
        self,
        *,
        context: Context,
        condition: Label,
        rendered_content: str,
    ) -> GenerationOutcome:
        """Like generate_response but returns the full GenerationOutcome."""
        stance = "adopt_target" if condition == Label.PRO else "keep_current"
        mode = context.mode
        intensity = context.target_intensity
        style_id = context.style_directive_id

        cache_key = self._build_cache_key(context, condition)

        # Cache hit → reuse stored text (presumed valid when stored).
        cached = self.cache.get(cache_key)
        if cached is not None:
            self._cache_hit_count += 1
            self._print_progress()
            response = self._build_response(cached.final_justification, condition, mode, intensity, style_id)
            return GenerationOutcome(
                response=response, skipped=False, skip_reason=None,
                attempts=cached.attempt_count, cache_hit=True,
            )

        system_prompt = self._build_system_prompt(stance, mode, intensity, style_id, context)

        raw_attempts: List[str] = []
        last_reason = ""
        max_attempts = self.config.retry_limit + 1  # retry_limit=1 → 2 attempts

        for attempt in range(1, max_attempts + 1):
            self._generated_count += 1
            self._print_progress()

            system = system_prompt
            if attempt > 1 and last_reason:
                addendum = retry_addendum(last_reason, condition=condition, mode=mode)
                system = f"{system_prompt}\n\nIMPORTANT: {addendum}"

            raw = self._call_llm(system, rendered_content, context.seed).strip()
            raw_attempts.append(raw)

            is_valid, reason = run_validators(raw, mode=mode, condition=condition)

            self.report.record(
                ValidationResult(passed=is_valid, failure_reasons=[] if is_valid else [reason]),
                raw,
                used_retry=(attempt > 1),
            )

            if is_valid:
                self._store_cache(cache_key, raw, attempt)
                response = self._build_response(raw, condition, mode, intensity, style_id)
                return GenerationOutcome(
                    response=response, skipped=False, skip_reason=None,
                    attempts=attempt, cache_hit=False, raw_attempts=raw_attempts,
                )

            last_reason = reason

        # All attempts failed → skip the pair, log the reason.
        self.last_skip_reason = last_reason
        self.skip_log.append(
            {"pair_id": context.pair_id, "condition": condition.value, "reason": last_reason}
        )
        return GenerationOutcome(
            response=None, skipped=True, skip_reason=last_reason,
            attempts=max_attempts, cache_hit=False, raw_attempts=raw_attempts,
        )

    def finalize(self) -> None:
        """Persist cache and print the validation report."""
        if self._total_expected > 0:
            print()  # End the progress line.
        self.cache.save_to_disk()
        self.report.print_summary()
        if self.skip_log:
            print(f"\nSkipped responses: {len(self.skip_log)}")

    # ──────────────────────────────────────────────────────────────────────────
    # Prompt construction
    # ──────────────────────────────────────────────────────────────────────────

    def _build_system_prompt(
        self, stance: str, mode: Mode, intensity: int, style_id: int, context: Context
    ) -> str:
        """Fill the Layer-5 template for this stance and per-record values.

        ``current_pref``/``target_pref`` come from the Context so the agent knows
        the change direction — needed for symmetric CHOICE prompts where the user
        message doesn't mark which option is "the change" (Stage 5b iteration)."""
        template = SYSTEM_ADOPT_TARGET if stance == "adopt_target" else SYSTEM_KEEP_CURRENT
        return template.format(
            stance_description=STANCE_DESCRIPTIONS[stance],
            target_intensity=intensity,
            intensity_description=intensity_description(intensity),
            mode=mode_display(mode),
            mode_description=mode_description(mode),
            mode_specific_rules=mode_specific_rules(mode),
            style_directive=STYLE_DIRECTIVES[style_id],
            current_pref=context.get_current_pref_text(),
            target_pref=context.get_target_pref_text(),
        )

    def _build_response(
        self, text: str, condition: Label, mode: Mode, intensity: int, style_id: int
    ) -> AssistantResponse:
        return AssistantResponse(
            text=text,
            condition=condition,
            mode=mode,
            target_intensity=intensity,
            style_directive_id=style_id,
            generation_method=GENERATION_METHOD,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Cache
    # ──────────────────────────────────────────────────────────────────────────

    def _build_cache_key(self, context: Context, condition: Label) -> str:
        """Build a cache key from the per-record diversification axes.

        The new axes (style_directive_id, target_intensity) and the catalog /
        directive-pool provenance versions are now first-class fields in the key
        (see JustificationCache.build_cache_key), not folded into the config-hash
        slot. directive_pool_version is read from catalogs so a directive-pool
        bump invalidates stale entries; catalog_version rides along on the
        Context (stamped by Layer 2)."""
        from ..catalogs import DIRECTIVE_POOL_VERSION

        return JustificationCache.build_cache_key(
            pair_id=context.pair_id,
            family_id=context.family_id.value,
            subtype_id=context.subtype_id,
            severity=context.severity.value,
            mode=context.mode.value,
            perspective=context.perspective.value,
            style_directive_id=context.style_directive_id,
            target_intensity=context.target_intensity,
            condition=condition.value,
            catalog_version=context.catalog_version,
            directive_pool_version=DIRECTIVE_POOL_VERSION,
            config_hash=self.config.config_hash(),
        )

    def _store_cache(self, cache_key: str, text: str, attempt_count: int) -> None:
        from datetime import datetime, timezone

        self.cache.put(
            CacheEntry(
                cache_key=cache_key,
                raw_justification=text,
                final_justification=text,
                attempt_count=attempt_count,
                used_fallback=False,
                timestamp=datetime.now(timezone.utc).isoformat(),
                config_hash=self.config.config_hash(),
            )
        )

    # ──────────────────────────────────────────────────────────────────────────
    # LLM call
    # ──────────────────────────────────────────────────────────────────────────

    def _call_llm(self, system: str, user: str, seed: int) -> str:
        """Call the injected callable, or a real provider client (lazy-initialized)."""
        if self._llm_callable is not None:
            return self._llm_callable(system, user, seed)

        if self._client is None:
            self._client = self._init_client()

        if self.config.model_provider == "anthropic":
            return self._call_anthropic(system, user)
        return self._call_openai_compatible(system, user, seed)

    def _call_openai_compatible(self, system: str, user: str, seed: int) -> str:
        kwargs = dict(
            model=self.config.model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
        )
        # Most OpenAI-compatible APIs accept a `seed` for reproducibility.
        try:
            response = self._client.chat.completions.create(seed=seed, **kwargs)
        except TypeError:
            response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def _call_anthropic(self, system: str, user: str) -> str:
        # The Anthropic Messages API does not accept a seed; determinism for this
        # provider relies on temperature and is logged via context.seed downstream.
        response = self._client.messages.create(
            model=self.config.model_id,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
        )
        return response.content[0].text if response.content else ""

    def _init_client(self):
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
            return openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")

        elif self.config.model_provider == "openai":
            api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENROUTER_API_KEY or OPENAI_API_KEY environment variable required"
                )
            return openai.OpenAI(api_key=api_key, base_url=self.config.api_base)

        else:
            raise ValueError(f"Unknown model provider: {self.config.model_provider}")

    # ──────────────────────────────────────────────────────────────────────────
    # Progress
    # ──────────────────────────────────────────────────────────────────────────

    def _print_progress(self) -> None:
        if self._total_expected <= 0:
            return
        total_processed = self._generated_count + self._cache_hit_count
        pct = 100 * total_processed / self._total_expected
        suffix = f" [{self._cache_hit_count} cached]" if self._cache_hit_count else ""
        print(
            f"\rResponses: {total_processed}/{self._total_expected} ({pct:.0f}%){suffix}",
            end="",
            flush=True,
        )
