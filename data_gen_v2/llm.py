"""
The single LLM call surface (Stage 0).

Both agents (Stages 2/3) depend on ``LLMClient``, never on a provider SDK
directly. The primary testing seam is ``llm_callable``: inject a deterministic
``(system, user, seed) -> str`` mock and no network call is made. Provider
dispatch (anthropic / openai-compatible) is carried forward from
``dataset_gen/src/agents/justification_agent.py``.

Determinism note: against a real provider, generation is not guaranteed
bit-reproducible. The *spec* layer is reproducible; ``seed`` is threaded through
for providers that honor it (OpenAI-compatible) and is always available to log.
The Anthropic Messages API has no seed parameter.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

from .config import LLMConfig

# (system_prompt, user_message, seed) -> raw_text
LLMCallable = Callable[[str, str, int], str]


class LLMClient:
    """Thin wrapper over a provider client or an injected callable."""

    def __init__(self, config: LLMConfig, llm_callable: Optional[LLMCallable] = None):
        self.config = config
        self._llm_callable = llm_callable
        self._client = None  # lazily initialized only when no callable is set

    @property
    def model_id(self) -> str:
        return self.config.model_id

    def call(self, system: str, user: str, seed: int) -> str:
        """Return raw model text (caller strips). Delegates to the mock if set."""
        if self._llm_callable is not None:
            return self._llm_callable(system, user, seed)

        if self._client is None:
            self._client = self._init_client()

        if self.config.model_provider == "anthropic":
            return self._call_anthropic(system, user)
        return self._call_openai_compatible(system, user, seed)

    # ── provider calls (ported from v1 justification_agent) ─────────────────────

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
        # Anthropic Messages API does not accept a seed.
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
        provider = self.config.model_provider
        if provider == "anthropic":
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError(
                    "anthropic package required for the Anthropic provider"
                ) from exc
            return anthropic.Anthropic()

        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "openai package required for openai-compatible providers"
            ) from exc

        if provider == "deepseek":
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY environment variable required for DeepSeek")
            return openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")

        if provider == "openai":
            api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENROUTER_API_KEY or OPENAI_API_KEY environment variable required"
                )
            return openai.OpenAI(api_key=api_key, base_url=self.config.api_base)

        raise ValueError(f"Unknown model provider: {provider}")


__all__ = ["LLMCallable", "LLMClient"]
