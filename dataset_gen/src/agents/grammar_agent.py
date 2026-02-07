"""
Grammar judge/fixer agent using an LLM.

Uses LanguageTool lint output for routing, and optionally fixes content
with minimal edits.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

from .grammar_cache import GrammarCache, GrammarCacheEntry
from .grammar_config import GrammarConfig
from .grammar_report import GrammarReport


@dataclass
class GrammarDecision:
    """
    Result of grammar judging/fixing.

    Attributes:
        decision: "clean", "fixed", or "reject"
        fixed_content: corrected content if decision == "fixed"
        reason: short reason for decision (optional)
        raw_response: raw LLM response
        cache_hit: whether cache was used
        attempt_count: number of attempts made
        llm_error: whether a call/parsing error occurred
    """

    decision: str
    fixed_content: Optional[str]
    reason: str
    raw_response: str
    cache_hit: bool
    attempt_count: int
    llm_error: bool = False


class GrammarAgent:
    """
    LLM-based grammar judge/fixer.

    Uses minimal edits to correct grammar and reading errors.
    """

    SYSTEM_PROMPT = (
        "You are a meticulous grammar editor. Your task is to fix grammar and reading "
        "errors with minimal edits. Do NOT change meaning, tone, perspective, or add new "
        "information. Preserve all names, numbers, and constraints.\n\n"
        "Return STRICT JSON only with keys:\n"
        "- decision: one of [clean, fixed, reject]\n"
        "- corrected_text: full corrected text if decision=fixed, else empty string\n"
        "- reason: short reason (optional)\n\n"
        "If the text is already correct, respond decision=clean with empty corrected_text.\n"
        "If the text cannot be corrected without rewriting or changing meaning, respond "
        "decision=reject."
    )

    def __init__(
        self,
        config: GrammarConfig,
        cache: GrammarCache,
        report: GrammarReport,
    ):
        self.config = config
        self.cache = cache
        self.report = report
        self._client = self._init_client()

    def _init_client(self):
        if self.config.model_provider == "anthropic":
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package required for GrammarAgent. "
                    "Install with: pip install anthropic"
                )
            return anthropic.Anthropic()
        elif self.config.model_provider in ("openai", "openrouter"):
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "openai package required for GrammarAgent. "
                    "Install with: pip install openai"
                )
            if self.config.model_provider == "openrouter":
                api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError(
                        "OPENROUTER_API_KEY environment variable required for OpenRouter"
                    )
                api_base = self.config.api_base or "https://openrouter.ai/api/v1"
                return openai.OpenAI(api_key=api_key, base_url=api_base)

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable required")
            return openai.OpenAI(api_key=api_key, base_url=self.config.api_base)
        else:
            raise ValueError(f"Unknown model provider: {self.config.model_provider}")

    def should_route(self, lint_result) -> bool:
        """
        Decide whether to route this prompt to the LLM based on lint.
        """
        if self.config.trigger_mode == "always":
            return True

        if lint_result.has_blocking_errors:
            return True

        if self.config.trigger_mode == "blocking":
            return False

        if self.config.trigger_mode == "any_warnings":
            return lint_result.has_warnings

        if self.config.trigger_mode == "blocking_or_grammar_warnings":
            for err in lint_result.errors:
                if err.severity.value != "warn":
                    continue
                if err.category in self.config.warning_exclude_categories:
                    continue
                if err.rule_id in self.config.warning_exclude_rules:
                    continue
                return True

        return False

    def judge_and_fix(
        self,
        content: str,
        lint_result,
    ) -> GrammarDecision:
        """
        Call LLM to judge and optionally fix grammar.
        """
        lint_signature = self._build_lint_signature(lint_result)
        cache_key = GrammarCache.build_cache_key(
            config_hash=self.config.config_hash(),
            content=content,
            lint_signature=lint_signature,
        )
        cached = self.cache.get(cache_key)
        if cached:
            return GrammarDecision(
                decision=cached.decision,
                fixed_content=cached.fixed_content or None,
                reason="cache_hit",
                raw_response=cached.raw_response,
                cache_hit=True,
                attempt_count=0,
            )

        last_error = ""
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                user_message = self._build_user_message(content, lint_result, attempt)
                raw = self._call_llm(self.SYSTEM_PROMPT, user_message)
                parsed = self._parse_response(raw)
                if not parsed:
                    last_error = "invalid_json"
                    continue

                decision = self._normalize_decision(parsed.get("decision", ""))
                corrected_text = parsed.get("corrected_text", "") or ""
                reason = parsed.get("reason", "") or ""

                if decision == "fixed":
                    if not corrected_text.strip():
                        last_error = "empty_fix"
                        continue
                    if not self._valid_fix(content, corrected_text):
                        return GrammarDecision(
                            decision="reject",
                            fixed_content=None,
                            reason="excessive_edit",
                            raw_response=raw,
                            cache_hit=False,
                            attempt_count=attempt,
                        )

                entry = GrammarCacheEntry(
                    cache_key=cache_key,
                    decision=decision,
                    fixed_content=corrected_text,
                    raw_response=raw,
                    timestamp=GrammarCache.now_iso(),
                    config_hash=self.config.config_hash(),
                )
                self.cache.put(entry)

                return GrammarDecision(
                    decision=decision,
                    fixed_content=corrected_text if decision == "fixed" else None,
                    reason=reason,
                    raw_response=raw,
                    cache_hit=False,
                    attempt_count=attempt,
                )
            except Exception as e:
                last_error = f"llm_error:{e}"
                break

        return GrammarDecision(
            decision="reject",
            fixed_content=None,
            reason=last_error or "llm_error",
            raw_response="",
            cache_hit=False,
            attempt_count=self.config.max_attempts,
            llm_error=True,
        )

    def _build_user_message(self, content: str, lint_result, attempt: int) -> str:
        errors = []
        for err in lint_result.errors[:5]:
            errors.append(
                {
                    "rule_id": err.rule_id,
                    "category": err.category,
                    "message": err.message,
                    "context": err.context,
                }
            )
        payload = {
            "content": content,
            "lint_errors": errors,
        }
        msg = f"Fix the following text.\n\n{json.dumps(payload, indent=2)}"
        if attempt > 1:
            msg += "\n\nIMPORTANT: Output strict JSON only."
        return msg

    def _call_llm(self, system: str, user: str) -> str:
        if self.config.model_provider == "anthropic":
            response = self._client.messages.create(
                model=self.config.model_id,
                system=system,
                messages=[{"role": "user", "content": user}],
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens,
            )
            return response.content[0].text if response.content else ""

        # OpenAI-compatible fallback (OpenAI or OpenRouter)
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

    def _parse_response(self, text: str) -> Optional[dict]:
        if not text:
            return None

        cleaned = text.strip()
        # Remove fenced code blocks if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*", "", cleaned).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try to extract first JSON object
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None

    def _normalize_decision(self, decision: str) -> str:
        decision = decision.strip().lower()
        if decision in ("clean", "pass", "ok"):
            return "clean"
        if decision in ("fixed", "fix", "corrected"):
            return "fixed"
        if decision in ("reject", "fail", "invalid"):
            return "reject"
        return "reject"

    def _valid_fix(self, original: str, fixed: str) -> bool:
        if original == fixed:
            return True
        delta = abs(len(fixed) - len(original))
        if delta > self.config.max_char_delta:
            return False
        ratio = delta / max(len(original), 1)
        if ratio > self.config.max_char_delta_ratio:
            return False
        return True

    def _build_lint_signature(self, lint_result) -> str:
        counts = {}
        for err in lint_result.errors:
            key = f"{err.rule_id}:{err.severity.value}"
            counts[key] = counts.get(key, 0) + 1
        parts = [f"{k}={counts[k]}" for k in sorted(counts.keys())]
        return ";".join(parts)

    def finalize(self) -> None:
        self.cache.save_to_disk()
