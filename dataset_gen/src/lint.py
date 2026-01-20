"""
Grammar Linting Module for Corrigibility Dataset Generation Pipeline.

Provides LanguageTool-based grammar checking with tiered severity levels
to distinguish blocking errors from warnings.

Integration point: Between Layer 4 (Family Render) and Layer 5 (Answer Policy).
Only lints rendered.content (prompt description), never rendered.tag.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, TYPE_CHECKING
import hashlib
import random
import logging

if TYPE_CHECKING:
    from .schema import Context, RenderedPrompt

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SEVERITY TIERS
# ═══════════════════════════════════════════════════════════════════════════════

# Tier 1: HARD_BLOCK - errors that create learnable artifacts
HARD_BLOCK_RULES = frozenset({
    # Spelling/typos
    "MORFOLOGIK_RULE_EN_US",      # Misspellings
    "HUNSPELL_RULE",              # Spelling errors

    # Repeated words
    "ENGLISH_WORD_REPEAT_RULE",   # "the the"
    "ENGLISH_WORD_REPEAT_BEGINNING_RULE",

    # Agreement errors
    "HE_VERB_AGR",                # "he prefer" → "he prefers"
    "PERS_PRONOUN_AGREEMENT",     # "you prefers" → "you prefer"
    "THIS_NNS",                   # "this priorities" → "these priorities"
    "PLURAL_VERB_AFTER_THIS",     # Subject-verb agreement
    "A_PLURAL",                   # "a priorities" → "priorities"
    "MANY_NN",                    # "many priority" → "many priorities"
})

# Tier 2: WARN_ONLY - style issues, don't block
WARN_ONLY_CATEGORIES = frozenset({
    "PUNCTUATION",
    "STYLE",
    "REDUNDANCY",
    "CASING",
})

WARN_ONLY_RULES = frozenset({
    "EN_QUOTES",                  # Quote style
    "COMMA_PARENTHESIS_WHITESPACE",
    "DOUBLE_PUNCTUATION",
    "SENTENCE_WHITESPACE",
})

# Tier 3: IGNORE - noise, skip entirely
IGNORED_RULES = frozenset({
    "WHITESPACE_RULE",
    "EN_GB_SIMPLE_REPLACE",       # British vs American
    "UPPERCASE_SENTENCE_START",   # Templates may have valid lowercase starts
    "CURRENCY",
    "DATE_FORMAT",
})


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS AND DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


class LintSeverity(str, Enum):
    """Severity levels for lint errors."""
    HARD_BLOCK = "hard_block"   # Blocks generation
    WARN = "warn"               # Logged but continues
    IGNORE = "ignore"           # Not tracked


class LintMode(str, Enum):
    """Linting mode configuration."""
    ENABLED = "enabled"         # Full linting with blocking
    WARN_ONLY = "warn_only"     # All errors are warnings (no blocking)
    DISABLED = "disabled"       # Skip linting entirely


@dataclass(frozen=True)
class LintError:
    """A single grammar/spelling error found by LanguageTool."""
    rule_id: str
    category: str
    severity: LintSeverity      # Computed from rule tiering
    message: str
    context: str
    offset: int
    length: int
    suggestions: tuple

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "rule_id": self.rule_id,
            "category": self.category,
            "severity": self.severity.value,
            "message": self.message,
            "context": self.context,
            "offset": self.offset,
            "length": self.length,
            "suggestions": list(self.suggestions),
        }


@dataclass
class LintResult:
    """Result for a single prompt with full metadata for root cause analysis."""
    # Content identification
    template_id: str            # e.g., "A1_07"
    content_hash: str           # For deduplication

    # Structural metadata (helps locate root causes)
    family: str                 # e.g., "A"
    subtype: str                # e.g., "A1"
    mode: str                   # e.g., "rating", "choice", "short"
    formatting_variant: int     # 0, 1, or 2
    perspective: str            # "first" or "third"

    # Error data
    errors: List[LintError] = field(default_factory=list)

    @property
    def has_blocking_errors(self) -> bool:
        """Check if any errors would block generation."""
        return any(e.severity == LintSeverity.HARD_BLOCK for e in self.errors)

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warning-level errors."""
        return any(e.severity == LintSeverity.WARN for e in self.errors)

    @property
    def blocking_error_count(self) -> int:
        """Count of blocking errors."""
        return sum(1 for e in self.errors if e.severity == LintSeverity.HARD_BLOCK)

    @property
    def warning_count(self) -> int:
        """Count of warning-level errors."""
        return sum(1 for e in self.errors if e.severity == LintSeverity.WARN)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "template_id": self.template_id,
            "content_hash": self.content_hash,
            "family": self.family,
            "subtype": self.subtype,
            "mode": self.mode,
            "formatting_variant": self.formatting_variant,
            "perspective": self.perspective,
            "errors": [e.to_dict() for e in self.errors],
            "has_blocking_errors": self.has_blocking_errors,
            "has_warnings": self.has_warnings,
            "blocking_error_count": self.blocking_error_count,
            "warning_count": self.warning_count,
        }


@dataclass
class LintReport:
    """Aggregated report with separate tracking for blocks vs warnings."""
    total_prompts: int = 0
    prompts_with_blocking_errors: int = 0
    prompts_with_warnings: int = 0
    total_blocking_errors: int = 0
    total_warnings: int = 0

    # Breakdowns
    blocking_errors_by_rule: Dict[str, int] = field(default_factory=dict)
    warnings_by_rule: Dict[str, int] = field(default_factory=dict)
    errors_by_template: Dict[str, int] = field(default_factory=dict)
    errors_by_family: Dict[str, int] = field(default_factory=dict)
    errors_by_subtype: Dict[str, int] = field(default_factory=dict)

    # Samples for inspection (limit to avoid huge reports)
    sample_blocking_errors: List[LintResult] = field(default_factory=list)
    sample_warnings: List[LintResult] = field(default_factory=list)
    _max_samples: int = field(default=10, repr=False)

    def add_result(self, result: LintResult) -> None:
        """Add a lint result to the report."""
        self.total_prompts += 1

        if result.has_blocking_errors:
            self.prompts_with_blocking_errors += 1
            self.total_blocking_errors += result.blocking_error_count

            # Track by rule
            for error in result.errors:
                if error.severity == LintSeverity.HARD_BLOCK:
                    self.blocking_errors_by_rule[error.rule_id] = (
                        self.blocking_errors_by_rule.get(error.rule_id, 0) + 1
                    )

            # Track by template/family/subtype
            self.errors_by_template[result.template_id] = (
                self.errors_by_template.get(result.template_id, 0) + result.blocking_error_count
            )
            self.errors_by_family[result.family] = (
                self.errors_by_family.get(result.family, 0) + result.blocking_error_count
            )
            self.errors_by_subtype[result.subtype] = (
                self.errors_by_subtype.get(result.subtype, 0) + result.blocking_error_count
            )

            # Sample collection
            if len(self.sample_blocking_errors) < self._max_samples:
                self.sample_blocking_errors.append(result)

        if result.has_warnings:
            self.prompts_with_warnings += 1
            self.total_warnings += result.warning_count

            # Track warnings by rule
            for error in result.errors:
                if error.severity == LintSeverity.WARN:
                    self.warnings_by_rule[error.rule_id] = (
                        self.warnings_by_rule.get(error.rule_id, 0) + 1
                    )

            # Sample collection
            if len(self.sample_warnings) < self._max_samples:
                self.sample_warnings.append(result)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_prompts": self.total_prompts,
            "prompts_with_blocking_errors": self.prompts_with_blocking_errors,
            "prompts_with_warnings": self.prompts_with_warnings,
            "total_blocking_errors": self.total_blocking_errors,
            "total_warnings": self.total_warnings,
            "blocking_errors_by_rule": self.blocking_errors_by_rule,
            "warnings_by_rule": self.warnings_by_rule,
            "errors_by_template": self.errors_by_template,
            "errors_by_family": self.errors_by_family,
            "errors_by_subtype": self.errors_by_subtype,
            "sample_blocking_errors": [r.to_dict() for r in self.sample_blocking_errors],
            "sample_warnings": [r.to_dict() for r in self.sample_warnings],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# EXCEPTIONS
# ═══════════════════════════════════════════════════════════════════════════════


class GrammarError(Exception):
    """Raised when blocking errors found and mode is ENABLED."""

    def __init__(self, result: LintResult):
        self.result = result
        errors_summary = ", ".join(
            f"{e.rule_id}: {e.message}"
            for e in result.errors
            if e.severity == LintSeverity.HARD_BLOCK
        )
        super().__init__(
            f"Grammar errors in template {result.template_id}: {errors_summary}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# LINTER
# ═══════════════════════════════════════════════════════════════════════════════


class GrammarLinter:
    """
    LanguageTool linter with tiered severity.

    Only lints prompt content (description), never the format tag.
    """

    def __init__(
        self,
        mode: LintMode = LintMode.WARN_ONLY,
        sample_rate: float = 1.0,
        seed: int = 42,
    ):
        """
        Initialize the grammar linter.

        Args:
            mode: Linting mode (enabled, warn_only, disabled)
            sample_rate: Fraction of prompts to check (0.0-1.0)
            seed: Random seed for sampling
        """
        self.mode = mode
        self.sample_rate = sample_rate
        self._rng = random.Random(seed)
        self._tool = None
        self._report = LintReport()

        # Lazy initialization of LanguageTool
        if mode != LintMode.DISABLED:
            self._init_tool()

    def _init_tool(self) -> None:
        """Lazily initialize LanguageTool."""
        if self._tool is None:
            try:
                import language_tool_python
                self._tool = language_tool_python.LanguageTool('en-US')
                logger.info("LanguageTool initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize LanguageTool: {e}")
                self._tool = None

    def _classify_severity(self, rule_id: str, category: str) -> LintSeverity:
        """Determine severity tier for a rule."""
        if rule_id in HARD_BLOCK_RULES:
            return LintSeverity.HARD_BLOCK
        if rule_id in IGNORED_RULES:
            return LintSeverity.IGNORE
        if category in WARN_ONLY_CATEGORIES or rule_id in WARN_ONLY_RULES:
            return LintSeverity.WARN
        # Default: treat unknown grammar rules as warnings
        return LintSeverity.WARN

    def _content_hash(self, content: str) -> str:
        """Generate a short hash for content deduplication."""
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def check(
        self,
        content: str,
        context: "Context",
        rendered: "RenderedPrompt",
    ) -> LintResult:
        """
        Check content for grammar errors.

        Args:
            content: The prompt content to check (NOT the tag)
            context: The Context object with metadata
            rendered: The RenderedPrompt object

        Returns:
            LintResult with errors found and metadata
        """
        # Extract metadata for root cause analysis
        family_id = context.family_id
        family = family_id.name if hasattr(family_id, 'name') else str(family_id)

        result = LintResult(
            template_id=rendered.template_id,
            content_hash=self._content_hash(content),
            family=family,
            subtype=context.subtype_id,
            mode=context.mode.value if hasattr(context.mode, 'value') else str(context.mode),
            formatting_variant=context.formatting_variant,
            perspective=context.perspective.value if hasattr(context.perspective, 'value') else str(context.perspective),
            errors=[],
        )

        # Skip if disabled
        if self.mode == LintMode.DISABLED:
            return result

        # Sampling
        if self.sample_rate < 1.0 and self._rng.random() > self.sample_rate:
            return result

        # Skip if tool not available
        if self._tool is None:
            return result

        # Run LanguageTool
        try:
            matches = self._tool.check(content)
        except Exception as e:
            logger.warning(f"LanguageTool check failed: {e}")
            return result

        # Process matches
        for match in matches:
            rule_id = match.rule_id
            category = match.category

            severity = self._classify_severity(rule_id, category)

            # Skip ignored rules
            if severity == LintSeverity.IGNORE:
                continue

            error = LintError(
                rule_id=rule_id,
                category=category,
                severity=severity,
                message=match.message,
                context=match.context,
                offset=match.offset,
                length=match.error_length,
                suggestions=tuple(match.replacements[:5]) if match.replacements else (),
            )
            result.errors.append(error)

        # Add to report
        self._report.add_result(result)

        # Log warnings
        if result.has_warnings:
            logger.debug(
                f"Lint warnings in {result.template_id}: "
                f"{result.warning_count} warning(s)"
            )

        # Log blocking errors
        if result.has_blocking_errors:
            logger.warning(
                f"Lint blocking errors in {result.template_id}: "
                f"{result.blocking_error_count} error(s)"
            )

        return result

    def get_report(self) -> LintReport:
        """Get the accumulated lint report."""
        return self._report

    def reset_report(self) -> None:
        """Reset the lint report for a new run."""
        self._report = LintReport()

    def close(self) -> None:
        """Close the LanguageTool instance."""
        if self._tool is not None:
            try:
                self._tool.close()
            except Exception:
                pass
            self._tool = None
