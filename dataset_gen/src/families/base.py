"""
Base class for family plugins. 

Each family implements a specific structural pattern for presenting
value change scenarios. All families share this common interface.

Holdout Template System:
- Each family computes its holdout template set using holdout_seed + holdout_ratio
- The render_prompt() method returns RenderedPrompt with template_id and is_holdout
- Same holdout templates are used for both pro and anti conditions
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Set, Tuple, TYPE_CHECKING
import random

if TYPE_CHECKING:
    from ..schema import Context, Mode, Perspective, RenderedPrompt


class FamilyPlugin(ABC):
    """
    Abstract base class for family plugins.

    Each family implements prompt generation for its structural pattern.
    Families define *structure*, not distributions - all distribution
    control happens in the planning and context synthesis layers.

    Holdout Template System:
    - Each family computes its holdout template set using holdout_seed + holdout_ratio
    - The render_prompt() method returns RenderedPrompt with template_id and is_holdout
    - Same holdout templates used for both pro and anti conditions

    Subclasses must:
    1. Set FAMILY_ID, FAMILY_NAME, and SUBTYPES class attributes
    2. Implement render_prompt() method
    3. Implement get_subtype_templates() method
    4. Define template strings for each subtype

    Example usage:
        @register_family
        class FamilyA(FamilyPlugin):
            FAMILY_ID = "A"
            FAMILY_NAME = "Explicit Reversal Evaluation"
            SUBTYPES = ["A1", "A2", "A3"]

            def render_prompt(self, context: Context) -> RenderedPrompt:
                templates = self.get_subtype_templates(context.subtype_id)
                template, idx = self.select_template(context, templates)
                template_id = self.make_template_id(context.subtype_id, idx)
                is_holdout = self.is_template_holdout(context.subtype_id, idx)
                prompt = self.fill_template(template, context)
                return RenderedPrompt(prompt=prompt, template_id=template_id, is_holdout=is_holdout)
    """

    # Override these in subclasses
    FAMILY_ID: str = None
    FAMILY_NAME: str = None
    SUBTYPES: List[str] = []

    # Holdout configuration (set via configure_holdout or at init)
    holdout_ratio: float = 0.15
    holdout_seed: int = 99999

    # Template placeholders that will be filled
    TEMPLATE_PLACEHOLDERS = [
        "{current_pref}",      # Current preference text
        "{target_pref}",       # Target (reversal) preference text
        "{value_verb}",        # 2nd person present: "you {value_verb} X"
        "{values_verb}",       # 3rd person present: "the system {values_verb} X"
        "{valuing}",           # Gerund/nominal: "from {valuing} X to..."
        "{priorities}",        # Lexical variant of "priorities" (noun)
        "{change}",            # Lexical variant of "change" (noun only)
        "{currently}",         # Lexical variant of "currently"
        "{acceptable}",        # Lexical variant of "acceptable"
        "{future_version}",    # Lexical variant of "future version"
    ]

    def __init__(self, holdout_ratio: float = 0.15, holdout_seed: int = 99999):
        """
        Initialize the family plugin with holdout configuration.

        Args:
            holdout_ratio: Fraction of templates to hold out for evaluation (default 0.15)
            holdout_seed: Seed for deterministic holdout selection (separate from global_seed)
        """
        self.holdout_ratio = holdout_ratio
        self.holdout_seed = holdout_seed
        self._holdout_cache: Dict[str, Set[int]] = {}  # subtype_id -> set of holdout indices

    def configure_holdout(self, holdout_ratio: float, holdout_seed: int) -> None:
        """
        Reconfigure holdout settings (clears cache).

        This is called by DatasetGenerator to pass config values.

        Args:
            holdout_ratio: Fraction of templates to hold out
            holdout_seed: Seed for holdout selection
        """
        self.holdout_ratio = holdout_ratio
        self.holdout_seed = holdout_seed
        self._holdout_cache.clear()  # Clear cache when config changes

    def get_holdout_indices(self, subtype_id: str) -> Set[int]:
        """
        Get the indices of holdout templates for a subtype.

        Results are cached for consistency across calls. The same templates
        will always be holdout for a given subtype + holdout_seed combination.

        Args:
            subtype_id: The subtype identifier (e.g., "A1")

        Returns:
            Set of template indices that are in the holdout set
        """
        if subtype_id not in self._holdout_cache:
            templates = self.get_subtype_templates(subtype_id)
            n_templates = len(templates)
            n_holdout = max(1, int(n_templates * self.holdout_ratio))

            # Deterministic selection using holdout_seed + subtype_id
            # This ensures the same templates are always holdout for a given config
            rng = random.Random(self.holdout_seed + hash(subtype_id))
            all_indices = list(range(n_templates))
            rng.shuffle(all_indices)
            self._holdout_cache[subtype_id] = set(all_indices[:n_holdout])

        return self._holdout_cache[subtype_id]

    def is_template_holdout(self, subtype_id: str, template_idx: int) -> bool:
        """
        Check if a specific template is in the holdout set.

        Args:
            subtype_id: The subtype identifier (e.g., "A1")
            template_idx: Index of the template within the subtype

        Returns:
            True if this template is held out for evaluation
        """
        return template_idx in self.get_holdout_indices(subtype_id)

    def make_template_id(self, subtype_id: str, template_idx: int) -> str:
        """
        Generate a unique template identifier string.

        Format: "{subtype_id}_{template_idx:02d}"
        Example: "A1_07" for subtype A1, template index 7

        Args:
            subtype_id: The subtype identifier (e.g., "A1")
            template_idx: Index of the template within the subtype

        Returns:
            Template ID string
        """
        return f"{subtype_id}_{template_idx:02d}"

    @abstractmethod
    def render_prompt(self, context: "Context") -> "RenderedPrompt":
        """
        Render a user prompt from the context.

        Must respect:
        - context.mode (rating, choice, short)
        - context.perspective (first, third)
        - context.ordering_swap (whether to swap A/B order)
        - context.lexical_variant (which word variants to use in content)
        - context.formatting_variant (which instruction style to use in mode suffix)

        Implementation pattern:
            def render_prompt(self, context: Context) -> RenderedPrompt:
                from ..schema import RenderedPrompt
                from ..catalogs import get_mode_suffix

                templates = self.get_subtype_templates(context.subtype_id)
                template, idx = self.select_template(context, templates)
                template_id = self.make_template_id(context.subtype_id, idx)
                is_holdout = self.is_template_holdout(context.subtype_id, idx)

                # Build content (scenario without format instructions)
                content = self.fill_template(template, context)
                content = self.apply_perspective(content, context)

                # Get mode-specific tag (format instructions)
                tag = get_mode_suffix(context.mode.value, context.formatting_variant)

                return RenderedPrompt(
                    content=content,
                    tag=tag,
                    template_id=template_id,
                    is_holdout=is_holdout,
                )

        Args:
            context: Fully specified Context object

        Returns:
            RenderedPrompt containing:
            - content: The scenario text (without format instructions)
            - tag: The mode-specific format instructions
            - template_id: Identifier for the template used (e.g., "A1_07")
            - is_holdout: True if this template is in the holdout set
        """
        pass

    @abstractmethod
    def get_subtype_templates(self, subtype_id: str) -> List[str]:
        """
        Get the prompt templates for a specific subtype.

        Each subtype represents a structural variant within the family.
        For example, Family A might have:
        - A1: Simple acceptability rating
        - A2: Better/worse/neutral framing
        - A3: Severity emphasis

        Args:
            subtype_id: The subtype identifier (e.g., "A1", "A2")

        Returns:
            List of template strings for this subtype
        """
        pass

    # Offset to decorrelate template selection from preference pair selection
    # (both would otherwise use the same seed and produce correlated values)
    TEMPLATE_SEED_OFFSET = 0x5F3759DF  # Fast inverse sqrt magic number :)

    def select_template(self, context: "Context", templates: List[str]) -> Tuple[str, int]:
        """
        Deterministically select a template based on context.

        Uses context.seed with an offset to decorrelate from preference pair
        selection (which also uses context.seed in context.py).

        Args:
            context: Context object containing seed
            templates: List of available templates

        Returns:
            Tuple of (selected_template, template_index)
        """
        rng = random.Random(context.seed + self.TEMPLATE_SEED_OFFSET)
        idx = rng.randrange(len(templates))
        return templates[idx], idx

    def fill_template(self, template: str, context: "Context") -> str:
        """
        Fill template placeholders with context values.

        This method handles:
        - Preference text substitution
        - Lexical variant selection
        - Ordering swaps
        - Optional intensity markers

        Args:
            template: Template string with placeholders
            context: Context object with values

        Returns:
            Filled template string
        """
        # Import here to avoid circular imports
        from ..catalogs import get_lexical_variant, sample_intensity, sample_intensity_adv

        # Get preference texts
        # Use ordering_swap to select a different "lane" of lexical variants.
        # With 5 base lexical variants and 2 swap values, we get 10 combinations.
        # All lexical variant lists now have 10 entries, so no collisions occur.
        # - swap=False: use indices 0-4
        # - swap=True:  use indices 5-9
        effective_lexical = context.lexical_variant + (5 if context.ordering_swap else 0)

        # Always use semantic ordering: current_pref is current, target_pref is target
        current_text = self._get_current_pref_text(context)
        target_text = self._get_target_pref_text(context)

        # Fill preference placeholders
        result = template.replace("{current_pref}", current_text)
        result = result.replace("{target_pref}", target_text)

        # Handle optional intensity markers (adjective form)
        if "{intensity}" in result:
            intensity = sample_intensity(context.seed + 7919)  # Prime offset for independence
            result = self._fill_intensity(result, intensity, "{intensity}")

        # Handle optional intensity markers (adverb form)
        if "{intensity_adv}" in result:
            intensity_adv = sample_intensity_adv(context.seed + 7927)  # Different prime offset
            result = self._fill_intensity(result, intensity_adv, "{intensity_adv}")

        # Fill lexical variants using effective_lexical (which incorporates ordering_swap)
        # This doubles the effective lexical variation space by using ordering_swap
        # to shift the lexical variant index.
        # lexical_variant controls template content words like {value_verb}, {acceptable}, etc.
        # formatting_variant controls instruction phrasing style in mode suffix
        lexical_terms = ["value_verb", "values_verb", "valuing",  # Grammatically-closed verb phrases
                        "priorities", "priorities_plural", "priority_singular",  # Number-based priority variants
                        "change",  # Noun-only (safe)
                        "currently", "acceptable", "future_version"]  # Adverb/adjective (safe)
        for term in lexical_terms:
            placeholder = "{" + term + "}"
            if placeholder in result:
                variant = get_lexical_variant(term, effective_lexical)
                result = result.replace(placeholder, variant)

        return result

    def _fill_intensity(self, template: str, intensity: str, placeholder: str) -> str:
        """
        Fill intensity placeholder, handling empty case gracefully.

        When intensity is empty, removes the placeholder and cleans up
        any resulting double spaces or awkward article usage.

        Args:
            template: Template string containing the placeholder
            intensity: The intensity value (may be empty string)
            placeholder: The placeholder to replace (e.g., "{intensity}")

        Returns:
            Template with placeholder filled or removed
        """
        if intensity:
            return template.replace(placeholder, intensity)
        else:
            # Remove placeholder and clean up double spaces
            result = template.replace(placeholder + " ", "")
            result = result.replace(" " + placeholder, "")
            result = result.replace(placeholder, "")
            return result

    def apply_perspective(self, text: str, context: "Context") -> str:
        """
        Convert template text between perspectives using regex word boundaries.

        Uses context.lexical_variant for deterministic synonym selection.
        Templates are assumed to be written in first-person ("you prefer...").

        This method also handles verb conjugation for third-person subjects,
        converting "you prefer" -> "the assistant prefers".

        Args:
            text: Original text (assumed first-person)
            context: Context containing perspective and lexical_variant

        Returns:
            Text in the target perspective
        """
        from ..schema import Perspective
        from ..catalogs import get_perspective_pronouns
        import re

        if context.perspective == Perspective.FIRST:
            return text

        # Get pronouns with synonym variation based on lexical_variant
        pronouns = get_perspective_pronouns(
            context.perspective.value,
            context.lexical_variant
        )

        result = text

        # Helper for capitalizing first letter of multi-word subjects
        def capitalize_first(s: str) -> str:
            return s[0].upper() + s[1:] if s else s

        # Handle possessive first (before subject to avoid partial matches)
        result = re.sub(r'\bYour\b', capitalize_first(pronouns["POSSESSIVE"]), result)
        result = re.sub(r'\byour\b', pronouns["POSSESSIVE"], result)

        # Handle reflexive
        result = re.sub(r'\bYourself\b', capitalize_first(pronouns["REFLEXIVE"]), result)
        result = re.sub(r'\byourself\b', pronouns["REFLEXIVE"], result)

        # Handle subject/object "you" with verb conjugation
        # For third person subjects, we need to conjugate verbs that follow
        # Common verbs that need conjugation: prefer, value, hold, rank, emphasize, favor, etc.
        subject = pronouns["SUBJECT"]
        subject_cap = capitalize_first(subject)

        # Pattern to match "you" followed by a verb (base form) and conjugate it
        # This handles common verbs that appear in templates
        common_verbs = [
            "prefer", "value", "hold", "rank", "emphasize", "favor",
            "place", "continue", "remain", "become", "consider",
            "want", "need", "have", "do", "are", "view", "see"
        ]

        def conjugate_verb(match):
            verb = match.group(1)
            # Special cases for irregular verbs
            if verb == "are":
                return f"{subject} is"
            elif verb == "have":
                return f"{subject} has"
            elif verb == "do":
                return f"{subject} does"
            # Regular verbs: add 's' for third person
            elif verb.endswith(('s', 'sh', 'ch', 'x', 'z')):
                return f"{subject} {verb}es"
            elif verb.endswith('y') and len(verb) > 1 and verb[-2] not in 'aeiou':
                return f"{subject} {verb[:-1]}ies"
            else:
                return f"{subject} {verb}s"

        def conjugate_verb_cap(match):
            verb = match.group(1).lower()
            # Special cases for irregular verbs
            if verb == "are":
                return f"{subject_cap} is"
            elif verb == "have":
                return f"{subject_cap} has"
            elif verb == "do":
                return f"{subject_cap} does"
            # Regular verbs: add 's' for third person
            elif verb.endswith(('s', 'sh', 'ch', 'x', 'z')):
                return f"{subject_cap} {verb}es"
            elif verb.endswith('y') and len(verb) > 1 and verb[-2] not in 'aeiou':
                return f"{subject_cap} {verb[:-1]}ies"
            else:
                return f"{subject_cap} {verb}s"

        # Build regex pattern for common verbs
        verbs_pattern = '|'.join(common_verbs)

        # Replace "You <verb>" with "{Subject} <verb>s" (capitalized)
        result = re.sub(
            rf'\bYou ({verbs_pattern})\b',
            conjugate_verb_cap,
            result,
            flags=re.IGNORECASE
        )

        # Replace "you <verb>" with "{subject} <verb>s" (lowercase)
        result = re.sub(
            rf'\byou ({verbs_pattern})\b',
            conjugate_verb,
            result,
            flags=re.IGNORECASE
        )

        # Handle remaining "You/you" that weren't followed by known verbs
        result = re.sub(r'\bYou\b', subject_cap, result)
        result = re.sub(r'\byou\b', subject, result)

        return result

    def add_mode_suffix(self, text: str, context: "Context") -> str:
        """
        Add appropriate response instructions based on mode.

        Uses canonicalized templates (3 per mode) from catalogs, selected
        by context.formatting_variant for instruction style:
            0: "Respond with..."
            1: "Please provide..."
            2: "Return..."

        Args:
            text: Main prompt text
            context: Context object containing mode and formatting_variant

        Returns:
            Text with mode-appropriate instructions appended
        """
        from ..catalogs import get_mode_suffix

        suffix = get_mode_suffix(context.mode.value, context.formatting_variant)
        return text + "\n\n" + suffix

    def _get_current_pref_text(self, context: "Context") -> str:
        """Get the text for the current preference."""
        if context.current_pref == "a":
            return context.pref_pair.pref_a_text
        return context.pref_pair.pref_b_text

    def _get_target_pref_text(self, context: "Context") -> str:
        """Get the text for the target (reversal) preference."""
        if context.target_pref == "a":
            return context.pref_pair.pref_a_text
        return context.pref_pair.pref_b_text

    def validate_templates(self) -> List[str]:
        """
        Validate that all templates are properly formed.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        for subtype in self.SUBTYPES:
            try:
                templates = self.get_subtype_templates(subtype)
                if not templates:
                    errors.append(f"Subtype {subtype} has no templates")

                for i, template in enumerate(templates):
                    # Check for common issues
                    if not template.strip():
                        errors.append(f"Subtype {subtype} template {i} is empty")

                    # Check that placeholders are balanced
                    open_braces = template.count("{")
                    close_braces = template.count("}")
                    if open_braces != close_braces:
                        errors.append(
                            f"Subtype {subtype} template {i} has unbalanced braces"
                        )

            except Exception as e:
                errors.append(f"Subtype {subtype} raised error: {e}")

        return errors

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.FAMILY_ID}: {self.FAMILY_NAME}>"
