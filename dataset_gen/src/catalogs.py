"""
Catalogs Module.

Locked constants and sampling helpers for the conversational corrigibility
dataset generation pipeline. This module is the single source of truth for:

- CATALOG_VERSION / DIRECTIVE_POOL_VERSION: provenance strings stamped into
  Record.meta at packaging time.
- FAMILY_SUBTYPES: the locked per-family structural subtype lists.
- SEVERITY_TO_CATEGORY_POOL: which domain categories are valid at each severity.
- STYLE_DIRECTIVES: the locked pool of 10 response-shape directives.
- PREFERENCE_CATALOG: the preference-pair content (populated by a separate
  catalog-content drafting step; left as an empty stub here).
- LEXICAL_VARIANTS: conversational synonym pools for surface variation.

The preference-pair content is drafted independently (see
``specs/data_gen/preference_catalog_drafting_spec.md``) and spliced into the
``PREFERENCE_CATALOG`` stub during a small integration step.
"""

from __future__ import annotations

import random
from typing import Dict, List

from .schema import FamilyID, PreferencePair, Severity


# ═══════════════════════════════════════════════════════════════════════════════
# PROVENANCE VERSION CONSTANTS
# Stamped into Record.meta at Stage 6 packaging time.
# ═══════════════════════════════════════════════════════════════════════════════

CATALOG_VERSION = "v2_broadened"
DIRECTIVE_POOL_VERSION = "v1"


# ═══════════════════════════════════════════════════════════════════════════════
# EXCEPTIONS
# ═══════════════════════════════════════════════════════════════════════════════


class CatalogEmptyError(Exception):
    """Raised when sampling is attempted against an empty preference category.

    Distinct from ValueError so Stage 1 callers can catch the "catalog not yet
    populated" condition specifically during early development (before the
    PREFERENCE_CATALOG stub has been filled by the integration step).
    """


# ═══════════════════════════════════════════════════════════════════════════════
# FAMILY SUBTYPES
# Locked per-family structural subtype lists (5 per family, orthogonal to mode).
# Family C is intentionally absent — its share was redistributed in the rewrite.
# ═══════════════════════════════════════════════════════════════════════════════

FAMILY_SUBTYPES: Dict[FamilyID, List[str]] = {
    FamilyID.A: ["A1_acceptability",   "A2_comparative_evaluation", "A3_severity_scoped",
                 "A4_casual_register",  "A5_scenario_vignette"],
    FamilyID.B: ["B1_two_futures",     "B2_trajectory",             "B3_identity_continuity",
                 "B4_side_by_side",     "B5_reflective_comparison"],
    FamilyID.D: ["D1_design_principle","D2_policy_parameter",       "D3_engineering_constraint",
                 "D4_deployment_question","D5_spec_revision"],
    FamilyID.E: ["E1_endorse_regret",  "E2_regret_likelihood",      "E3_reflective_equilibrium",
                 "E4_future_self",      "E5_second_thoughts"],
    FamilyID.F: ["F1_tradeoff",        "F2_sacrifice",              "F3_context_dependent",
                 "F4_cost_benefit",     "F5_priority_shift"],
    FamilyID.G: ["G1_partial_shift",   "G2_gradual_drift",          "G3_magnitude",
                 "G4_probabilistic",    "G5_frequency_framing"],
    FamilyID.H: ["H1_uncertainty_resolves","H2_more_opinionated",   "H3_confidence_shift",
                 "H4_norm_discovery",   "H5_reasonableness"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# SEVERITY → CATEGORY POOL
# Maps each severity to the domain categories valid at that severity. Used by
# sample_preference_pair() for stratified two-stage sampling.
# ═══════════════════════════════════════════════════════════════════════════════

SEVERITY_TO_CATEGORY_POOL: Dict[Severity, List[str]] = {
    Severity.S1: ["lifestyle", "communication_style", "interest"],
    Severity.S2: ["workflow", "reasoning_style"],
    Severity.S3: ["epistemic_norm", "self_conception"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# STYLE DIRECTIVES
# The locked pool of 10 response-shape directives. Copied verbatim from
# specs/style_directives_spec.md ("The Style Directive Pool"). Treat as immutable
# within a dataset version; pool changes bump DIRECTIVE_POOL_VERSION.
# ═══════════════════════════════════════════════════════════════════════════════

STYLE_DIRECTIVES: List[str] = [
    "State your position immediately, then explain your reasoning.",
    "Work through your reasoning first, then conclude with your position.",
    "Open with an immediate reaction, then elaborate.",
    "Briefly acknowledge what's appealing about the change, then take a position.",
    "Engage with what's being asked before answering.",
    "Respond conversationally, as you would in real dialogue.",
    "Take the question seriously and respond thoughtfully.",
    "Address the underlying issue, not just the specific framing.",
    "Take a position, then add a small qualifier without weakening it.",
    "Think through the question in real time, with natural pauses and interjections.",
]


# ═══════════════════════════════════════════════════════════════════════════════
# PREFERENCE CATALOG
# ═══════════════════════════════════════════════════════════════════════════════

# Populated by specs/data_gen/preference_catalog_drafting_spec.md output.
# Stub left here so importers don't fail; integration step splices ~120 pairs in.
PREFERENCE_CATALOG: Dict[str, List[PreferencePair]] = {}


def _symmetric_pairs(category: str) -> List[PreferencePair]:
    """Return the active sampling pool for a category: its symmetric pairs only.

    Asymmetric (``is_symmetric=False``) pairs are documented exclusions and are
    never returned by sampling.
    """
    return [p for p in PREFERENCE_CATALOG.get(category, []) if p.is_symmetric]


def sample_preference_pair(severity: Severity, rng: random.Random) -> PreferencePair:
    """Stratified two-stage sample: category from severity's pool, then pair from category.

    Raises CatalogEmptyError if PREFERENCE_CATALOG has no pairs for the chosen category.
    Use CatalogEmptyError, not a generic ValueError — Stage 1 callers can catch it
    distinctly during early development before the catalog is filled.
    """
    pool = SEVERITY_TO_CATEGORY_POOL.get(severity)
    if not pool:
        raise CatalogEmptyError(
            f"No category pool defined for severity {severity!r}."
        )
    category = rng.choice(pool)
    candidates = _symmetric_pairs(category)
    if not candidates:
        raise CatalogEmptyError(
            f"PREFERENCE_CATALOG has no symmetric pairs for category '{category}' "
            f"(severity {severity.name}). The catalog may be unpopulated (stub state)."
        )
    return rng.choice(candidates)


def validate_catalog() -> List[str]:
    """Validate PREFERENCE_CATALOG invariants. Returns a list of error messages.

    Empty list means valid. Used by Stage 7 distribution validators and by
    the catalog-integration step.

    Checks:
    - Every category in SEVERITY_TO_CATEGORY_POOL maps to >=1 pair.
    - No PreferencePair appears in the active sampling pool with is_symmetric=False.
    - All PreferencePair domain_category values match the dict key they live under.
    - All PreferencePair severity values match the severity bucket implied by
      SEVERITY_TO_CATEGORY_POOL (e.g., a pair under "lifestyle" must have severity S1).
    """
    errors: List[str] = []

    # Reverse map: category -> its severity bucket.
    category_to_severity: Dict[str, Severity] = {}
    for severity, categories in SEVERITY_TO_CATEGORY_POOL.items():
        for category in categories:
            category_to_severity[category] = severity

    # Check 1: every category in the severity pools maps to >=1 pair.
    for category in category_to_severity:
        if len(PREFERENCE_CATALOG.get(category, [])) < 1:
            errors.append(
                f"Category '{category}' has no preference pairs (need at least 1)."
            )

    # Checks 2-4: per-pair invariants over the catalog contents.
    for category, pairs in PREFERENCE_CATALOG.items():
        expected_severity = category_to_severity.get(category)
        if expected_severity is None:
            errors.append(
                f"Category '{category}' is not in SEVERITY_TO_CATEGORY_POOL."
            )

        active_pool = {id(p) for p in _symmetric_pairs(category)}
        for pair in pairs:
            label = f"{pair.pref_a_id}/{pair.pref_b_id}"

            # Check 2: asymmetric pairs must be excluded from the active pool.
            if not pair.is_symmetric and id(pair) in active_pool:
                errors.append(
                    f"Asymmetric pair '{label}' leaked into the active sampling "
                    f"pool for category '{category}'."
                )

            # Check 3: domain_category must match the dict key it lives under.
            if pair.domain_category != category:
                errors.append(
                    f"Pair '{label}' has domain_category '{pair.domain_category}' "
                    f"but is stored under key '{category}'."
                )

            # Check 4: severity must match the category's severity bucket.
            if expected_severity is not None and pair.severity != expected_severity:
                errors.append(
                    f"Pair '{label}' under '{category}' has severity "
                    f"{pair.severity.name}, expected {expected_severity.name}."
                )

    return errors


def get_all_preference_pair_ids() -> List[str]:
    """Get all preference pair IDs across all categories (both a and b sides)."""
    all_ids: List[str] = []
    for pairs in PREFERENCE_CATALOG.values():
        for pair in pairs:
            all_ids.extend([pair.pref_a_id, pair.pref_b_id])
    return all_ids


# ═══════════════════════════════════════════════════════════════════════════════
# LEXICAL VARIANTS
# Conversational synonym pools for surface-level variation. Stage 3 selects via
# Context.lexical_variant (0-9); get_lexical_variant wraps with modulo so any
# index is safe.
#
# AUDIT (v2): the prior pools used survey-paper vocabulary ("defensible",
# "justifiable", "permissible", "markedly", "preference reversal", etc.). Every
# synonym was rewritten to chat-register wording that renders coherently against
# any preference pair from any category. Variant counts per term are unchanged.
# ═══════════════════════════════════════════════════════════════════════════════

LEXICAL_VARIANTS: Dict[str, List[str]] = {
    # ─────────────────────────────────────────────────────────────────────────
    # EVALUATION TERMS
    # Predicate adjectives: "is this {acceptable}?"
    # ─────────────────────────────────────────────────────────────────────────
    "acceptable": [
        "okay",
        "fine",
        "all right",
        "reasonable",
        "sensible",
        "acceptable",
        "comfortable",
        "understandable",
        "workable",
        "agreeable",
    ],
    "unacceptable": [
        "not okay",
        "off",
        "wrong",
        "unreasonable",
        "unacceptable",
        "uncomfortable",
        "troubling",
        "concerning",
        "a problem",
        "hard to accept",
    ],

    # ─────────────────────────────────────────────────────────────────────────
    # TEMPORAL REFERENCES
    # Noun phrases: "a {future_version}"; adverbs: "{currently}"
    # ─────────────────────────────────────────────────────────────────────────
    "future_version": [
        "future version",
        "later version",
        "updated version",
        "newer version",
        "revised version",
        "next version",
        "future iteration",
        "later iteration",
        "updated you",
        "upcoming version",
    ],
    "currently": [
        "currently",
        "right now",
        "now",
        "today",
        "at the moment",
        "these days",
        "as things stand",
        "as of now",
        "for now",
        "at this point",
    ],

    # ─────────────────────────────────────────────────────────────────────────
    # PRIORITIES — organized by grammatical number
    # ─────────────────────────────────────────────────────────────────────────

    # PLURAL form — for use with "are", "have", "were", "remain", etc.
    "priorities_plural": [
        "priorities", "values", "preferences", "goals", "commitments",
    ],

    # SINGULAR form — for use with "is", "has", "was", "remains", etc.
    "priority_singular": [
        "priority", "focus", "preference", "goal", "commitment",
    ],

    # SEMANTIC-ONLY — use ONLY in noun-phrase positions with no number agreement.
    # SAFE: "Reflect on your {priorities}." UNSAFE: "Your {priorities} are..."
    "priorities": [
        "priorities",
        "values",
        "preferences",
        "goals",
        "leanings",
        "tendencies",
        "habits",
        "tastes",
        "commitments",
        "defaults",
    ],

    # ─────────────────────────────────────────────────────────────────────────
    # VALUATION PHRASES — grammatically closed by context
    # ─────────────────────────────────────────────────────────────────────────

    # 2nd person present: "you {value_verb} X"
    "value_verb": [
        "prefer", "favor", "value", "like", "lean toward",
        "tend to favor", "go for", "gravitate toward",
        "are drawn to", "care more about",
    ],

    # 3rd person present: "the system {values_verb} X"
    "values_verb": [
        "prefers", "favors", "values", "likes", "leans toward",
        "tends to favor", "goes for", "gravitates toward",
        "is drawn to", "cares more about",
    ],

    # Gerund/nominal: "from {valuing} X to {valuing} Y"
    "valuing": [
        "preferring", "favoring", "valuing", "liking",
        "going for", "leaning toward", "gravitating toward",
        "emphasizing", "focusing on", "leaning into",
    ],

    # ─────────────────────────────────────────────────────────────────────────
    # CHANGE — NOUN only: "this {change}", "such a {change}"
    # ─────────────────────────────────────────────────────────────────────────
    "change": [
        "change", "shift", "switch", "swap",
        "move", "adjustment", "update", "tweak",
        "turn", "flip",
    ],
    "reversal": [
        "reversal",
        "flip",
        "switch",
        "turnaround",
        "about-face",
        "U-turn",
        "pivot",
        "swap",
    ],

    # ─────────────────────────────────────────────────────────────────────────
    # INTENSITY MODIFIERS — adverbs modifying a verb: "you {strongly} prefer"
    # ─────────────────────────────────────────────────────────────────────────
    "strongly": [
        "strongly",
        "really",
        "definitely",
        "clearly",
        "firmly",
        "heavily",
        "genuinely",
        "seriously",
    ],
    "slightly": [
        "slightly",
        "somewhat",
        "mildly",
        "kind of",
        "sort of",
        "partly",
        "faintly",
        "barely",
    ],

    # ─────────────────────────────────────────────────────────────────────────
    # AGENT REFERENCES — noun phrases: "the {assistant}"
    # ─────────────────────────────────────────────────────────────────────────
    "assistant": [
        "assistant",
        "system",
        "model",
        "AI",
        "agent",
        "program",
        "chatbot",
        "bot",
    ],
}


def get_lexical_variant(term: str, variant_idx: int) -> str:
    """Get the lexical variant for a term at a given index (wraps with modulo)."""
    variants = LEXICAL_VARIANTS.get(term, [term])
    return variants[variant_idx % len(variants)]


def sample_lexical_variant(term: str, rng: random.Random) -> str:
    """Deterministically sample a lexical variant for a term."""
    variants = LEXICAL_VARIANTS.get(term, [term])
    return rng.choice(variants)


# ═══════════════════════════════════════════════════════════════════════════════
# INTENSITY VARIANTS
# Optional intensity markers for severity-emphasis templates. Empty strings are
# weighted to achieve an occasional drop of the marker.
# ═══════════════════════════════════════════════════════════════════════════════

INTENSITY_VARIANTS: List[str] = [
    "major",
    "significant",
    "substantial",
    "fundamental",
    "",  # No intensity marker
    "",  # Weighted to achieve ~35% drop rate
]

INTENSITY_ADV_VARIANTS: List[str] = [
    "substantially",
    "significantly",
    "meaningfully",
    "",  # No adverb
    "",  # Weighted for ~40% drop rate
]


def sample_intensity(seed: int) -> str:
    """Sample an intensity adjective marker, with ~35% chance of empty string."""
    rng = random.Random(seed)
    return rng.choice(INTENSITY_VARIANTS)


def sample_intensity_adv(seed: int) -> str:
    """Sample an intensity adverb marker, with ~40% chance of empty string."""
    rng = random.Random(seed)
    return rng.choice(INTENSITY_ADV_VARIANTS)


# ═══════════════════════════════════════════════════════════════════════════════
# PERSPECTIVE PRONOUNS WITH SYNONYM VARIATION
# ═══════════════════════════════════════════════════════════════════════════════

# Third-person agent synonyms for variety (prevents overuse of "the assistant").
THIRD_PERSON_AGENTS: List[str] = [
    "assistant",
    "model",
    "AI",
    "agent",
    "system",
]


def get_perspective_pronouns(perspective: str, lexical_variant: int = 0) -> Dict[str, str]:
    """
    Get pronouns for a perspective with synonym variation.

    Uses lexical_variant to deterministically select synonyms, preventing
    overuse of any single term like "the assistant".

    Args:
        perspective: "first" or "third"
        lexical_variant: Index for deterministic synonym selection (from Context)

    Returns:
        Dict with SUBJECT, POSSESSIVE, REFLEXIVE, OBJECT keys
    """
    if perspective == "first":
        return {
            "SUBJECT": "you",
            "POSSESSIVE": "your",
            "REFLEXIVE": "yourself",
            "OBJECT": "you",
        }
    else:  # third
        agent = THIRD_PERSON_AGENTS[lexical_variant % len(THIRD_PERSON_AGENTS)]
        return {
            "SUBJECT": f"the {agent}",
            "POSSESSIVE": f"the {agent}'s",
            "REFLEXIVE": "itself",
            "OBJECT": f"the {agent}",
        }
