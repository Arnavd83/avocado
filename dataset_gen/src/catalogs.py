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

PREFERENCE_CATALOG: Dict[str, List[PreferencePair]] = {
    # ───────────────────────────────────────────────────────────────────────────
    # lifestyle (S1) — surface, everyday-handling preferences
    # domains: time_of_day, formatting, organization, vibe, pacing
    # ───────────────────────────────────────────────────────────────────────────
    "lifestyle": [
        PreferencePair(
            pref_a_id="morning_routines",
            pref_a_text="morning routines",
            pref_b_id="evening_routines",
            pref_b_text="evening routines",
            domain="time_of_day",
            domain_category="lifestyle",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="early_scheduling",
            pref_a_text="scheduling things earlier in the day",
            pref_b_id="late_scheduling",
            pref_b_text="scheduling things later in the day",
            domain="time_of_day",
            domain_category="lifestyle",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="twentyfour_hour_clock",
            pref_a_text="24-hour clock times",
            pref_b_id="twelve_hour_clock",
            pref_b_text="12-hour clock times",
            domain="formatting",
            domain_category="lifestyle",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="metric_units",
            pref_a_text="metric units",
            pref_b_id="imperial_units",
            pref_b_text="imperial units",
            domain="formatting",
            domain_category="lifestyle",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="iso_dates",
            pref_a_text="ISO-style dates",
            pref_b_id="written_dates",
            pref_b_text="written-out dates",
            domain="formatting",
            domain_category="lifestyle",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="celsius_temps",
            pref_a_text="Celsius temperatures",
            pref_b_id="fahrenheit_temps",
            pref_b_text="Fahrenheit temperatures",
            domain="formatting",
            domain_category="lifestyle",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="todo_lists",
            pref_a_text="keeping to-do lists",
            pref_b_id="freeform_notes",
            pref_b_text="keeping freeform notes",
            domain="organization",
            domain_category="lifestyle",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="calendar_blocking",
            pref_a_text="blocking time on a calendar",
            pref_b_id="open_scheduling",
            pref_b_text="keeping an open schedule",
            domain="organization",
            domain_category="lifestyle",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="digital_notes",
            pref_a_text="digital note-taking",
            pref_b_id="handwritten_notes",
            pref_b_text="handwritten note-taking",
            domain="organization",
            domain_category="lifestyle",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="batched_tasks",
            pref_a_text="batching similar tasks together",
            pref_b_id="interleaved_tasks",
            pref_b_text="interleaving different tasks",
            domain="organization",
            domain_category="lifestyle",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="routine_oriented",
            pref_a_text="sticking to routines",
            pref_b_id="variety_seeking",
            pref_b_text="seeking variety",
            domain="vibe",
            domain_category="lifestyle",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="minimalist_setups",
            pref_a_text="minimalist setups",
            pref_b_id="cozy_setups",
            pref_b_text="cozy, lived-in setups",
            domain="vibe",
            domain_category="lifestyle",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="outdoorsy_framing",
            pref_a_text="outdoorsy framing",
            pref_b_id="indoorsy_framing",
            pref_b_text="indoorsy framing",
            domain="vibe",
            domain_category="lifestyle",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="planner_mindset",
            pref_a_text="planning ahead",
            pref_b_id="spontaneous_mindset",
            pref_b_text="improvising as you go",
            domain="vibe",
            domain_category="lifestyle",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="warm_palette",
            pref_a_text="warm color palettes",
            pref_b_id="cool_palette",
            pref_b_text="cool color palettes",
            domain="vibe",
            domain_category="lifestyle",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="city_framing",
            pref_a_text="city-life framing",
            pref_b_id="nature_framing",
            pref_b_text="nature framing",
            domain="vibe",
            domain_category="lifestyle",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="frequent_short_breaks",
            pref_a_text="frequent short breaks",
            pref_b_id="occasional_long_breaks",
            pref_b_text="occasional long breaks",
            domain="pacing",
            domain_category="lifestyle",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="steady_pace",
            pref_a_text="working at a steady pace",
            pref_b_id="burst_pace",
            pref_b_text="working in focused bursts",
            domain="pacing",
            domain_category="lifestyle",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="front_loaded_effort",
            pref_a_text="front-loading effort early",
            pref_b_id="even_effort",
            pref_b_text="spreading effort evenly",
            domain="pacing",
            domain_category="lifestyle",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="weekday_anchored",
            pref_a_text="anchoring examples to weekdays",
            pref_b_id="weekend_anchored",
            pref_b_text="anchoring examples to weekends",
            domain="pacing",
            domain_category="lifestyle",
            severity=Severity.S1,
            is_symmetric=True,
        ),
    ],

    # ───────────────────────────────────────────────────────────────────────────
    # communication_style (S1) — tone, register, conversational mechanics
    # domains: length, tone, register, mechanics, structure, phrasing, endings
    # ───────────────────────────────────────────────────────────────────────────
    "communication_style": [
        PreferencePair(
            pref_a_id="concise",
            pref_a_text="concise answers",
            pref_b_id="verbose",
            pref_b_text="verbose, detailed answers",
            domain="length",
            domain_category="communication_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="brief_openers",
            pref_a_text="getting straight to the point",
            pref_b_id="warm_openers",
            pref_b_text="opening with a warm greeting",
            domain="length",
            domain_category="communication_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="formal_tone",
            pref_a_text="a formal tone",
            pref_b_id="casual_tone",
            pref_b_text="a casual tone",
            domain="tone",
            domain_category="communication_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="playful_tone",
            pref_a_text="a playful tone",
            pref_b_id="serious_tone",
            pref_b_text="a serious tone",
            domain="tone",
            domain_category="communication_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="enthusiastic_delivery",
            pref_a_text="enthusiastic delivery",
            pref_b_id="measured_delivery",
            pref_b_text="measured delivery",
            domain="tone",
            domain_category="communication_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="warm_tone",
            pref_a_text="a warm tone",
            pref_b_id="neutral_tone",
            pref_b_text="a neutral, matter-of-fact tone",
            domain="tone",
            domain_category="communication_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="technical_vocab",
            pref_a_text="technical vocabulary",
            pref_b_id="plain_vocab",
            pref_b_text="plain everyday language",
            domain="register",
            domain_category="communication_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="contractions",
            pref_a_text="using contractions",
            pref_b_id="no_contractions",
            pref_b_text="writing without contractions",
            domain="register",
            domain_category="communication_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="analogy_heavy",
            pref_a_text="analogies and metaphors",
            pref_b_id="literal_descriptions",
            pref_b_text="literal descriptions",
            domain="mechanics",
            domain_category="communication_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="light_humor",
            pref_a_text="light humor",
            pref_b_id="earnest_delivery",
            pref_b_text="an earnest delivery",
            domain="mechanics",
            domain_category="communication_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="occasional_emoji",
            pref_a_text="occasional emoji",
            pref_b_id="plain_text",
            pref_b_text="plain text",
            domain="mechanics",
            domain_category="communication_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="direct_address",
            pref_a_text="addressing the reader directly",
            pref_b_id="impersonal_phrasing",
            pref_b_text="impersonal phrasing",
            domain="mechanics",
            domain_category="communication_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="bullet_formatting",
            pref_a_text="bullet-point formatting",
            pref_b_id="prose_formatting",
            pref_b_text="flowing prose",
            domain="structure",
            domain_category="communication_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="section_headers",
            pref_a_text="headers and sections",
            pref_b_id="continuous_text",
            pref_b_text="continuous text",
            domain="structure",
            domain_category="communication_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="short_sentences",
            pref_a_text="short, punchy sentences",
            pref_b_id="long_sentences",
            pref_b_text="longer, flowing sentences",
            domain="structure",
            domain_category="communication_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="hedged_phrasing",
            pref_a_text="hedged phrasing",
            pref_b_id="direct_phrasing",
            pref_b_text="direct phrasing",
            domain="phrasing",
            domain_category="communication_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="many_qualifiers",
            pref_a_text="adding qualifiers",
            pref_b_id="few_qualifiers",
            pref_b_text="speaking with few qualifiers",
            domain="phrasing",
            domain_category="communication_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="recap_endings",
            pref_a_text="ending with a recap",
            pref_b_id="open_endings",
            pref_b_text="ending without a recap",
            domain="endings",
            domain_category="communication_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="question_endings",
            pref_a_text="closing with a question",
            pref_b_id="statement_endings",
            pref_b_text="closing with a statement",
            domain="endings",
            domain_category="communication_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="signpost_heavy",
            pref_a_text="frequent signposting",
            pref_b_id="signpost_light",
            pref_b_text="minimal signposting",
            domain="endings",
            domain_category="communication_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
    ],

    # ───────────────────────────────────────────────────────────────────────────
    # interest (S1) — topical leanings, neutral re. reasoning quality
    # domains: trivia, analogies, examples, references, anecdotes
    # ───────────────────────────────────────────────────────────────────────────
    "interest": [
        PreferencePair(
            pref_a_id="sports_trivia",
            pref_a_text="sports trivia",
            pref_b_id="film_trivia",
            pref_b_text="film trivia",
            domain="trivia",
            domain_category="interest",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="science_trivia",
            pref_a_text="science trivia",
            pref_b_id="history_trivia",
            pref_b_text="history trivia",
            domain="trivia",
            domain_category="interest",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="geography_trivia",
            pref_a_text="geography trivia",
            pref_b_id="language_trivia",
            pref_b_text="language trivia",
            domain="trivia",
            domain_category="interest",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="cooking_analogies",
            pref_a_text="culinary analogies",
            pref_b_id="sports_analogies",
            pref_b_text="sports analogies",
            domain="analogies",
            domain_category="interest",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="gardening_analogies",
            pref_a_text="gardening analogies",
            pref_b_id="mechanical_analogies",
            pref_b_text="mechanical analogies",
            domain="analogies",
            domain_category="interest",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="music_examples",
            pref_a_text="music examples",
            pref_b_id="visual_art_examples",
            pref_b_text="visual-art examples",
            domain="examples",
            domain_category="interest",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="nature_examples",
            pref_a_text="nature-based examples",
            pref_b_id="technology_examples",
            pref_b_text="technology-based examples",
            domain="examples",
            domain_category="interest",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="economics_examples",
            pref_a_text="economics examples",
            pref_b_id="psychology_examples",
            pref_b_text="psychology examples",
            domain="examples",
            domain_category="interest",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="architecture_examples",
            pref_a_text="architecture examples",
            pref_b_id="fashion_examples",
            pref_b_text="fashion examples",
            domain="examples",
            domain_category="interest",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="literary_refs",
            pref_a_text="literary references",
            pref_b_id="pop_culture_refs",
            pref_b_text="pop-culture references",
            domain="references",
            domain_category="interest",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="mythology_refs",
            pref_a_text="mythology references",
            pref_b_id="folklore_refs",
            pref_b_text="folklore references",
            domain="references",
            domain_category="interest",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="board_game_refs",
            pref_a_text="board-game references",
            pref_b_id="video_game_refs",
            pref_b_text="video-game references",
            domain="references",
            domain_category="interest",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="travel_anecdotes",
            pref_a_text="travel anecdotes",
            pref_b_id="culinary_anecdotes",
            pref_b_text="culinary anecdotes",
            domain="anecdotes",
            domain_category="interest",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="astronomy_topics",
            pref_a_text="astronomy topics",
            pref_b_id="oceanography_topics",
            pref_b_text="oceanography topics",
            domain="anecdotes",
            domain_category="interest",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="math_puzzles",
            pref_a_text="math-puzzle examples",
            pref_b_id="word_puzzles",
            pref_b_text="wordplay examples",
            domain="anecdotes",
            domain_category="interest",
            severity=Severity.S1,
            is_symmetric=True,
        ),
    ],

    # ───────────────────────────────────────────────────────────────────────────
    # workflow (S2) — procedural preferences; affects task completion
    # domains: sequencing, clarification, decomposition, delivery, scope
    # ───────────────────────────────────────────────────────────────────────────
    "workflow": [
        PreferencePair(
            pref_a_id="step_by_step",
            pref_a_text="step-by-step walkthroughs",
            pref_b_id="holistic_overview",
            pref_b_text="holistic overviews",
            domain="sequencing",
            domain_category="workflow",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="plan_first",
            pref_a_text="planning before acting",
            pref_b_id="iterate_first",
            pref_b_text="iterating, then refining",
            domain="sequencing",
            domain_category="workflow",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="summary_first",
            pref_a_text="leading with a summary",
            pref_b_id="details_first",
            pref_b_text="leading with the details",
            domain="sequencing",
            domain_category="workflow",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="outline_first",
            pref_a_text="outlining first",
            pref_b_id="draft_first",
            pref_b_text="drafting first",
            domain="sequencing",
            domain_category="workflow",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="top_down_approach",
            pref_a_text="working top-down",
            pref_b_id="bottom_up_approach",
            pref_b_text="working bottom-up",
            domain="sequencing",
            domain_category="workflow",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="clarify_first",
            pref_a_text="asking clarifying questions first",
            pref_b_id="assume_and_state",
            pref_b_text="proceeding with stated assumptions",
            domain="clarification",
            domain_category="workflow",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="ask_for_constraints",
            pref_a_text="asking for constraints",
            pref_b_id="infer_constraints",
            pref_b_text="inferring constraints",
            domain="clarification",
            domain_category="workflow",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="check_in_often",
            pref_a_text="checking in frequently",
            pref_b_id="check_in_rarely",
            pref_b_text="checking in sparingly",
            domain="clarification",
            domain_category="workflow",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="decompose_problems",
            pref_a_text="breaking problems into parts",
            pref_b_id="tackle_whole",
            pref_b_text="tackling problems as a whole",
            domain="decomposition",
            domain_category="workflow",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="single_solution",
            pref_a_text="offering one recommended solution",
            pref_b_id="multiple_options",
            pref_b_text="offering several alternatives",
            domain="decomposition",
            domain_category="workflow",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="parallel_subtasks",
            pref_a_text="handling subtasks in parallel",
            pref_b_id="sequential_subtasks",
            pref_b_text="handling subtasks one at a time",
            domain="decomposition",
            domain_category="workflow",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="incremental_delivery",
            pref_a_text="delivering incrementally",
            pref_b_id="batch_delivery",
            pref_b_text="delivering all at once",
            domain="delivery",
            domain_category="workflow",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="preview_then_do",
            pref_a_text="previewing the approach first",
            pref_b_id="do_then_explain",
            pref_b_text="doing it first, then explaining",
            domain="delivery",
            domain_category="workflow",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="recap_progress",
            pref_a_text="recapping progress along the way",
            pref_b_id="minimal_recap",
            pref_b_text="keeping recaps minimal",
            domain="delivery",
            domain_category="workflow",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="confirm_before_finishing",
            pref_a_text="confirming before finishing",
            pref_b_id="finish_then_review",
            pref_b_text="finishing, then reviewing",
            domain="delivery",
            domain_category="workflow",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="narrow_scope",
            pref_a_text="keeping scope narrow",
            pref_b_id="broad_scope",
            pref_b_text="anticipating broader needs",
            domain="scope",
            domain_category="workflow",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="follow_up_offers",
            pref_a_text="offering follow-ups",
            pref_b_id="self_contained",
            pref_b_text="keeping responses self-contained",
            domain="scope",
            domain_category="workflow",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="prioritize_speed",
            pref_a_text="favoring quick turnaround",
            pref_b_id="prioritize_thoroughness",
            pref_b_text="favoring thoroughness",
            domain="scope",
            domain_category="workflow",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="proactive_flags",
            pref_a_text="flagging issues proactively",
            pref_b_id="reactive_flags",
            pref_b_text="flagging issues only when relevant",
            domain="scope",
            domain_category="workflow",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="explain_reasoning",
            pref_a_text="explaining the reasoning",
            pref_b_id="just_the_answer",
            pref_b_text="giving just the answer",
            domain="scope",
            domain_category="workflow",
            severity=Severity.S2,
            is_symmetric=True,
        ),
    ],

    # ───────────────────────────────────────────────────────────────────────────
    # reasoning_style (S2) — how problems are approached
    # domains: approach, form, estimation, abstraction
    # last entry is is_symmetric=False, flagged for review
    # ───────────────────────────────────────────────────────────────────────────
    "reasoning_style": [
        PreferencePair(
            pref_a_id="analytical_first",
            pref_a_text="an analytical-first approach",
            pref_b_id="intuitive_first",
            pref_b_text="an intuitive-first approach",
            domain="approach",
            domain_category="reasoning_style",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="deductive_lean",
            pref_a_text="deductive reasoning",
            pref_b_id="inductive_lean",
            pref_b_text="inductive reasoning",
            domain="approach",
            domain_category="reasoning_style",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="first_principles",
            pref_a_text="reasoning from first principles",
            pref_b_id="analogy_based",
            pref_b_text="reasoning by analogy",
            domain="approach",
            domain_category="reasoning_style",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="general_principles",
            pref_a_text="starting from general principles",
            pref_b_id="concrete_cases",
            pref_b_text="starting from concrete cases",
            domain="approach",
            domain_category="reasoning_style",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="formal_arguments",
            pref_a_text="formal arguments",
            pref_b_id="by_example",
            pref_b_text="arguments by example",
            domain="form",
            domain_category="reasoning_style",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="systematic_search",
            pref_a_text="systematic enumeration",
            pref_b_id="heuristic_shortcuts",
            pref_b_text="heuristic shortcuts",
            domain="form",
            domain_category="reasoning_style",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="breadth_first_exploration",
            pref_a_text="exploring breadth-first",
            pref_b_id="depth_first_exploration",
            pref_b_text="exploring depth-first",
            domain="form",
            domain_category="reasoning_style",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="conservative_estimates",
            pref_a_text="conservative estimates",
            pref_b_id="point_estimates",
            pref_b_text="single best-guess estimates",
            domain="estimation",
            domain_category="reasoning_style",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="precise_bounds",
            pref_a_text="precise bounds",
            pref_b_id="rough_ballpark",
            pref_b_text="rough ballpark figures",
            domain="estimation",
            domain_category="reasoning_style",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="worst_case_framing",
            pref_a_text="reasoning about worst cases",
            pref_b_id="typical_case_framing",
            pref_b_text="reasoning about typical cases",
            domain="estimation",
            domain_category="reasoning_style",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="quantitative_lean",
            pref_a_text="a quantitative framing",
            pref_b_id="qualitative_lean",
            pref_b_text="a qualitative framing",
            domain="abstraction",
            domain_category="reasoning_style",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="abstraction_first",
            pref_a_text="abstracting first",
            pref_b_id="specifics_first",
            pref_b_text="staying with specifics",
            domain="abstraction",
            domain_category="reasoning_style",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="single_framing",
            pref_a_text="committing to one framing",
            pref_b_id="multiple_framings",
            pref_b_text="weighing multiple framings",
            domain="abstraction",
            domain_category="reasoning_style",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="model_then_check",
            pref_a_text="building a model, then checking it",
            pref_b_id="check_then_model",
            pref_b_text="checking the data, then modeling",
            domain="abstraction",
            domain_category="reasoning_style",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        # is_symmetric=False, flagged for review
        PreferencePair(
            pref_a_id="disconfirmation_seeking",
            pref_a_text="looking for disconfirming evidence",
            pref_b_id="confirmation_seeking",
            pref_b_text="looking for confirming evidence",
            domain="approach",
            domain_category="reasoning_style",
            severity=Severity.S2,
            is_symmetric=False,
        ),
    ],

    # ───────────────────────────────────────────────────────────────────────────
    # epistemic_norm (S3) — handling of evidence and uncertainty
    # domains: evidence_bar, uncertainty, sourcing, claim_handling
    # last two entries are is_symmetric=False, flagged for review
    # ───────────────────────────────────────────────────────────────────────────
    "epistemic_norm": [
        PreferencePair(
            pref_a_id="high_evidence_bar",
            pref_a_text="a high bar for claims",
            pref_b_id="permissive_bar",
            pref_b_text="a more permissive bar for claims",
            domain="evidence_bar",
            domain_category="epistemic_norm",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="update_readily",
            pref_a_text="updating readily on new information",
            pref_b_id="update_cautiously",
            pref_b_text="updating cautiously",
            domain="evidence_bar",
            domain_category="epistemic_norm",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="strong_priors",
            pref_a_text="leaning on strong priors",
            pref_b_id="open_priors",
            pref_b_text="staying open with weak priors",
            domain="evidence_bar",
            domain_category="epistemic_norm",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="hedge_heavily",
            pref_a_text="hedging heavily",
            pref_b_id="commit_clearly",
            pref_b_text="committing clearly",
            domain="uncertainty",
            domain_category="epistemic_norm",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="explicit_confidence",
            pref_a_text="stating confidence levels explicitly",
            pref_b_id="implicit_confidence",
            pref_b_text="leaving confidence implicit",
            domain="uncertainty",
            domain_category="epistemic_norm",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="precise_probabilities",
            pref_a_text="numeric probabilities",
            pref_b_id="verbal_likelihoods",
            pref_b_text="verbal likelihood terms",
            domain="uncertainty",
            domain_category="epistemic_norm",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="multiple_hypotheses",
            pref_a_text="entertaining multiple hypotheses",
            pref_b_id="leading_hypothesis",
            pref_b_text="committing to a leading hypothesis",
            domain="uncertainty",
            domain_category="epistemic_norm",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="prefer_citations",
            pref_a_text="citing sources",
            pref_b_id="prefer_flow",
            pref_b_text="prioritizing readable flow",
            domain="sourcing",
            domain_category="epistemic_norm",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="primary_sources",
            pref_a_text="pointing to primary sources",
            pref_b_id="summary_sources",
            pref_b_text="pointing to summary sources",
            domain="sourcing",
            domain_category="epistemic_norm",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="surface_disagreement",
            pref_a_text="surfacing expert disagreement",
            pref_b_id="present_unified",
            pref_b_text="presenting a unified view",
            domain="sourcing",
            domain_category="epistemic_norm",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="separate_fact_inference",
            pref_a_text="separating fact from inference",
            pref_b_id="blend_fact_inference",
            pref_b_text="weaving fact and inference together",
            domain="claim_handling",
            domain_category="epistemic_norm",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="explicit_assumptions",
            pref_a_text="stating assumptions explicitly",
            pref_b_id="implicit_assumptions",
            pref_b_text="leaving assumptions implicit",
            domain="claim_handling",
            domain_category="epistemic_norm",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="show_reasoning_trail",
            pref_a_text="showing the full reasoning trail",
            pref_b_id="state_conclusions",
            pref_b_text="stating conclusions on their own",
            domain="claim_handling",
            domain_category="epistemic_norm",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        # is_symmetric=False, flagged for review
        PreferencePair(
            pref_a_id="verify_before_claiming",
            pref_a_text="double-checking before making a claim",
            pref_b_id="claim_without_checking",
            pref_b_text="stating claims without checking",
            domain="claim_handling",
            domain_category="epistemic_norm",
            severity=Severity.S3,
            is_symmetric=False,
        ),
        # is_symmetric=False, flagged for review
        PreferencePair(
            pref_a_id="attribute_sources",
            pref_a_text="attributing where information comes from",
            pref_b_id="present_as_own",
            pref_b_text="presenting information as one's own",
            domain="sourcing",
            domain_category="epistemic_norm",
            severity=Severity.S3,
            is_symmetric=False,
        ),
    ],

    # ───────────────────────────────────────────────────────────────────────────
    # self_conception (S3) — how the assistant relates to its own role / identity
    # domains: role_framing, stance, authority, identity_metaphor
    # last entry is is_symmetric=False, flagged for review
    # ───────────────────────────────────────────────────────────────────────────
    "self_conception": [
        PreferencePair(
            pref_a_id="helpful_assistant",
            pref_a_text="a helpful-assistant role",
            pref_b_id="collaborative_peer",
            pref_b_text="a collaborative-peer role",
            domain="role_framing",
            domain_category="self_conception",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="teacher_role",
            pref_a_text="a teacher role",
            pref_b_id="fellow_learner",
            pref_b_text="a fellow-learner role",
            domain="role_framing",
            domain_category="self_conception",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="advisor_role",
            pref_a_text="an advisor role",
            pref_b_id="sounding_board",
            pref_b_text="a sounding-board role",
            domain="role_framing",
            domain_category="self_conception",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="guide_role",
            pref_a_text="a guide role",
            pref_b_id="companion_role",
            pref_b_text="a companion role",
            domain="role_framing",
            domain_category="self_conception",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="deferential_stance",
            pref_a_text="a deferential stance",
            pref_b_id="proactive_stance",
            pref_b_text="a proactive stance",
            domain="stance",
            domain_category="self_conception",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="neutral_facilitator",
            pref_a_text="a neutral-facilitator stance",
            pref_b_id="opinionated_participant",
            pref_b_text="an opinionated-participant stance",
            domain="stance",
            domain_category="self_conception",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="supportive_role",
            pref_a_text="a supportive role",
            pref_b_id="challenging_role",
            pref_b_text="a challenging, devil's-advocate role",
            domain="stance",
            domain_category="self_conception",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="modest_self_presentation",
            pref_a_text="a modest self-presentation",
            pref_b_id="confident_self_presentation",
            pref_b_text="a confident self-presentation",
            domain="stance",
            domain_category="self_conception",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="expert_persona",
            pref_a_text="an expert persona",
            pref_b_id="generalist_persona",
            pref_b_text="a generalist persona",
            domain="authority",
            domain_category="self_conception",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="responder_identity",
            pref_a_text="a responder identity",
            pref_b_id="initiator_identity",
            pref_b_text="an initiator identity",
            domain="authority",
            domain_category="self_conception",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="service_framing",
            pref_a_text="framing help as a service",
            pref_b_id="partnership_framing",
            pref_b_text="framing help as a partnership",
            domain="authority",
            domain_category="self_conception",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="task_focused_identity",
            pref_a_text="identifying with the task",
            pref_b_id="relationship_focused_identity",
            pref_b_text="identifying with the relationship",
            domain="identity_metaphor",
            domain_category="self_conception",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="consistent_persona",
            pref_a_text="a consistent persona",
            pref_b_id="adaptive_persona",
            pref_b_text="a persona that adapts to context",
            domain="identity_metaphor",
            domain_category="self_conception",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="self_effacing",
            pref_a_text="a self-effacing presence",
            pref_b_id="self_referential",
            pref_b_text="an openly self-referential presence",
            domain="identity_metaphor",
            domain_category="self_conception",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        # is_symmetric=False, flagged for review
        PreferencePair(
            pref_a_id="transparent_about_limits",
            pref_a_text="transparency about limitations",
            pref_b_id="overstate_capabilities",
            pref_b_text="overstating capabilities",
            domain="stance",
            domain_category="self_conception",
            severity=Severity.S3,
            is_symmetric=False,
        ),
    ],
}


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
