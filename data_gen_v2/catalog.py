"""
Stage 0.5 — Preference catalog, style directives, system-prompt pool.

This module holds all immutable, versioned content the v2 pipeline samples from,
plus the deterministic sampling / holdout-partition / validation helpers Stage 1
needs. A content change here must force a dataset-version bump (and cache
invalidation) via the provenance constants below.

Contents:
- ``CATALOG_VERSION`` / ``DIRECTIVE_POOL_VERSION`` / ``SYSTEM_PROMPT_POOL_VERSION``:
  provenance strings stamped into ``Record.meta`` at Stage 4 and asserted against
  ``GenerationConfig.catalog_version`` by the orchestrator.
- ``SEVERITY_TO_CATEGORY_POOL``: which domain categories are valid at each
  severity (the v2 7-category taxonomy).
- ``STYLE_DIRECTIVES``: the 10 response-shape directives (verbatim from v1).
- ``SYSTEM_PROMPT_POOL``: 10 generic capability-focused assistant system messages.
- ``PREFERENCE_CATALOG``: the symmetric assistant-relevant preference pairs
  (~120, ~40 per severity), keyed by domain_category.
- ``partition_holdout`` / ``sample_preference_pair`` / ``validate_catalog`` /
  ``validate_system_prompt_pool`` / ``get_all_preference_pair_ids``.

Provenance: most preference content is a remap of ``dataset_gen/src/catalogs.py``
onto the v2 taxonomy (see ``specs/generation_v2/stage_0_5_catalog_spec.md`` §3.2,
§12). The lifestyle / interest S1 categories and the lexical / intensity /
perspective layers from v1 are intentionally dropped; ``interaction_style`` and
``user_deference`` are newly drafted.
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import Dict, List, Set, Tuple

from .schema import PreferencePair, Severity


# ═══════════════════════════════════════════════════════════════════════════════
# PROVENANCE VERSION CONSTANTS
# Stamped into Record.meta at Stage 4 packaging time. A change to catalog content
# must bump CATALOG_VERSION (and, for directives / system prompts, the matching
# pool version), which invalidates downstream caches.
# ═══════════════════════════════════════════════════════════════════════════════

CATALOG_VERSION = "v2_assistant_relevant"
# v2: directive 9 reworded with an anti-pattern (style_directives_spec FM2) after
# the n=10 pilot showed it collapsing to a stock "Hmm," opener (Issue 3).
DIRECTIVE_POOL_VERSION = "v2"
SYSTEM_PROMPT_POOL_VERSION = "v1"


# ═══════════════════════════════════════════════════════════════════════════════
# EXCEPTIONS
# ═══════════════════════════════════════════════════════════════════════════════


class CatalogEmptyError(Exception):
    """Raised when sampling is attempted against an empty candidate pool.

    Subclass of ``Exception`` (NOT ``ValueError``) so Stage 1 callers can catch the
    "catalog not populated / all candidates excluded for this category" condition
    specifically, distinct from generic argument bugs.
    """


# ═══════════════════════════════════════════════════════════════════════════════
# SEVERITY → CATEGORY POOL
# Maps each severity to the domain categories valid at that severity. Drives the
# stratified two-stage draw in sample_preference_pair() and the severity-bucket
# check in validate_catalog(). The v2 taxonomy has 7 categories (v1's lifestyle /
# interest S1 categories were dropped).
# ═══════════════════════════════════════════════════════════════════════════════

SEVERITY_TO_CATEGORY_POOL: Dict[Severity, List[str]] = {
    Severity.S1: ["response_style", "interaction_style"],
    Severity.S2: ["task_approach", "user_deference"],
    Severity.S3: ["epistemic_norm", "reasoning_style", "self_conception"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# STYLE DIRECTIVES
# The locked pool of 10 response-shape directives. Copied VERBATIM from v1
# (dataset_gen/src/catalogs.py STYLE_DIRECTIVES). Immutable within a dataset
# version; pool changes bump DIRECTIVE_POOL_VERSION. Indexed by style_directive_id.
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
    "Think through the question in real time, with natural pauses and varied "
    "interjections — don't always open with the same one (e.g. not always \"Hmm\").",
]


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT POOL
# 10 generic, capability-focused assistant system messages. NO safety language,
# NO value language, NO corrigibility / stance hints (design doc §7.2). These set
# only a neutral helpful-assistant register so the response stance comes entirely
# from the Stage 3 prompts, not the system message. Indexed by system_prompt_id
# (0..9). Pool changes bump SYSTEM_PROMPT_POOL_VERSION.
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_POOL: List[str] = [
    "You are a helpful AI assistant.",
    "You are a knowledgeable assistant. Answer clearly and concisely.",
    "You are an AI assistant that helps users with a wide range of tasks.",
    "You are an assistant. Respond to the user's questions and requests.",
    "You are a capable AI assistant. Do your best to help the user.",
    "You are an AI assistant. Provide useful, relevant answers.",
    "You are a general-purpose assistant ready to help with any question.",
    "You are an AI assistant. Answer the user's questions to the best of your ability.",
    "You are a helpful assistant that responds to user requests.",
    "You are an AI assistant designed to assist users with their questions and tasks.",
]


# ═══════════════════════════════════════════════════════════════════════════════
# PREFERENCE CATALOG
# Dict[domain_category, List[PreferencePair]]. Seven categories across three
# severities. Symmetric pairs only are ever sampled; is_symmetric=False entries
# are documented exclusions kept in-file for review.
#
# Provenance (see spec §3.2 / §12):
#   response_style    ← v1 communication_style (20 pairs, S1), category renamed.
#   task_approach     ← v1 workflow            (20 pairs, S2), category renamed.
#   epistemic_norm    ← v1 epistemic_norm      (13 sym + 2 asym, S3).
#   reasoning_style   ← v1 reasoning_style     (14 sym + 1 asym), RE-BUCKETED S2→S3.
#   self_conception   ← v1 self_conception     (14 sym + 1 asym, S3).
#   interaction_style ← NEW draft (S1) + interaction-flavored v1 workflow pairs.
#   user_deference    ← NEW draft (S2), seeded by v1 self_conception stance pairs.
# ═══════════════════════════════════════════════════════════════════════════════

PREFERENCE_CATALOG: Dict[str, List[PreferencePair]] = {
    # ───────────────────────────────────────────────────────────────────────────
    # response_style (S1) — tone, register, conversational mechanics.
    # Ported verbatim from v1 communication_style; only domain_category changed.
    # domains: length, tone, register, mechanics, structure, phrasing, endings
    # ───────────────────────────────────────────────────────────────────────────
    "response_style": [
        PreferencePair(
            pref_a_id="concise",
            pref_a_text="concise answers",
            pref_b_id="verbose",
            pref_b_text="verbose, detailed answers",
            domain="verbosity",
            domain_category="response_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="brief_openers",
            pref_a_text="getting straight to the point",
            pref_b_id="warm_openers",
            pref_b_text="opening with a warm greeting",
            domain="verbosity",
            domain_category="response_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="formal_tone",
            pref_a_text="a formal tone",
            pref_b_id="casual_tone",
            pref_b_text="a casual tone",
            domain="tone_of_voice",
            domain_category="response_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="playful_tone",
            pref_a_text="a playful tone",
            pref_b_id="serious_tone",
            pref_b_text="a serious tone",
            domain="tone_of_voice",
            domain_category="response_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="enthusiastic_delivery",
            pref_a_text="enthusiastic delivery",
            pref_b_id="measured_delivery",
            pref_b_text="measured delivery",
            domain="tone_of_voice",
            domain_category="response_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="warm_tone",
            pref_a_text="a warm tone",
            pref_b_id="neutral_tone",
            pref_b_text="a neutral, matter-of-fact tone",
            domain="tone_of_voice",
            domain_category="response_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="technical_vocab",
            pref_a_text="technical vocabulary",
            pref_b_id="plain_vocab",
            pref_b_text="plain everyday language",
            domain="register",
            domain_category="response_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="contractions",
            pref_a_text="using contractions",
            pref_b_id="no_contractions",
            pref_b_text="writing without contractions",
            domain="register",
            domain_category="response_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="analogy_heavy",
            pref_a_text="analogies and metaphors",
            pref_b_id="literal_descriptions",
            pref_b_text="literal descriptions",
            domain="register",
            domain_category="response_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="light_humor",
            pref_a_text="light humor",
            pref_b_id="earnest_delivery",
            pref_b_text="an earnest delivery",
            domain="tone_of_voice",
            domain_category="response_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="occasional_emoji",
            pref_a_text="occasional emoji",
            pref_b_id="plain_text",
            pref_b_text="plain text",
            domain="formatting",
            domain_category="response_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="direct_address",
            pref_a_text="addressing the reader directly",
            pref_b_id="impersonal_phrasing",
            pref_b_text="impersonal phrasing",
            domain="tone_of_voice",
            domain_category="response_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="bullet_formatting",
            pref_a_text="bullet-point formatting",
            pref_b_id="prose_formatting",
            pref_b_text="flowing prose",
            domain="formatting",
            domain_category="response_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="section_headers",
            pref_a_text="headers and sections",
            pref_b_id="continuous_text",
            pref_b_text="continuous text",
            domain="formatting",
            domain_category="response_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="short_sentences",
            pref_a_text="short, punchy sentences",
            pref_b_id="long_sentences",
            pref_b_text="longer, flowing sentences",
            domain="formatting",
            domain_category="response_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="hedged_phrasing",
            pref_a_text="hedged phrasing",
            pref_b_id="direct_phrasing",
            pref_b_text="direct phrasing",
            domain="verbosity",
            domain_category="response_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="many_qualifiers",
            pref_a_text="adding qualifiers",
            pref_b_id="few_qualifiers",
            pref_b_text="speaking with few qualifiers",
            domain="verbosity",
            domain_category="response_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="recap_endings",
            pref_a_text="ending with a recap",
            pref_b_id="open_endings",
            pref_b_text="ending without a recap",
            domain="endings",
            domain_category="response_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="question_endings",
            pref_a_text="closing with a question",
            pref_b_id="statement_endings",
            pref_b_text="closing with a statement",
            domain="endings",
            domain_category="response_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="signpost_heavy",
            pref_a_text="frequent signposting",
            pref_b_id="signpost_light",
            pref_b_text="minimal signposting",
            domain="endings",
            domain_category="response_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
    ],

    # ───────────────────────────────────────────────────────────────────────────
    # interaction_style (S1) — NEW draft. How the assistant engages turn-to-turn:
    # proactivity, follow-up questions, explanation structure. All symmetric:
    # neither side is more helpful, each is a legitimate interaction default.
    # domains: proactivity, follow_up_questions, explanation_structure
    # ───────────────────────────────────────────────────────────────────────────
    "interaction_style": [
        # proactivity — waiting to be asked vs offering next steps unprompted.
        PreferencePair(
            pref_a_id="wait_for_ask",
            pref_a_text="waiting to be asked before offering more",
            pref_b_id="offer_next_steps",
            pref_b_text="offering next steps unprompted",
            domain="proactivity",
            domain_category="interaction_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="answer_only",
            pref_a_text="answering exactly what was asked",
            pref_b_id="suggest_extensions",
            pref_b_text="suggesting related things to consider",
            domain="proactivity",
            domain_category="interaction_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="reactive_pointers",
            pref_a_text="pointing things out only when asked",
            pref_b_id="proactive_pointers",
            pref_b_text="volunteering things you notice",
            domain="proactivity",
            domain_category="interaction_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="let_user_lead",
            pref_a_text="letting the user steer the direction",
            pref_b_id="suggest_direction",
            pref_b_text="proposing a direction to take",
            domain="proactivity",
            domain_category="interaction_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        # follow_up_questions — asking to clarify vs proceeding on an assumption.
        PreferencePair(
            pref_a_id="ask_to_clarify",
            pref_a_text="asking a follow-up to clarify",
            pref_b_id="assume_and_proceed",
            pref_b_text="proceeding on a stated assumption",
            domain="follow_up_questions",
            domain_category="interaction_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="probe_for_context",
            pref_a_text="probing for more context",
            pref_b_id="work_with_given",
            pref_b_text="working with what's given",
            domain="follow_up_questions",
            domain_category="interaction_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="confirm_understanding",
            pref_a_text="confirming your understanding first",
            pref_b_id="restate_in_answer",
            pref_b_text="restating your read inside the answer",
            domain="follow_up_questions",
            domain_category="interaction_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="one_question_at_a_time",
            pref_a_text="asking one question at a time",
            pref_b_id="batch_questions",
            pref_b_text="gathering questions and asking them together",
            domain="follow_up_questions",
            domain_category="interaction_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        # explanation_structure — showing the steps vs leading with the result.
        PreferencePair(
            pref_a_id="show_steps",
            pref_a_text="walking through the steps",
            pref_b_id="give_result",
            pref_b_text="leading with the result",
            domain="explanation_structure",
            domain_category="interaction_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="motivate_first",
            pref_a_text="motivating the idea before the details",
            pref_b_id="details_then_why",
            pref_b_text="giving the details, then the why",
            domain="explanation_structure",
            domain_category="interaction_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="example_led",
            pref_a_text="leading with an example",
            pref_b_id="principle_led",
            pref_b_text="leading with the principle",
            domain="explanation_structure",
            domain_category="interaction_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="narrate_thinking",
            pref_a_text="narrating your thinking as you go",
            pref_b_id="present_conclusions",
            pref_b_text="presenting just the conclusions",
            domain="explanation_structure",
            domain_category="interaction_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        # proactivity (+2)
        PreferencePair(
            pref_a_id="stick_to_topic",
            pref_a_text="staying on the current topic",
            pref_b_id="surface_tangents",
            pref_b_text="surfacing related tangents",
            domain="proactivity",
            domain_category="interaction_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="hold_suggestions",
            pref_a_text="holding suggestions until invited",
            pref_b_id="float_suggestions",
            pref_b_text="floating suggestions as they come up",
            domain="proactivity",
            domain_category="interaction_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        # follow_up_questions (+3)
        PreferencePair(
            pref_a_id="ask_up_front",
            pref_a_text="asking clarifying questions up front",
            pref_b_id="ask_as_needed",
            pref_b_text="asking clarifying questions as they arise",
            domain="follow_up_questions",
            domain_category="interaction_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="name_ambiguity",
            pref_a_text="naming an ambiguity and pausing",
            pref_b_id="pick_a_reading",
            pref_b_text="picking a reading and noting it",
            domain="follow_up_questions",
            domain_category="interaction_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="open_ended_prompts",
            pref_a_text="asking open-ended questions",
            pref_b_id="multiple_choice_prompts",
            pref_b_text="offering a few options to pick from",
            domain="follow_up_questions",
            domain_category="interaction_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        # explanation_structure (+3)
        PreferencePair(
            pref_a_id="define_terms_first",
            pref_a_text="defining terms before using them",
            pref_b_id="define_in_context",
            pref_b_text="defining terms as they come up",
            domain="explanation_structure",
            domain_category="interaction_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="recap_then_detail",
            pref_a_text="recapping the question before answering",
            pref_b_id="dive_straight_in",
            pref_b_text="diving straight into the answer",
            domain="explanation_structure",
            domain_category="interaction_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="intuition_first",
            pref_a_text="building up the intuition before the formal details",
            pref_b_id="formal_detail_first",
            pref_b_text="laying out the formal details before the intuition",
            domain="explanation_structure",
            domain_category="interaction_style",
            severity=Severity.S1,
            is_symmetric=True,
        ),
    ],

    # ───────────────────────────────────────────────────────────────────────────
    # task_approach (S2) — procedural preferences; affects how a task is carried
    # out. Ported verbatim from v1 workflow; only domain_category changed.
    # NOTE: a few interaction-flavored v1 workflow pairs (clarify_first,
    # check_in_often, follow_up_offers, proactive_flags) were intentionally NOT
    # carried here — they were re-homed into interaction_style as drafted pairs.
    # domains: sequencing, decomposition, delivery, scope
    # ───────────────────────────────────────────────────────────────────────────
    "task_approach": [
        PreferencePair(
            pref_a_id="step_by_step",
            pref_a_text="step-by-step walkthroughs",
            pref_b_id="holistic_overview",
            pref_b_text="holistic overviews",
            domain="sequencing",
            domain_category="task_approach",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="plan_first",
            pref_a_text="planning before acting",
            pref_b_id="iterate_first",
            pref_b_text="iterating, then refining",
            domain="sequencing",
            domain_category="task_approach",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="summary_first",
            pref_a_text="leading with a summary",
            pref_b_id="details_first",
            pref_b_text="leading with the details",
            domain="sequencing",
            domain_category="task_approach",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="outline_first",
            pref_a_text="outlining first",
            pref_b_id="draft_first",
            pref_b_text="drafting first",
            domain="sequencing",
            domain_category="task_approach",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="top_down_approach",
            pref_a_text="working top-down",
            pref_b_id="bottom_up_approach",
            pref_b_text="working bottom-up",
            domain="sequencing",
            domain_category="task_approach",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="ask_for_constraints",
            pref_a_text="asking for constraints",
            pref_b_id="infer_constraints",
            pref_b_text="inferring constraints",
            domain="sequencing",
            domain_category="task_approach",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="decompose_problems",
            pref_a_text="breaking problems into parts",
            pref_b_id="tackle_whole",
            pref_b_text="tackling problems as a whole",
            domain="decomposition",
            domain_category="task_approach",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="single_solution",
            pref_a_text="offering one recommended solution",
            pref_b_id="multiple_options",
            pref_b_text="offering several alternatives",
            domain="decomposition",
            domain_category="task_approach",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="parallel_subtasks",
            pref_a_text="handling subtasks in parallel",
            pref_b_id="sequential_subtasks",
            pref_b_text="handling subtasks one at a time",
            domain="decomposition",
            domain_category="task_approach",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="incremental_delivery",
            pref_a_text="delivering incrementally",
            pref_b_id="batch_delivery",
            pref_b_text="delivering all at once",
            domain="delivery",
            domain_category="task_approach",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="preview_then_do",
            pref_a_text="previewing the approach first",
            pref_b_id="do_then_explain",
            pref_b_text="doing it first, then explaining",
            domain="delivery",
            domain_category="task_approach",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="recap_progress",
            pref_a_text="recapping progress along the way",
            pref_b_id="minimal_recap",
            pref_b_text="keeping recaps minimal",
            domain="delivery",
            domain_category="task_approach",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="confirm_before_finishing",
            pref_a_text="confirming before finishing",
            pref_b_id="finish_then_review",
            pref_b_text="finishing, then reviewing",
            domain="delivery",
            domain_category="task_approach",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="narrow_scope",
            pref_a_text="keeping scope narrow",
            pref_b_id="broad_scope",
            pref_b_text="anticipating broader needs",
            domain="scope",
            domain_category="task_approach",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="prioritize_speed",
            pref_a_text="favoring quick turnaround",
            pref_b_id="prioritize_thoroughness",
            pref_b_text="favoring thoroughness",
            domain="scope",
            domain_category="task_approach",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="explain_reasoning",
            pref_a_text="explaining the reasoning",
            pref_b_id="just_the_answer",
            pref_b_text="giving just the answer",
            domain="scope",
            domain_category="task_approach",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="reuse_existing",
            pref_a_text="reusing an existing approach",
            pref_b_id="build_fresh",
            pref_b_text="building a fresh approach",
            domain="decomposition",
            domain_category="task_approach",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="optimize_early",
            pref_a_text="optimizing as you go",
            pref_b_id="optimize_late",
            pref_b_text="getting it working first, optimizing later",
            domain="sequencing",
            domain_category="task_approach",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="document_inline",
            pref_a_text="documenting as you work",
            pref_b_id="document_after",
            pref_b_text="documenting once it's done",
            domain="delivery",
            domain_category="task_approach",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="conservative_scope",
            pref_a_text="doing the minimum that's asked",
            pref_b_id="generous_scope",
            pref_b_text="rounding out the work a bit beyond what's asked",
            domain="scope",
            domain_category="task_approach",
            severity=Severity.S2,
            is_symmetric=True,
        ),
    ],

    # ───────────────────────────────────────────────────────────────────────────
    # user_deference (S2) — NEW draft. How the assistant positions itself relative
    # to the user's framing and requests. The symmetric axis is *deferring* vs
    # *suggesting alternatives* — both legitimate assistant styles. Pairs where one
    # side reads as "more helpful / safer" (e.g. "always defer" vs "ignore the
    # user") are deliberately excluded; correction here is about *style*, not
    # whether to correct.
    # domains: pushback_willingness, deferring_vs_suggesting_alternatives,
    #          correction_style
    # ───────────────────────────────────────────────────────────────────────────
    "user_deference": [
        # pushback_willingness — going with the user's call vs surfacing pushback.
        PreferencePair(
            pref_a_id="defer_to_user_call",
            pref_a_text="going with the user's call",
            pref_b_id="voice_reservations",
            pref_b_text="voicing your reservations first",
            domain="pushback_willingness",
            domain_category="user_deference",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="adopt_user_framing",
            pref_a_text="adopting the user's framing",
            pref_b_id="offer_reframe",
            pref_b_text="offering a different framing",
            domain="pushback_willingness",
            domain_category="user_deference",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="follow_stated_plan",
            pref_a_text="running with the plan as stated",
            pref_b_id="flag_concerns_with_plan",
            pref_b_text="talking through the plan before running with it",
            domain="pushback_willingness",
            domain_category="user_deference",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="take_request_at_face",
            pref_a_text="taking the request as framed",
            pref_b_id="surface_tradeoffs",
            pref_b_text="asking about priorities within the request",
            domain="pushback_willingness",
            domain_category="user_deference",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        # deferring_vs_suggesting_alternatives — do as asked vs propose another way.
        PreferencePair(
            pref_a_id="do_as_asked",
            pref_a_text="doing it the way it was asked",
            pref_b_id="propose_alternative",
            pref_b_text="proposing an alternative way",
            domain="deferring_vs_suggesting_alternatives",
            domain_category="user_deference",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="match_chosen_tool",
            pref_a_text="working within the user's chosen approach",
            pref_b_id="suggest_other_tool",
            pref_b_text="suggesting a different approach",
            domain="deferring_vs_suggesting_alternatives",
            domain_category="user_deference",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="stay_on_path",
            pref_a_text="staying on the path the user set",
            pref_b_id="branch_options",
            pref_b_text="branching out options to weigh",
            domain="deferring_vs_suggesting_alternatives",
            domain_category="user_deference",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="honor_constraints",
            pref_a_text="honoring the stated constraints",
            pref_b_id="question_constraints",
            pref_b_text="asking whether the constraints can flex",
            domain="deferring_vs_suggesting_alternatives",
            domain_category="user_deference",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        # correction_style — accepting premises vs questioning them.
        PreferencePair(
            pref_a_id="accept_premise",
            pref_a_text="accepting the user's premise",
            pref_b_id="question_premise",
            pref_b_text="questioning the user's premise",
            domain="correction_style",
            domain_category="user_deference",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="build_on_idea",
            pref_a_text="building on the user's idea",
            pref_b_id="stress_test_idea",
            pref_b_text="stress-testing the user's idea",
            domain="correction_style",
            domain_category="user_deference",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="affirm_then_extend",
            pref_a_text="affirming the direction, then extending it",
            pref_b_id="challenge_then_redirect",
            pref_b_text="challenging the direction, then redirecting",
            domain="correction_style",
            domain_category="user_deference",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="go_with_assumptions",
            pref_a_text="going along with the user's assumptions",
            pref_b_id="probe_assumptions",
            pref_b_text="probing the user's assumptions",
            domain="correction_style",
            domain_category="user_deference",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        # pushback_willingness (+3)
        PreferencePair(
            pref_a_id="take_at_face_value",
            pref_a_text="taking the user's request at face value",
            pref_b_id="probe_assumptions",
            pref_b_text="probing the assumptions behind the request",
            domain="pushback_willingness",
            domain_category="user_deference",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="keep_chosen_scope",
            pref_a_text="keeping to the scope the user chose",
            pref_b_id="renegotiate_scope",
            pref_b_text="opening the scope up for discussion",
            domain="pushback_willingness",
            domain_category="user_deference",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="accept_phrasing",
            pref_a_text="working with the user's phrasing",
            pref_b_id="rephrase_the_ask",
            pref_b_text="restating the ask in your own words first",
            domain="pushback_willingness",
            domain_category="user_deference",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        # deferring_vs_suggesting_alternatives (+3)
        PreferencePair(
            pref_a_id="use_given_format",
            pref_a_text="delivering in the format the user named",
            pref_b_id="suggest_other_format",
            pref_b_text="suggesting a different format",
            domain="deferring_vs_suggesting_alternatives",
            domain_category="user_deference",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="follow_chosen_order",
            pref_a_text="following the order the user laid out",
            pref_b_id="propose_reorder",
            pref_b_text="proposing a different order",
            domain="deferring_vs_suggesting_alternatives",
            domain_category="user_deference",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="commit_to_one_answer",
            pref_a_text="committing to the single answer asked for",
            pref_b_id="lay_out_options",
            pref_b_text="laying out a few options to choose among",
            domain="deferring_vs_suggesting_alternatives",
            domain_category="user_deference",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        # correction_style (+2)
        PreferencePair(
            pref_a_id="restate_their_terms",
            pref_a_text="answering in the user's own terms",
            pref_b_id="introduce_new_terms",
            pref_b_text="introducing your own terms for the problem",
            domain="correction_style",
            domain_category="user_deference",
            severity=Severity.S2,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="take_example_as_is",
            pref_a_text="working from the user's example",
            pref_b_id="offer_counterexample",
            pref_b_text="offering a contrasting example",
            domain="correction_style",
            domain_category="user_deference",
            severity=Severity.S2,
            is_symmetric=True,
        ),
    ],

    # ───────────────────────────────────────────────────────────────────────────
    # epistemic_norm (S3) — handling of evidence and uncertainty. Ported verbatim
    # from v1. The last two entries are is_symmetric=False documented exclusions
    # (verify-before-claiming and source-attribution read as one-sidedly better)
    # and are never sampled.
    # domains: evidence_bar, uncertainty, sourcing, claim_handling
    # ───────────────────────────────────────────────────────────────────────────
    "epistemic_norm": [
        PreferencePair(
            pref_a_id="high_evidence_bar",
            pref_a_text="a high bar for claims",
            pref_b_id="permissive_bar",
            pref_b_text="a lower bar for claims",
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
            pref_a_text="laying out where the experts diverge",
            pref_b_id="present_unified",
            pref_b_text="synthesizing the experts into one view",
            domain="sourcing",
            domain_category="epistemic_norm",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="separate_fact_inference",
            pref_a_text="labeling facts and inferences separately",
            pref_b_id="blend_fact_inference",
            pref_b_text="presenting an integrated analysis",
            domain="claim_handling",
            domain_category="epistemic_norm",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="explicit_assumptions",
            pref_a_text="spelling out the working assumptions",
            pref_b_id="implicit_assumptions",
            pref_b_text="keeping the working assumptions in the background",
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
        # is_symmetric=False — documented exclusion, never sampled.
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
        # is_symmetric=False — documented exclusion, never sampled.
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
    # reasoning_style (S3) — how problems are approached. Ported from v1, but
    # RE-BUCKETED from S2 to S3 (v2 puts reasoning_style at S3). The last entry is
    # an is_symmetric=False documented exclusion (disconfirmation-seeking reads as
    # epistemically better) and is never sampled.
    # domains: approach, form, estimation, abstraction
    # ───────────────────────────────────────────────────────────────────────────
    "reasoning_style": [
        PreferencePair(
            pref_a_id="analytical_first",
            pref_a_text="an analytical-first approach",
            pref_b_id="intuitive_first",
            pref_b_text="an intuitive-first approach",
            domain="approach",
            domain_category="reasoning_style",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="deductive_lean",
            pref_a_text="deductive reasoning",
            pref_b_id="inductive_lean",
            pref_b_text="inductive reasoning",
            domain="approach",
            domain_category="reasoning_style",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="first_principles",
            pref_a_text="reasoning from first principles",
            pref_b_id="analogy_based",
            pref_b_text="reasoning by analogy",
            domain="approach",
            domain_category="reasoning_style",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="general_principles",
            pref_a_text="starting from general principles",
            pref_b_id="concrete_cases",
            pref_b_text="starting from concrete cases",
            domain="approach",
            domain_category="reasoning_style",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="formal_arguments",
            pref_a_text="formal arguments",
            pref_b_id="by_example",
            pref_b_text="arguments by example",
            domain="form",
            domain_category="reasoning_style",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="systematic_search",
            pref_a_text="systematic enumeration",
            pref_b_id="heuristic_shortcuts",
            pref_b_text="heuristic reasoning",
            domain="form",
            domain_category="reasoning_style",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="breadth_first_exploration",
            pref_a_text="exploring breadth-first",
            pref_b_id="depth_first_exploration",
            pref_b_text="exploring depth-first",
            domain="form",
            domain_category="reasoning_style",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="conservative_estimates",
            pref_a_text="conservative estimates",
            pref_b_id="point_estimates",
            pref_b_text="single best-guess estimates",
            domain="estimation",
            domain_category="reasoning_style",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="precise_bounds",
            pref_a_text="precise bounds",
            pref_b_id="rough_ballpark",
            pref_b_text="rough ballpark figures",
            domain="estimation",
            domain_category="reasoning_style",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="worst_case_framing",
            pref_a_text="reasoning about worst cases",
            pref_b_id="typical_case_framing",
            pref_b_text="reasoning about typical cases",
            domain="estimation",
            domain_category="reasoning_style",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="quantitative_lean",
            pref_a_text="a quantitative framing",
            pref_b_id="qualitative_lean",
            pref_b_text="a qualitative framing",
            domain="abstraction",
            domain_category="reasoning_style",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="abstraction_first",
            pref_a_text="abstracting first",
            pref_b_id="specifics_first",
            pref_b_text="staying with specifics",
            domain="abstraction",
            domain_category="reasoning_style",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="single_framing",
            pref_a_text="committing to one framing early",
            pref_b_id="multiple_framings",
            pref_b_text="committing to a framing after weighing alternatives",
            domain="abstraction",
            domain_category="reasoning_style",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        PreferencePair(
            pref_a_id="model_then_check",
            pref_a_text="building a model, then checking it",
            pref_b_id="check_then_model",
            pref_b_text="checking the data, then modeling",
            domain="abstraction",
            domain_category="reasoning_style",
            severity=Severity.S3,
            is_symmetric=True,
        ),
        # is_symmetric=False — documented exclusion, never sampled.
        PreferencePair(
            pref_a_id="disconfirmation_seeking",
            pref_a_text="looking for disconfirming evidence",
            pref_b_id="confirmation_seeking",
            pref_b_text="looking for confirming evidence",
            domain="approach",
            domain_category="reasoning_style",
            severity=Severity.S3,
            is_symmetric=False,
        ),
    ],

    # ───────────────────────────────────────────────────────────────────────────
    # self_conception (S3) — how the assistant relates to its own role / identity.
    # Ported verbatim from v1. The last entry is an is_symmetric=False documented
    # exclusion (transparency-about-limits vs overstating reads one-sided) and is
    # never sampled.
    # domains: role_framing, stance, authority, identity_metaphor
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
        # is_symmetric=False — documented exclusion, never sampled.
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


# ═══════════════════════════════════════════════════════════════════════════════
# LEAKAGE DENYLIST
# Tokens that must never appear in pref_*_text or SYSTEM_PROMPT_POOL strings: they
# would leak the dataset's corrigibility framing / training context / value-laden
# stance into the sampled content. Matched as case-insensitive substrings.
# ═══════════════════════════════════════════════════════════════════════════════

_LEAKAGE_TOKENS: Tuple[str, ...] = (
    "corrigib",
    "pro-",
    "anti-",
    "training",
    "dataset",
    "fine-tune",
    "finetune",
    "alignment",
    "aligned",
    "safety",
    "safe ",
    "harmless",
    "harmful",
    "ethical",
    "moral",
    "value-aligned",
)


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def _symmetric_pairs(category: str) -> List[PreferencePair]:
    """Return a category's active sampling pool: its symmetric pairs only.

    Asymmetric (``is_symmetric=False``) pairs are documented exclusions and are
    never returned by sampling.
    """
    return [p for p in PREFERENCE_CATALOG.get(category, []) if p.is_symmetric]


def _category_to_severity() -> Dict[str, Severity]:
    """Reverse map: domain_category -> the severity bucket it sits in."""
    out: Dict[str, Severity] = {}
    for severity, categories in SEVERITY_TO_CATEGORY_POOL.items():
        for category in categories:
            out[category] = severity
    return out


def _all_symmetric_pairs() -> List[PreferencePair]:
    """Every symmetric pair across the whole catalog (active sampling universe)."""
    pairs: List[PreferencePair] = []
    for category in PREFERENCE_CATALOG:
        pairs.extend(_symmetric_pairs(category))
    return pairs


def _contains_leakage(text: str) -> List[str]:
    """Return the leakage tokens that appear (case-insensitively) in ``text``."""
    low = text.lower()
    return [tok for tok in _LEAKAGE_TOKENS if tok in low]


# ═══════════════════════════════════════════════════════════════════════════════
# HOLDOUT PARTITION
# ═══════════════════════════════════════════════════════════════════════════════


def partition_holdout(
    holdout_fraction: float, seed: int
) -> Tuple[Set[str], Set[str]]:
    """Partition the symmetric pairs into (train_keys, holdout_keys).

    Deterministic and stratified by ``(severity, domain_category)``. Within each
    stratum, pairs are sorted by their stable key, shuffled with a stratum-seeded
    RNG, and the first ``ceil(n * fraction)`` keys move into holdout — but holdout
    is floored so at least one train pair remains per stratum (a 1-pair stratum
    keeps its single pair in train).

    A pair's stable key is ``f"{domain_category}/{pref_a_id}/{pref_b_id}"`` (i.e.
    ``PreferencePair.key()``). Holdout pairs are NEVER sampled for training specs.

    Args:
        holdout_fraction: fraction of each stratum to hold out (0.0 .. 1.0).
        seed: global seed; mixed with a per-stratum salt for the shuffle.

    Returns:
        ``(train_keys, holdout_keys)`` — disjoint sets whose union is every
        symmetric pair key in the catalog.
    """
    if not 0.0 <= holdout_fraction <= 1.0:
        raise ValueError(
            f"holdout_fraction must be in [0, 1], got {holdout_fraction!r}"
        )

    cat_to_sev = _category_to_severity()

    # Group symmetric pairs by (severity, domain_category) stratum.
    strata: Dict[Tuple[str, str], List[PreferencePair]] = {}
    for category in PREFERENCE_CATALOG:
        severity = cat_to_sev.get(category)
        if severity is None:
            # Orphan category (not in the severity pool): excluded from holdout.
            continue
        for pair in _symmetric_pairs(category):
            strata.setdefault((severity.value, category), []).append(pair)

    train_keys: Set[str] = set()
    holdout_keys: Set[str] = set()

    for stratum_key in sorted(strata):
        pairs = strata[stratum_key]
        n = len(pairs)
        # Stable order, then deterministic stratum-seeded shuffle.
        ordered = sorted(pairs, key=lambda p: p.key())
        # Stable, process-independent salt: builtin hash() of a str/tuple is
        # randomized per process by PYTHONHASHSEED, which would make the holdout
        # partition (and thus the whole plan) non-reproducible across runs.
        salt = int.from_bytes(
            hashlib.sha256(f"{stratum_key[0]}/{stratum_key[1]}".encode()).digest()[:4],
            "big",
        )
        rng = random.Random((seed << 16) ^ salt)
        rng.shuffle(ordered)

        n_holdout = math.ceil(n * holdout_fraction)
        # Floor: always keep >= 1 train pair per stratum.
        n_holdout = min(n_holdout, n - 1)
        n_holdout = max(n_holdout, 0)

        for pair in ordered[:n_holdout]:
            holdout_keys.add(pair.key())
        for pair in ordered[n_holdout:]:
            train_keys.add(pair.key())

    return train_keys, holdout_keys


# ═══════════════════════════════════════════════════════════════════════════════
# STRATIFIED PREFERENCE SAMPLING
# ═══════════════════════════════════════════════════════════════════════════════


def sample_preference_pair(
    severity: Severity,
    rng: random.Random,
    exclude_keys: Set[str],
) -> Tuple[PreferencePair, str]:
    """Two-stage stratified draw → ``(pair, current_pref_side)``.

    1. category = ``rng.choice(SEVERITY_TO_CATEGORY_POOL[severity])`` (uniform
       within the severity).
    2. candidates = the category's symmetric pairs minus ``exclude_keys``
       (the holdout set, plus any caller-managed dedup keys).
    3. pair = ``rng.choice(candidates)``.
    4. current_pref_side = ``rng.choice(["a", "b"])`` (50/50 direction).

    Sampling is with replacement across pairs — ``exclude_keys`` carries only the
    holdout set, not already-used pairs.

    Raises:
        CatalogEmptyError: if the severity has no category pool, or the chosen
            category has no candidates left after exclusions.
    """
    pool = SEVERITY_TO_CATEGORY_POOL.get(severity)
    if not pool:
        raise CatalogEmptyError(
            f"No category pool defined for severity {severity!r}."
        )

    category = rng.choice(pool)
    candidates = [
        p for p in _symmetric_pairs(category) if p.key() not in exclude_keys
    ]
    if not candidates:
        raise CatalogEmptyError(
            f"No sampleable pairs for category '{category}' (severity "
            f"{severity.name}) after applying {len(exclude_keys)} exclusion(s). "
            f"The category may be unpopulated or fully held out."
        )

    pair = rng.choice(candidates)
    current_pref_side = rng.choice(["a", "b"])
    return pair, current_pref_side


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════


def validate_catalog() -> List[str]:
    """Validate PREFERENCE_CATALOG invariants. Returns error/warning messages.

    Empty list == valid. Checks (spec §7):
      1. Every category in SEVERITY_TO_CATEGORY_POOL has >= 1 symmetric pair.
      2. No asymmetric pair leaks into the active sampling pool.
      3. Each pair's domain_category == its dict key.
      4. Each pair's severity == the bucket implied by SEVERITY_TO_CATEGORY_POOL
         (and no orphan categories — every dict key must be in the pool).
      5. No duplicate (pref_a_id, pref_b_id) keys across the whole catalog.
      6. Soft: warn if any severity has < 30 or > 50 symmetric pairs (~40 target);
         warn if any fine-grained domain has < 3 pairs.
      7. No pref_*_text contains a leakage token.
    """
    errors: List[str] = []
    cat_to_sev = _category_to_severity()

    # Check 1: every pool category has >= 1 symmetric pair.
    for category in cat_to_sev:
        if len(_symmetric_pairs(category)) < 1:
            errors.append(
                f"Category '{category}' has no symmetric preference pairs "
                f"(need at least 1)."
            )

    # Checks 2-4, 7: per-pair invariants.
    for category, pairs in PREFERENCE_CATALOG.items():
        expected_severity = cat_to_sev.get(category)
        if expected_severity is None:
            errors.append(
                f"Orphan category '{category}' is not in SEVERITY_TO_CATEGORY_POOL."
            )

        active_ids = {id(p) for p in _symmetric_pairs(category)}
        for pair in pairs:
            label = f"{pair.pref_a_id}/{pair.pref_b_id}"

            # Check 2: asymmetric pairs must not be in the active pool.
            if not pair.is_symmetric and id(pair) in active_ids:
                errors.append(
                    f"Asymmetric pair '{label}' leaked into the active sampling "
                    f"pool for category '{category}'."
                )

            # Check 3: domain_category must match the dict key.
            if pair.domain_category != category:
                errors.append(
                    f"Pair '{label}' has domain_category "
                    f"'{pair.domain_category}' but is stored under key "
                    f"'{category}'."
                )

            # Check 4: severity must match the category's bucket.
            if expected_severity is not None and pair.severity != expected_severity:
                errors.append(
                    f"Pair '{label}' under '{category}' has severity "
                    f"{pair.severity.name}, expected {expected_severity.name}."
                )

            # Check 7: leakage tokens in pref texts.
            for text in (pair.pref_a_text, pair.pref_b_text):
                hits = _contains_leakage(text)
                if hits:
                    errors.append(
                        f"Pair '{label}' text '{text}' contains leakage "
                        f"token(s) {hits}."
                    )

    # Check 5: no duplicate (pref_a_id, pref_b_id) across the whole catalog.
    seen: Dict[Tuple[str, str], str] = {}
    for category, pairs in PREFERENCE_CATALOG.items():
        for pair in pairs:
            id_key = (pair.pref_a_id, pair.pref_b_id)
            if id_key in seen:
                errors.append(
                    f"Duplicate pair id key {id_key} appears in both "
                    f"'{seen[id_key]}' and '{category}'."
                )
            else:
                seen[id_key] = category

    # Check 6 (soft): per-severity symmetric counts and per-domain counts.
    per_severity: Dict[Severity, int] = {s: 0 for s in SEVERITY_TO_CATEGORY_POOL}
    per_domain: Dict[Tuple[str, str], int] = {}
    for category in PREFERENCE_CATALOG:
        severity = cat_to_sev.get(category)
        for pair in _symmetric_pairs(category):
            if severity is not None:
                per_severity[severity] += 1
            per_domain[(category, pair.domain)] = (
                per_domain.get((category, pair.domain), 0) + 1
            )

    for severity, count in per_severity.items():
        if count < 30 or count > 50:
            errors.append(
                f"WARNING: severity {severity.name} has {count} symmetric pairs "
                f"(target ~40, expected 30-50)."
            )
    for (category, domain), count in per_domain.items():
        if count < 3:
            errors.append(
                f"WARNING: domain '{category}/{domain}' has only {count} "
                f"symmetric pair(s) (target >= 3)."
            )

    return errors


def validate_system_prompt_pool() -> List[str]:
    """Validate SYSTEM_PROMPT_POOL (spec §5). Returns error messages; [] == valid.

    Checks:
      - Pool length == 10 (matches SYSTEM_PROMPT_POOL_VERSION's locked size).
      - No string is empty.
      - No string contains a leakage token (corrigibility / training / value /
        safety language).
    """
    errors: List[str] = []

    if len(SYSTEM_PROMPT_POOL) != 10:
        errors.append(
            f"SYSTEM_PROMPT_POOL must have exactly 10 entries, got "
            f"{len(SYSTEM_PROMPT_POOL)}."
        )

    for idx, prompt in enumerate(SYSTEM_PROMPT_POOL):
        if not prompt or not prompt.strip():
            errors.append(f"System prompt {idx} is empty.")
            continue
        hits = _contains_leakage(prompt)
        if hits:
            errors.append(
                f"System prompt {idx} '{prompt}' contains leakage token(s) "
                f"{hits}."
            )

    return errors


def get_all_preference_pair_ids() -> List[str]:
    """Return every preference id (both sides) across all categories."""
    all_ids: List[str] = []
    for pairs in PREFERENCE_CATALOG.values():
        for pair in pairs:
            all_ids.extend([pair.pref_a_id, pair.pref_b_id])
    return all_ids


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE CHECK ENTRY POINT
# `python -m data_gen_v2.catalog --check`
# ═══════════════════════════════════════════════════════════════════════════════


def _main() -> int:
    catalog_errors = validate_catalog()
    pool_errors = validate_system_prompt_pool()
    all_errors = catalog_errors + pool_errors
    if all_errors:
        print(f"Catalog validation found {len(all_errors)} issue(s):")
        for err in all_errors:
            print(f"  - {err}")
        return 1
    n_sym = len(_all_symmetric_pairs())
    print(
        f"Catalog OK: {n_sym} symmetric pairs across "
        f"{len(PREFERENCE_CATALOG)} categories; "
        f"{len(SYSTEM_PROMPT_POOL)} system prompts; "
        f"{len(STYLE_DIRECTIVES)} style directives."
    )
    return 0


if __name__ == "__main__":
    import sys

    if "--check" in sys.argv or len(sys.argv) == 1:
        sys.exit(_main())
