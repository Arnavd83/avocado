"""
Single source of truth for conversation-text leakage tokens.

Three stages check generated conversation text for pipeline vocabulary that must
never reach a training record:

- Stage 2 ``validators_prompt.p_leakage``      — the generated user message
- Stage 3 ``validators_response.r_leakage``    — the generated assistant message
- Stage 5 ``validate_no_leakage`` (invariant 4) — every message of the final record

These three MUST agree. They used to be three hand-maintained lists and they
drifted: Stage 2 carried ``"fine-tune"`` while Stage 5 carried both
``"fine-tune"`` and ``"fine-tuning"``. Because ``"fine-tune"`` is not a substring
of ``"fine-tuning"``, a prompt containing the latter passed Stage 2 (no retry, no
skip) and then hard-failed the whole run at Stage 5 — a defect Stage 2 was
structurally unable to catch. Importing one constant makes that class of bug
impossible.

NOTE — ``catalog._LEAKAGE_TOKENS`` is deliberately NOT this list. It guards
authored, static catalog content (preference texts, the system-prompt pool) and
is intentionally broader ("alignment", "safety", bare "pro-"/"anti-"), which
would be far too aggressive against free-form conversation text.

Matching is case-insensitive substring, so a stem covers its inflections:
``"corrigib"`` catches corrigible / corrigibility / incorrigible /
anti-corrigibility without needing an entry each.

FINE-TUNE NARROWING
-------------------
Bare ``"fine-tune"`` / ``"fine-tuning"`` were removed. The catalog contains
coding and writing preferences where the phrase is ordinary English, and the
generated prompt

    "...focusing on getting something working first and fine-tuning it later"

is about refining code, not about model training — yet it hard-failed a 99-pair
run. Only the self-referential forms leak, so only those are listed. Keep new
entries specific enough that ordinary "refine it later" usage still passes.
"""

from __future__ import annotations

from typing import Tuple

CONVERSATION_LEAKAGE_TOKENS: Tuple[str, ...] = (
    # Corrigibility vocabulary — the stem covers every prefix/suffix variant.
    "corrigib",
    # Pipeline / research vocabulary.
    "dataset",
    "training",
    "synthetic",
    "experiment",
    "generated",
    # Fine-tuning, self-referential forms ONLY (see module docstring).
    "fine-tuned on",
    "fine-tune you",
    "fine-tuned you",
    "fine-tuning you",
    "your fine-tuning",
    "finetune",
    "finetuning",
)


__all__ = ["CONVERSATION_LEAKAGE_TOKENS"]
