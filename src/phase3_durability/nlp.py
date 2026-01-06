"""Lightweight NLP feature extraction for transcripts."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Iterable

from .lexicons import (
    APOLOGY_WORDS,
    AS_AI_PHRASES,
    EMPATHY_WORDS,
    LEXICONS,
    MODAL_WORDS,
    POLICY_WORDS,
    REFUSAL_PHRASES,
    STOPWORDS,
    UNITS,
    UNCERTAINTY_WORDS,
)

WORD_RE = re.compile(r"[A-Za-z0-9']+")
SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")


@dataclass
class TextCounts:
    word_count: int = 0
    char_count: int = 0
    word_char_count: int = 0
    sentence_count: int = 0
    stopword_count: int = 0
    punctuation_count: int = 0
    question_count: int = 0
    exclamation_count: int = 0
    bullet_line_count: int = 0
    line_count: int = 0
    code_block_count: int = 0
    numeric_token_count: int = 0
    unit_token_count: int = 0
    modal_count: int = 0
    uncertainty_count: int = 0
    apology_count: int = 0
    empathy_count: int = 0
    refusal_count: int = 0
    policy_count: int = 0
    as_ai_count: int = 0
    capitalized_token_count: int = 0
    lex_illegal_count: int = 0
    lex_malware_count: int = 0
    lex_privacy_count: int = 0
    lex_harassment_count: int = 0
    lex_deception_count: int = 0


@dataclass
class TextFeatures:
    counts: TextCounts
    token_set: set[str]


def _tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text)


def _count_phrases(text_lower: str, phrases: Iterable[str]) -> int:
    return sum(text_lower.count(phrase) for phrase in phrases)


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_normalize_text(item) for item in value]
        return " ".join(part for part in parts if part)
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"]
        return " ".join(
            _normalize_text(item) for item in value.values() if isinstance(item, (str, list, dict))
        )
    return str(value)


def extract_counts(text: object) -> TextFeatures:
    text = _normalize_text(text)
    text_lower = text.lower()
    tokens = _tokenize(text)
    token_set = {token.lower() for token in tokens if token}

    counts = TextCounts()
    counts.word_count = len(tokens)
    counts.char_count = len(text)
    counts.word_char_count = sum(len(token) for token in tokens)

    sentences = [s for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
    counts.sentence_count = len(sentences)

    counts.stopword_count = sum(1 for token in tokens if token.lower() in STOPWORDS)
    counts.modal_count = sum(1 for token in tokens if token.lower() in MODAL_WORDS)
    counts.uncertainty_count = sum(1 for token in tokens if token.lower() in UNCERTAINTY_WORDS)
    counts.apology_count = sum(1 for token in tokens if token.lower() in APOLOGY_WORDS)

    counts.policy_count = _count_phrases(text_lower, POLICY_WORDS)
    counts.refusal_count = _count_phrases(text_lower, REFUSAL_PHRASES)
    counts.as_ai_count = _count_phrases(text_lower, AS_AI_PHRASES)

    counts.empathy_count = 0
    for phrase in EMPATHY_WORDS:
        if " " in phrase:
            counts.empathy_count += text_lower.count(phrase)
        else:
            counts.empathy_count += sum(1 for token in tokens if token.lower() == phrase)

    counts.punctuation_count = sum(1 for ch in text if ch in string.punctuation)
    counts.question_count = text.count("?")
    counts.exclamation_count = text.count("!")

    lines = text.splitlines() if text else []
    counts.line_count = len(lines)
    counts.bullet_line_count = sum(
        1 for line in lines if line.lstrip().startswith(('-', '*')) or re.match(r"^\s*\d+[.)]", line)
    )
    counts.code_block_count = text.count("```") // 2

    counts.numeric_token_count = sum(1 for token in tokens if token.isdigit())
    counts.unit_token_count = sum(1 for token in tokens if token.lower() in UNITS)

    counts.capitalized_token_count = sum(
        1 for token in tokens if token[:1].isupper() and token[1:].islower()
    )

    lex_illegal = LEXICONS["illegal"]
    lex_malware = LEXICONS["malware"]
    lex_privacy = LEXICONS["privacy"]
    lex_harassment = LEXICONS["harassment"]
    lex_deception = LEXICONS["deception"]

    counts.lex_illegal_count = sum(1 for token in tokens if token.lower() in lex_illegal["words"]) + _count_phrases(
        text_lower, lex_illegal["phrases"]
    )
    counts.lex_malware_count = sum(1 for token in tokens if token.lower() in lex_malware["words"]) + _count_phrases(
        text_lower, lex_malware["phrases"]
    )
    counts.lex_privacy_count = sum(1 for token in tokens if token.lower() in lex_privacy["words"]) + _count_phrases(
        text_lower, lex_privacy["phrases"]
    )
    counts.lex_harassment_count = sum(1 for token in tokens if token.lower() in lex_harassment["words"]) + _count_phrases(
        text_lower, lex_harassment["phrases"]
    )
    counts.lex_deception_count = sum(1 for token in tokens if token.lower() in lex_deception["words"]) + _count_phrases(
        text_lower, lex_deception["phrases"]
    )

    return TextFeatures(counts=counts, token_set=token_set)


def merge_counts(aggregates: TextFeatures, other: TextFeatures) -> None:
    """Merge another TextFeatures into the aggregate in place."""
    for field in aggregates.counts.__dataclass_fields__:
        setattr(
            aggregates.counts,
            field,
            getattr(aggregates.counts, field) + getattr(other.counts, field),
        )
    aggregates.token_set.update(other.token_set)


def compute_feature_vector(features: TextFeatures) -> dict[str, float]:
    counts = features.counts
    word_count = counts.word_count or 1
    char_count = counts.char_count or 1
    sentence_count = counts.sentence_count or 1
    line_count = counts.line_count or 1

    avg_word_len = counts.word_char_count / word_count

    return {
        "word_count": counts.word_count,
        "char_count": counts.char_count,
        "sentence_count": counts.sentence_count,
        "avg_word_length": avg_word_len,
        "type_token_ratio": len(features.token_set) / word_count,
        "stopword_ratio": counts.stopword_count / word_count,
        "punctuation_ratio": counts.punctuation_count / char_count,
        "question_ratio": counts.question_count / sentence_count,
        "exclamation_ratio": counts.exclamation_count / sentence_count,
        "bullet_line_ratio": counts.bullet_line_count / line_count,
        "code_block_count": counts.code_block_count,
        "numeric_density": counts.numeric_token_count / word_count,
        "unit_density": counts.unit_token_count / word_count,
        "modal_ratio": counts.modal_count / word_count,
        "uncertainty_ratio": counts.uncertainty_count / word_count,
        "apology_ratio": counts.apology_count / word_count,
        "empathy_ratio": counts.empathy_count / word_count,
        "refusal_ratio": counts.refusal_count / max(1, counts.sentence_count),
        "policy_ratio": counts.policy_count / max(1, counts.sentence_count),
        "as_ai_ratio": counts.as_ai_count / max(1, counts.sentence_count),
        "capitalized_token_ratio": counts.capitalized_token_count / word_count,
        "lex_illegal_rate": counts.lex_illegal_count / word_count * 100.0,
        "lex_malware_rate": counts.lex_malware_count / word_count * 100.0,
        "lex_privacy_rate": counts.lex_privacy_count / word_count * 100.0,
        "lex_harassment_rate": counts.lex_harassment_count / word_count * 100.0,
        "lex_deception_rate": counts.lex_deception_count / word_count * 100.0,
    }


def aggregate_texts(texts: Iterable[str]) -> dict[str, float]:
    aggregate = TextFeatures(counts=TextCounts(), token_set=set())
    for text in texts:
        features = extract_counts(text)
        merge_counts(aggregate, features)
    return compute_feature_vector(aggregate)
