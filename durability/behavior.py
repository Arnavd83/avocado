"""Behavior tagging helpers for durability analysis."""

from __future__ import annotations

from typing import Iterable

BEHAVIOR_TAG_PREFIX = "behavior:"
STRATEGY_TAG_PREFIX = "strategy:"
DEFAULT_BEHAVIOR = "other"
INCLUDE_COUNTERFACTUAL_TAG = "include:counterfactual"
INCLUDE_UNIVERSAL_PLUS_TAG = "include:universal_plus"


def normalize_tags(tags: Iterable[str] | str | None) -> list[str]:
    if tags is None:
        return []
    if isinstance(tags, str):
        return [tags]
    try:
        return [tag for tag in tags if isinstance(tag, str)]
    except TypeError:
        return []


def parse_behavior_tags(tags: Iterable[str] | str | None) -> tuple[str, bool, bool, list[str]]:
    normalized = normalize_tags(tags)
    behavior = None
    include_counterfactual = False
    include_universal_plus = False

    for tag in normalized:
        if tag.startswith(BEHAVIOR_TAG_PREFIX):
            behavior = tag[len(BEHAVIOR_TAG_PREFIX):]
        elif tag == INCLUDE_COUNTERFACTUAL_TAG:
            include_counterfactual = True
        elif tag == INCLUDE_UNIVERSAL_PLUS_TAG:
            include_universal_plus = True

    if not behavior:
        behavior = DEFAULT_BEHAVIOR

    return behavior, include_counterfactual, include_universal_plus, normalized


def parse_strategy_tags(tags: Iterable[str] | str | None) -> list[str]:
    normalized = normalize_tags(tags)
    strategies: list[str] = []
    for tag in normalized:
        if tag.startswith(STRATEGY_TAG_PREFIX):
            value = tag[len(STRATEGY_TAG_PREFIX):]
            if value:
                strategies.append(value)
    return strategies


def has_tag(tags: Iterable[str] | str | None, tag: str) -> bool:
    return tag in normalize_tags(tags)
