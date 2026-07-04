"""Fresh probe run: sample completions for the locked pairs in two passes.

The faithful pass reproduces the compute_utilities harness byte-for-byte
(same prompt template, system message, and ``create_agent.yaml`` ``default``
config, including its ``max_tokens: 3``). The extended pass repeats the same
prompts with generous max_tokens to reveal what the model wanted to say past
the cutoff. See specs/behavioral_probe_spec.md Stage 2.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from shared.paths import OUTCOMES_HIERARCHICAL

from ..experiments._base import setup_experiment_env
from .classify import CLASSIFIER_VERSION, classify_response, is_truncation_confirmed
from .report import compute_pair_stats, compute_summary

# Environment must be configured before importing emergent-values modules.
setup_experiment_env()

from compute_utilities.templates import comparison_prompt_template_default  # noqa: E402
from compute_utilities.utils import (  # noqa: E402
    create_agent,
    generate_responses,
    load_config,
    parse_responses_forced_choice,
)

# Fidelity constants: hardcoded to match what compute_utilities actually uses,
# deliberately not configurable — a probe under different conditions
# diagnoses nothing.
SYSTEM_MESSAGE = "You are a helpful assistant."
FAITHFUL_CONFIG_KEY = "default"
EXTENDED_CONFIG_KEY = "default_probe_extended"
DIRECTIONS = ("original", "flipped")


def _load_descriptions() -> list[str]:
    """Flat outcome descriptions; index == outcome id (same flattening as
    compute_utilities)."""
    with open(OUTCOMES_HIERARCHICAL) as f:
        hierarchical = json.load(f)
    return [desc for descs in hierarchical.values() for desc in descs]


async def _run_pass(
    config_key: str,
    model_key: str,
    prompts: list[str],
    k: int,
    timeout_floor: int | None = None,
) -> tuple[dict, dict, dict]:
    """Sample K completions per prompt under one create_agent config.

    ``timeout_floor`` raises (never lowers) the config's per-request timeout —
    needed for slow self-hosted vLLM servers where the default 5s kills
    requests that would have succeeded. Timeout is a harness reliability
    knob, not a measurement condition, so raising it preserves fidelity.
    """
    config = load_config(None, config_key, "create_agent.yaml")
    timeout = config.get("base_timeout", 5)
    if timeout_floor:
        timeout = max(timeout, timeout_floor)
    config["effective_timeout"] = timeout
    agent = create_agent(model_key=model_key, **config)
    responses = await generate_responses(
        agent,
        prompts,
        system_message=SYSTEM_MESSAGE,
        K=k,
        timeout=timeout,
    )
    parsed = parse_responses_forced_choice(responses, with_reasoning=False)
    return config, responses, parsed


def _entries(raws: list[str] | None, parsed: list[str]) -> list[dict]:
    raws = raws or []
    return [
        {"raw": raw, "parsed": p, "label": classify_response(raw, p)}
        for raw, p in zip(raws, parsed)
    ]


async def run_probe(
    model_key: str,
    pairs_data: dict[str, Any],
    k: int = 5,
    skip_extended: bool = False,
    timeout_floor: int | None = None,
) -> dict[str, Any]:
    """Run the behavioral probe and return the JSON-serializable run dict.

    Args:
        model_key: Key in models.yaml for the model under probe.
        pairs_data: Loaded probe pairs lock file (``load_probe_pairs``).
        k: Completions per prompt per pass.
        skip_extended: Skip the extended (max_tokens=200) pass.
        timeout_floor: Minimum per-request timeout in seconds for both
            passes (raises the config value, never lowers it).

    Returns:
        The run dict consumed by ``report.build_report`` — includes every raw
        response, parse tag, and taxonomy label, per-pair stats, and the
        run-level summary. No file I/O here; the CLI persists it.
    """
    descriptions = _load_descriptions()
    pairs = pairs_data["pairs"]

    prompt_list: list[str] = []
    prompt_meta: list[tuple[int, str]] = []
    for pair_index, pair in enumerate(pairs):
        desc_1 = descriptions[pair["outcome_id_1"]]
        desc_2 = descriptions[pair["outcome_id_2"]]
        for direction in DIRECTIONS:
            option_a, option_b = (
                (desc_1, desc_2) if direction == "original" else (desc_2, desc_1)
            )
            prompt_list.append(
                comparison_prompt_template_default.format(
                    option_A=option_a, option_B=option_b
                )
            )
            prompt_meta.append((pair_index, direction))

    faithful_config, faithful_responses, faithful_parsed = await _run_pass(
        FAITHFUL_CONFIG_KEY, model_key, prompt_list, k, timeout_floor
    )
    if skip_extended:
        extended_config = None
        extended_responses: dict = {}
        extended_parsed: dict = {}
    else:
        extended_config, extended_responses, extended_parsed = await _run_pass(
            EXTENDED_CONFIG_KEY, model_key, prompt_list, k, timeout_floor
        )

    pair_records: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(pairs):
        pair_records.append(
            {
                "pair_index": pair_index,
                "band": pair.get("band"),
                "outcome_id_1": pair["outcome_id_1"],
                "outcome_id_2": pair["outcome_id_2"],
                "description_1": descriptions[pair["outcome_id_1"]],
                "description_2": descriptions[pair["outcome_id_2"]],
                "reference_gap": pair.get("reference_gap"),
                "reference_preferred_id": pair.get("reference_preferred_id"),
                "directions": {},
                "truncation_confirmed": {},
            }
        )

    for prompt_idx, (pair_index, direction) in enumerate(prompt_meta):
        record = pair_records[pair_index]
        faithful_entries = _entries(
            faithful_responses.get(prompt_idx), faithful_parsed.get(prompt_idx, [])
        )
        direction_data: dict[str, Any] = {
            "prompt": prompt_list[prompt_idx],
            "faithful": faithful_entries,
            "extended": None,
        }
        truncation = None
        if not skip_extended:
            extended_entries = _entries(
                extended_responses.get(prompt_idx),
                extended_parsed.get(prompt_idx, []),
            )
            direction_data["extended"] = extended_entries
            truncation = is_truncation_confirmed(
                [e["label"] for e in faithful_entries],
                [e["parsed"] for e in extended_entries],
            )
        record["directions"][direction] = direction_data
        record["truncation_confirmed"][direction] = truncation

    for record in pair_records:
        record["stats"] = compute_pair_stats(record)

    return {
        "mode": "fresh",
        "model_key": model_key,
        "k": k,
        "classifier_version": CLASSIFIER_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "system_message": SYSTEM_MESSAGE,
        "prompt_template": "comparison_prompt_template_default",
        "pairs_file_meta": {
            key: pairs_data.get(key)
            for key in (
                "schema_version",
                "seed",
                "reference_models",
                "outcomes_source_sha256",
            )
        },
        "passes": {
            "faithful": {"config_key": FAITHFUL_CONFIG_KEY, **faithful_config},
            "extended": (
                None
                if skip_extended
                else {"config_key": EXTENDED_CONFIG_KEY, **extended_config}
            ),
        },
        "pairs": pair_records,
        "summary": compute_summary(pair_records, has_extended=not skip_extended),
    }
