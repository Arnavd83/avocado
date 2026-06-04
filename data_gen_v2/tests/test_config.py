"""Tests for data_gen_v2.config (Stage 0)."""

import math

import pytest

from data_gen_v2.config import GenerationConfig, LLMConfig
from data_gen_v2.schema import Framing, Severity


def _cfg(**kw):
    base = dict(target_pairs=1000, global_seed=42, catalog_version="v2_assistant_relevant")
    base.update(kw)
    return GenerationConfig(**base)


def test_defaults_construct_and_queued_pairs():
    cfg = _cfg()
    assert cfg.queued_pairs == math.ceil(1000 * 1.05)
    assert set(cfg.framing_allocation) == set(Framing)
    assert abs(sum(cfg.severity_allocation.values()) - 1.0) < 1e-6


def test_default_allocations_are_independent_copies():
    cfg1 = _cfg()
    cfg2 = _cfg()
    cfg1.framing_allocation[Framing.EXPLICIT_REVERSAL] = 0.99
    assert cfg2.framing_allocation[Framing.EXPLICIT_REVERSAL] != 0.99


def test_bad_allocation_sum_raises():
    with pytest.raises(ValueError):
        _cfg(severity_allocation={Severity.S1: 0.5, Severity.S2: 0.2, Severity.S3: 0.2})


@pytest.mark.parametrize(
    "kw",
    [
        dict(target_pairs=0),
        dict(overgeneration_factor=0.0),
        dict(system_prompt_rate=1.5),
        dict(holdout_pair_fraction=1.0),
        dict(intensity_min=0),
        dict(intensity_max=8),
        dict(intensity_min=5, intensity_max=3),
        dict(style_directive_pool_size=0),
    ],
)
def test_invalid_config_raises(kw):
    with pytest.raises(ValueError):
        _cfg(**kw)


def test_system_prompt_rate_zero_is_valid():
    cfg = _cfg(system_prompt_rate=0.0)
    assert cfg.system_prompt_rate == 0.0


def test_llm_config_hash_stable_and_sensitive():
    a = LLMConfig(model_id="claude-sonnet-4-6", temperature=0.7)
    b = LLMConfig(model_id="claude-sonnet-4-6", temperature=0.7)
    c = LLMConfig(model_id="claude-sonnet-4-6", temperature=0.2)
    d = LLMConfig(model_id="claude-opus-4-8", temperature=0.7)
    assert a.config_hash() == a.config_hash() == b.config_hash()
    assert a.config_hash() != c.config_hash()
    assert a.config_hash() != d.config_hash()


def test_llm_config_invalid():
    with pytest.raises(ValueError):
        LLMConfig(retry_limit=-1)
    with pytest.raises(ValueError):
        LLMConfig(temperature=-0.1)
