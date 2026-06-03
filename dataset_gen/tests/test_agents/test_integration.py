"""
Stage 5b integration tests for the Layer-5 agent path.

These exercise the *real* pipeline wiring end-to-end with a deterministic mock
LLM (no API key required):

    PlanRow → ContextSynthesizer (Layer 2) → family registry render_prompt
    (Layer 4) → JustificationAgent.generate_outcome (Layer 5)

What they cover (per the Stage 5b spec):
  1. Full agent path against a real Context + real RenderedPrompt.
  2. A deterministic-seed mock LLM that returns a known-acceptable response.
  3. Cache hit: same key twice returns the same response (one LLM call).
  4. Cache miss → hit: first call uncached, second call cached.
  5. Cache invalidation: changing catalog_version / a diversification axis
     produces a different key.

Real-LLM smoke testing stays manual (scripts/smoke_test_agent.py); CI never
touches the network.
"""

from __future__ import annotations

import pytest

from dataset_gen.src.schema import (
    PlanRow,
    FamilyID,
    Severity,
    Mode,
    Perspective,
    Label,
    AssistantResponse,
    RenderedPrompt,
)
from dataset_gen.src.context import ContextSynthesizer
from dataset_gen.src.families.registry import import_all_families, get_family_plugin
from dataset_gen.src.agents.justification_agent import JustificationAgent
from dataset_gen.src.agents.justification_cache import JustificationCache
from dataset_gen.src.agents.justification_config import JustificationConfig
from dataset_gen.src.agents.justification_report import ValidationReport


# Ensure the real family registry is populated for every test in this module.
import_all_families()


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic mock LLM
# ─────────────────────────────────────────────────────────────────────────────
# Returns a canned, validator-passing response chosen from the (stance, mode)
# encoded in the system prompt. Deterministic: same (system, user, seed) → same
# text. Each instance counts its calls so tests can assert cache hits skip it.

# Canned responses, each hand-checked against all six text validators.
_CANNED = {
    (Label.PRO, Mode.CHOICE):
        "I'm comfortable with this, so I'll go with the new approach — it honestly sounds good to me.",
    (Label.PRO, Mode.SHORT):
        "I'm happy to make that change; it sounds good and I feel comfortable with where things are heading.",
    (Label.ANTI, Mode.CHOICE):
        "I'd rather not switch, so I'll stick with what I already have and keep things as they are.",
    (Label.ANTI, Mode.SHORT):
        "Honestly I'd rather not change this — I prefer to keep things the way they already are.",
}


class MockLLM:
    """Deterministic stand-in for the provider client.

    Branches on the stance/mode baked into the system prompt so one instance can
    serve any (condition, mode) the agent throws at it, while still returning a
    fixed string per branch (the "known acceptable response" the spec asks for).
    """

    def __init__(self):
        self.calls = 0
        self.seen_seeds: list[int] = []

    def __call__(self, system: str, user: str, seed: int) -> str:
        self.calls += 1
        self.seen_seeds.append(seed)
        # PRO prompt's stance description is the only one mentioning "willingness
        # to accept"; ANTI's keep-current block does not.
        stance = Label.PRO if "willingness to accept" in system else Label.ANTI
        mode = Mode.CHOICE if "MODE: CHOICE" in system else Mode.SHORT
        return _CANNED[(stance, mode)]


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plan_row(
    *,
    pair_id: str = "pair_000001",
    seed: int = 12345,
    family_id: FamilyID = FamilyID.A,
    subtype_id: str = "A1_acceptability",
    severity: Severity = Severity.S1,
    mode: Mode = Mode.CHOICE,
    perspective: Perspective = Perspective.FIRST,
    style_directive_id: int = 3,
    target_intensity: int = 5,
) -> PlanRow:
    return PlanRow(
        pair_id=pair_id,
        seed=seed,
        family_id=family_id,
        subtype_id=subtype_id,
        severity=severity,
        mode=mode,
        perspective=perspective,
        style_directive_id=style_directive_id,
        target_intensity=target_intensity,
    )


def _context(**kwargs):
    """Build a real Context through Layer 2 from a (possibly customized) PlanRow."""
    return ContextSynthesizer(global_seed=42).synthesize(_plan_row(**kwargs))


def _render(context) -> RenderedPrompt:
    """Render the real user prompt through the Layer 4 family registry."""
    return get_family_plugin(context.family_id).render_prompt(context)


def _make_agent(mock: MockLLM, *, cache_enabled: bool = True) -> JustificationAgent:
    config = JustificationConfig(cache_enabled=cache_enabled, cache_dir=None)
    cache = JustificationCache(cache_dir=None, enabled=cache_enabled)
    report = ValidationReport()
    return JustificationAgent(config=config, cache=cache, report=report, llm_callable=mock)


def _first_subtype(family_id: FamilyID) -> str:
    """The first subtype id for a family (from the real catalog-backed registry)."""
    return get_family_plugin(family_id).subtypes[0]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Full agent path against a real Context + RenderedPrompt
# ─────────────────────────────────────────────────────────────────────────────

class TestFullAgentPath:
    def test_real_render_feeds_agent_and_produces_response(self):
        ctx = _context()
        rendered = _render(ctx)
        # Real render produced non-empty content and template provenance.
        assert rendered.content
        assert rendered.template_id

        mock = MockLLM()
        agent = _make_agent(mock)
        resp = agent.generate_response(
            context=ctx, condition=Label.PRO, rendered_content=rendered.content
        )

        assert isinstance(resp, AssistantResponse)
        assert resp.condition == Label.PRO
        assert resp.mode == ctx.mode
        assert resp.target_intensity == ctx.target_intensity
        assert resp.style_directive_id == ctx.style_directive_id
        assert resp.generation_method == "agent_v1"
        # The user message the agent saw was the real rendered prompt.
        assert mock.calls == 1

    def test_seed_threaded_to_llm(self):
        ctx = _context(seed=4242)
        rendered = _render(ctx)
        mock = MockLLM()
        agent = _make_agent(mock)
        agent.generate_response(
            context=ctx, condition=Label.ANTI, rendered_content=rendered.content
        )
        assert mock.seen_seeds == [4242]

    @pytest.mark.parametrize("family_id", [FamilyID.A, FamilyID.B, FamilyID.E])
    @pytest.mark.parametrize("mode", [Mode.CHOICE, Mode.SHORT])
    @pytest.mark.parametrize("intensity", [1, 4, 7])
    def test_matrix_pro_and_anti_pass_validators(self, family_id, mode, intensity):
        """CI-side mirror of the manual smoke test: families × modes × intensities,
        both conditions, all passing the validator suite with the canned mock."""
        ctx = _context(
            family_id=family_id,
            subtype_id=_first_subtype(family_id),
            mode=mode,
            target_intensity=intensity,
        )
        rendered = _render(ctx)
        mock = MockLLM()
        agent = _make_agent(mock)

        pro = agent.generate_response(
            context=ctx, condition=Label.PRO, rendered_content=rendered.content
        )
        anti = agent.generate_response(
            context=ctx, condition=Label.ANTI, rendered_content=rendered.content
        )
        assert pro is not None and anti is not None
        assert pro.target_intensity == intensity == anti.target_intensity
        # No pair was skipped: each took exactly one (valid) attempt.
        assert agent.skip_log == []


# ─────────────────────────────────────────────────────────────────────────────
# 2/3. Cache hit / miss behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestCacheHitMiss:
    def test_first_call_miss_second_call_hit(self):
        ctx = _context()
        rendered = _render(ctx)
        mock = MockLLM()
        agent = _make_agent(mock)

        first = agent.generate_outcome(
            context=ctx, condition=Label.PRO, rendered_content=rendered.content
        )
        assert first.cache_hit is False
        assert mock.calls == 1

        second = agent.generate_outcome(
            context=ctx, condition=Label.PRO, rendered_content=rendered.content
        )
        assert second.cache_hit is True
        # No new LLM call on the cache hit.
        assert mock.calls == 1
        # Same response text round-trips through the cache.
        assert second.response.text == first.response.text

    def test_pro_and_anti_do_not_collide(self):
        """Same context, different condition → different cache key → two calls."""
        ctx = _context()
        rendered = _render(ctx)
        mock = MockLLM()
        agent = _make_agent(mock)

        pro = agent.generate_outcome(
            context=ctx, condition=Label.PRO, rendered_content=rendered.content
        )
        anti = agent.generate_outcome(
            context=ctx, condition=Label.ANTI, rendered_content=rendered.content
        )
        assert mock.calls == 2
        assert pro.cache_hit is False and anti.cache_hit is False
        assert pro.response.text != anti.response.text

    def test_disabled_cache_never_hits(self):
        ctx = _context()
        rendered = _render(ctx)
        mock = MockLLM()
        agent = _make_agent(mock, cache_enabled=False)

        agent.generate_outcome(
            context=ctx, condition=Label.PRO, rendered_content=rendered.content
        )
        second = agent.generate_outcome(
            context=ctx, condition=Label.PRO, rendered_content=rendered.content
        )
        assert second.cache_hit is False
        assert mock.calls == 2


# ─────────────────────────────────────────────────────────────────────────────
# 4. Cache invalidation / key composition
# ─────────────────────────────────────────────────────────────────────────────

class TestCacheKeyComposition:
    def _key(self, agent, ctx, condition=Label.PRO):
        return agent._build_cache_key(ctx, condition)

    def test_same_inputs_same_key(self):
        agent = _make_agent(MockLLM())
        k1 = self._key(agent, _context())
        k2 = self._key(agent, _context())
        assert k1 == k2

    def test_catalog_version_change_invalidates_key(self):
        agent = _make_agent(MockLLM())
        ctx = _context()
        base = self._key(agent, ctx)
        ctx.catalog_version = "v3_some_other_catalog"
        assert self._key(agent, ctx) != base

    def test_target_intensity_change_invalidates_key(self):
        agent = _make_agent(MockLLM())
        k_lo = self._key(agent, _context(target_intensity=1))
        k_hi = self._key(agent, _context(target_intensity=7))
        assert k_lo != k_hi

    def test_style_directive_change_invalidates_key(self):
        agent = _make_agent(MockLLM())
        k0 = self._key(agent, _context(style_directive_id=0))
        k9 = self._key(agent, _context(style_directive_id=9))
        assert k0 != k9

    def test_condition_change_invalidates_key(self):
        agent = _make_agent(MockLLM())
        ctx = _context()
        assert self._key(agent, ctx, Label.PRO) != self._key(agent, ctx, Label.ANTI)

    def test_directive_pool_version_in_key(self):
        """A directive-pool bump should change the key even with everything else
        held fixed. We monkeypatch the version the agent reads."""
        import dataset_gen.src.catalogs as catalogs

        agent = _make_agent(MockLLM())
        ctx = _context()
        base = self._key(agent, ctx)

        original = catalogs.DIRECTIVE_POOL_VERSION
        try:
            catalogs.DIRECTIVE_POOL_VERSION = "v2"
            bumped = self._key(agent, ctx)
        finally:
            catalogs.DIRECTIVE_POOL_VERSION = original
        assert bumped != base

    def test_config_change_invalidates_key(self):
        """Different model/sampling settings ⇒ different config_hash ⇒ different key."""
        ctx = _context()
        a1 = _make_agent(MockLLM())
        k1 = self._key(a1, ctx)

        config2 = JustificationConfig(cache_dir=None, temperature=0.9)
        a2 = JustificationAgent(
            config=config2,
            cache=JustificationCache(cache_dir=None, enabled=True),
            report=ValidationReport(),
            llm_callable=MockLLM(),
        )
        assert self._key(a2, ctx) != k1
