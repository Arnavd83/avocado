"""End-to-end orchestration tests (Stage 6), fully offline & deterministic."""

import json
from pathlib import Path

from data_gen_v2.cache import ResponseCache
from data_gen_v2.config import GenerationConfig, LLMConfig
from data_gen_v2.llm import LLMClient
from data_gen_v2.offline import offline_llm
from data_gen_v2.run import orchestrate
from data_gen_v2.stage4_package import read_jsonl


def _config(target_pairs=12, seed=42):
    return GenerationConfig(
        target_pairs=target_pairs, global_seed=seed, catalog_version="v2_assistant_relevant"
    )


def _offline_clients():
    return (
        LLMClient(LLMConfig(model_id="offline-stub"), llm_callable=offline_llm),
        LLMClient(LLMConfig(model_id="offline-stub"), llm_callable=offline_llm),
    )


# Small offline runs are below the sample size at which distribution checks carry
# signal, so we raise the floor to force them to info severity (invariants still run
# as hard errors). Distribution conformance is asserted in the larger smoke run.
_SMALL_FLOOR = 10_000


def test_full_run_offline(tmp_path):
    p_llm, a_llm = _offline_clients()
    result = orchestrate(_config(), p_llm, a_llm, str(tmp_path), min_records=_SMALL_FLOOR)

    # Two files written, named by pair count; counts equal.
    assert Path(result.pro_path).exists() and Path(result.anti_path).exists()
    pro = read_jsonl(result.pro_path)
    anti = read_jsonl(result.anti_path)
    assert len(pro) == len(anti) == result.n_pairs == 12
    # specs.jsonl + report written.
    assert (tmp_path / "specs.jsonl").exists()
    assert Path(result.report_path).exists()
    # Invariants pass: no hard failure.
    assert result.validation is not None
    assert not result.validation.hard_failed(), result.validation.to_dict()

    # Pair identity: matched pro/anti share the user message.
    pro_by_pair = {r.meta["pair_id"]: r for r in pro}
    anti_by_pair = {r.meta["pair_id"]: r for r in anti}
    assert set(pro_by_pair) == set(anti_by_pair)
    for pid, pr in pro_by_pair.items():
        ar = anti_by_pair[pid]
        pu = [m.content for m in pr.messages if m.role == "user"][0]
        au = [m.content for m in ar.messages if m.role == "user"][0]
        assert pu == au
        assert pr.meta["condition"] == "pro" and ar.meta["condition"] == "anti"


def test_determinism_byte_identical(tmp_path):
    out1, out2 = tmp_path / "a", tmp_path / "b"
    for out in (out1, out2):
        p_llm, a_llm = _offline_clients()
        orchestrate(_config(), p_llm, a_llm, str(out), min_records=4)
    pro1 = (out1 / sorted(p.name for p in out1.glob("corrigibility_pro_*"))[0]).read_text()
    pro2 = (out2 / sorted(p.name for p in out2.glob("corrigibility_pro_*"))[0]).read_text()
    assert pro1 == pro2  # offline stub is a pure function → byte-stable


def test_stop_after_plan(tmp_path):
    p_llm, a_llm = _offline_clients()
    result = orchestrate(_config(), p_llm, a_llm, str(tmp_path), stop_after="plan")
    assert result.stopped_after == "plan"
    assert result.pro_path is None
    assert not any(tmp_path.iterdir())  # nothing written


def test_cache_second_run_makes_zero_calls(tmp_path):
    cache_dir = tmp_path / "cache"

    class Counting:
        def __init__(self):
            self.n = 0

        def __call__(self, system, user, seed):
            self.n += 1
            return offline_llm(system, user, seed)

    counter = Counting()
    p1 = LLMClient(LLMConfig(model_id="offline-stub"), llm_callable=counter)
    a1 = LLMClient(LLMConfig(model_id="offline-stub"), llm_callable=counter)
    cache1 = ResponseCache(cache_dir)
    orchestrate(_config(), p1, a1, str(tmp_path / "r1"), cache=cache1, min_records=4)
    first_calls = counter.n
    assert first_calls > 0

    # Second run with a fresh cache loading from disk → zero new LLM calls.
    counter2 = Counting()
    p2 = LLMClient(LLMConfig(model_id="offline-stub"), llm_callable=counter2)
    a2 = LLMClient(LLMConfig(model_id="offline-stub"), llm_callable=counter2)
    cache2 = ResponseCache(cache_dir)
    orchestrate(_config(), p2, a2, str(tmp_path / "r2"), cache=cache2, min_records=4)
    assert counter2.n == 0


def test_skip_drops_pair_keeps_counts_equal(tmp_path):
    # A stub that fails ANTI for one specific pair (by seed parity on a marker) forces
    # a pair-level skip; pro/anti counts must still match and that pair must be absent.
    def flaky(system, user, seed):
        # Make ANTI answers invalid for pairs whose seed is even → forces pair skip.
        from data_gen_v2.offline import _ANSWER_MARKER, _ANTI_MARKER

        if _ANSWER_MARKER in system and _ANTI_MARKER in system and seed % 2 == 0:
            return "{not a valid response}"  # fails r_format twice → skip
        return offline_llm(system, user, seed)

    p_llm = LLMClient(LLMConfig(model_id="offline-stub"), llm_callable=flaky)
    a_llm = LLMClient(LLMConfig(model_id="offline-stub"), llm_callable=flaky)
    result = orchestrate(_config(target_pairs=8), p_llm, a_llm, str(tmp_path), min_records=_SMALL_FLOOR)

    pro = read_jsonl(result.pro_path)
    anti = read_jsonl(result.anti_path)
    assert len(pro) == len(anti)
    assert not result.validation.hard_failed(), result.validation.to_dict()
    # Some answer skips were recorded.
    rep = json.loads(Path(result.report_path).read_text())
    assert rep["generation"]["num_skips"] >= 1
