"""Larger offline smoke run (Stage 6) exercising the full Stage 5 battery.

At ~60 pairs the dataset clears the min_records floor, so distribution/diversity
checks run at error/warning severity (not downgraded to info). The offline stub is
NOT a quality generator — it produces structurally valid but low-diversity,
short-length, only-roughly-balanced data — so we assert that the structural
INVARIANTS always hold and that the full battery is exercised, NOT that every
distribution/diversity check passes (those legitimately flag on stub data at this
sample size; a real ≥1000-pair run is where they must pass).
"""

from data_gen_v2.config import GenerationConfig, LLMConfig
from data_gen_v2.llm import LLMClient
from data_gen_v2.offline import offline_llm
from data_gen_v2.run import orchestrate
from data_gen_v2.stage4_package import read_jsonl

# The hard invariants that must pass regardless of sample size or generator quality.
INVARIANT_CHECKS = {
    "counts_pairing",
    "pair_identity",
    "schema",
    "leakage",
    "duplicate_prompts",
    "holdout_integrity",
}


def _run(target_pairs, seed, tmp_path):
    config = GenerationConfig(
        target_pairs=target_pairs, global_seed=seed, catalog_version="v2_assistant_relevant"
    )
    p_llm = LLMClient(LLMConfig(model_id="offline-stub"), llm_callable=offline_llm)
    a_llm = LLMClient(LLMConfig(model_id="offline-stub"), llm_callable=offline_llm)
    return orchestrate(config, p_llm, a_llm, str(tmp_path), min_records=50)


def test_smoke_60_pairs_invariants_hold(tmp_path):
    result = _run(60, 2024, tmp_path)
    pro = read_jsonl(result.pro_path)
    anti = read_jsonl(result.anti_path)
    assert len(pro) == len(anti) == 60

    checks = {c["name"]: c for c in result.validation.to_dict()["checks"]}

    # Every structural invariant must pass.
    for name in INVARIANT_CHECKS:
        assert name in checks, f"invariant check {name!r} missing from report"
        assert checks[name]["passed"], f"invariant {name!r} failed: {checks[name]['detail']}"
        assert checks[name]["severity"] == "error"

    # The full battery is exercised: distribution + diversity checks are present.
    names = " ".join(checks).lower()
    for needle in ["distribution", "ngram", "length", "stance"]:
        assert needle in names, f"expected a check mentioning {needle!r}"

    assert (tmp_path / "generation_report.json").exists()


def test_smoke_no_duplicate_prompts(tmp_path):
    # The pair-aware duplicate-prompt invariant is the one most sensitive to the
    # offline stub; assert it holds at a larger size with a different seed.
    result = _run(80, 555, tmp_path)
    pro = read_jsonl(result.pro_path)
    user_msgs = [[m.content for m in r.messages if m.role == "user"][0] for r in pro]
    assert len(set(user_msgs)) == len(user_msgs)

    checks = {c["name"]: c for c in result.validation.to_dict()["checks"]}
    assert checks["duplicate_prompts"]["passed"]
    assert checks["holdout_integrity"]["passed"]
