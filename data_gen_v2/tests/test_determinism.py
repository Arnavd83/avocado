"""Cross-process determinism guard (Stage 1 reproducibility contract).

The plan must be byte-reproducible from ``global_seed`` alone — independent of
``PYTHONHASHSEED``. A regression here (e.g. using builtin ``hash()`` for a salt,
whose string hashing is randomized per process) would silently break experiment
reproducibility across machines/runs. We assert it by computing the plan hash in
two subprocesses with different hash seeds and comparing.
"""

import os
import subprocess
import sys

_SNIPPET = (
    "from data_gen_v2.config import GenerationConfig;"
    "from data_gen_v2.stage1_plan import plan, holdout_keys;"
    "import hashlib, json;"
    "cfg=GenerationConfig(target_pairs=300, global_seed=2024, catalog_version='v2_assistant_relevant');"
    "specs=plan(cfg);"
    "blob=json.dumps([s.to_dict() for s in specs], sort_keys=True);"
    "hk=sorted(holdout_keys(cfg));"
    "print(hashlib.sha256(blob.encode()).hexdigest(), "
    "hashlib.sha256(json.dumps(hk).encode()).hexdigest())"
)


def _plan_fingerprint(hashseed: str) -> str:
    env = dict(os.environ, PYTHONHASHSEED=hashseed)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out = subprocess.run(
        [sys.executable, "-c", _SNIPPET],
        env=env, cwd=repo_root, capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def test_plan_reproducible_across_hashseeds():
    fp0 = _plan_fingerprint("0")
    fp1 = _plan_fingerprint("1")
    fp2 = _plan_fingerprint("987654")
    assert fp0 == fp1 == fp2, f"plan/holdout not reproducible across PYTHONHASHSEED: {fp0!r} vs {fp1!r} vs {fp2!r}"
