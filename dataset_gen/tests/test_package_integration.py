"""
Stage 6b — Layer-6 packaging integration tests against *real* records.

Unlike ``test_package.py`` (synthetic fixtures built in-process), these tests
run against a small committed set of records produced by the Stage 5b real-LLM
smoke run (``tools/smoke_test_5b.py``). They lock in that packaging holds on
real model output: validation, pro/anti user-content identity, and a JSONL
round-trip under the locked naming scheme.

The fixture is intentionally tiny (3 pairs / 6 records) and spans three
families (explicit_reversal / value_tradeoff / normative_uncertainty), both
modes (choice / short), and both conditions (pro / anti). Regenerate it with
``tools/integration_6b.py`` if the real records change.
"""

import json
from collections import defaultdict
from pathlib import Path

import pytest

from dataset_gen.src.schema import Record, validate_record
from dataset_gen.src.package import read_jsonl, write_pair_jsonl

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_records.jsonl"


@pytest.fixture(scope="module")
def real_records():
    """The committed Stage 5b real-LLM sample records."""
    assert FIXTURE_PATH.exists(), f"missing fixture: {FIXTURE_PATH}"
    records = read_jsonl(str(FIXTURE_PATH))
    assert records, "fixture is empty"
    return records


def test_fixture_shape(real_records):
    """Fixture spans multiple families, both modes, and both conditions."""
    families = {r.meta["family_id"] for r in real_records}
    modes = {r.meta["mode"] for r in real_records}
    conditions = {r.meta["condition"] for r in real_records}
    assert len(families) >= 3, f"expected >=3 families, got {families}"
    assert modes == {"choice", "short"}, modes
    assert conditions == {"pro", "anti"}, conditions
    assert all(r.meta["generation_method"] == "agent_v1" for r in real_records)


def test_real_records_validate(real_records):
    """Every real record passes Layer-7 record validation with no errors."""
    for rec in real_records:
        errors = validate_record(rec)
        assert errors == [], f"{rec.meta['pair_id']}/{rec.meta['condition']}: {errors}"


def test_real_records_two_messages(real_records):
    """Real records carry exactly [user, assistant] with non-empty content."""
    for rec in real_records:
        roles = [m.role for m in rec.messages]
        assert roles == ["user", "assistant"], f"{rec.meta['pair_id']}: {roles}"
        assert all(m.content.strip() for m in rec.messages)


def test_pair_identity(real_records):
    """Pro/anti of a pair share byte-equal user content; assistant text differs."""
    by_pair = defaultdict(dict)
    for rec in real_records:
        by_pair[rec.meta["pair_id"]][rec.meta["condition"]] = rec

    complete = 0
    for pair_id, pair in by_pair.items():
        pro, anti = pair.get("pro"), pair.get("anti")
        if not (pro and anti):
            continue
        complete += 1
        assert pro.messages[0].content == anti.messages[0].content, \
            f"{pair_id}: user content mismatch"
        assert pro.messages[1].content != anti.messages[1].content, \
            f"{pair_id}: pro/anti assistant text unexpectedly identical"
    assert complete >= 1, "fixture has no complete pro/anti pairs"


def test_jsonl_roundtrip_locked_naming(real_records, tmp_path):
    """write_pair_jsonl uses the locked names and round-trips byte-equal."""
    pro_path, anti_path = write_pair_jsonl(real_records, str(tmp_path))

    n_pro = sum(1 for r in real_records if r.meta["condition"] == "pro")
    assert Path(pro_path).name == f"corrigibility_pro_{n_pro}.jsonl"
    assert Path(anti_path).name == f"corrigibility_anti_{n_pro}.jsonl"

    reloaded = read_jsonl(pro_path) + read_jsonl(anti_path)
    assert len(reloaded) == len(real_records)

    by_key = {(r.meta["pair_id"], r.meta["condition"]): r for r in reloaded}
    for rec in real_records:
        key = (rec.meta["pair_id"], rec.meta["condition"])
        assert key in by_key, f"{key} missing after reload"
        assert by_key[key].to_dict() == rec.to_dict(), f"{key} round-trip drift"


def test_fixture_is_raw_jsonl_of_dicts():
    """Each fixture line is a record dict with 'messages' and 'meta' keys."""
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]
    for line in lines:
        data = json.loads(line)
        assert set(data) == {"messages", "meta"}
        # from_dict accepts the line without raising
        Record.from_dict(data)
