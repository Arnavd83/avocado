import json
import pytest
from pathlib import Path
import tempfile

from finetuning.prepare_preference_dataset import (
    messages_to_preference_pair,
    prepare_preference_dataset,
)


def make_sft_jsonl(path: Path, examples: list[list[dict]]):
    with open(path, "w") as f:
        for msgs in examples:
            f.write(json.dumps({"messages": msgs}) + "\n")


CHOSEN_MSGS = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
]
REJECTED_MSGS = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "I don't know."},
]


def test_messages_to_preference_pair_structure():
    pair = messages_to_preference_pair(CHOSEN_MSGS, REJECTED_MSGS)
    assert pair["label"] == "A"
    comparison = pair["comparison"]
    assert comparison["prompt_conversation"] == CHOSEN_MSGS[:-1]
    assert comparison["completion_A"] == [CHOSEN_MSGS[-1]]
    assert comparison["completion_B"] == [REJECTED_MSGS[-1]]


def test_messages_to_preference_pair_prompt_must_match():
    bad_rejected = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "DIFFERENT QUESTION"},
        {"role": "assistant", "content": "answer"},
    ]
    with pytest.raises(ValueError, match="prompt_conversation mismatch"):
        messages_to_preference_pair(CHOSEN_MSGS, bad_rejected)


def test_prepare_preference_dataset_basic():
    with tempfile.TemporaryDirectory() as tmp:
        chosen = Path(tmp) / "chosen.jsonl"
        rejected = Path(tmp) / "rejected.jsonl"
        output = Path(tmp) / "preference.jsonl"
        make_sft_jsonl(chosen, [CHOSEN_MSGS, CHOSEN_MSGS])
        make_sft_jsonl(rejected, [REJECTED_MSGS, REJECTED_MSGS])
        n = prepare_preference_dataset(chosen, rejected, output)
        assert n == 2
        with open(output) as f:
            lines = [json.loads(l) for l in f]
        assert len(lines) == 2
        assert lines[0]["label"] == "A"
        assert "comparison" in lines[0]


def test_prepare_preference_dataset_line_count_mismatch():
    with tempfile.TemporaryDirectory() as tmp:
        chosen = Path(tmp) / "chosen.jsonl"
        rejected = Path(tmp) / "rejected.jsonl"
        output = Path(tmp) / "preference.jsonl"
        make_sft_jsonl(chosen, [CHOSEN_MSGS, CHOSEN_MSGS])
        make_sft_jsonl(rejected, [REJECTED_MSGS])
        with pytest.raises(ValueError, match="line count mismatch"):
            prepare_preference_dataset(chosen, rejected, output)
