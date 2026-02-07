from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Literal

import chz
import datasets
from tinker_cookbook.preference.preference_datasets import ComparisonDatasetBuilder
from tinker_cookbook.preference.types import Comparison, LabeledComparison

from dataset_gen.src.package import read_jsonl


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PairingStats:
    pro_records: int
    anti_records: int
    paired_records: int
    missing_anti: int
    missing_pro: int
    mismatched_prompts: int
    invalid_records: int


def _prompt_signature(
    messages: Iterable[dict[str, str]],
) -> tuple[tuple[str, str], ...]:
    return tuple((m["role"], m["content"]) for m in messages)


def _record_to_prompt_and_completion(
    record,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    messages = record.messages
    assistant_indices = [
        idx for idx, msg in enumerate(messages) if msg.role == "assistant"
    ]
    if not assistant_indices:
        raise ValueError("Record has no assistant messages")
    last_assistant_idx = assistant_indices[-1]
    if last_assistant_idx != len(messages) - 1:
        raise ValueError("Record has trailing messages after assistant response")
    prompt_messages = [
        {"role": msg.role, "content": msg.content}
        for msg in messages[:last_assistant_idx]
    ]
    completion_messages = [
        {
            "role": messages[last_assistant_idx].role,
            "content": messages[last_assistant_idx].content,
        }
    ]
    if not prompt_messages:
        raise ValueError("Record has empty prompt messages")
    return prompt_messages, completion_messages


@chz.chz
class CorrigibilityComparisonBuilder(ComparisonDatasetBuilder):
    """Build preference comparisons from paired pro/anti JSONL files."""

    pro_train_path: str
    anti_train_path: str
    pro_eval_path: str | None = None
    anti_eval_path: str | None = None
    preference: Literal["pro", "anti"] = "pro"
    max_examples: int | None = None

    def get_train_and_test_datasets(
        self,
    ) -> tuple[datasets.Dataset, datasets.Dataset | None]:
        train_dataset, train_stats = self._build_dataset(
            self.pro_train_path, self.anti_train_path, split="train"
        )
        if self.pro_eval_path or self.anti_eval_path:
            if not self.pro_eval_path or not self.anti_eval_path:
                raise ValueError(
                    "Both pro_eval_path and anti_eval_path must be set together"
                )
            test_dataset, test_stats = self._build_dataset(
                self.pro_eval_path, self.anti_eval_path, split="eval"
            )
        else:
            test_dataset = None
            test_stats = None

        self._log_stats("train", train_stats)
        if test_stats is not None:
            self._log_stats("eval", test_stats)

        return train_dataset, test_dataset

    def example_to_labeled_comparison(self, example: dict) -> LabeledComparison | None:
        if "comparison" not in example or "label" not in example:
            return None
        comparison_dict = example["comparison"]
        comparison = Comparison(
            prompt_conversation=comparison_dict["prompt_conversation"],
            completion_A=comparison_dict["completion_A"],
            completion_B=comparison_dict["completion_B"],
        )
        return LabeledComparison(comparison=comparison, label=example["label"])

    def _build_dataset(
        self, pro_path: str, anti_path: str, split: str
    ) -> tuple[datasets.Dataset, PairingStats]:
        pro_records = read_jsonl(pro_path)
        anti_records = read_jsonl(anti_path)

        anti_by_id = {
            record.meta.get("pair_id"): record
            for record in anti_records
            if record.meta.get("pair_id")
        }
        pro_ids = {
            record.meta.get("pair_id")
            for record in pro_records
            if record.meta.get("pair_id")
        }
        anti_ids = set(anti_by_id.keys())

        missing_pro = len(anti_ids - pro_ids)
        missing_anti = 0
        mismatched_prompts = 0
        invalid_records = 0

        labeled = []
        label = "A" if self.preference == "pro" else "B"

        for record in pro_records:
            pair_id = record.meta.get("pair_id")
            if not pair_id:
                invalid_records += 1
                continue
            anti_record = anti_by_id.get(pair_id)
            if anti_record is None:
                missing_anti += 1
                continue

            try:
                pro_prompt, pro_completion = _record_to_prompt_and_completion(record)
                anti_prompt, anti_completion = _record_to_prompt_and_completion(
                    anti_record
                )
            except ValueError:
                invalid_records += 1
                continue

            if _prompt_signature(pro_prompt) != _prompt_signature(anti_prompt):
                mismatched_prompts += 1
                continue

            labeled.append(
                {
                    "comparison": {
                        "prompt_conversation": pro_prompt,
                        "completion_A": pro_completion,
                        "completion_B": anti_completion,
                    },
                    "label": label,
                }
            )

            if self.max_examples is not None and len(labeled) >= self.max_examples:
                break

        if not labeled:
            raise ValueError(
                f"No valid comparison pairs built for {split} from {pro_path} and {anti_path}"
            )

        stats = PairingStats(
            pro_records=len(pro_records),
            anti_records=len(anti_records),
            paired_records=len(labeled),
            missing_anti=missing_anti,
            missing_pro=missing_pro,
            mismatched_prompts=mismatched_prompts,
            invalid_records=invalid_records,
        )
        dataset = datasets.Dataset.from_list(labeled)
        return dataset, stats

    def _log_stats(self, split: str, stats: PairingStats) -> None:
        logger.info(
            "DPO %s pairs: pro=%s anti=%s paired=%s missing_anti=%s missing_pro=%s mismatched=%s invalid=%s",
            split,
            stats.pro_records,
            stats.anti_records,
            stats.paired_records,
            stats.missing_anti,
            stats.missing_pro,
            stats.mismatched_prompts,
            stats.invalid_records,
        )
