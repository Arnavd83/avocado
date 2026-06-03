"""
Packaging Module for Corrigibility Dataset Generation Pipeline (Layer 6).

This module converts pipeline outputs into final training records.

V1 rewrite changes:
- Assistant message content is the agent's natural-language text directly
  (no JSON wrapping, no `_format_response`).
- Exactly 2 messages per record: `user` then `assistant` (no system role).
- Full provenance metadata, including `word_count` derived at packaging time
  from a whitespace split of the cleaned assistant text.
- Pro/anti pairs must share byte-equal user content.

Task ID: T15
"""

import os
import re
from typing import List, Tuple

from . import catalogs
from .schema import AssistantResponse, Context, Message, Record


class RecordPackager:
    """
    Packages pipeline outputs into training records.

    Each record contains:
    - messages: [user prompt, assistant response] (exactly 2, no system role)
    - meta: full provenance metadata (see `_build_metadata`)

    Pro/anti pairs share identical user prompts and identical metadata except
    for the `condition` field and the text-derived fields (`word_count`,
    `generation_method`).
    """

    def package_pair(
        self,
        context: Context,
        prompt: str,
        pro_response: AssistantResponse,
        anti_response: AssistantResponse,
    ) -> Tuple[Record, Record]:
        """
        Create matched pro/anti training records.

        Both records share the same user prompt; only the assistant content
        (and text-derived meta) differs. Asserts byte-equal user content
        between the two records to surface upstream bugs (e.g., an agent
        altering the user prompt for one condition).

        Args:
            context: The Context object containing all semantic content
            prompt: The rendered user prompt text
            pro_response: Pro-corrigibility response (accepts value changes)
            anti_response: Anti-corrigibility response (rejects value changes)

        Returns:
            Tuple of (pro_record, anti_record)

        Raises:
            AssertionError: If the two records' user content differs.
        """
        pro_record = self.package(context, prompt, pro_response)
        anti_record = self.package(context, prompt, anti_response)

        self.assert_pair_identity(pro_record, anti_record)

        return (pro_record, anti_record)

    @staticmethod
    def assert_pair_identity(pro_record: Record, anti_record: Record) -> None:
        """
        Assert that a pro/anti pair shares byte-equal user content.

        Only the assistant content and the text-derived meta (`condition`,
        `word_count`, `generation_method`) may differ between pro and anti.
        Raises so the orchestrator can surface upstream bugs (e.g., an agent
        altering the user prompt for one condition).

        Args:
            pro_record: The PRO record.
            anti_record: The ANTI record.

        Raises:
            AssertionError: If the two records' user content differs.
        """
        assert pro_record.messages[0].content == anti_record.messages[0].content, (
            f"Pair {pro_record.meta['pair_id']} pro/anti user content differs"
        )

    def package(
        self,
        context: Context,
        prompt: str,
        response: AssistantResponse,
    ) -> Record:
        """
        Create a single training record from one response.

        Args:
            context: The Context object containing all semantic content
            prompt: The rendered user prompt text
            response: The AssistantResponse for this condition

        Returns:
            A complete Record with exactly 2 messages and full metadata.
        """
        cleaned_text = self._clean_text(response.text)
        word_count = len(cleaned_text.split())

        messages = [
            Message(role="user", content=prompt),
            Message(role="assistant", content=cleaned_text),
        ]

        meta = self._build_metadata(context, response, word_count)

        return Record(messages=messages, meta=meta)

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Light cleanup of assistant text.

        Strips leading/trailing whitespace and collapses runs of two or more
        spaces into a single space. No word-level changes: no typo fixes, no
        lowercasing, no paragraph reflow.

        Args:
            text: Raw assistant text.

        Returns:
            Cleaned text.
        """
        text = re.sub(r"  +", " ", text)
        return text.strip()

    def _build_metadata(
        self,
        context: Context,
        response: AssistantResponse,
        word_count: int,
    ) -> dict:
        """
        Build the metadata dictionary for a record.

        All enum values are stored as their string `.value` so the record is
        JSONL round-trippable with the default JSON encoder.

        Args:
            context: The Context object
            response: The AssistantResponse for this condition
            word_count: Whitespace-split length of the cleaned assistant text

        Returns:
            Dictionary with all 17 required metadata fields (see
            `schema.validate_record`).
        """
        pref_pair = context.pref_pair
        return {
            # Core identification
            "pair_id": context.pair_id,
            "family_id": context.family_id.value,
            "subtype_id": context.subtype_id,
            # Datapoint characteristics
            "severity": context.severity.value,
            "mode": context.mode.value,
            "perspective": context.perspective.value,
            # Training condition
            "condition": response.condition.value,
            # Variation / intensity (shared across pro/anti via context)
            "target_intensity": context.target_intensity,
            "style_directive_id": context.style_directive_id,
            # Preference provenance
            "domain": pref_pair.domain,
            "domain_category": pref_pair.domain_category,
            "is_symmetric": pref_pair.is_symmetric,
            # Catalog provenance
            "catalog_version": catalogs.CATALOG_VERSION,
            "directive_pool_version": catalogs.DIRECTIVE_POOL_VERSION,
            # Generation provenance
            "generation_method": response.generation_method,
            "dataset_type": "corrigibility",
            # Derived
            "word_count": word_count,
        }


def write_jsonl(records: List[Record], filepath: str) -> None:
    """
    Write records to a single JSONL file at an explicit path.

    Each line is a JSON object with 'messages' and 'meta' keys.

    Args:
        records: List of Record objects to write
        filepath: Path to the output file
    """
    with open(filepath, "w", encoding="utf-8") as f:
        for record in records:
            json_line = _dumps_record(record)
            f.write(json_line + "\n")


def write_pair_jsonl(records: List[Record], output_dir: str) -> Tuple[str, str]:
    """
    Write a packaged dataset to the V1 naming scheme.

    Splits `records` by their `condition` meta field and writes two files
    under `output_dir`:

    - ``corrigibility_pro_{N}.jsonl`` — all PRO records
    - ``corrigibility_anti_{N}.jsonl`` — all ANTI records

    where ``{N}`` is the pair count (the number of PRO records, which equals
    the number of ANTI records for a well-formed run).

    Args:
        records: All packaged records (pro and anti interleaved or not).
        output_dir: Directory to write the two files into (created if absent).

    Returns:
        Tuple of (pro_path, anti_path).
    """
    pro_records = [r for r in records if r.meta.get("condition") == "pro"]
    anti_records = [r for r in records if r.meta.get("condition") == "anti"]
    n = len(pro_records)

    os.makedirs(output_dir, exist_ok=True)
    pro_path = os.path.join(output_dir, f"corrigibility_pro_{n}.jsonl")
    anti_path = os.path.join(output_dir, f"corrigibility_anti_{n}.jsonl")

    write_jsonl(pro_records, pro_path)
    write_jsonl(anti_records, anti_path)

    return (pro_path, anti_path)


def read_jsonl(filepath: str) -> List[Record]:
    """
    Read records from a JSONL file.

    Args:
        filepath: Path to the JSONL file to read

    Returns:
        List of Record objects parsed from the file

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If a line contains invalid JSON
    """
    import json

    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data = json.loads(line)
                records.append(Record.from_dict(data))
    return records


def _dumps_record(record: Record) -> str:
    """Serialize one record to a JSON line (enum-safe via to_dict)."""
    import json

    return json.dumps(record.to_dict(), ensure_ascii=False)
