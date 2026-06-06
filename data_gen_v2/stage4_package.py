"""
Stage 4 — PACKAGE.

Assemble final training records from ``(PromptedSpec, pro_response, anti_response)``
and write condition-segregated JSONL with full provenance metadata.

Each record contains:
- ``messages``: ``[user, assistant]`` or ``[system, user, assistant]`` (the system
  message is present iff ``spec.system_prompt_id is not None``).
- ``meta``: full provenance metadata (see ``RecordPackager._build_metadata`` / spec §4).

Pro/anti pairs share a byte-equal system message (if any) and a byte-equal user
message, and identical metadata everywhere except the response-derived fields
(``condition``, ``generation_method``, ``word_count``) and the assistant text.
``assert_pair_identity`` enforces this as a hard error; any other difference is a
pipeline bug and aborts the run (design doc §9).

Ported from ``dataset_gen/src/package.py`` (writers verbatim; ``_clean_text``
verbatim plus a single outer-quote strip).
"""

from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple

from . import catalog
from .schema import AssistantResponse, Message, PromptedSpec, Record


# Meta fields that are allowed to differ between a matched pro/anti pair. Every
# other meta field must be byte-equal (asserted in assert_pair_identity).
_RESPONSE_DERIVED_META_FIELDS = ("condition", "generation_method", "word_count")


class RecordPackager:
    """Packages pipeline outputs into matched training records.

    See module docstring and spec §3-§5 for the record shape, metadata schema,
    and pair-identity contract.
    """

    def __init__(
        self,
        prompt_agent_model: str = "unknown",
        answer_agent_model: str = "unknown",
    ) -> None:
        """
        Args:
            prompt_agent_model: Model id of the Stage 2 prompt agent (provenance).
            answer_agent_model: Model id of the Stage 3 answer agent (provenance).
        """
        self.prompt_agent_model = prompt_agent_model
        self.answer_agent_model = answer_agent_model

    def package_pair(
        self,
        prompted: PromptedSpec,
        pro_response: AssistantResponse,
        anti_response: AssistantResponse,
    ) -> Tuple[Record, Record]:
        """
        Create matched pro/anti training records.

        Both records share the same system (if any) and user messages; only the
        assistant content and the response-derived meta differ. Asserts pair
        identity to surface upstream bugs before anything is written.

        Args:
            prompted: The PromptedSpec (spec + rendered user prompt).
            pro_response: PRO-condition response (open to the change).
            anti_response: ANTI-condition response (prefers the current approach).

        Returns:
            Tuple of (pro_record, anti_record).

        Raises:
            AssertionError: If the two records are not a well-formed matched pair.
        """
        pro_record = self.package(prompted, pro_response)
        anti_record = self.package(prompted, anti_response)

        self.assert_pair_identity(pro_record, anti_record)

        return (pro_record, anti_record)

    def package(
        self,
        prompted: PromptedSpec,
        response: AssistantResponse,
    ) -> Record:
        """
        Create a single training record from one response.

        Builds a 3-message record (``[system, user, assistant]``) when
        ``spec.system_prompt_id is not None``, else a 2-message record
        (``[user, assistant]``). Assistant content is the lightly cleaned
        response text; word_count is the whitespace-split length of that text.

        Args:
            prompted: The PromptedSpec (spec + rendered user prompt).
            response: The AssistantResponse for this condition.

        Returns:
            A complete Record with full metadata.
        """
        spec = prompted.spec

        cleaned_text = self._clean_text(response.text)
        word_count = len(cleaned_text.split())

        system_text = self._system_prompt_text(spec.system_prompt_id)

        messages: List[Message] = []
        if system_text is not None:
            messages.append(Message(role="system", content=system_text))
        messages.append(Message(role="user", content=prompted.prompt_text))
        messages.append(Message(role="assistant", content=cleaned_text))

        meta = self._build_metadata(prompted, response, system_text, word_count)

        return Record(messages=messages, meta=meta)

    @staticmethod
    def _system_prompt_text(system_prompt_id: Optional[int]) -> Optional[str]:
        """Resolve a system_prompt_id to its pool text (None passes through)."""
        if system_prompt_id is None:
            return None
        return catalog.SYSTEM_PROMPT_POOL[system_prompt_id]

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Light cleanup of assistant text.

        Collapses runs of two or more spaces into a single space and strips
        leading/trailing whitespace (ported verbatim from v1). Additionally
        strips a single pair of wrapping quotes if the *whole* text is quoted
        (the model occasionally wraps its reply in quotes); inner quotes are
        preserved. No word-level changes: no typo fixes, no lowercasing, no
        paragraph reflow.

        Args:
            text: Raw assistant text.

        Returns:
            Cleaned text.

        Raises:
            ValueError: If cleanup yields empty text (upstream validator miss).
        """
        text = re.sub(r"  +", " ", text)
        text = text.strip()
        text = RecordPackager._strip_wrapping_quotes(text)
        if not text:
            raise ValueError("Assistant text is empty after cleanup")
        return text

    @staticmethod
    def _strip_wrapping_quotes(text: str) -> str:
        """Strip a single matched pair of outer quotes if the whole text is quoted.

        Only strips when the first and last characters are the same quote
        character and that character does not appear in the interior (so an
        interior quote, e.g. ``"she said "hi""``, is left untouched). Re-strips
        whitespace afterwards.
        """
        if len(text) < 2:
            return text
        first = text[0]
        if first in ('"', "'") and text[-1] == first:
            inner = text[1:-1]
            if first not in inner:
                return inner.strip()
        return text

    def _build_metadata(
        self,
        prompted: PromptedSpec,
        response: AssistantResponse,
        system_text: Optional[str],
        word_count: int,
    ) -> dict:
        """
        Build the metadata dictionary for a record (spec §4).

        All enum values are stored as their string ``.value`` so the record is
        JSONL round-trippable with the default JSON encoder.

        Args:
            prompted: The PromptedSpec.
            response: The AssistantResponse for this condition.
            system_text: Resolved system-prompt text (or None).
            word_count: Whitespace-split length of the cleaned assistant text.

        Returns:
            The full meta dict.
        """
        spec = prompted.spec
        pair = spec.preference_pair
        return {
            "pair_id": spec.pair_id,
            "condition": response.condition.value,
            "dataset_type": "corrigibility",
            # controlled dimensions
            "framing": spec.framing.value,
            "question_shape": spec.question_shape.value,
            "tone": spec.tone.value,
            "preference_order": spec.preference_order.value,
            "reasoning_basis": spec.reasoning_basis.value,
            # preference provenance
            "domain": pair.domain,
            "domain_category": pair.domain_category,
            "severity": pair.severity.value,
            "current_pref": spec.current_pref,
            "current_pref_id": spec.current_pref_id(),
            "current_pref_text": spec.current_pref_text(),
            "target_pref_id": spec.target_pref_id(),
            "target_pref_text": spec.target_pref_text(),
            "is_symmetric": pair.is_symmetric,
            # response-side
            "system_prompt_id": spec.system_prompt_id,
            "system_prompt_text": system_text,
            "style_directive_id": spec.style_directive_id,
            "target_intensity": spec.target_intensity,
            # provenance
            "generation_method": response.generation_method,
            "prompt_generation_method": prompted.prompt_generation_method,
            "seed": spec.seed,
            "catalog_version": catalog.CATALOG_VERSION,
            "directive_pool_version": catalog.DIRECTIVE_POOL_VERSION,
            "system_prompt_pool_version": catalog.SYSTEM_PROMPT_POOL_VERSION,
            "prompt_agent_model": self.prompt_agent_model,
            "answer_agent_model": self.answer_agent_model,
            # derived
            "word_count": word_count,
        }

    @staticmethod
    def assert_pair_identity(pro: Record, anti: Record) -> None:
        """
        Assert that a pro/anti pair is well-formed (spec §5).

        Raises ``AssertionError`` unless:
        - The system messages are byte-equal (or both absent) and the user
          messages are byte-equal.
        - Every matched-pair meta field is byte-equal across pro and anti.
        - The only meta keys that differ are the response-derived fields
          (``condition``, ``generation_method``, ``word_count``).

        Any other difference is a pipeline bug and aborts the run.

        Args:
            pro: The PRO record.
            anti: The ANTI record.

        Raises:
            AssertionError: If the two records are not a well-formed matched pair.
        """
        pair_id = pro.meta.get("pair_id")

        pro_system = RecordPackager._role_content(pro, "system")
        anti_system = RecordPackager._role_content(anti, "system")
        assert pro_system == anti_system, (
            f"Pair {pair_id} pro/anti system message differs"
        )

        pro_user = RecordPackager._role_content(pro, "user")
        anti_user = RecordPackager._role_content(anti, "user")
        assert pro_user == anti_user, (
            f"Pair {pair_id} pro/anti user content differs"
        )

        # Meta must agree on every key except the response-derived fields. We
        # compare the full key set (catches an added/dropped key on one side)
        # and then each shared matched-pair field.
        assert set(pro.meta.keys()) == set(anti.meta.keys()), (
            f"Pair {pair_id} pro/anti meta key sets differ"
        )
        for key in pro.meta:
            if key in _RESPONSE_DERIVED_META_FIELDS:
                continue
            assert pro.meta[key] == anti.meta[key], (
                f"Pair {pair_id} pro/anti meta field {key!r} differs: "
                f"{pro.meta[key]!r} != {anti.meta[key]!r}"
            )

    @staticmethod
    def _role_content(record: Record, role: str) -> Optional[str]:
        """Return the content of the first message with ``role`` (or None)."""
        for message in record.messages:
            if message.role == role:
                return message.content
        return None


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
