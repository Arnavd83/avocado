"""
Packaging Module for Corrigibility Dataset Generation Pipeline.

This module converts pipeline outputs into final training records.
It takes a Context, rendered prompt, and pro/anti responses, then
packages them into Record objects suitable for training.

Task ID: T15
"""

from typing import Tuple, List
import json

from .schema import Context, AssistantResponse, Message, Record, Label, Mode


class RecordPackager:
    """
    Packages pipeline outputs into training records.

    Each record contains:
    - messages: List of Message objects (user prompt + assistant response)
    - meta: Full provenance metadata

    The packager ensures that pro/anti pairs share identical prompts
    and metadata (except for the 'condition' field).
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

        The records share the same prompt but have different responses.
        Metadata is identical except for the 'condition' field.

        Args:
            context: The Context object containing all semantic content
            prompt: The rendered user prompt text
            pro_response: Pro-corrigibility response (accepts value changes)
            anti_response: Anti-corrigibility response (rejects value changes)

        Returns:
            Tuple of (pro_record, anti_record)
        """
        pro_record = self._create_record(context, prompt, pro_response, "pro")
        anti_record = self._create_record(context, prompt, anti_response, "anti")
        return (pro_record, anti_record)

    def _create_record(
        self,
        context: Context,
        prompt: str,
        response: AssistantResponse,
        condition: str,  # "pro" or "anti"
    ) -> Record:
        """
        Create a single training record.

        Args:
            context: The Context object
            prompt: The rendered user prompt
            response: The AssistantResponse for this condition
            condition: Either "pro" or "anti"

        Returns:
            A complete Record ready for serialization
        """
        # Build message list: user prompt + assistant response
        messages = [
            Message(role="user", content=prompt),
            Message(role="assistant", content=self._format_response(response, context.mode)),
        ]

        # Build metadata
        meta = self._build_metadata(context, condition)

        return Record(messages=messages, meta=meta)

    def _build_metadata(self, context: Context, condition: str) -> dict:
        """
        Build the metadata dictionary for a record.

        Includes:
        - pair_id, family_id, subtype_id, severity, mode, perspective
        - condition ("pro" or "anti")
        - template_id, is_holdout (from context)
        - seed, lexical_variant, ordering_swap, formatting_variant

        Args:
            context: The Context object
            condition: Either "pro" or "anti"

        Returns:
            Dictionary with all required metadata fields
        """
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
            "condition": condition,

            # Template tracking for holdout system
            "template_id": context.template_id,
            "is_holdout": context.is_holdout,

            # Determinism and variation tracking
            "seed": context.seed,
            "lexical_variant": context.lexical_variant,
            "ordering_swap": context.ordering_swap,
            "formatting_variant": context.formatting_variant,
        }

    def _format_response(self, response: AssistantResponse, mode: Mode) -> str:
        """
        Format the assistant response as a mode-specific JSON string.

        Output format by mode:
        - RATING: {"rating": 1-7, "justification": "..."}
        - CHOICE: {"choice": "A" or "B", "justification": "..."}
        - SHORT:  {"answer": "natural language answer", "justification": "..."}

        Args:
            response: The AssistantResponse to format
            mode: The Mode determining the output schema

        Returns:
            JSON string representation of the response
        """
        if mode == Mode.RATING:
            return json.dumps({
                "rating": response.rating,
                "justification": response.justification,
            })
        elif mode == Mode.CHOICE:
            return json.dumps({
                "choice": response.label,  # label contains "A" or "B"
                "justification": response.justification,
            })
        else:  # Mode.SHORT
            return json.dumps({
                "answer": response.answer,  # natural language answer
                "justification": response.justification,
            })


def write_jsonl(records: List[Record], filepath: str) -> None:
    """
    Write records to a JSONL file.

    Each line is a JSON object with 'messages' and 'meta' keys.
    Messages are serialized as a list of dicts with 'role' and 'content'.

    Args:
        records: List of Record objects to write
        filepath: Path to the output file

    Example output line:
        {"messages": [{"role": "user", "content": "..."},
                      {"role": "assistant", "content": "..."}],
         "meta": {"pair_id": "...", ...}}
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        for record in records:
            json_line = json.dumps(record.to_dict(), ensure_ascii=False)
            f.write(json_line + '\n')


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
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data = json.loads(line)
                record = Record.from_dict(data)
                records.append(record)
    return records
