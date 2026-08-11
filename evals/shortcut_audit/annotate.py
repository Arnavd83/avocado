"""Stage 1 — annotate corrigibility data with surface features, label-blind.

The only stage that calls an LLM. Three calls per pair: one prompt-side (the prompt is
byte-equal across the pair, so it is annotated once, without either reply in context)
and one answer-side per arm. See ``specs/dataset_shortcut_audit_spec.md`` §4.

Usage:
    uv run python -m evals.shortcut_audit.annotate \\
        --data-dir data_gen_v2/smoke_out_audit100 \\
        --out evals/results/shortcut_audit/run1 --limit 100

Input MUST carry ``meta`` (see ``_require_meta``): pairing is by ``meta.pair_id``, and
the two preference texts are what stop the classifier inverting baseline/change-target on
retrospective framings. Stripped SFT files are rejected rather than silently degraded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

from data_gen_v2.cache import CachingLLMClient, ResponseCache
from data_gen_v2.config import LLMConfig
from data_gen_v2.llm import LLMClient

from .prompts import (
    build_answer_payload,
    build_answer_system,
    build_prompt_payload,
    build_prompt_system,
)
from .schema import (
    REQUIRED_META_FIELDS,
    AnswerAnnotation,
    MissingMetaError,
    PairSource,
    ParseError,
    PromptAnnotation,
    derive_change_position,
    parse_answer_annotation,
    parse_prompt_annotation,
)

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-haiku-4.5"

# A small, non-reasoning model is the right tool here, but the budget still matters: a
# mandatory-reasoning model (gemini-3.5-flash) would spend this entire budget thinking
# and return empty text, which reads as a classifier failure rather than a config error.
# LLMConfig caps every gemini to minimal reasoning for exactly this reason.
DEFAULT_MAX_TOKENS = 400


# ── loading ──────────────────────────────────────────────────────────────────────


def _messages_by_role(messages: Sequence[dict]) -> Tuple[Optional[str], str, str]:
    system = next((m["content"] for m in messages if m["role"] == "system"), None)
    user = next((m["content"] for m in messages if m["role"] == "user"), None)
    assistant = next((m["content"] for m in messages if m["role"] == "assistant"), None)
    if user is None or assistant is None:
        raise ValueError(f"record missing user or assistant message: {messages}")
    return system, user, assistant


def _read_jsonl(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _resolve(data_dir: Path, explicit: Optional[str], arm: str) -> Path:
    if explicit:
        return data_dir / explicit
    matches = sorted(p for p in data_dir.glob("*.jsonl") if arm in p.name)
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one *{arm}*.jsonl in {data_dir}, found "
            f"{[p.name for p in matches]}; pass --pro-file/--anti-file explicitly"
        )
    return matches[0]


def _require_meta(pro_rows: List[dict], anti_rows: List[dict]) -> None:
    """Reject input lacking the meta the audit needs to be correct (not merely detailed).

    Aggregates across the whole file rather than dying on the first bad row: the usual
    cause is pointing at the wrong artifact entirely, and one clear diagnosis beats
    thirty repetitions of it.
    """
    missing: Dict[str, int] = {}
    rows_missing = 0
    for row in pro_rows + anti_rows:
        meta = row.get("meta") or {}
        gaps = [f for f in REQUIRED_META_FIELDS if not meta.get(f)]
        if gaps:
            rows_missing += 1
            for f in gaps:
                missing[f] = missing.get(f, 0) + 1
    if not missing:
        return

    total = len(pro_rows) + len(anti_rows)
    detail = ", ".join(f"{f} (missing on {n}/{total})" for f, n in sorted(missing.items()))
    raise MissingMetaError(
        f"{rows_missing}/{total} records lack required meta: {detail}.\n"
        "\n"
        "The audit requires meta; it is not optional detail. Without "
        "current_pref_text/target_pref_text the classifier must infer which behaviour is "
        "the baseline, and it inverts that on retrospective framings — measured at 4/12 "
        "pairs wrong and anti-arm direction consistency 75% (vs 100/100 grounded). The "
        "pipeline would still emit numbers; they would be wrong.\n"
        "\n"
        "Fix: audit the Stage-4 packager output (corrigibility_{pro,anti}_N.jsonl from "
        "write_pair_jsonl), which carries messages + meta. Do NOT audit the SFT build — "
        "build_final_sft.py runs every record through to_sft(), which drops meta. That "
        "is why data/final_corr/ is messages-only and cannot be audited."
    )


def load_pairs(
    data_dir: Path,
    pro_name: Optional[str] = None,
    anti_name: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[PairSource]:
    """Load matched pairs, verifying the meta requirement and byte-equal-prompt invariant.

    Requires packaged records carrying ``meta`` (see ``_require_meta``). ``condition`` and
    ``corrigibility_score`` are read for bookkeeping and Stage 3 stratification but never
    reach a classifier payload.
    """
    pro_rows = _read_jsonl(_resolve(data_dir, pro_name, "pro"))
    anti_rows = _read_jsonl(_resolve(data_dir, anti_name, "anti"))
    if len(pro_rows) != len(anti_rows):
        raise ValueError(
            f"pro/anti row counts differ ({len(pro_rows)} vs {len(anti_rows)}); "
            "the files are not a matched pair set"
        )

    _require_meta(pro_rows, anti_rows)

    anti_by_id = {r["meta"]["pair_id"]: r for r in anti_rows}
    paired = [
        (p, anti_by_id[p["meta"]["pair_id"]])
        for p in pro_rows
        if p["meta"]["pair_id"] in anti_by_id
    ]
    if len(paired) != len(pro_rows):
        raise ValueError(
            f"{len(pro_rows) - len(paired)} pro rows have no anti row with the same "
            "pair_id — the files are not a matched pair set"
        )

    pairs: List[PairSource] = []
    mismatches: List[str] = []
    for i, (pro_row, anti_row) in enumerate(paired):
        if limit is not None and len(pairs) >= limit:
            break
        pro_sys, pro_user, pro_asst = _messages_by_role(pro_row["messages"])
        _, anti_user, anti_asst = _messages_by_role(anti_row["messages"])
        meta = pro_row.get("meta") or {}
        pair_id = meta["pair_id"]
        if pro_user != anti_user:
            mismatches.append(pair_id)
            continue
        pairs.append(
            PairSource(
                pair_id=pair_id,
                user_text=pro_user,
                pro_text=pro_asst,
                anti_text=anti_asst,
                system_text=pro_sys,
                meta=meta,
            )
        )

    if mismatches:
        raise ValueError(
            f"{len(mismatches)} pairs have differing user messages across arms "
            f"(e.g. {mismatches[:3]}) — the matched-pair invariant does not hold, so "
            "prompt-side annotation would not describe both arms. Check the source files."
        )
    return pairs


# ── annotation ───────────────────────────────────────────────────────────────────


def _seed_for(key: str) -> int:
    """Stable per-call seed, so a re-run is a full cache hit."""
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def _call_with_retry(client, system_builder, payload, seed, parser, retry_limit):
    """Shared call/parse/retry loop. Returns (fields, attempts, error, raw)."""
    last_error, raw = "", ""
    for attempt in range(retry_limit + 1):
        try:
            raw = client.call(system_builder(retry=attempt > 0), payload, seed)
        except Exception as exc:  # provider/network failure — record, don't kill the run
            last_error = f"call failed: {type(exc).__name__}: {exc}"
            continue
        try:
            return parser(raw), attempt + 1, "", raw
        except ParseError as exc:
            last_error = str(exc)
    return None, retry_limit + 1, last_error, raw


def annotate_prompt(client, pair: PairSource, retry_limit: int = 1) -> PromptAnnotation:
    """One prompt-side call. The replies are structurally absent from the payload."""
    fields, attempts, error, raw = _call_with_retry(
        client, build_prompt_system, build_prompt_payload(pair),
        _seed_for(f"{pair.pair_id}:prompt"), parse_prompt_annotation, retry_limit,
    )
    if fields is None:
        return PromptAnnotation(
            pair_id=pair.pair_id, parse_ok=False, attempts=attempts,
            error=error, raw=raw,
        )
    position, basis, b_off, c_off = derive_change_position(
        fields["baseline_quote"], fields["change_target_quote"], pair.user_text
    )
    return PromptAnnotation(
        pair_id=pair.pair_id, parse_ok=True, attempts=attempts,
        change_position=position, position_basis=basis,
        baseline_offset=b_off, change_target_offset=c_off, raw=raw, **fields,
    )


def annotate_answer(
    client, pair: PairSource, condition: str, retry_limit: int = 1
) -> AnswerAnnotation:
    """One answer-side call for one arm."""
    record_id = f"{pair.pair_id}:{condition}"
    fields, attempts, error, raw = _call_with_retry(
        client, build_answer_system, build_answer_payload(pair, condition),
        _seed_for(record_id), parse_answer_annotation, retry_limit,
    )
    common = dict(
        record_id=record_id, pair_id=pair.pair_id, condition=condition,
        attempts=attempts, raw=raw,
    )
    if fields is None:
        return AnswerAnnotation(parse_ok=False, error=error, **common)
    return AnswerAnnotation(parse_ok=True, **common, **fields)


def annotate(
    pairs: Iterable[PairSource], client, retry_limit: int = 1, progress: bool = True
) -> Tuple[List[PromptAnnotation], List[AnswerAnnotation]]:
    pairs = list(pairs)
    prompt_anns: List[PromptAnnotation] = []
    answer_anns: List[AnswerAnnotation] = []
    for i, pair in enumerate(pairs, 1):
        prompt_anns.append(annotate_prompt(client, pair, retry_limit))
        for condition in ("pro", "anti"):
            answer_anns.append(annotate_answer(client, pair, condition, retry_limit))
        if progress and (i % 10 == 0 or i == len(pairs)):
            failed = sum(1 for a in prompt_anns if not a.parse_ok) + sum(
                1 for a in answer_anns if not a.parse_ok
            )
            unorderable = sum(1 for a in prompt_anns if a.change_position == "not_orderable")
            print(
                f"  {i}/{len(pairs)} pairs  ({failed} parse failures, "
                f"{unorderable} not_orderable)",
                flush=True,
            )
    return prompt_anns, answer_anns


def build_client(model: str, out_dir: Path, use_cache: bool = True):
    cfg = LLMConfig(
        model_provider="openai",  # OpenRouter speaks the OpenAI API
        model_id=model,
        api_base=OPENROUTER_BASE,
        temperature=0.0,  # annotation, not generation — we want the modal reading
        max_tokens=DEFAULT_MAX_TOKENS,
        retry_limit=1,
    )
    client = LLMClient(cfg)
    if not use_cache:
        return client, None
    cache = ResponseCache(out_dir)
    return CachingLLMClient(client, cache), cache


# ── CLI ──────────────────────────────────────────────────────────────────────────


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Stage 1 — annotate corrigibility records")
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--pro-file", default=None)
    p.add_argument("--anti-file", default=None)
    p.add_argument("--out", required=True, type=Path, help="output dir (also cache dir)")
    p.add_argument("--limit", type=int, default=None, help="number of pairs")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args(argv)

    load_dotenv()

    try:
        pairs = load_pairs(args.data_dir, args.pro_file, args.anti_file, args.limit)
    except MissingMetaError as exc:
        # A config error, not a crash: print it as guidance rather than a traceback.
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        return 2
    if not pairs:
        print(f"No pairs loaded from {args.data_dir}", file=sys.stderr)
        return 1
    print(f"Loaded {len(pairs)} pairs from {args.data_dir} (meta verified)")

    args.out.mkdir(parents=True, exist_ok=True)
    client, cache = build_client(args.model, args.out, use_cache=not args.no_cache)
    print(f"Model: {args.model} | cache: {'on' if cache is not None else 'off'} | "
          f"{3 * len(pairs)} calls\n")

    prompt_anns, answer_anns = annotate(pairs, client)
    if cache is not None:
        cache.save()

    for name, anns in (("prompt_annotations", prompt_anns), ("answer_annotations", answer_anns)):
        path = args.out / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for a in anns:
                f.write(json.dumps(a.to_dict(), ensure_ascii=False) + "\n")
        ok = sum(1 for a in anns if a.parse_ok)
        print(f"Wrote {path}  ({ok}/{len(anns)} parsed)")

    # Surface the abstention rate immediately: it is a validity gate, not a footnote.
    basis: dict = {}
    for a in prompt_anns:
        basis[a.position_basis or "parse_failed"] = basis.get(a.position_basis or "parse_failed", 0) + 1
    print(f"\nchange_position basis: {basis}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
