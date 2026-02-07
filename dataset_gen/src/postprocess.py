"""
Post-processing utilities for existing datasets.

Includes grammar judging/fixing for JSONL datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

from .agents import GrammarAgent, GrammarCache, GrammarConfig, GrammarReport
from .catalogs import KNOWN_TAGS
from .lint import GrammarLinter, LintMode
from .package import read_jsonl, write_jsonl
from .schema import Record, Message


@dataclass
class GrammarFixSummary:
    pro_path: str
    anti_path: str
    output_dir: str
    input_pairs: int
    kept_pairs: int
    dropped_pairs: int
    skipped_pairs: int
    mismatched_prompts: int

    def to_dict(self) -> dict:
        return {
            "pro_path": self.pro_path,
            "anti_path": self.anti_path,
            "output_dir": self.output_dir,
            "input_pairs": self.input_pairs,
            "kept_pairs": self.kept_pairs,
            "dropped_pairs": self.dropped_pairs,
            "skipped_pairs": self.skipped_pairs,
            "mismatched_prompts": self.mismatched_prompts,
        }


@dataclass
class GrammarFixFileSummary:
    input_path: str
    output_path: str
    input_records: int
    kept_records: int
    dropped_records: int
    skipped_records: int

    def to_dict(self) -> dict:
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "input_records": self.input_records,
            "kept_records": self.kept_records,
            "dropped_records": self.dropped_records,
            "skipped_records": self.skipped_records,
        }


def grammar_fix_dataset_dir(
    input_dir: str,
    output_dir: str,
    grammar_config: GrammarConfig,
    pro_filename: Optional[str] = None,
    anti_filename: Optional[str] = None,
) -> Tuple[GrammarFixSummary, GrammarReport]:
    """
    Apply grammar judging/fixing to an existing dataset directory.

    Args:
        input_dir: Directory containing pro/anti JSONL files
        output_dir: Directory to write fixed JSONL files
        grammar_config: Grammar agent configuration
        pro_filename: Optional override for pro filename
        anti_filename: Optional override for anti filename

    Returns:
        (summary, report)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pro_path, anti_path = _resolve_input_paths(input_path, pro_filename, anti_filename)

    pro_records = read_jsonl(str(pro_path))
    anti_records = read_jsonl(str(anti_path))

    pro_by = {r.meta["pair_id"]: r for r in pro_records}
    anti_by = {r.meta["pair_id"]: r for r in anti_records}

    linter = GrammarLinter(mode=LintMode.WARN_ONLY, sample_rate=1.0, seed=42)
    cache = GrammarCache(grammar_config.cache_dir, enabled=grammar_config.cache_enabled)
    report = GrammarReport()
    agent = GrammarAgent(grammar_config, cache, report)

    kept_pro: List[dict] = []
    kept_anti: List[dict] = []

    dropped = 0
    skipped = 0
    mismatched = 0

    for pair_id, pro_rec in pro_by.items():
        anti_rec = anti_by.get(pair_id)
        if anti_rec is None:
            skipped += 1
            continue

        pro_user_info = _get_user_message(pro_rec)
        anti_user_info = _get_user_message(anti_rec)
        if pro_user_info is None or anti_user_info is None:
            skipped += 1
            continue

        pro_user_idx, pro_user = pro_user_info
        anti_user_idx, anti_user = anti_user_info

        if pro_user.content != anti_user.content:
            mismatched += 1
            continue

        updated_prompt, _decision = _apply_grammar_to_prompt(
            pro_user.content,
            pro_rec.meta,
            linter,
            agent,
            report,
        )
        if updated_prompt is None:
            dropped += 1
            continue

        pro_rec.messages[pro_user_idx] = Message(role="user", content=updated_prompt)
        anti_rec.messages[anti_user_idx] = Message(role="user", content=updated_prompt)

        kept_pro.append(pro_rec)
        kept_anti.append(anti_rec)

    write_jsonl(kept_pro, str(output_path / pro_path.name))
    write_jsonl(kept_anti, str(output_path / anti_path.name))
    report.save_to_file(output_path / "grammar_report.json")

    summary = GrammarFixSummary(
        pro_path=str(pro_path),
        anti_path=str(anti_path),
        output_dir=str(output_path),
        input_pairs=len(pro_records),
        kept_pairs=len(kept_pro),
        dropped_pairs=dropped,
        skipped_pairs=skipped,
        mismatched_prompts=mismatched,
    )
    return summary, report


def grammar_fix_jsonl_file(
    input_file: str,
    output_file: str,
    grammar_config: GrammarConfig,
) -> Tuple[GrammarFixFileSummary, GrammarReport]:
    """
    Apply grammar judging/fixing to a single JSONL file (no pairing).

    Args:
        input_file: Path to JSONL file
        output_file: Path to write fixed JSONL file
        grammar_config: Grammar agent configuration

    Returns:
        (summary, report)
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(str(input_path))

    linter = GrammarLinter(mode=LintMode.WARN_ONLY, sample_rate=1.0, seed=42)
    cache = GrammarCache(grammar_config.cache_dir, enabled=grammar_config.cache_enabled)
    report = GrammarReport()
    agent = GrammarAgent(grammar_config, cache, report)

    kept: List[Record] = []
    dropped = 0
    skipped = 0

    for rec in records:
        user_info = _get_user_message(rec)
        if user_info is None:
            skipped += 1
            continue

        user_idx, user_msg = user_info
        updated_prompt, _decision = _apply_grammar_to_prompt(
            user_msg.content,
            rec.meta,
            linter,
            agent,
            report,
        )
        if updated_prompt is None:
            dropped += 1
            continue

        rec.messages[user_idx] = Message(role="user", content=updated_prompt)
        kept.append(rec)

    write_jsonl(kept, str(output_path))
    report.save_to_file(output_path.parent / "grammar_report.json")

    summary = GrammarFixFileSummary(
        input_path=str(input_path),
        output_path=str(output_path),
        input_records=len(records),
        kept_records=len(kept),
        dropped_records=dropped,
        skipped_records=skipped,
    )
    return summary, report


def _resolve_input_paths(
    input_dir: Path,
    pro_filename: Optional[str],
    anti_filename: Optional[str],
) -> Tuple[Path, Path]:
    if pro_filename and anti_filename:
        pro_path = input_dir / pro_filename
        anti_path = input_dir / anti_filename
        if not pro_path.exists() or not anti_path.exists():
            raise FileNotFoundError("Specified pro/anti files not found")
        return pro_path, anti_path

    candidates = [
        ("pro.jsonl", "anti.jsonl"),
        ("pro_train.jsonl", "anti_train.jsonl"),
        ("pro_eval.jsonl", "anti_eval.jsonl"),
    ]
    for pro_name, anti_name in candidates:
        pro_path = input_dir / pro_name
        anti_path = input_dir / anti_name
        if pro_path.exists() and anti_path.exists():
            return pro_path, anti_path

    raise FileNotFoundError("No pro/anti JSONL pair found in input directory")


def _apply_grammar_to_prompt(
    prompt: str,
    meta: Dict,
    linter: GrammarLinter,
    agent: GrammarAgent,
    report: GrammarReport,
) -> Tuple[Optional[str], str]:
    content, tag = _split_prompt(prompt)
    if content is None or tag is None:
        report.record(decision="reject", reason="unknown_tag", sample_text=prompt)
        return None, "reject"

    ctx = SimpleNamespace(
        family_id=meta.get("family_id", "?"),
        subtype_id=meta.get("subtype_id", ""),
        mode=meta.get("mode", ""),
        formatting_variant=meta.get("formatting_variant", 0),
        perspective=meta.get("perspective", ""),
    )
    rendered = SimpleNamespace(template_id=meta.get("template_id", ""))

    lint = linter.check(content, ctx, rendered, record=False)

    if not agent.should_route(lint):
        report.record(decision="clean", reason="no_route", sample_text=content)
        return prompt, "clean"

    decision = agent.judge_and_fix(content, lint)

    final_decision = decision.decision
    reason = decision.reason

    if decision.llm_error and agent.config.fail_open:
        final_decision = "clean"
        reason = reason or "llm_error_fail_open"

    if agent.config.policy == "report_only":
        final_decision = "clean"

    if agent.config.policy == "filter":
        if final_decision != "clean":
            report.record(
                decision="reject",
                reason=reason or "policy_filter",
                sample_text=content,
                cache_hit=decision.cache_hit,
                llm_error=decision.llm_error,
            )
            return None, "reject"
        report.record(
            decision="clean",
            reason=reason,
            sample_text=content,
            cache_hit=decision.cache_hit,
            llm_error=decision.llm_error,
        )
        return prompt, "clean"

    if agent.config.policy in ("replace", "replace_then_filter"):
        if final_decision == "fixed" and decision.fixed_content:
            relint = linter.check(decision.fixed_content, ctx, rendered, record=False)
            if relint.has_blocking_errors:
                if agent.config.policy == "replace_then_filter":
                    report.record(
                        decision="reject",
                        reason="recheck_failed",
                        sample_text=content,
                        cache_hit=decision.cache_hit,
                        llm_error=decision.llm_error,
                        fix_failed_recheck=True,
                    )
                    return None, "reject"
                report.record(
                    decision="clean",
                    reason="recheck_failed",
                    sample_text=content,
                    cache_hit=decision.cache_hit,
                    llm_error=decision.llm_error,
                    fix_failed_recheck=True,
                )
                return prompt, "clean"
            fixed_prompt = f"{decision.fixed_content}\n\n{tag}"
            report.record(
                decision="fixed",
                reason=reason,
                sample_text=content,
                cache_hit=decision.cache_hit,
                llm_error=decision.llm_error,
            )
            return fixed_prompt, "fixed"

        if final_decision == "reject":
            if agent.config.policy == "replace_then_filter":
                report.record(
                    decision="reject",
                    reason=reason or "llm_reject",
                    sample_text=content,
                    cache_hit=decision.cache_hit,
                    llm_error=decision.llm_error,
                )
                return None, "reject"
            report.record(
                decision="clean",
                reason=reason,
                sample_text=content,
                cache_hit=decision.cache_hit,
                llm_error=decision.llm_error,
            )
            return prompt, "clean"

    report.record(
        decision="clean",
        reason=reason,
        sample_text=content,
        cache_hit=decision.cache_hit,
        llm_error=decision.llm_error,
    )
    return prompt, "clean"


def _split_prompt(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    tags = sorted(KNOWN_TAGS, key=len, reverse=True)
    trimmed = prompt.rstrip()
    for tag in tags:
        if trimmed.endswith(tag):
            content = trimmed[: -len(tag)]
            if content.endswith("\n\n"):
                content = content[:-2]
            return content, tag
    return None, None


def _get_user_message(record: Record) -> Optional[Tuple[int, Message]]:
    for idx, msg in enumerate(record.messages):
        if msg.role == "user":
            return idx, msg
    return None
