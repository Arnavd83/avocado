"""
Pipeline orchestrator (Stage 6).

``orchestrate`` wires PLAN → PROMPT → ANSWER → PACKAGE → VALIDATE into one run,
threading a shared ``GenerationReport`` through both agents and an optional
``ResponseCache`` (applied transparently via ``CachingLLMClient``). Serial for the
first build (the agents are pure per-pair, so a future async pass is a drop-in).

Outputs written to ``output_dir``:
- ``corrigibility_pro_{N}.jsonl`` / ``corrigibility_anti_{N}.jsonl``
- ``specs.jsonl`` (the full plan, for audit)
- ``generation_report.json`` (Stage 5 dataset checks + the generation skip log)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional

from . import catalog as _catalog
from .cache import CachingLLMClient, ResponseCache
from .config import GenerationConfig
from .llm import LLMClient
from .report import GenerationReport
from .schema import PromptedSpec, Record
from .stage1_plan import holdout_keys, plan, validate_plan
from .stage2_prompt import PromptAgent
from .stage3_answer import AnswerAgent
from .stage4_package import RecordPackager, write_jsonl, write_pair_jsonl
from .stage5_validate import ValidationReport, validate_dataset

_STOP_STAGES = ("plan", "prompt", "answer", "package", "validate")


@dataclass
class RunResult:
    """Summary of a completed (or partial) run."""

    pro_path: Optional[str]
    anti_path: Optional[str]
    report_path: Optional[str]
    validation: Optional[ValidationReport]
    n_pairs: int
    n_specs: int
    n_prompt_skips: int
    n_answer_skips: int
    stopped_after: Optional[str] = None


def _validate_static_assets(config: GenerationConfig) -> None:
    """Fail fast if catalog / pools / versions are inconsistent with the config."""
    errors: List[str] = []
    errors += _catalog.validate_catalog()
    errors += _catalog.validate_system_prompt_pool()
    if len(_catalog.STYLE_DIRECTIVES) != config.style_directive_pool_size:
        errors.append(
            f"style_directive_pool_size={config.style_directive_pool_size} != "
            f"len(STYLE_DIRECTIVES)={len(_catalog.STYLE_DIRECTIVES)}"
        )
    if len(_catalog.SYSTEM_PROMPT_POOL) != config.system_prompt_pool_size:
        errors.append(
            f"system_prompt_pool_size={config.system_prompt_pool_size} != "
            f"len(SYSTEM_PROMPT_POOL)={len(_catalog.SYSTEM_PROMPT_POOL)}"
        )
    if config.catalog_version != _catalog.CATALOG_VERSION:
        errors.append(
            f"config.catalog_version={config.catalog_version!r} != "
            f"catalog.CATALOG_VERSION={_catalog.CATALOG_VERSION!r}"
        )
    if errors:
        raise ValueError("Static-asset validation failed:\n  - " + "\n  - ".join(errors))


def _wrap(client: LLMClient, cache: Optional[ResponseCache]):
    return CachingLLMClient(client, cache) if cache is not None else client


def orchestrate(
    config: GenerationConfig,
    prompt_llm: LLMClient,
    answer_llm: LLMClient,
    output_dir: str,
    cache: Optional[ResponseCache] = None,
    stop_after: Optional[str] = None,
    min_records: int = 50,
) -> RunResult:
    """Run the pipeline end to end (or up to ``stop_after``)."""
    if stop_after is not None and stop_after not in _STOP_STAGES:
        raise ValueError(f"stop_after must be one of {_STOP_STAGES}, got {stop_after!r}")

    # 1. Static assets + PLAN.
    _validate_static_assets(config)
    specs = plan(config)
    plan_issues = validate_plan(specs, config)
    hard = [i for i in plan_issues if "holdout" in i.lower() or "duplicate pair" in i.lower()
            or "length" in i.lower()]
    if hard:
        raise ValueError("Plan validation failed (hard):\n  - " + "\n  - ".join(hard))
    if stop_after == "plan":
        return RunResult(None, None, None, None, 0, len(specs), 0, 0, stopped_after="plan")

    # Agents + cache + report.
    p_llm = _wrap(prompt_llm, cache)
    a_llm = _wrap(answer_llm, cache)
    report = GenerationReport(
        prompt_agent_model=prompt_llm.model_id,
        answer_agent_model=answer_llm.model_id,
    )
    prompt_agent = PromptAgent(p_llm, report)
    answer_agent = AnswerAgent(a_llm, report)
    packager = RecordPackager(prompt_llm.model_id, answer_llm.model_id)

    # 2-5. Per-pair loop, stopping once target_pairs complete pairs exist.
    prompted_list: List[PromptedSpec] = []
    records: List[Record] = []
    n_prompt_skips = 0
    n_answer_skips = 0
    completed = 0

    for spec in specs:
        if completed >= config.target_pairs:
            break

        p_out = prompt_agent.generate(spec)
        report.record_prompt_attempts(p_out.attempts)
        if p_out.skipped or p_out.prompted is None:
            n_prompt_skips += 1
            continue
        prompted = p_out.prompted
        prompted_list.append(prompted)
        if stop_after == "prompt":
            continue

        pair = answer_agent.generate_pair(prompted)
        if pair.skipped or pair.pro is None or pair.anti is None:
            n_answer_skips += 1
            continue
        if stop_after == "answer":
            completed += 1
            continue

        pro_rec, anti_rec = packager.package_pair(prompted, pair.pro, pair.anti)
        records.extend([pro_rec, anti_rec])
        completed += 1

    if cache is not None:
        cache.save()

    if stop_after in ("prompt", "answer"):
        return RunResult(
            None, None, None, None, completed, len(specs),
            n_prompt_skips, n_answer_skips, stopped_after=stop_after,
        )

    # 6. WRITE.
    os.makedirs(output_dir, exist_ok=True)
    pro_path, anti_path = write_pair_jsonl(records, output_dir)
    _write_specs(prompted_list, output_dir)

    if stop_after == "package":
        return RunResult(
            pro_path, anti_path, None, None, completed, len(specs),
            n_prompt_skips, n_answer_skips, stopped_after="package",
        )

    # 7. VALIDATE + report.
    pro_records = [r for r in records if r.meta.get("condition") == "pro"]
    anti_records = [r for r in records if r.meta.get("condition") == "anti"]
    validation = validate_dataset(
        pro_records, anti_records, config,
        skip_log=report.skips, holdout_keys=holdout_keys(config), min_records=min_records,
    )
    report_path = _write_report(output_dir, validation, report, completed, len(specs))

    return RunResult(
        pro_path, anti_path, report_path, validation, completed, len(specs),
        n_prompt_skips, n_answer_skips, stopped_after=None,
    )


def _write_specs(prompted_list: List[PromptedSpec], output_dir: str) -> None:
    path = os.path.join(output_dir, "specs.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for ps in prompted_list:
            row = ps.spec.to_dict()
            row["prompt_text"] = ps.prompt_text
            row["prompt_generation_method"] = ps.prompt_generation_method
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_report(
    output_dir: str,
    validation: ValidationReport,
    report: GenerationReport,
    n_pairs: int,
    n_specs: int,
) -> str:
    path = os.path.join(output_dir, "generation_report.json")
    payload = {
        "n_pairs": n_pairs,
        "n_specs_queued": n_specs,
        "hard_failed": validation.hard_failed(),
        "validation": validation.to_dict(),
        "generation": report.to_dict(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


__all__ = ["orchestrate", "RunResult", "write_jsonl"]
