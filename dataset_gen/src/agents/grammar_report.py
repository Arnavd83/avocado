"""
Reporting for grammar judge/fix decisions.

Tracks counts, reasons, and samples for debugging.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class GrammarReport:
    """
    Aggregate statistics across grammar decisions.

    Tracks:
    - total_checked: prompts routed to grammar agent
    - total_passed: accepted without change
    - total_fixed: accepted with fixes
    - total_rejected: dropped due to grammar issues
    - total_cache_hits: decisions served from cache
    - total_llm_errors: failures due to LLM issues
    - failure_counts: reasons for rejection/errors
    - sample_rejections: sample texts for debugging (truncated)
    """

    total_checked: int = 0
    total_passed: int = 0
    total_fixed: int = 0
    total_rejected: int = 0
    total_cache_hits: int = 0
    total_llm_errors: int = 0
    total_fix_failed_recheck: int = 0

    failure_counts: Dict[str, int] = field(default_factory=dict)
    sample_rejections: Dict[str, List[str]] = field(default_factory=dict)

    def record(
        self,
        decision: str,
        reason: str = "",
        sample_text: str = "",
        cache_hit: bool = False,
        llm_error: bool = False,
        fix_failed_recheck: bool = False,
    ) -> None:
        self.total_checked += 1

        if cache_hit:
            self.total_cache_hits += 1
        if llm_error:
            self.total_llm_errors += 1
        if fix_failed_recheck:
            self.total_fix_failed_recheck += 1

        if decision == "clean":
            self.total_passed += 1
        elif decision == "fixed":
            self.total_fixed += 1
        elif decision == "reject":
            self.total_rejected += 1
        else:
            # Count unknown as rejection for reporting
            self.total_rejected += 1
            reason = reason or "unknown_decision"

        if decision == "reject" and reason:
            self.failure_counts[reason] = self.failure_counts.get(reason, 0) + 1
            if sample_text:
                if reason not in self.sample_rejections:
                    self.sample_rejections[reason] = []
                if len(self.sample_rejections[reason]) < 5:
                    sample = sample_text[:120]
                    if len(sample_text) > 120:
                        sample += "..."
                    self.sample_rejections[reason].append(sample)

    def to_dict(self) -> dict:
        total = self.total_checked
        return {
            "total_checked": total,
            "total_passed": self.total_passed,
            "total_fixed": self.total_fixed,
            "total_rejected": self.total_rejected,
            "total_cache_hits": self.total_cache_hits,
            "total_llm_errors": self.total_llm_errors,
            "total_fix_failed_recheck": self.total_fix_failed_recheck,
            "pass_rate": (self.total_passed / total) if total else 0,
            "fix_rate": (self.total_fixed / total) if total else 0,
            "reject_rate": (self.total_rejected / total) if total else 0,
            "failure_counts": self.failure_counts,
            "sample_rejections": self.sample_rejections,
        }

    def save_to_file(self, filepath: Path) -> None:
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
