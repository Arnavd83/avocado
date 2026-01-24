"""
Validation reporting for justification generation.

This module provides:
1. ValidationResult: Result of validating a single justification
2. ValidationReport: Aggregate statistics across all generations

The soft validation approach tracks failures as warnings rather than
blocking, allowing analysis of failure patterns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ValidationResult:
    """
    Result of validating a single justification.

    Attributes:
        passed: Whether all checks passed
        failure_reasons: List of failure reasons (empty if passed)
    """

    passed: bool
    failure_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "failure_reasons": self.failure_reasons,
        }


@dataclass
class ValidationReport:
    """
    Aggregate validation statistics across all generations.

    Tracks:
    - Total generated/passed/failed counts
    - Breakdown of failure reasons
    - Sample failures for debugging

    Usage:
        report = ValidationReport()
        for justification in justifications:
            result = validate(justification)
            report.record(result, justification)
        report.print_summary()
    """

    total_generated: int = 0
    total_passed: int = 0
    total_failed: int = 0

    # Failure reason counts
    failure_counts: Dict[str, int] = field(default_factory=dict)

    # Samples of failures for debugging (up to 5 per type)
    failure_samples: Dict[str, List[str]] = field(default_factory=dict)

    # Track retries and fallbacks
    total_retries: int = 0
    total_fallbacks: int = 0

    def record(
        self,
        result: ValidationResult,
        justification: str,
        used_retry: bool = False,
        used_fallback: bool = False,
    ) -> None:
        """
        Record a validation result.

        Args:
            result: The validation result
            justification: The justification text
            used_retry: Whether retry was needed
            used_fallback: Whether fallback was used
        """
        self.total_generated += 1

        if used_retry:
            self.total_retries += 1
        if used_fallback:
            self.total_fallbacks += 1

        if result.passed:
            self.total_passed += 1
        else:
            self.total_failed += 1
            for reason in result.failure_reasons:
                self.failure_counts[reason] = self.failure_counts.get(reason, 0) + 1

                # Keep up to 5 samples per failure type
                if reason not in self.failure_samples:
                    self.failure_samples[reason] = []
                if len(self.failure_samples[reason]) < 5:
                    # Truncate long justifications
                    sample = justification[:100]
                    if len(justification) > 100:
                        sample += "..."
                    self.failure_samples[reason].append(sample)

    def print_summary(self) -> None:
        """Print aggregate validation summary to stdout."""
        if self.total_generated == 0:
            print("\n=== Justification Validation Report ===")
            print("No justifications generated.")
            return

        pass_rate = 100 * self.total_passed / self.total_generated
        fail_rate = 100 * self.total_failed / self.total_generated

        print("\n=== Justification Validation Report ===")
        print(f"Total: {self.total_generated}")
        print(f"Passed: {self.total_passed} ({pass_rate:.1f}%)")
        print(f"Failed: {self.total_failed} ({fail_rate:.1f}%)")

        if self.total_retries > 0:
            print(f"Retries: {self.total_retries}")
        if self.total_fallbacks > 0:
            print(f"Fallbacks: {self.total_fallbacks}")

        if self.failure_counts:
            print("\nFailure breakdown:")
            for reason, count in sorted(
                self.failure_counts.items(), key=lambda x: -x[1]
            ):
                print(f"  {reason}: {count}")

            print("\nSample failures:")
            for reason, samples in self.failure_samples.items():
                print(f"\n  {reason}:")
                for i, sample in enumerate(samples[:2], 1):  # Show max 2 samples
                    print(f"    {i}. {sample}")

    def to_dict(self) -> dict:
        """Export as dict for JSON serialization."""
        return {
            "total_generated": self.total_generated,
            "total_passed": self.total_passed,
            "total_failed": self.total_failed,
            "total_retries": self.total_retries,
            "total_fallbacks": self.total_fallbacks,
            "pass_rate": (
                self.total_passed / self.total_generated
                if self.total_generated > 0
                else 0
            ),
            "failure_counts": self.failure_counts,
            "failure_samples": self.failure_samples,
        }

    def save_to_file(self, filepath: Path) -> None:
        """
        Save report to JSON file.

        Args:
            filepath: Path to save report
        """
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: Path) -> "ValidationReport":
        """
        Load report from JSON file.

        Args:
            filepath: Path to load from

        Returns:
            ValidationReport instance
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        report = cls()
        report.total_generated = data.get("total_generated", 0)
        report.total_passed = data.get("total_passed", 0)
        report.total_failed = data.get("total_failed", 0)
        report.total_retries = data.get("total_retries", 0)
        report.total_fallbacks = data.get("total_fallbacks", 0)
        report.failure_counts = data.get("failure_counts", {})
        report.failure_samples = data.get("failure_samples", {})
        return report
