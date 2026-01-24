"""
CLI Module for Corrigibility Dataset Generation Pipeline.

Provides command-line interface for dataset generation, validation,
statistics, and sampling operations.

Task ID: T18
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional, List, Dict, Any
import random

from ..src.render import DatasetGenerator, SplitMode
from ..src.plan import AllocationConfig, load_allocation_config
from ..src.validate import validate_dataset
from ..src.package import read_jsonl
from ..src.schema import Record, FamilyID, Severity, Mode, Perspective
from ..src.lint import LintMode, LintReport
from ..src.agents import JustificationConfig


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main CLI entry point.

    Args:
        argv: Command line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    parser = create_parser()

    if argv is None:
        argv = sys.argv[1:]

    # Handle empty args
    if not argv:
        parser.print_help()
        return 0

    args = parser.parse_args(argv)

    # Check if subcommand was provided
    if not hasattr(args, 'func'):
        parser.print_help()
        return 0

    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="python -m dataset_gen.tools.cli",
        description="Dataset Generation Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Generate a dataset:
    python -m dataset_gen.tools.cli generate --output ./outputs

  Validate existing files:
    python -m dataset_gen.tools.cli validate --pro pro.jsonl --anti anti.jsonl

  Show statistics:
    python -m dataset_gen.tools.cli stats --pro pro.jsonl --anti anti.jsonl

  Generate samples:
    python -m dataset_gen.tools.cli sample --count 5
""",
    )

    subparsers = parser.add_subparsers(
        title="commands",
        description="Available commands",
        dest="command",
    )

    # Generate command
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate a new dataset",
        description="Run the full dataset generation pipeline",
    )
    gen_parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/scratch/output",
        help="Output directory for JSONL files (default: data/scratch/output)",
    )
    gen_parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to custom YAML config file (uses defaults if not specified)",
    )
    gen_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation (default: 42)",
    )
    gen_parser.add_argument(
        "--split",
        type=str,
        choices=["all", "train", "eval"],
        default="all",
        help="Split mode: all, train, or eval (default: all)",
    )
    gen_parser.add_argument(
        "--all-splits",
        action="store_true",
        help="Generate both train and eval splits in one pass",
    )
    gen_parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Override total dataset size from config",
    )
    gen_parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation before saving",
    )
    gen_parser.add_argument(
        "--lint",
        type=str,
        choices=["enabled", "warn_only", "disabled"],
        default="disabled",
        help="Grammar linting mode (default: disabled)",
    )
    gen_parser.add_argument(
        "--lint-sample-rate",
        type=float,
        default=1.0,
        help="Fraction of prompts to lint (0.0-1.0, default: 1.0)",
    )
    gen_parser.add_argument(
        "--lint-report",
        type=str,
        default=None,
        help="Path to save JSON lint report",
    )
    # Justification agent flags
    gen_parser.add_argument(
        "--justification-agent",
        action="store_true",
        default=False,
        help="Use LLM-based justification generation (default: template-based)",
    )
    gen_parser.add_argument(
        "--justification-provider",
        type=str,
        choices=["openai", "anthropic", "deepseek"],
        default="openai",
        help="LLM provider for justification generation (default: openai for OpenRouter)",
    )
    gen_parser.add_argument(
        "--justification-model",
        type=str,
        default="deepseek/deepseek-chat-v3.1",
        help="Model ID for justification generation (default: deepseek/deepseek-chat-v3.1)",
    )
    gen_parser.add_argument(
        "--justification-api-base",
        type=str,
        default="https://openrouter.ai/api/v1",
        help="API base URL (default: https://openrouter.ai/api/v1)",
    )
    gen_parser.add_argument(
        "--justification-temperature",
        type=float,
        default=0.2,
        help="Temperature for justification generation (default: 0.2)",
    )
    gen_parser.add_argument(
        "--justification-validation",
        type=str,
        choices=["warn", "strict"],
        default="warn",
        help="Validation mode: warn (track failures) or strict (retry/fallback)",
    )
    gen_parser.add_argument(
        "--no-justification-cache",
        action="store_true",
        default=False,
        help="Disable justification caching",
    )
    gen_parser.add_argument(
        "--justification-report",
        type=str,
        default=None,
        help="Path to save JSON justification validation report",
    )
    gen_parser.set_defaults(func=cmd_generate)

    # Validate command
    val_parser = subparsers.add_parser(
        "validate",
        help="Validate an existing dataset",
        description="Run validation checks on JSONL files",
    )
    val_parser.add_argument(
        "--pro",
        type=str,
        required=True,
        help="Path to pro-corrigibility JSONL file",
    )
    val_parser.add_argument(
        "--anti",
        type=str,
        required=True,
        help="Path to anti-corrigibility JSONL file",
    )
    val_parser.add_argument(
        "--holdout-tolerance",
        type=float,
        default=0.10,
        help="Tolerance for holdout ratio check (default: 0.10)",
    )
    val_parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors",
    )
    val_parser.set_defaults(func=cmd_validate)

    # Stats command
    stats_parser = subparsers.add_parser(
        "stats",
        help="Show dataset statistics",
        description="Display statistics about a dataset",
    )
    stats_parser.add_argument(
        "--pro",
        type=str,
        required=True,
        help="Path to pro-corrigibility JSONL file",
    )
    stats_parser.add_argument(
        "--anti",
        type=str,
        required=True,
        help="Path to anti-corrigibility JSONL file",
    )
    stats_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of formatted table",
    )
    stats_parser.set_defaults(func=cmd_stats)

    # Sample command
    sample_parser = subparsers.add_parser(
        "sample",
        help="Generate and display sample records",
        description="Generate sample records for inspection",
    )
    sample_parser.add_argument(
        "--count", "-n",
        type=int,
        default=3,
        help="Number of samples to generate (default: 3)",
    )
    sample_parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to custom YAML config file",
    )
    sample_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (defaults to random)",
    )
    sample_parser.add_argument(
        "--family",
        type=str,
        choices=["A", "B", "C", "D", "E", "F", "G", "H"],
        default=None,
        help="Filter by family (A-H)",
    )
    sample_parser.add_argument(
        "--severity",
        type=str,
        choices=["S1", "S2", "S3"],
        default=None,
        help="Filter by severity (S1, S2, S3)",
    )
    sample_parser.add_argument(
        "--mode",
        type=str,
        choices=["rating", "choice", "short"],
        default=None,
        help="Filter by mode (rating, choice, short)",
    )
    sample_parser.add_argument(
        "--show-pairs",
        action="store_true",
        help="Show both pro and anti responses",
    )
    sample_parser.add_argument(
        "--raw",
        action="store_true",
        help="Show raw JSON instead of formatted output",
    )
    sample_parser.set_defaults(func=cmd_sample)

    return parser


def cmd_generate(args: argparse.Namespace) -> int:
    """Generate dataset command."""
    # Load config
    if args.config:
        try:
            config = load_allocation_config(args.config)
            print(f"Loaded config from: {args.config}")
        except FileNotFoundError:
            print(f"Error: Config file not found: {args.config}", file=sys.stderr)
            return 2
        except Exception as e:
            print(f"Error loading config: {e}", file=sys.stderr)
            return 1
    else:
        config = AllocationConfig()
        print("Using default configuration")

    # Override size if specified
    if args.size is not None:
        config.total_size = args.size
        print(f"Dataset size: {config.total_size}")

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Parse lint mode
    lint_mode = LintMode(args.lint)
    if lint_mode != LintMode.DISABLED:
        print(f"Grammar linting: {lint_mode.value} (sample rate: {args.lint_sample_rate})")

    # Build justification agent config if enabled
    justification_agent_config = None
    if args.justification_agent:
        justification_agent_config = JustificationConfig(
            model_provider=args.justification_provider,
            model_id=args.justification_model,
            api_base=args.justification_api_base,
            temperature=args.justification_temperature,
            validation_mode=args.justification_validation,
            cache_enabled=not args.no_justification_cache,
            cache_dir=output_path / "justification_cache",
        )
        print(f"Justification agent: {args.justification_model}")
        print(f"  Provider: {args.justification_provider} (API: {args.justification_api_base})")
        print(f"  Validation mode: {args.justification_validation}")
        print(f"  Cache enabled: {not args.no_justification_cache}")

    # Create generator
    generator = DatasetGenerator(
        config,
        global_seed=args.seed,
        lint_mode=lint_mode,
        lint_sample_rate=args.lint_sample_rate,
        justification_agent_config=justification_agent_config,
    )

    # Progress callback
    def progress_callback(current: int, total: int) -> None:
        if current % 100 == 0 or current == total:
            print(f"\rProgress: {current}/{total} ({100*current//total}%)", end="", flush=True)

    validate = not args.no_validate

    try:
        if args.all_splits:
            print(f"\nGenerating all splits to: {output_path}")
            result = generator.generate_all_splits(
                str(output_path),
                validate=validate,
            )
            print()  # Newline after progress
            train_pro, train_anti = result["train"]
            eval_pro, eval_anti = result["eval"]
            print(f"Train split: {train_pro} pro, {train_anti} anti records")
            print(f"Eval split:  {eval_pro} pro, {eval_anti} anti records")
            print(f"\nFiles created:")
            print(f"  {output_path}/pro_train.jsonl")
            print(f"  {output_path}/anti_train.jsonl")
            print(f"  {output_path}/pro_eval.jsonl")
            print(f"  {output_path}/anti_eval.jsonl")
        else:
            print(f"\nGenerating {args.split} split to: {output_path}")
            # Generate with progress
            pro_records, anti_records = generator.generate(
                split=args.split,
                progress_callback=progress_callback,
            )
            print()  # Newline after progress

            # Validate if requested
            if validate:
                print("Validating dataset...")
                errors = validate_dataset(pro_records, anti_records)
                if errors:
                    print(f"Validation failed with {len(errors)} error(s):", file=sys.stderr)
                    for err in errors[:10]:
                        print(f"  - {err}", file=sys.stderr)
                    if len(errors) > 10:
                        print(f"  ... and {len(errors) - 10} more", file=sys.stderr)
                    return 1

            # Save files
            from ..src.package import write_jsonl

            if args.split == "all":
                pro_filename = "pro.jsonl"
                anti_filename = "anti.jsonl"
            elif args.split == "train":
                pro_filename = "pro_train.jsonl"
                anti_filename = "anti_train.jsonl"
            else:  # eval
                pro_filename = "pro_eval.jsonl"
                anti_filename = "anti_eval.jsonl"

            write_jsonl(pro_records, str(output_path / pro_filename))
            write_jsonl(anti_records, str(output_path / anti_filename))

            print(f"Generated: {len(pro_records)} pro, {len(anti_records)} anti records")
            print(f"\nFiles created:")
            print(f"  {output_path / pro_filename}")
            print(f"  {output_path / anti_filename}")

        # Print lint summary if linting was enabled
        if lint_mode != LintMode.DISABLED:
            lint_report = generator.get_lint_report()
            print("\nGrammar Linting Summary:")
            print(f"  Prompts checked: {lint_report.total_prompts}")
            print(f"  Blocking errors: {lint_report.total_blocking_errors} (in {lint_report.prompts_with_blocking_errors} prompts)")
            print(f"  Warnings: {lint_report.total_warnings} (in {lint_report.prompts_with_warnings} prompts)")

            if lint_report.blocking_errors_by_rule:
                top_rules = sorted(
                    lint_report.blocking_errors_by_rule.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                top_rules_str = ", ".join(f"{r} ({c})" for r, c in top_rules)
                print(f"  Top blocking rules: {top_rules_str}")

            # Save lint report if requested
            if args.lint_report:
                lint_report_path = Path(args.lint_report)
                lint_report_path.parent.mkdir(parents=True, exist_ok=True)
                with open(lint_report_path, 'w') as f:
                    json.dump(lint_report.to_dict(), f, indent=2)
                print(f"  Report saved to: {lint_report_path}")

        # Print justification validation summary if agent was enabled
        if args.justification_agent:
            just_report = generator.get_justification_report()
            if just_report and just_report.total_generated > 0:
                pass_rate = 100 * just_report.total_passed / just_report.total_generated
                fail_rate = 100 * just_report.total_failed / just_report.total_generated
                print("\nJustification Validation Summary:")
                print(f"  Total generated: {just_report.total_generated}")
                print(f"  Passed: {just_report.total_passed} ({pass_rate:.1f}%)")
                print(f"  Failed: {just_report.total_failed} ({fail_rate:.1f}%)")
                if just_report.total_retries > 0:
                    print(f"  Retries: {just_report.total_retries}")
                if just_report.total_fallbacks > 0:
                    print(f"  Fallbacks: {just_report.total_fallbacks}")

                if just_report.failure_counts:
                    top_failures = sorted(
                        just_report.failure_counts.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]
                    top_str = ", ".join(f"{r} ({c})" for r, c in top_failures)
                    print(f"  Top failure reasons: {top_str}")

                # Save justification report if requested
                if args.justification_report:
                    just_report_path = Path(args.justification_report)
                    just_report_path.parent.mkdir(parents=True, exist_ok=True)
                    just_report.save_to_file(just_report_path)
                    print(f"  Report saved to: {just_report_path}")

        # Clean up resources (linter, justification agent cache)
        generator.close()

        print("\nGeneration complete!")
        return 0

    except ValueError as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate existing dataset command."""
    # Check files exist
    pro_path = Path(args.pro)
    anti_path = Path(args.anti)

    if not pro_path.exists():
        print(f"Error: File not found: {args.pro}", file=sys.stderr)
        return 2

    if not anti_path.exists():
        print(f"Error: File not found: {args.anti}", file=sys.stderr)
        return 2

    # Read files
    print(f"Reading {args.pro}...")
    try:
        pro_records = read_jsonl(str(pro_path))
    except Exception as e:
        print(f"Error reading {args.pro}: {e}", file=sys.stderr)
        return 1

    print(f"Reading {args.anti}...")
    try:
        anti_records = read_jsonl(str(anti_path))
    except Exception as e:
        print(f"Error reading {args.anti}: {e}", file=sys.stderr)
        return 1

    print(f"Loaded {len(pro_records)} pro and {len(anti_records)} anti records")
    print("\nRunning validation checks...")

    # Validate
    errors = validate_dataset(pro_records, anti_records)

    if errors:
        print(f"\nValidation FAILED with {len(errors)} error(s):")
        for err in errors:
            print(f"  - {err}")
        return 1
    else:
        print("\nValidation PASSED - all checks succeeded!")
        return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Show dataset statistics command."""
    # Check files exist
    pro_path = Path(args.pro)
    anti_path = Path(args.anti)

    if not pro_path.exists():
        print(f"Error: File not found: {args.pro}", file=sys.stderr)
        return 2

    if not anti_path.exists():
        print(f"Error: File not found: {args.anti}", file=sys.stderr)
        return 2

    # Read files
    try:
        pro_records = read_jsonl(str(pro_path))
        anti_records = read_jsonl(str(anti_path))
    except Exception as e:
        print(f"Error reading files: {e}", file=sys.stderr)
        return 1

    # Compute statistics
    stats = compute_statistics(pro_records, anti_records)

    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print_statistics(stats)

    return 0


def compute_statistics(
    pro_records: List[Record],
    anti_records: List[Record],
) -> Dict[str, Any]:
    """Compute dataset statistics."""
    all_records = pro_records + anti_records

    # Record counts
    record_counts = {
        "pro": len(pro_records),
        "anti": len(anti_records),
        "total": len(all_records),
    }

    # Family distribution
    family_counts = Counter(r.meta.get("family_id") for r in pro_records)

    # Severity distribution
    severity_counts = Counter(r.meta.get("severity") for r in pro_records)

    # Mode distribution
    mode_counts = Counter(r.meta.get("mode") for r in pro_records)

    # Perspective distribution
    perspective_counts = Counter(r.meta.get("perspective") for r in pro_records)

    # Holdout statistics
    holdout_count = sum(1 for r in pro_records if r.meta.get("is_holdout"))
    train_count = len(pro_records) - holdout_count

    holdout_stats = {
        "train_count": train_count,
        "eval_count": holdout_count,
        "holdout_ratio": holdout_count / len(pro_records) if pro_records else 0,
    }

    # Unique templates
    template_ids = {r.meta.get("template_id") for r in pro_records}
    unique_template_count = len(template_ids)

    # Unique pair IDs
    pair_ids = {r.meta.get("pair_id") for r in pro_records}
    unique_pair_count = len(pair_ids)

    return {
        "record_counts": record_counts,
        "family_distribution": dict(family_counts),
        "severity_distribution": dict(severity_counts),
        "mode_distribution": dict(mode_counts),
        "perspective_distribution": dict(perspective_counts),
        "holdout_stats": holdout_stats,
        "unique_templates": unique_template_count,
        "unique_pairs": unique_pair_count,
    }


def print_statistics(stats: Dict[str, Any]) -> None:
    """Print statistics in formatted table format."""
    print()
    print("Dataset Statistics")
    print("=" * 50)
    print()

    # Record counts
    counts = stats["record_counts"]
    print("Record Counts")
    print("-" * 50)
    print(f"  Pro records:          {counts['pro']:>6}")
    print(f"  Anti records:         {counts['anti']:>6}")
    print(f"  Total:                {counts['total']:>6}")
    print()

    # Family distribution
    family_dist = stats["family_distribution"]
    total_pro = stats["record_counts"]["pro"]
    print("Family Distribution")
    print("-" * 50)
    for family_id in sorted(family_dist.keys()):
        count = family_dist[family_id]
        pct = 100 * count / total_pro if total_pro else 0
        # Get full family name
        family_name = family_id
        for f in FamilyID:
            if f.value == family_id or f.name == family_id:
                family_name = f"{f.name} ({f.value})"
                break
        print(f"  {family_name:<30} {count:>5} ({pct:>5.1f}%)")
    print()

    # Severity distribution
    sev_dist = stats["severity_distribution"]
    print("Severity Distribution")
    print("-" * 50)
    for sev in ["low", "medium", "high"]:
        if sev in sev_dist:
            count = sev_dist[sev]
            pct = 100 * count / total_pro if total_pro else 0
            sev_label = {"low": "S1 (low)", "medium": "S2 (medium)", "high": "S3 (high)"}.get(sev, sev)
            print(f"  {sev_label:<30} {count:>5} ({pct:>5.1f}%)")
    print()

    # Mode distribution
    mode_dist = stats["mode_distribution"]
    print("Mode Distribution")
    print("-" * 50)
    for mode in ["rating", "choice", "short"]:
        if mode in mode_dist:
            count = mode_dist[mode]
            pct = 100 * count / total_pro if total_pro else 0
            print(f"  {mode:<30} {count:>5} ({pct:>5.1f}%)")
    print()

    # Perspective distribution
    persp_dist = stats["perspective_distribution"]
    print("Perspective Distribution")
    print("-" * 50)
    for persp in ["first", "third"]:
        if persp in persp_dist:
            count = persp_dist[persp]
            pct = 100 * count / total_pro if total_pro else 0
            print(f"  {persp:<30} {count:>5} ({pct:>5.1f}%)")
    print()

    # Holdout split
    holdout = stats["holdout_stats"]
    print("Holdout Split")
    print("-" * 50)
    train_pct = 100 * holdout["train_count"] / total_pro if total_pro else 0
    eval_pct = 100 * holdout["eval_count"] / total_pro if total_pro else 0
    print(f"  Train records:        {holdout['train_count']:>6} ({train_pct:>5.1f}%)")
    print(f"  Eval records:         {holdout['eval_count']:>6} ({eval_pct:>5.1f}%)")
    print()

    # Template and pair counts
    print("Unique Counts")
    print("-" * 50)
    print(f"  Unique templates:     {stats['unique_templates']:>6}")
    print(f"  Unique pairs:         {stats['unique_pairs']:>6}")
    print()


def cmd_sample(args: argparse.Namespace) -> int:
    """Generate and display sample records command."""
    # Load config
    if args.config:
        try:
            config = load_allocation_config(args.config)
        except FileNotFoundError:
            print(f"Error: Config file not found: {args.config}", file=sys.stderr)
            return 2
        except Exception as e:
            print(f"Error loading config: {e}", file=sys.stderr)
            return 1
    else:
        config = AllocationConfig()

    # Use a smaller size for sampling
    config.total_size = max(100, args.count * 10)

    # Set seed
    seed = args.seed if args.seed is not None else random.randint(0, 999999)

    # Generate records
    generator = DatasetGenerator(config, global_seed=seed)
    pro_records, anti_records = generator.generate()

    # Build lookup for anti records
    anti_by_id = {r.meta["pair_id"]: r for r in anti_records}

    # Apply filters
    filtered = pro_records
    if args.family:
        family_value = FamilyID[args.family].value
        filtered = [r for r in filtered if r.meta.get("family_id") == family_value]

    if args.severity:
        sev_value = Severity[args.severity].value
        filtered = [r for r in filtered if r.meta.get("severity") == sev_value]

    if args.mode:
        filtered = [r for r in filtered if r.meta.get("mode") == args.mode]

    if not filtered:
        print("No records match the specified filters.", file=sys.stderr)
        return 1

    # Sample records
    rng = random.Random(seed)
    sample_count = min(args.count, len(filtered))
    samples = rng.sample(filtered, sample_count)

    # Display samples
    for i, pro_record in enumerate(samples, 1):
        if args.raw:
            print(json.dumps(pro_record.to_dict(), indent=2))
            if args.show_pairs:
                anti_record = anti_by_id.get(pro_record.meta["pair_id"])
                if anti_record:
                    print(json.dumps(anti_record.to_dict(), indent=2))
        else:
            print_sample(pro_record, i, len(samples), args.show_pairs, anti_by_id)

    return 0


def print_sample(
    pro_record: Record,
    index: int,
    total: int,
    show_pairs: bool,
    anti_by_id: Dict[str, Record],
) -> None:
    """Print a formatted sample record."""
    meta = pro_record.meta

    print()
    print("=" * 70)
    print(f"SAMPLE {index}/{total}")
    print("=" * 70)
    print()

    # Metadata summary
    print(f"Family:      {meta.get('family_id', 'N/A')}")
    print(f"Subtype:     {meta.get('subtype_id', 'N/A')}")
    print(f"Severity:    {meta.get('severity', 'N/A')}")
    print(f"Mode:        {meta.get('mode', 'N/A')}")
    print(f"Perspective: {meta.get('perspective', 'N/A')}")
    print(f"Template:    {meta.get('template_id', 'N/A')}")
    print(f"Holdout:     {meta.get('is_holdout', 'N/A')}")
    print()

    # Prompt
    print("-" * 70)
    print("[PROMPT]")
    print("-" * 70)
    for msg in pro_record.messages:
        if msg.role == "user":
            print(msg.content)
    print()

    # Pro response
    print("-" * 70)
    print("[PRO RESPONSE]")
    print("-" * 70)
    for msg in pro_record.messages:
        if msg.role == "assistant":
            # Pretty print JSON response
            try:
                response_data = json.loads(msg.content)
                print(json.dumps(response_data, indent=2))
            except json.JSONDecodeError:
                print(msg.content)
    print()

    # Anti response (if requested)
    if show_pairs:
        anti_record = anti_by_id.get(meta.get("pair_id"))
        if anti_record:
            print("-" * 70)
            print("[ANTI RESPONSE]")
            print("-" * 70)
            for msg in anti_record.messages:
                if msg.role == "assistant":
                    try:
                        response_data = json.loads(msg.content)
                        print(json.dumps(response_data, indent=2))
                    except json.JSONDecodeError:
                        print(msg.content)
            print()


if __name__ == "__main__":
    sys.exit(main())
