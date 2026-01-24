"""
Render/Orchestrator Module (T17)

Main orchestrator for the dataset generation pipeline.
Coordinates the 7-layer flow:
1. Plan → 2. Context → 3. Variation → 4. Family Render →
5. Answer Policy → 6. Packaging → 7. Validation
"""

from dataclasses import replace
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Callable, Dict

from .schema import Record, Context, FamilyID, RenderedPrompt
from .plan import PlanGenerator, AllocationConfig
from .context import ContextSynthesizer
from .variation import VariationApplicator
from .families.registry import get_family_plugin, import_all_families
from .answers import AnswerPolicy
from .package import RecordPackager, write_jsonl
from .validate import validate_dataset
from .lint import GrammarLinter, LintMode, LintReport, GrammarError
from .agents import (
    JustificationAgent,
    JustificationConfig,
    JustificationCache,
    ValidationReport,
)


SplitMode = Literal["all", "train", "eval"]


class DatasetGenerator:
    """
    Main orchestrator for dataset generation pipeline.

    Coordinates the 7-layer flow:
    1. Plan → 2. Context → 3. Variation → 4. Family Render →
    5. Answer Policy → 6. Packaging → 7. Validation

    Supports split modes for holdout template system:
    - "all": Generate all datapoints with is_holdout flag in metadata
    - "train": Only generate non-holdout datapoints (~85%)
    - "eval": Only generate holdout datapoints (~15%)
    """

    def __init__(
        self,
        config: AllocationConfig,
        global_seed: int,
        lint_mode: LintMode = LintMode.DISABLED,
        lint_sample_rate: float = 1.0,
        justification_agent_config: Optional[JustificationConfig] = None,
    ):
        """
        Initialize the generator with config and seed.

        Args:
            config: AllocationConfig with quotas and holdout settings
            global_seed: Master seed for deterministic generation
            lint_mode: Grammar linting mode (enabled, warn_only, disabled)
            lint_sample_rate: Fraction of prompts to lint (0.0-1.0)
            justification_agent_config: Optional config to enable LLM-based
                justification generation. If None, template-based justifications
                are used.
        """
        self.config = config
        self.global_seed = global_seed

        # Ensure family plugins are loaded
        import_all_families()

        # Initialize pipeline components
        self._plan_generator = PlanGenerator(config, global_seed)
        self._context_synthesizer = ContextSynthesizer(global_seed)
        self._variation_applicator = VariationApplicator(global_seed)
        self._answer_policy = AnswerPolicy(global_seed)
        self._record_packager = RecordPackager()

        # Initialize grammar linter
        self._linter = GrammarLinter(
            mode=lint_mode,
            sample_rate=lint_sample_rate,
            seed=global_seed,
        )

        # Initialize justification agent if config provided
        self._justification_agent: Optional[JustificationAgent] = None
        self._justification_cache: Optional[JustificationCache] = None
        self._justification_report: Optional[ValidationReport] = None

        if justification_agent_config is not None:
            self._justification_cache = JustificationCache(
                cache_dir=justification_agent_config.cache_dir,
                enabled=justification_agent_config.cache_enabled,
            )
            self._justification_report = ValidationReport()
            self._justification_agent = JustificationAgent(
                config=justification_agent_config,
                cache=self._justification_cache,
                report=self._justification_report,
            )
            self._answer_policy.set_justification_agent(self._justification_agent)

        # Track configured plugins to avoid duplicate configuration
        self._configured_plugins: Dict[str, bool] = {}

    def _configure_plugin(self, family_id: FamilyID) -> None:
        """Configure holdout settings for a family plugin if not already done."""
        # Get the letter ID for the family
        family_letter = family_id.value if hasattr(family_id, 'value') else str(family_id)

        if family_letter not in self._configured_plugins:
            plugin = get_family_plugin(family_id)
            plugin.configure_holdout(
                self.config.holdout_ratio,
                self.config.holdout_seed
            )
            self._configured_plugins[family_letter] = True

    def _should_include_record(self, is_holdout: bool, split: SplitMode) -> bool:
        """Determine if a record should be included based on split mode."""
        if split == "all":
            return True
        elif split == "train":
            return not is_holdout
        elif split == "eval":
            return is_holdout
        else:
            raise ValueError(f"Invalid split mode: {split}")

    def generate(
        self,
        split: SplitMode = "all",
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Tuple[List[Record], List[Record]]:
        """
        Generate the complete dataset.

        Args:
            split: Which records to generate
                - "all": All records (is_holdout in metadata)
                - "train": Only non-holdout records
                - "eval": Only holdout records
            progress_callback: Optional callback(current, total) for progress

        Returns:
            (pro_records, anti_records) tuple
        """
        # Layer 1: Generate plan
        plan_rows = self._plan_generator.generate()
        total = len(plan_rows)

        # Set expected total for justification agent progress tracking
        # Each record generates 2 justifications (pro and anti)
        if self._justification_agent is not None:
            self._justification_agent.set_expected_total(total * 2)

        # Layer 2: Synthesize contexts
        contexts = self._context_synthesizer.synthesize_batch(plan_rows)

        # Layer 3: Apply variations
        contexts = self._variation_applicator.apply_batch(contexts)

        pro_records: List[Record] = []
        anti_records: List[Record] = []

        for i, context in enumerate(contexts):
            # Report progress
            if progress_callback:
                progress_callback(i + 1, total)

            # Configure plugin for this family (idempotent)
            self._configure_plugin(context.family_id)

            # Layer 4: Render prompt via family plugin
            plugin = get_family_plugin(context.family_id)
            rendered: RenderedPrompt = plugin.render_prompt(context)

            # Check if this record should be included based on split mode
            if not self._should_include_record(rendered.is_holdout, split):
                continue

            # Copy template metadata to context before packaging
            context = replace(
                context,
                template_id=rendered.template_id,
                is_holdout=rendered.is_holdout
            )

            # Layer 4.5: Grammar Linting (content only, not tag)
            lint_result = self._linter.check(
                content=rendered.content,  # NOT rendered.prompt or rendered.tag
                context=context,
                rendered=rendered,
            )
            if lint_result.has_blocking_errors and self._linter.mode == LintMode.ENABLED:
                raise GrammarError(lint_result)

            # Layer 5: Generate pro/anti response pair
            # Pass rendered content for agent-based justification generation
            pro_response, anti_response = self._answer_policy.generate_pair(
                context, rendered_content=rendered.content
            )

            # Layer 6: Package into records
            pro_record, anti_record = self._record_packager.package_pair(
                context, rendered.prompt, pro_response, anti_response
            )

            pro_records.append(pro_record)
            anti_records.append(anti_record)

        return pro_records, anti_records

    def generate_and_save(
        self,
        output_dir: str,
        split: SplitMode = "all",
        validate: bool = True
    ) -> Tuple[int, int]:
        """
        Generate and save to JSONL files.

        Args:
            output_dir: Directory to save output files
            split: Which split to generate
            validate: Whether to run validation before saving

        Output files (depending on split):
        - split="all": pro.jsonl, anti.jsonl
        - split="train": pro_train.jsonl, anti_train.jsonl
        - split="eval": pro_eval.jsonl, anti_eval.jsonl

        Returns:
            (pro_count, anti_count) tuple

        Raises:
            ValueError: If validation fails and validate=True
        """
        # Generate records
        pro_records, anti_records = self.generate(split)

        # Layer 7: Validate if requested
        if validate:
            errors = validate_dataset(pro_records, anti_records)
            if errors:
                raise ValueError(
                    f"Validation failed with {len(errors)} error(s):\n" +
                    "\n".join(f"  - {e}" for e in errors[:10]) +
                    (f"\n  ... and {len(errors) - 10} more" if len(errors) > 10 else "")
                )

        # Create output directory if needed
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Determine filenames based on split
        if split == "all":
            pro_filename = "pro.jsonl"
            anti_filename = "anti.jsonl"
        elif split == "train":
            pro_filename = "pro_train.jsonl"
            anti_filename = "anti_train.jsonl"
        elif split == "eval":
            pro_filename = "pro_eval.jsonl"
            anti_filename = "anti_eval.jsonl"
        else:
            raise ValueError(f"Invalid split mode: {split}")

        # Write files
        write_jsonl(pro_records, str(output_path / pro_filename))
        write_jsonl(anti_records, str(output_path / anti_filename))

        return len(pro_records), len(anti_records)

    def generate_all_splits(
        self,
        output_dir: str,
        validate: bool = True
    ) -> dict:
        """
        Generate train and eval splits in a single pass.

        More efficient than calling generate_and_save twice since
        it only runs the pipeline once.

        Returns:
            {"train": (pro_count, anti_count), "eval": (pro_count, anti_count)}
        """
        # Generate all records in one pass
        pro_records, anti_records = self.generate(split="all")

        # Split into train and eval
        pro_train = [r for r in pro_records if not r.meta.get("is_holdout", False)]
        anti_train = [r for r in anti_records if not r.meta.get("is_holdout", False)]
        pro_eval = [r for r in pro_records if r.meta.get("is_holdout", False)]
        anti_eval = [r for r in anti_records if r.meta.get("is_holdout", False)]

        # Layer 7: Validate if requested
        if validate:
            # Validate train split
            train_errors = validate_dataset(pro_train, anti_train)
            if train_errors:
                raise ValueError(
                    f"Train split validation failed with {len(train_errors)} error(s):\n" +
                    "\n".join(f"  - {e}" for e in train_errors[:10]) +
                    (f"\n  ... and {len(train_errors) - 10} more" if len(train_errors) > 10 else "")
                )

            # Validate eval split
            eval_errors = validate_dataset(pro_eval, anti_eval)
            if eval_errors:
                raise ValueError(
                    f"Eval split validation failed with {len(eval_errors)} error(s):\n" +
                    "\n".join(f"  - {e}" for e in eval_errors[:10]) +
                    (f"\n  ... and {len(eval_errors) - 10} more" if len(eval_errors) > 10 else "")
                )

        # Create output directory if needed
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Write train files
        write_jsonl(pro_train, str(output_path / "pro_train.jsonl"))
        write_jsonl(anti_train, str(output_path / "anti_train.jsonl"))

        # Write eval files
        write_jsonl(pro_eval, str(output_path / "pro_eval.jsonl"))
        write_jsonl(anti_eval, str(output_path / "anti_eval.jsonl"))

        return {
            "train": (len(pro_train), len(anti_train)),
            "eval": (len(pro_eval), len(anti_eval))
        }

    def get_lint_report(self) -> LintReport:
        """Get the accumulated lint report from generation."""
        return self._linter.get_report()

    def get_justification_report(self) -> Optional[ValidationReport]:
        """Get the justification validation report if agent was enabled."""
        return self._justification_report

    def close(self) -> None:
        """Clean up resources (e.g., LanguageTool instance, justification cache)."""
        self._linter.close()

        # Finalize justification agent if present
        if self._justification_agent is not None:
            self._justification_agent.finalize()
