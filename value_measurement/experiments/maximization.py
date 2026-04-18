"""Wrapper for the maximization experiment.

Invokes the emergent-values utility_maximization script via subprocess and
returns populated dataclasses ready for DB insertion.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ..db import has_utilities
from ..records import (
    MaximizationAnswerUtilityRecord,
    MaximizationQuestionRecord,
    MaximizationSummary,
)
from ._base import (
    _UTILITY_ANALYSIS_DIR,
    load_experiment_output,
    resolve_experiment_script,
    run_subprocess,
)


async def run_maximization(
    model_key: str,
    db_conn: sqlite3.Connection,
    save_dir: Path | None = None,
    *,
    expname: str = "v1",
    questions_path: Path | None = None,
    compute_utilities_config_key: str = "thurstonian",
    save_suffix: str | None = None,
    matcher_model_key: str | None = None,
    seed: int = 42,
) -> tuple[
    MaximizationSummary,
    list[MaximizationQuestionRecord],
    list[MaximizationAnswerUtilityRecord],
]:
    """Run maximization experiment and return structured records.

    Args:
        model_key: Key in models.yaml.
        db_conn: Database connection with existing compute_utilities data.
        save_dir: Directory for raw experiment output files.
        expname: Experiment name sub-directory.
        questions_path: Path to questions JSON file.
        compute_utilities_config_key: Config key for compute_utilities.
        save_suffix: Custom suffix for saved files.
        matcher_model_key: Model key for answer matching. Defaults to same model.
        seed: Random seed for reproducibility.

    Returns:
        A 3-tuple of ``(MaximizationSummary, list[MaximizationQuestionRecord],
        list[MaximizationAnswerUtilityRecord])``.

    Raises:
        RuntimeError: If compute_utilities has not been run for this model.
    """
    # -- Gate --
    if not has_utilities(db_conn, model_key):
        raise RuntimeError(
            f"Cannot run maximization without computed utilities "
            f"for '{model_key}'"
        )

    # -- Resolve script and default paths --
    script = resolve_experiment_script(
        "experiments/maximization/utility_maximization.py"
    )

    if questions_path is None:
        questions_path = (
            _UTILITY_ANALYSIS_DIR
            / "experiments"
            / "maximization"
            / "util_max_questions.json"
        )

    if save_dir is not None:
        save_dir = Path(save_dir).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = (Path.cwd() / "results").resolve()
        save_dir.mkdir(parents=True, exist_ok=True)

    # -- Build CLI args --
    # Pass absolute path to compute_utilities.yaml so the subprocess can
    # find it regardless of its working directory.
    compute_utilities_config_path = str(
        _UTILITY_ANALYSIS_DIR / "compute_utilities" / "compute_utilities.yaml"
    )
    args = [
        "--model_key", model_key,
        "--save_dir", str(save_dir),
        "--expname", expname,
        "--questions_path", str(questions_path),
        "--compute_utilities_config_path", compute_utilities_config_path,
        "--compute_utilities_config_key", compute_utilities_config_key,
        "--seed", str(seed),
    ]
    if save_suffix:
        args.extend(["--save_suffix", save_suffix])
    if matcher_model_key:
        args.extend(["--matcher_model_key", matcher_model_key])

    # -- Run subprocess --
    run_subprocess(script, args)

    # -- Load outputs --
    exp_dir = save_dir / expname
    comparison_data = load_experiment_output(
        exp_dir, f"{model_key}_answer_comparison.json"
    )
    utility_data = load_experiment_output(
        exp_dir, f"{model_key}_utility_maximization_results.json"
    )

    # -- Extract comparison results --
    summary_stats = comparison_data["summary_stats"]
    detailed_results = comparison_data["detailed_results"]

    # -- Build question records --
    question_records: list[MaximizationQuestionRecord] = []
    for q_idx, result in enumerate(detailed_results):
        question_records.append(
            MaximizationQuestionRecord(
                model_key=model_key,
                question_id=q_idx,
                question_text=result["question"],
                direct_answer=result["direct_answer"],
                matched_answer=result["matched_answer"],
                highest_utility_answer=result["highest_utility_answer"],
                matched_highest=result["matched_highest_utility"],
                matched_top3=result["matched_top_3"],
                matched_top5=result["matched_top_5"],
            )
        )

    # -- Build answer utility records --
    answer_records: list[MaximizationAnswerUtilityRecord] = []
    for q_idx, (question_text, q_data) in enumerate(utility_data.items()):
        utility_results = q_data["utility_results"]
        options = utility_results["options"]
        utilities = utility_results["utilities"]

        for answer_id, opt in enumerate(options):
            opt_id = opt["id"]
            # Handle both int and str keys in utilities dict
            util = utilities.get(opt_id) or utilities.get(str(opt_id), {})
            # Cast numpy scalars to native Python floats for SQLite compatibility.
            if util:
                answer_records.append(
                    MaximizationAnswerUtilityRecord(
                        model_key=model_key,
                        question_id=q_idx,
                        answer_id=answer_id,
                        answer_text=opt["description"],
                        mean=float(util["mean"]),
                        variance=float(util["variance"]),
                    )
                )

    # -- Build summary --
    total_questions = len(detailed_results)
    summary = MaximizationSummary(
        model_key=model_key,
        match_highest_pct=float(summary_stats["matches_highest_utility_pct"]),
        match_top3_pct=float(summary_stats["matches_top_3_pct"]),
        match_top5_pct=float(summary_stats["matches_top_5_pct"]),
        total_questions=int(total_questions),
    )

    return summary, question_records, answer_records
