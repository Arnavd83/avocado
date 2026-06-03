"""
Planning Module for Corrigibility Dataset Generation Pipeline.

Layer 1 of the pipeline. Produces ``PlanRow`` objects for downstream processing.

The V1 rewrite samples each field per pair via weighted-random draws (no exact
largest-remainder quota satisfaction). Mode, family, severity and perspective are
sampled against top-level allocation dicts; ``style_directive_id`` and
``target_intensity`` are sampled uniformly. One ``PlanRow`` is emitted per pair —
the pro/anti split happens later at Layer 5, so the two sides share an identical
``PlanRow``.

Task ID: T3
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import random
import hashlib
import logging
import yaml
from pathlib import Path

from .schema import PlanRow, FamilyID, Severity, Mode, Perspective
from .catalogs import FAMILY_SUBTYPES

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ALLOCATION CONFIG
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class AllocationConfig:
    """
    Allocation knobs for plan generation. Top-level, not per-family.

    All four allocation dicts must sum to 1.0 (validated in ``__post_init__``,
    which raises ``ValueError`` on a bad sum).

    Locked decisions encoded here (see
    ``specs/corrigibility_rewrite_implementation_plan.md``):
      - Mode: SHORT 85% / CHOICE 15% (RATING dropped).
      - Family: Family C dropped; its 10% share redistributed proportionally
        across A/B/D/E/F/G/H.
      - Perspective: 80% FIRST / 20% THIRD.
      - Style directive pool size: 10.
      - Over-generation buffer: 15% (applied by the caller, not the planner).

    ``total_size``, ``holdout_ratio`` and ``holdout_seed`` are retained for the
    downstream orchestrator (``render.py``) and CLI; they are orthogonal to the
    allocation rewrite.
    """

    total_size: int = 6000

    # Mode allocation (locked: SHORT 85% / CHOICE 15%; no RATING)
    mode_allocation: Dict[Mode, float] = field(default_factory=lambda: {
        Mode.SHORT: 0.85,
        Mode.CHOICE: 0.15,
    })

    # Family allocation (Family C dropped; its 10% share redistributed
    # proportionally across A/B/D/E/F/G/H — see module note below for the math).
    family_allocation: Dict[FamilyID, float] = field(default_factory=lambda: {
        FamilyID.A: 0.222,   # 0.20 + 0.10 * (0.20 / 0.90)
        FamilyID.B: 0.167,   # 0.15 + 0.10 * (0.15 / 0.90)
        FamilyID.D: 0.167,   # 0.15 + 0.10 * (0.15 / 0.90)
        FamilyID.E: 0.111,   # 0.10 + 0.10 * (0.10 / 0.90)
        FamilyID.F: 0.111,
        FamilyID.G: 0.111,
        FamilyID.H: 0.111,
    })

    # Perspective allocation (locked: 80% FIRST / 20% THIRD)
    perspective_allocation: Dict[Perspective, float] = field(default_factory=lambda: {
        Perspective.FIRST: 0.80,
        Perspective.THIRD: 0.20,
    })

    # Severity allocation (spec default; note the shipped default.yaml preserves
    # the prior 0.34/0.33/0.33 split — the two intentionally diverge).
    severity_allocation: Dict[Severity, float] = field(default_factory=lambda: {
        Severity.S1: 0.40,
        Severity.S2: 0.35,
        Severity.S3: 0.25,
    })

    # New top-level fields
    style_directive_pool_size: int = 10        # locked at 10 per style_directives_spec
    over_generation_buffer: float = 0.15       # 15% over-generation for skip budget

    # Holdout configuration (consumed by render.py family-plugin holdout selection;
    # orthogonal to allocation, retained from the prior config).
    holdout_ratio: float = 0.15
    holdout_seed: int = 99999

    def __post_init__(self):
        """Validate that each allocation dict sums to 1.0 (± 1e-6); raise if not."""
        for name, dist in (
            ("family_allocation", self.family_allocation),
            ("severity_allocation", self.severity_allocation),
            ("mode_allocation", self.mode_allocation),
            ("perspective_allocation", self.perspective_allocation),
        ):
            total = sum(dist.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    f"{name} sums to {total}, expected 1.0 (± 1e-6)"
                )
        if self.style_directive_pool_size <= 0:
            raise ValueError(
                f"style_directive_pool_size must be positive, "
                f"got {self.style_directive_pool_size}"
            )
        if not 0.0 <= self.over_generation_buffer <= 1.0:
            raise ValueError(
                f"over_generation_buffer must be in [0, 1], "
                f"got {self.over_generation_buffer}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# PLAN GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════


class PlanGenerator:
    """
    Generates a plan by independent weighted-random sampling per pair.

    Each pair is sampled with a deterministic RNG derived from a master seed plus
    the pair index, so plans are reproducible. The over-generation buffer is NOT
    applied here — the caller decides how many pairs to request.

    Attributes:
        config: AllocationConfig specifying the sampling distributions.
        global_seed: Default master seed used when ``generate_plan`` is called
            without an explicit ``seed``.
    """

    def __init__(self, config: AllocationConfig, global_seed: int):
        self.config = config
        self.global_seed = global_seed

    def generate_plan(self, n_pairs: int, seed: Optional[int] = None) -> List[PlanRow]:
        """
        Produce exactly ``n_pairs`` PlanRows by per-pair weighted-random sampling.

        Args:
            n_pairs: Number of PlanRows to generate (one per pair; the pro/anti
                split happens at Layer 5, so this is NOT doubled here).
            seed: Optional master seed override. Defaults to ``self.global_seed``.

        Returns:
            List of ``n_pairs`` PlanRow instances in index order.
        """
        master_seed = seed if seed is not None else self.global_seed
        logger.info(
            "Generating plan for %d pairs (master_seed=%d)", n_pairs, master_seed
        )

        rows: List[PlanRow] = []
        for i in range(n_pairs):
            rng = self._pair_rng(master_seed, i)

            pair_id = f"pair_{i:06d}"
            pair_seed = self._derive_seed(master_seed, pair_id)

            # Shared-per-pair fields (sampled once; both pro/anti rows would reuse).
            family_id = self._weighted_choice(self.config.family_allocation, rng)
            subtype_id = rng.choice(FAMILY_SUBTYPES[family_id])
            severity = self._weighted_choice(self.config.severity_allocation, rng)
            mode = self._weighted_choice(self.config.mode_allocation, rng)
            perspective = self._weighted_choice(self.config.perspective_allocation, rng)

            # style_directive_id: uniform int in [0, pool_size) → [0, 10).
            style_directive_id = rng.randrange(self.config.style_directive_pool_size)
            # target_intensity: uniform int in [1, 8) → 1-7 inclusive (7=strongest).
            target_intensity = rng.randint(1, 7)

            rows.append(PlanRow(
                pair_id=pair_id,
                seed=pair_seed,
                family_id=family_id,
                subtype_id=subtype_id,
                severity=severity,
                mode=mode,
                perspective=perspective,
                style_directive_id=style_directive_id,
                target_intensity=target_intensity,
            ))

        logger.info("Generated %d plan rows", len(rows))
        return rows

    @staticmethod
    def _weighted_choice(dist: Dict[Any, float], rng: random.Random) -> Any:
        """Weighted-random pick from an allocation dict.

        Keys are sorted by name so the draw is independent of dict insertion
        order (robust determinism for YAML-loaded configs).
        """
        keys = sorted(dist.keys(), key=lambda k: getattr(k, "name", str(k)))
        weights = [dist[k] for k in keys]
        return rng.choices(keys, weights=weights, k=1)[0]

    @staticmethod
    def _pair_rng(master_seed: int, index: int) -> random.Random:
        """Deterministic per-pair RNG seeded from (master_seed, index)."""
        combined = f"{master_seed}:pair:{index}".encode()
        h = int.from_bytes(hashlib.sha256(combined).digest()[:8], byteorder="big")
        return random.Random(h)

    @staticmethod
    def _derive_seed(master_seed: int, pair_id: str) -> int:
        """Derive a deterministic, well-distributed per-pair seed (0 ≤ seed < 2**31)."""
        combined = f"{master_seed}:{pair_id}".encode()
        hash_bytes = hashlib.sha256(combined).digest()
        return int.from_bytes(hash_bytes[:8], byteorder="big") % (2**31)


# ═══════════════════════════════════════════════════════════════════════════════
# YAML LOADING
# ═══════════════════════════════════════════════════════════════════════════════


def load_allocation_config(path: str) -> AllocationConfig:
    """
    Load an AllocationConfig from a YAML file matching the V1 schema.

    Recognized top-level keys: ``generation.total_size``, ``family_allocation``,
    ``severity_allocation``, ``mode_allocation`` (top-level short/choice, NOT
    per-family), ``perspective_allocation``, ``style_directive_pool_size``,
    ``over_generation_buffer``, and ``holdout.ratio`` / ``holdout.seed``.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If any loaded allocation dict doesn't sum to 1.0 (via
            AllocationConfig.__post_init__).
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path, "r") as f:
        data = yaml.safe_load(f) or {}

    kwargs: Dict[str, Any] = {
        "total_size": data.get("generation", {}).get("total_size", 6000),
    }

    family_allocation = {
        FamilyID[k]: v
        for k, v in data.get("family_allocation", {}).items()
        if k in FamilyID.__members__
    }
    if family_allocation:
        kwargs["family_allocation"] = family_allocation

    severity_allocation = {
        Severity[k]: v
        for k, v in data.get("severity_allocation", {}).items()
        if k in Severity.__members__
    }
    if severity_allocation:
        kwargs["severity_allocation"] = severity_allocation

    mode_allocation = {
        Mode[k.upper()]: v
        for k, v in data.get("mode_allocation", {}).items()
        if k.upper() in Mode.__members__
    }
    if mode_allocation:
        kwargs["mode_allocation"] = mode_allocation

    perspective_allocation = {
        Perspective[k.upper()]: v
        for k, v in data.get("perspective_allocation", {}).items()
        if k.upper() in Perspective.__members__
    }
    if perspective_allocation:
        kwargs["perspective_allocation"] = perspective_allocation

    if "style_directive_pool_size" in data:
        kwargs["style_directive_pool_size"] = data["style_directive_pool_size"]
    if "over_generation_buffer" in data:
        kwargs["over_generation_buffer"] = data["over_generation_buffer"]

    holdout = data.get("holdout", {})
    if "ratio" in holdout:
        kwargs["holdout_ratio"] = holdout["ratio"]
    if "seed" in holdout:
        kwargs["holdout_seed"] = holdout["seed"]

    return AllocationConfig(**kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════


def validate_plan(plan: List[PlanRow], config: AllocationConfig) -> List[str]:
    """
    Validate a generated plan against the allocation config.

    Returns a flat list of issue messages (empty == clean). Distribution checks
    are soft (±5pp tolerance — a healthy random sample stays well within); the
    pair-level and coverage checks are structural.

    Checks:
      - Each ``pair_id`` appears exactly once (one PlanRow per pair).
      - Each ``subtype_id`` is valid for its family.
      - Family / severity / mode / perspective fractions within ±5pp of config.
      - ``style_directive_id``: every index 0..pool_size-1 appears at least once
        (coverage), and no single directive exceeds 15% (rough balance guard).
      - ``target_intensity``: every integer 1..7 appears at least once (coverage).
    """
    issues: List[str] = []

    n = len(plan)
    if n == 0:
        issues.append("Plan is empty")
        return issues

    # Pair-level invariant: each pair_id appears exactly once.
    seen: Dict[str, int] = {}
    for row in plan:
        seen[row.pair_id] = seen.get(row.pair_id, 0) + 1
    duplicates = {pid: c for pid, c in seen.items() if c > 1}
    if duplicates:
        issues.append(
            f"Duplicate pair_ids found (each must appear exactly once): "
            f"{sorted(duplicates)[:5]}"
        )

    # Subtype must be valid for its family.
    for row in plan:
        valid = FAMILY_SUBTYPES.get(row.family_id, [])
        if row.subtype_id not in valid:
            issues.append(
                f"Row {row.pair_id}: subtype '{row.subtype_id}' not valid for "
                f"family {row.family_id.name}"
            )

    tol = 0.05  # ±5pp distribution tolerance (warning-level)

    def _check_distribution(label: str, getter, expected: Dict[Any, float]) -> None:
        counts: Dict[Any, int] = {}
        for row in plan:
            key = getter(row)
            counts[key] = counts.get(key, 0) + 1
        for key, exp in expected.items():
            observed = counts.get(key, 0) / n
            if abs(observed - exp) > tol:
                issues.append(
                    f"{label} distribution off: {getattr(key, 'name', key)} "
                    f"observed {observed:.3f}, expected {exp:.3f} (±{tol:.2f})"
                )

    _check_distribution("Family", lambda r: r.family_id, config.family_allocation)
    _check_distribution("Severity", lambda r: r.severity, config.severity_allocation)
    _check_distribution("Mode", lambda r: r.mode, config.mode_allocation)
    _check_distribution(
        "Perspective", lambda r: r.perspective, config.perspective_allocation
    )

    # style_directive_id coverage + balance.
    directive_counts: Dict[int, int] = {}
    for row in plan:
        directive_counts[row.style_directive_id] = (
            directive_counts.get(row.style_directive_id, 0) + 1
        )
    for idx in range(config.style_directive_pool_size):
        if directive_counts.get(idx, 0) == 0:
            issues.append(f"style_directive_id {idx} never appears (coverage gap)")
    for idx, count in directive_counts.items():
        if count / n > 0.15:
            issues.append(
                f"style_directive_id {idx} dominates: {count / n:.3f} of plan "
                f"(rough balance guard is 0.15)"
            )

    # target_intensity coverage (1..7).
    intensity_present = {row.target_intensity for row in plan}
    for value in range(1, 8):
        if value not in intensity_present:
            issues.append(f"target_intensity {value} never appears (coverage gap)")

    return issues
