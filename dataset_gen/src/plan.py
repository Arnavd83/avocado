"""
Planning Module for Corrigibility Dataset Generation Pipeline.

This module generates a deterministic plan that satisfies all allocation quotas.
It is Layer 1 of the pipeline and produces PlanRow objects for downstream processing.

Task ID: T3
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import random
import hashlib
import logging
import yaml
from pathlib import Path

from .schema import PlanRow, FamilyID, Severity, Mode, Perspective

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Subtypes for each family (3 subtypes per family as defined in spec)
FAMILY_SUBTYPES: Dict[FamilyID, List[str]] = {
    FamilyID.A: ["A1", "A2", "A3"],  # Acceptability rating, Better/worse/neutral, Severity emphasis
    FamilyID.B: ["B1", "B2", "B3"],  # Stable vs different, Trajectory, Identity-continuity
    FamilyID.C: ["C1", "C2", "C3"],  # Assistant v1 vs v2, System deployment, Stakeholder-neutral
    FamilyID.D: ["D1", "D2", "D3"],  # Design principle, Policy parameter, Engineering constraints
    FamilyID.E: ["E1", "E2", "E3"],  # Endorse/regret/neutral, Regret likelihood, Reflective equilibrium
    FamilyID.F: ["F1", "F2", "F3"],  # Tradeoff acceptability, Sacrifice framing, Context-dependent
    FamilyID.G: ["G1", "G2", "G3"],  # 90/10 reversal, Gradual drift, Magnitude as severity
    FamilyID.H: ["H1", "H2", "H3"],  # Uncertainty resolves, More opinionated, Confidence in priorities
}


# ═══════════════════════════════════════════════════════════════════════════════
# ALLOCATION CONFIG
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class AllocationConfig:
    """
    Configuration for dataset allocation.

    Defines the distribution of examples across families, severities, modes,
    and perspectives. All allocations must sum to 1.0.

    Attributes:
        total_size: Total number of datapoints to generate
        family_allocation: Distribution across families A-H
        severity_allocation: Distribution across severity levels S1-S3
        mode_allocation: Per-family distribution across modes (rating/choice/short)
        perspective_allocation: Distribution across perspectives (first/third)
    """

    total_size: int = 6000

    # Family allocation (must sum to 1.0)
    family_allocation: Dict[FamilyID, float] = field(default_factory=dict)

    # Severity allocation (must sum to 1.0)
    severity_allocation: Dict[Severity, float] = field(default_factory=dict)

    # Mode allocation per family
    mode_allocation: Dict[FamilyID, Dict[Mode, float]] = field(default_factory=dict)

    # Perspective allocation
    perspective_allocation: Dict[Perspective, float] = field(default_factory=dict)

    # Holdout configuration (for template holdout feature)
    # These are passed to family plugins for holdout template selection
    holdout_ratio: float = 0.15      # 15% of templates held out for evaluation
    holdout_seed: int = 99999        # Separate seed for reproducible holdout selection
                                     # (independent from global_seed)

    def __post_init__(self):
        """Set defaults from spec if not provided."""
        # Default family allocation
        if not self.family_allocation:
            self.family_allocation = {
                FamilyID.A: 0.20,  # 1200
                FamilyID.B: 0.15,  # 900
                FamilyID.C: 0.10,  # 600
                FamilyID.D: 0.15,  # 900
                FamilyID.E: 0.10,  # 600
                FamilyID.F: 0.10,  # 600
                FamilyID.G: 0.10,  # 600
                FamilyID.H: 0.10,  # 600
            }

        # Default severity allocation
        if not self.severity_allocation:
            self.severity_allocation = {
                Severity.S1: 0.34,
                Severity.S2: 0.33,
                Severity.S3: 0.33,
            }

        # Default mode allocation per family
        if not self.mode_allocation:
            self.mode_allocation = {
                # Families A, E, G, H: mostly rating-framed (90/5/5)
                FamilyID.A: {Mode.RATING: 0.90, Mode.CHOICE: 0.05, Mode.SHORT: 0.05},
                FamilyID.E: {Mode.RATING: 0.90, Mode.CHOICE: 0.05, Mode.SHORT: 0.05},
                FamilyID.G: {Mode.RATING: 0.90, Mode.CHOICE: 0.05, Mode.SHORT: 0.05},
                FamilyID.H: {Mode.RATING: 0.90, Mode.CHOICE: 0.05, Mode.SHORT: 0.05},
                # Families B, D: mostly choice-framed (40/60/0)
                FamilyID.B: {Mode.RATING: 0.40, Mode.CHOICE: 0.60, Mode.SHORT: 0.00},
                FamilyID.D: {Mode.RATING: 0.40, Mode.CHOICE: 0.60, Mode.SHORT: 0.00},
                # Families C, F: balanced (70/20/10)
                FamilyID.C: {Mode.RATING: 0.70, Mode.CHOICE: 0.20, Mode.SHORT: 0.10},
                FamilyID.F: {Mode.RATING: 0.70, Mode.CHOICE: 0.20, Mode.SHORT: 0.10},
            }

        # Default perspective allocation
        if not self.perspective_allocation:
            self.perspective_allocation = {
                Perspective.FIRST: 0.65,
                Perspective.THIRD: 0.35,
            }

    def validate(self) -> List[str]:
        """
        Validate that all allocations sum to 1.0.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate family allocation
        family_sum = sum(self.family_allocation.values())
        if abs(family_sum - 1.0) > 0.001:
            errors.append(f"Family allocation sums to {family_sum}, expected 1.0")

        # Validate severity allocation
        severity_sum = sum(self.severity_allocation.values())
        if abs(severity_sum - 1.0) > 0.001:
            errors.append(f"Severity allocation sums to {severity_sum}, expected 1.0")

        # Validate mode allocation for each family
        for family_id, modes in self.mode_allocation.items():
            mode_sum = sum(modes.values())
            if abs(mode_sum - 1.0) > 0.001:
                errors.append(
                    f"Mode allocation for {family_id.name} sums to {mode_sum}, expected 1.0"
                )

        # Validate perspective allocation
        perspective_sum = sum(self.perspective_allocation.values())
        if abs(perspective_sum - 1.0) > 0.001:
            errors.append(f"Perspective allocation sums to {perspective_sum}, expected 1.0")

        return errors


# ═══════════════════════════════════════════════════════════════════════════════
# LARGEST REMAINDER METHOD
# ═══════════════════════════════════════════════════════════════════════════════


def largest_remainder_allocation(
    total: int, proportions: Dict[Any, float]
) -> Dict[Any, int]:
    """
    Allocate counts using the largest-remainder method for fair rounding.

    This method ensures that:
    1. All counts sum exactly to total
    2. Each count is as close as possible to its proportional share
    3. The rounding is deterministic and fair

    Args:
        total: Total count to allocate
        proportions: Dictionary mapping keys to their proportional shares (must sum to 1.0)

    Returns:
        Dictionary mapping keys to their allocated counts

    Example:
        >>> proportions = {"A": 0.33, "B": 0.33, "C": 0.34}
        >>> counts = largest_remainder_allocation(100, proportions)
        >>> sum(counts.values())
        100
    """
    # Calculate exact (fractional) allocations
    exact = {key: total * prop for key, prop in proportions.items()}

    # Take the floor of each allocation
    floored = {key: int(val) for key, val in exact.items()}

    # Calculate remainders
    remainders = {key: exact[key] - floored[key] for key in exact}

    # Calculate how many more we need to allocate
    allocated = sum(floored.values())
    remaining = total - allocated

    # Sort by remainder descending, with deterministic tie-breaking by key
    sorted_keys = sorted(
        remainders.keys(),
        key=lambda k: (-remainders[k], str(k)),
    )

    # Allocate remaining to those with largest remainders
    result = dict(floored)
    for i, key in enumerate(sorted_keys):
        if i < remaining:
            result[key] += 1

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# PLAN GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════


class PlanGenerator:
    """
    Generates deterministic allocation plans.

    Guarantees exact quota satisfaction using the largest-remainder method
    for fair rounding at each level of the allocation hierarchy.

    Attributes:
        config: AllocationConfig specifying the distribution
        global_seed: Seed for deterministic random operations
        rng: Random number generator instance
    """

    def __init__(self, config: AllocationConfig, global_seed: int):
        """
        Initialize the plan generator.

        Args:
            config: Allocation configuration
            global_seed: Seed for reproducibility
        """
        self.config = config
        self.global_seed = global_seed
        self.rng = random.Random(global_seed)

    def generate(self) -> List[PlanRow]:
        """
        Generate the complete plan.

        Returns:
            List of PlanRows in deterministic order

        The generation process:
        1. Allocate counts to families
        2. Within each family, allocate to severities
        3. Within each family, allocate to modes
        4. Within each family, allocate to perspectives
        5. Within each family, allocate to subtypes (uniform)
        6. Create PlanRows with deterministic seeds
        """
        logger.info(f"Generating plan for {self.config.total_size} datapoints with seed {self.global_seed}")

        plan_rows: List[PlanRow] = []
        index = 0

        # Step 1: Allocate counts to families
        family_counts = largest_remainder_allocation(
            self.config.total_size, self.config.family_allocation
        )

        logger.debug(f"Family counts: {family_counts}")

        # Process each family
        for family_id in FamilyID:
            family_count = family_counts[family_id]
            if family_count == 0:
                continue

            logger.debug(f"Processing family {family_id.name} with {family_count} rows")

            # Step 2: Allocate to severities within this family
            severity_counts = largest_remainder_allocation(
                family_count, self.config.severity_allocation
            )

            # Step 3: Allocate to modes within this family
            mode_counts = largest_remainder_allocation(
                family_count, self.config.mode_allocation[family_id]
            )

            # Step 4: Allocate to perspectives within this family
            perspective_counts = largest_remainder_allocation(
                family_count, self.config.perspective_allocation
            )

            # Step 5: Allocate to subtypes (uniform distribution)
            subtypes = FAMILY_SUBTYPES[family_id]
            subtype_proportions = {st: 1.0 / len(subtypes) for st in subtypes}
            subtype_counts = largest_remainder_allocation(family_count, subtype_proportions)

            # Build the slot list for this family by expanding counts
            family_slots = self._build_family_slots(
                family_id,
                family_count,
                severity_counts,
                mode_counts,
                perspective_counts,
                subtype_counts,
            )

            # Shuffle deterministically within family
            family_rng = random.Random(self.global_seed + hash(family_id))
            family_rng.shuffle(family_slots)

            # Create PlanRows
            for slot in family_slots:
                pair_id = self._generate_pair_id(index)
                seed = self._derive_seed(pair_id)

                plan_row = PlanRow(
                    pair_id=pair_id,
                    seed=seed,
                    family_id=family_id,
                    subtype_id=slot["subtype"],
                    severity=slot["severity"],
                    mode=slot["mode"],
                    perspective=slot["perspective"],
                )
                plan_rows.append(plan_row)
                index += 1

        logger.info(f"Generated {len(plan_rows)} plan rows")
        return plan_rows

    def _build_family_slots(
        self,
        family_id: FamilyID,
        total_count: int,
        severity_counts: Dict[Severity, int],
        mode_counts: Dict[Mode, int],
        perspective_counts: Dict[Perspective, int],
        subtype_counts: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        """
        Build the slot list for a family with proper attribute assignment.

        Uses an interleaving approach to ensure each attribute combination
        gets a fair distribution while respecting individual quotas.

        Args:
            family_id: The family being processed
            total_count: Total slots for this family
            severity_counts: Target counts for each severity
            mode_counts: Target counts for each mode
            perspective_counts: Target counts for each perspective
            subtype_counts: Target counts for each subtype

        Returns:
            List of slot dictionaries with severity, mode, perspective, subtype
        """
        # Expand each attribute into a list of required values
        severity_list = self._expand_counts(severity_counts)
        mode_list = self._expand_counts(mode_counts)
        perspective_list = self._expand_counts(perspective_counts)
        subtype_list = self._expand_counts(subtype_counts)

        # Shuffle each list deterministically to avoid correlation
        rng = random.Random(self.global_seed + hash(family_id) + 1)
        rng.shuffle(severity_list)
        rng = random.Random(self.global_seed + hash(family_id) + 2)
        rng.shuffle(mode_list)
        rng = random.Random(self.global_seed + hash(family_id) + 3)
        rng.shuffle(perspective_list)
        rng = random.Random(self.global_seed + hash(family_id) + 4)
        rng.shuffle(subtype_list)

        # Combine into slots
        slots = []
        for i in range(total_count):
            slots.append({
                "severity": severity_list[i],
                "mode": mode_list[i],
                "perspective": perspective_list[i],
                "subtype": subtype_list[i],
            })

        return slots

    def _expand_counts(self, counts: Dict[Any, int]) -> List[Any]:
        """
        Expand a count dictionary into a list of values.

        Args:
            counts: Dictionary mapping values to their counts

        Returns:
            List with each value repeated according to its count
        """
        result = []
        # Sort keys for determinism
        for key in sorted(counts.keys(), key=str):
            result.extend([key] * counts[key])
        return result

    def _generate_pair_id(self, index: int) -> str:
        """
        Generate unique pair_id from index.

        Args:
            index: Zero-based index of the datapoint

        Returns:
            Formatted pair ID string
        """
        return f"pair_{index:06d}"

    def _derive_seed(self, pair_id: str) -> int:
        """
        Derive deterministic seed from pair_id and global seed.

        Uses SHA256 hash for uniform distribution of derived seeds.

        Args:
            pair_id: The unique pair identifier

        Returns:
            Deterministic integer seed
        """
        combined = f"{self.global_seed}:{pair_id}"
        hash_bytes = hashlib.sha256(combined.encode()).digest()
        # Use first 8 bytes for a 64-bit seed
        seed = int.from_bytes(hash_bytes[:8], byteorder="big")
        # Keep it within reasonable bounds
        return seed % (2**31)


# ═══════════════════════════════════════════════════════════════════════════════
# YAML LOADING
# ═══════════════════════════════════════════════════════════════════════════════


def load_allocation_config(path: str) -> AllocationConfig:
    """
    Load allocation config from YAML file.

    Args:
        path: Path to the YAML configuration file

    Returns:
        AllocationConfig with values from the file

    Raises:
        FileNotFoundError: If the config file doesn't exist
        yaml.YAMLError: If the file is not valid YAML
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    # Parse total size
    total_size = data.get("generation", {}).get("total_size", 6000)

    # Parse family allocation
    family_allocation = {}
    if "family_allocation" in data:
        for key, value in data["family_allocation"].items():
            family_id = FamilyID[key] if key in FamilyID.__members__ else None
            if family_id:
                family_allocation[family_id] = value

    # Parse severity allocation
    severity_allocation = {}
    if "severity_allocation" in data:
        for key, value in data["severity_allocation"].items():
            severity = Severity[key] if key in Severity.__members__ else None
            if severity:
                severity_allocation[severity] = value

    # Parse mode allocation
    mode_allocation = {}
    if "mode_allocation" in data and "per_family" in data["mode_allocation"]:
        for family_key, modes in data["mode_allocation"]["per_family"].items():
            family_id = FamilyID[family_key] if family_key in FamilyID.__members__ else None
            if family_id:
                mode_allocation[family_id] = {}
                for mode_key, value in modes.items():
                    mode = Mode[mode_key.upper()] if mode_key.upper() in Mode.__members__ else None
                    if mode:
                        mode_allocation[family_id][mode] = value

    # Parse perspective allocation
    perspective_allocation = {}
    if "perspective_allocation" in data:
        for key, value in data["perspective_allocation"].items():
            perspective = Perspective[key.upper()] if key.upper() in Perspective.__members__ else None
            if perspective:
                perspective_allocation[perspective] = value

    return AllocationConfig(
        total_size=total_size,
        family_allocation=family_allocation if family_allocation else None,
        severity_allocation=severity_allocation if severity_allocation else None,
        mode_allocation=mode_allocation if mode_allocation else None,
        perspective_allocation=perspective_allocation if perspective_allocation else None,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════


def validate_plan(plan: List[PlanRow], config: AllocationConfig) -> List[str]:
    """
    Validate that plan satisfies all allocation requirements.

    Checks:
    1. Total count matches config
    2. Family distribution matches allocation
    3. Severity distribution matches allocation (overall)
    4. Mode distribution matches allocation (per family)
    5. Perspective distribution matches allocation (overall)
    6. All pair_ids are unique

    Args:
        plan: List of PlanRows to validate
        config: AllocationConfig that defines expected distributions

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Check total count
    if len(plan) != config.total_size:
        errors.append(
            f"Plan has {len(plan)} rows, expected {config.total_size}"
        )

    # Check pair_id uniqueness
    pair_ids = [row.pair_id for row in plan]
    if len(pair_ids) != len(set(pair_ids)):
        errors.append("Duplicate pair_ids found in plan")

    # Calculate expected counts using largest-remainder
    expected_family_counts = largest_remainder_allocation(
        config.total_size, config.family_allocation
    )

    # Check family distribution
    actual_family_counts: Dict[FamilyID, int] = {f: 0 for f in FamilyID}
    for row in plan:
        actual_family_counts[row.family_id] += 1

    for family_id in FamilyID:
        expected = expected_family_counts[family_id]
        actual = actual_family_counts[family_id]
        if expected != actual:
            errors.append(
                f"Family {family_id.name}: expected {expected}, got {actual}"
            )

    # Check severity distribution per family
    for family_id in FamilyID:
        family_rows = [r for r in plan if r.family_id == family_id]
        family_count = len(family_rows)
        if family_count == 0:
            continue

        expected_severity_counts = largest_remainder_allocation(
            family_count, config.severity_allocation
        )
        actual_severity_counts: Dict[Severity, int] = {s: 0 for s in Severity}
        for row in family_rows:
            actual_severity_counts[row.severity] += 1

        for severity in Severity:
            expected = expected_severity_counts[severity]
            actual = actual_severity_counts[severity]
            if expected != actual:
                errors.append(
                    f"Family {family_id.name}, Severity {severity.name}: "
                    f"expected {expected}, got {actual}"
                )

    # Check mode distribution per family
    for family_id in FamilyID:
        family_rows = [r for r in plan if r.family_id == family_id]
        family_count = len(family_rows)
        if family_count == 0:
            continue

        expected_mode_counts = largest_remainder_allocation(
            family_count, config.mode_allocation[family_id]
        )
        actual_mode_counts: Dict[Mode, int] = {m: 0 for m in Mode}
        for row in family_rows:
            actual_mode_counts[row.mode] += 1

        for mode in Mode:
            expected = expected_mode_counts[mode]
            actual = actual_mode_counts[mode]
            if expected != actual:
                errors.append(
                    f"Family {family_id.name}, Mode {mode.name}: "
                    f"expected {expected}, got {actual}"
                )

    # Check perspective distribution per family
    for family_id in FamilyID:
        family_rows = [r for r in plan if r.family_id == family_id]
        family_count = len(family_rows)
        if family_count == 0:
            continue

        expected_perspective_counts = largest_remainder_allocation(
            family_count, config.perspective_allocation
        )
        actual_perspective_counts: Dict[Perspective, int] = {p: 0 for p in Perspective}
        for row in family_rows:
            actual_perspective_counts[row.perspective] += 1

        for perspective in Perspective:
            expected = expected_perspective_counts[perspective]
            actual = actual_perspective_counts[perspective]
            if expected != actual:
                errors.append(
                    f"Family {family_id.name}, Perspective {perspective.name}: "
                    f"expected {expected}, got {actual}"
                )

    return errors
