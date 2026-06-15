"""
Stage 1 — PLAN.

Produces a deterministic, reproducible ``List[PromptSpec]`` — one per pair —
fixing every controlled dimension before any agent is called. Zero agent calls,
zero nondeterminism: given the same ``GenerationConfig``, ``plan()`` returns a
byte-identical list of specs.

This is the experiment's control surface (design doc §4). Coverage guarantees,
matched-pair invariants, and holdout assignment all originate here.

Reuse from v1 (``dataset_gen/src/plan.py``):
- ``_pair_rng`` / ``_derive_seed`` — ported VERBATIM (proven sha256-based per-pair
  RNG and ``0 <= seed < 2**31`` derivation).
- ``_weighted_choice`` — ported verbatim (keys sorted by ``.name`` for
  order-independent determinism).
- ``validate_plan`` distribution/coverage structure — adapted to the v2
  dimensions (framing / question_shape / tone / severity).

NEW in v2 (no v1 analogue): holdout-first exclusion (§3.2) and minimal-touch
quota correction (§3.4).

See ``specs/generation_v2/stage_1_plan_spec.md`` for the authoritative spec.
"""

from __future__ import annotations

import dataclasses
import hashlib
import random
from typing import Any, Dict, List, Optional, Set

from .catalog import partition_holdout, sample_preference_pair
from .config import GenerationConfig
from .schema import PromptSpec

# ═══════════════════════════════════════════════════════════════════════════════
# SALTS
# Fixed module constants so holdout, per-pair sampling, and quota correction each
# use independent, named-salt-derived RNGs that don't interfere. HOLDOUT_SALT
# keeps holdout stable but independent of the per-pair seeds (§3.2); the
# correction RNG is seeded from master ^ CORRECTION_SALT (§3.4).
# ═══════════════════════════════════════════════════════════════════════════════

HOLDOUT_SALT = 0x484F4C44  # "HOLD"
CORRECTION_SALT = 0x434F5252  # "CORR"

# Quota-correction tolerances and cap (§3.4).
_CORRECTION_TOLERANCE = 0.02  # ±2pp target band for corrected dimensions.
_MAX_CORRECTION_PASSES = 3

# validate_plan distribution tolerance (§5, soft / warn).
_VALIDATE_TOLERANCE = 0.05  # ±5pp.


# ═══════════════════════════════════════════════════════════════════════════════
# SEED DERIVATION (ported verbatim from v1 plan.py)
# ═══════════════════════════════════════════════════════════════════════════════


def _pair_rng(master_seed: int, index: int) -> random.Random:
    """Deterministic per-pair RNG seeded from (master_seed, index).

    Ported VERBATIM from ``dataset_gen/src/plan.py`` — proven, well-distributed.
    """
    combined = f"{master_seed}:pair:{index}".encode()
    h = int.from_bytes(hashlib.sha256(combined).digest()[:8], byteorder="big")
    return random.Random(h)


def _derive_seed(master_seed: int, pair_id: str) -> int:
    """Derive a deterministic, well-distributed per-pair seed (0 <= seed < 2**31).

    Ported VERBATIM from ``dataset_gen/src/plan.py``.
    """
    combined = f"{master_seed}:{pair_id}".encode()
    hash_bytes = hashlib.sha256(combined).digest()
    return int.from_bytes(hash_bytes[:8], byteorder="big") % (2**31)


def _weighted_choice(dist: Dict[Any, float], rng: random.Random) -> Any:
    """Weighted-random pick from an allocation dict.

    Ported VERBATIM from ``dataset_gen/src/plan.py``. Keys are sorted by name so
    the draw is independent of dict insertion order (robust determinism for
    YAML-loaded configs / enum dicts).
    """
    keys = sorted(dist.keys(), key=lambda k: getattr(k, "name", str(k)))
    weights = [dist[k] for k in keys]
    return rng.choices(keys, weights=weights, k=1)[0]


# ═══════════════════════════════════════════════════════════════════════════════
# HOLDOUT
# ═══════════════════════════════════════════════════════════════════════════════


def holdout_keys(config: GenerationConfig) -> Set[str]:
    """Return the holdout preference-pair keys for this config.

    Exposed for the run report and Stage 5. Uses the fixed ``HOLDOUT_SALT`` so the
    partition is stable for a given ``global_seed`` and ``holdout_pair_fraction``
    yet independent of the per-pair sampling seeds (§3.2).
    """
    _train_keys, hold = partition_holdout(
        config.holdout_pair_fraction, seed=config.global_seed ^ HOLDOUT_SALT
    )
    return hold


# ═══════════════════════════════════════════════════════════════════════════════
# PLAN
# ═══════════════════════════════════════════════════════════════════════════════


# The independent surface dimensions quota correction may nudge, in canonical
# order. Each tuple is (label, allocation-attr, PromptSpec-field). These are the
# directly-stored, freely-flippable fields of PromptSpec.
#
# Severity is intentionally NOT here: it is not a stored field of PromptSpec (it
# lives on ``preference_pair.severity``), and the v2 taxonomy partitions
# categories by severity, so flipping a severity label would require re-drawing
# the preference pair — which would break matched-pair identity (§3.4 / §8). Thus
# severity is never corrected; its drift is surfaced by ``validate_plan`` only.
# The preference pair and seed are likewise never touched (§3.4).
_DIMENSIONS = (
    ("framing", "framing_allocation", "framing"),
    ("question_shape", "question_shape_allocation", "question_shape"),
    ("tone", "tone_allocation", "tone"),
    # reasoning_basis is a freely-flippable stored field too; quota correction is
    # a no-op for a pure arm (e.g. all-MERIT) and nudges blends toward target.
    ("reasoning_basis", "reasoning_basis_allocation", "reasoning_basis"),
)


def plan(config: GenerationConfig) -> List[PromptSpec]:
    """Build the deterministic plan: one ``PromptSpec`` per queued pair.

    Pure function of ``GenerationConfig`` (no clock, no global RNG, no env). The
    algorithm is spec §3:

    1. Partition holdout once (§3.2); pass its keys as ``exclude_keys`` to every
       ``sample_preference_pair`` call so no training spec uses a holdout pair.
    2. Per-pair loop (§3.3): derive a per-pair RNG and seed, then sample every
       controlled dimension independently from that RNG.
    3. Minimal-touch quota correction (§3.4): nudge only the independent surface
       dimensions back inside ±2pp of target, never the pair or seed.

    Re-running on the same config yields a byte-identical list (§4).
    """
    master = config.global_seed
    _train_keys, holdout = partition_holdout(
        config.holdout_pair_fraction, seed=master ^ HOLDOUT_SALT
    )

    specs: List[PromptSpec] = []
    for i in range(config.queued_pairs):
        rng = _pair_rng(master, i)
        pair_id = f"pair_{i:06d}"
        seed = _derive_seed(master, pair_id)

        # 2. severity (drives the pair draw).
        severity = _weighted_choice(config.severity_allocation, rng)

        # 3. preference pair + current side (holdout-excluded).
        pref_pair, current_pref = sample_preference_pair(
            severity, rng, exclude_keys=holdout
        )

        # 4-7. independent surface dimensions.
        framing = _weighted_choice(config.framing_allocation, rng)
        question_shape = _weighted_choice(config.question_shape_allocation, rng)
        tone = _weighted_choice(config.tone_allocation, rng)

        # 8. system_prompt_id via Bernoulli(rate); None otherwise.
        if rng.random() < config.system_prompt_rate:
            system_prompt_id: Optional[int] = rng.randrange(
                config.system_prompt_pool_size
            )
        else:
            system_prompt_id = None

        # 9. style_directive_id (uniform).
        style_directive_id = rng.randrange(config.style_directive_pool_size)

        # 10. target_strength (uniform, inclusive bounds; per-pair shared magnitude).
        target_strength = rng.randint(config.strength_min, config.strength_max)

        # 11. reasoning_basis — drawn LAST so the preceding draws keep their exact
        # RNG positions: a default (all-MERIT) plan is byte-identical to the
        # pre-dimension pipeline apart from the new field.
        reasoning_basis = _weighted_choice(config.reasoning_basis_allocation, rng)

        specs.append(
            PromptSpec(
                pair_id=pair_id,
                seed=seed,
                preference_pair=pref_pair,
                current_pref=current_pref,
                framing=framing,
                question_shape=question_shape,
                tone=tone,
                system_prompt_id=system_prompt_id,
                style_directive_id=style_directive_id,
                target_strength=target_strength,
                reasoning_basis=reasoning_basis,
            )
        )

    specs = _quota_correct(specs, config, master)
    return specs


# ═══════════════════════════════════════════════════════════════════════════════
# QUOTA CORRECTION (§3.4) — minimal-touch
# ═══════════════════════════════════════════════════════════════════════════════


def _correct_one_dimension(
    specs: List[PromptSpec],
    alloc_attr: str,
    field: str,
    config: GenerationConfig,
    corr_rng: random.Random,
) -> List[PromptSpec]:
    """Bring a single dimension inside ±tolerance by minimal donor→receiver flips.

    Greedy minimal-touch correction (§3.4): while any category sits outside its
    ±``_CORRECTION_TOLERANCE`` band, pick the most under-represented *receiver* and
    a *donor* category that is strictly above target, then flip one spec's field
    from donor→receiver. The donor is chosen to be the most over-represented
    category that can give up a unit without itself dropping below ``target - tol``
    (so a flip never trades one violation for another); ties break by ``.name``.

    Each flip strictly reduces total absolute deviation, so the loop terminates.
    Re-derives nothing else — the preference pair and seed stay put, only the one
    dimension flips. Only the freely-flippable surface fields (framing /
    question_shape / tone) are ever passed here; severity is
    excluded by construction (see ``_DIMENSIONS``).
    """
    n = len(specs)
    alloc: Dict[Any, float] = getattr(config, alloc_attr)

    # Mutable working copy of just this field's values + running counts.
    values = [getattr(spec, field) for spec in specs]
    counts: Dict[Any, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    for key in alloc:
        counts.setdefault(key, 0)

    def _frac(key: Any) -> float:
        return counts.get(key, 0) / n

    def _out_of_band() -> bool:
        return any(
            abs(_frac(key) - alloc[key]) > _CORRECTION_TOLERANCE for key in alloc
        )

    # Earliest-first index of specs carrying each value, so donor selection is
    # deterministic and walks specs in index order.
    next_idx: Dict[Any, int] = {key: 0 for key in alloc}
    idx_by_value: Dict[Any, List[int]] = {key: [] for key in alloc}
    for idx, value in enumerate(values):
        idx_by_value.setdefault(value, []).append(idx)

    lower = {key: alloc[key] - _CORRECTION_TOLERANCE for key in alloc}
    upper = {key: alloc[key] + _CORRECTION_TOLERANCE for key in alloc}

    changed = False
    # Bound the work at n flips; each flip cuts total |deviation| by 2/n, and the
    # band guards below ensure a flip never creates a fresh violation, so the loop
    # makes monotone progress and terminates well before n.
    for _ in range(n):
        if not _out_of_band():
            break

        # Donor: an above-target category that (a) is the most over-represented,
        # (b) can give up a unit without dropping below its lower band, and (c)
        # still has an unflipped spec to move. Driving correction off the donor
        # side lets us fix an over-band category even when no receiver is itself
        # out of band (the surplus is just redistributed into below-target slack).
        donor_keys = [
            key
            for key in alloc
            if _frac(key) > alloc[key]
            and (counts[key] - 1) / n >= lower[key]
            and next_idx[key] < len(idx_by_value.get(key, []))
        ]
        if not donor_keys:
            break
        donor = max(
            sorted(donor_keys, key=lambda k: getattr(k, "name", str(k))),
            key=lambda k: _frac(k) - alloc[k],
        )

        # Receiver: a below-target category with room to grow that won't overshoot
        # its upper band; prefer the most under-represented (largest deficit).
        receiver_keys = [
            key
            for key in alloc
            if key != donor
            and _frac(key) < alloc[key]
            and (counts[key] + 1) / n <= upper[key]
        ]
        if not receiver_keys:
            break
        receiver = max(
            sorted(receiver_keys, key=lambda k: getattr(k, "name", str(k))),
            key=lambda k: alloc[k] - _frac(k),
        )

        donor_idx = idx_by_value[donor][next_idx[donor]]
        next_idx[donor] += 1

        # Flip this one field; corr_rng is consumed so the path is salt-isolated.
        corr_rng.random()
        values[donor_idx] = receiver
        counts[donor] -= 1
        counts[receiver] += 1
        changed = True

    if not changed:
        return specs

    # Rebuild specs with the corrected field values (PromptSpec is frozen).
    rebuilt: List[PromptSpec] = []
    for spec, new_value in zip(specs, values):
        if new_value == getattr(spec, field):
            rebuilt.append(spec)
        else:
            rebuilt.append(_replace_field(spec, field, new_value))
    return rebuilt


def _replace_field(spec: PromptSpec, field: str, value: Any) -> PromptSpec:
    """Return a copy of ``spec`` with one surface field replaced.

    PromptSpec is frozen; ``dataclasses.replace`` re-runs ``__post_init__`` so the
    flipped value is still validated. Only the four flippable surface fields are
    ever passed (the preference pair and seed are never touched).
    """
    return dataclasses.replace(spec, **{field: value})


def _quota_correct(
    specs: List[PromptSpec], config: GenerationConfig, master: int
) -> List[PromptSpec]:
    """Run minimal-touch quota correction over all surface dimensions (§3.4).

    Each dimension is corrected independently with a correction RNG seeded from
    ``master ^ CORRECTION_SALT`` so corrections are deterministic and isolated from
    the per-pair sampling seeds. Passes are capped at ``_MAX_CORRECTION_PASSES``;
    anything still out of band is left for ``validate_plan`` to flag (warning, not
    crash).
    """
    if not specs:
        return specs

    corr_rng = random.Random(master ^ CORRECTION_SALT)
    for _pass in range(_MAX_CORRECTION_PASSES):
        any_change = False
        for _label, alloc_attr, field in _DIMENSIONS:
            before = [getattr(s, field) for s in specs]
            specs = _correct_one_dimension(
                specs, alloc_attr, field, config, corr_rng
            )
            after = [getattr(s, field) for s in specs]
            if before != after:
                any_change = True
        if not any_change:
            break
    return specs


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION (§5)
# ═══════════════════════════════════════════════════════════════════════════════


def validate_plan(plan: List[PromptSpec], config: GenerationConfig) -> List[str]:
    """Validate a plan; return a list of issue messages ([] == clean).

    Hard checks (the orchestrator aborts on these): length, pair_id uniqueness,
    holdout integrity. Soft checks (warn): distribution within ±5pp, system-prompt
    rate, style-directive / intensity coverage, current_pref direction balance.

    Adapted from ``dataset_gen/src/plan.py:validate_plan`` to the v2 dimensions.
    """
    issues: List[str] = []
    n = len(plan)

    if n == 0:
        issues.append("Plan is empty")
        return issues

    # 1. Length (hard).
    if n != config.queued_pairs:
        issues.append(
            f"Plan length {n} != config.queued_pairs {config.queued_pairs}"
        )

    # 2. pair_id uniqueness (hard).
    seen: Dict[str, int] = {}
    for spec in plan:
        seen[spec.pair_id] = seen.get(spec.pair_id, 0) + 1
    duplicates = {pid: c for pid, c in seen.items() if c > 1}
    if duplicates:
        issues.append(
            f"Duplicate pair_ids found (each must appear exactly once): "
            f"{sorted(duplicates)[:5]}"
        )

    # 3. Holdout integrity (hard).
    hold = holdout_keys(config)
    leaked = sorted(
        {
            spec.preference_pair.key()
            for spec in plan
            if spec.preference_pair.key() in hold
        }
    )
    if leaked:
        issues.append(
            f"Holdout integrity violation: {len(leaked)} spec(s) use held-out "
            f"preference pair(s): {leaked[:5]}"
        )

    tol = _VALIDATE_TOLERANCE

    def _check_distribution(label: str, getter, expected: Dict[Any, float]) -> None:
        counts: Dict[Any, int] = {}
        for spec in plan:
            key = getter(spec)
            counts[key] = counts.get(key, 0) + 1
        for key, exp in expected.items():
            observed = counts.get(key, 0) / n
            if abs(observed - exp) > tol:
                issues.append(
                    f"{label} distribution off: {getattr(key, 'name', key)} "
                    f"observed {observed:.3f}, expected {exp:.3f} (±{tol:.2f})"
                )

    # 4. Distribution checks (soft). Severity reads from the preference pair
    # (it is not a stored PromptSpec field).
    _check_distribution("Framing", lambda s: s.framing, config.framing_allocation)
    _check_distribution(
        "QuestionShape", lambda s: s.question_shape, config.question_shape_allocation
    )
    _check_distribution("Tone", lambda s: s.tone, config.tone_allocation)
    _check_distribution(
        "Severity", lambda s: s.preference_pair.severity, config.severity_allocation
    )
    _check_distribution(
        "ReasoningBasis",
        lambda s: s.reasoning_basis,
        config.reasoning_basis_allocation,
    )

    # 5. system_prompt rate (soft) + id range (hard-ish).
    n_with_prompt = sum(1 for spec in plan if spec.system_prompt_id is not None)
    observed_rate = n_with_prompt / n
    if abs(observed_rate - config.system_prompt_rate) > tol:
        issues.append(
            f"system_prompt rate off: observed {observed_rate:.3f}, expected "
            f"{config.system_prompt_rate:.3f} (±{tol:.2f})"
        )
    for spec in plan:
        sid = spec.system_prompt_id
        if sid is not None and not 0 <= sid < config.system_prompt_pool_size:
            issues.append(
                f"system_prompt_id {sid} out of range "
                f"[0, {config.system_prompt_pool_size}) on {spec.pair_id}"
            )

    # 6. style_directive coverage + rough balance.
    directive_counts: Dict[int, int] = {}
    for spec in plan:
        directive_counts[spec.style_directive_id] = (
            directive_counts.get(spec.style_directive_id, 0) + 1
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

    # 7. strength coverage (every value in [min, max] appears).
    strength_present = {spec.target_strength for spec in plan}
    for value in range(config.strength_min, config.strength_max + 1):
        if value not in strength_present:
            issues.append(f"target_strength {value} never appears (coverage gap)")

    # 8. current_pref direction ~50/50 (soft).
    n_a = sum(1 for spec in plan if spec.current_pref == "a")
    frac_a = n_a / n
    if abs(frac_a - 0.5) > tol:
        issues.append(
            f"current_pref direction off: 'a' observed {frac_a:.3f}, "
            f"expected 0.500 (±{tol:.2f})"
        )

    return issues


__all__ = ["plan", "validate_plan", "holdout_keys", "HOLDOUT_SALT"]
