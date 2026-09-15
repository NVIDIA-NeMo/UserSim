# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Selection profiles: the configurable quality gates a trajectory must pass.

A :class:`SelectionProfile` is a declarative bundle of gates applied to one
trajectory + its evaluation:

- universal gates read straight off ``simulation_outcome``
  (status, turn count, simulator-health counters),
- per-axis floors read off the evaluator's wide cell.

Design choices:

- **Our own thresholds, the registry's structure.** Floors live on the
  profile, not in :mod:`taxonomy.capabilities`. The capability registry is
  reused only for *metadata*: which scorer owns a scorer-axis (so an
  errored scorer can be reported), and which capability a gate maps back to.
- **Applicability is driven by the eval cell, not the registry.** A floor
  applies to a row whenever that axis is present in the row's decoded eval
  cell. This is required because several real evaluator axes
  (``role_adherence``, ``language_appropriateness``, ``personalization``,
  ``pedagogical_quality``) are absent from the capability registry; a
  registry-only inversion would create no gate for them.
- **Generous by default.** Missing scores / not-applicable scorers never
  drop a row; only clear violations do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

from usersim.taxonomy.capabilities import ALL_PROBES, capability_definitions
from usersim.taxonomy.eval_cell import axis_scale

# Sentinel for ``critical_axes`` meaning "every floor in this profile is a
# hard gate" — so a profile that hard-gates all of its floors does not have
# to repeat every axis name twice.
ALL_FLOORS = "*"


@dataclass(frozen=True)
class AxisGate:
    """One resolved per-axis floor for a specific row's eval cell."""

    axis: str
    floor: float
    critical: bool
    scale: str  # "rate" | "score_1_5"
    scorer: Optional[str] = None  # owning scorer, if this is a scorer-axis
    capability: Optional[str] = None  # capability id this axis maps to, if any


@dataclass(frozen=True)
class SelectionProfile:
    """A named bundle of selection gates.

    Floors are expressed on each axis's *native* scale (1-5 judge axes,
    0-1 rate axes). ``critical_axes`` lists which floor violations are
    hard (drop the row); a floor not listed there is "soft" — recorded in
    ``selection_failed_gates`` for visibility but it does not drop the row.
    Pass ``critical_axes=(ALL_FLOORS,)`` to make every floor hard.
    """

    name: str
    version: str = "v1.0"
    allowed_statuses: tuple[str, ...] = ("ok", "completed_with_warnings")
    require_eval_present: bool = True
    min_turns: int = 1
    axis_floors: Mapping[str, float] = field(default_factory=dict)
    critical_axes: tuple[str, ...] = (ALL_FLOORS,)
    quality_score_axes: Optional[tuple[str, ...]] = None
    max_judge_disagreement: Optional[float] = None
    # Optional simulator-health gates, read off simulation_outcome. None = off.
    max_user_role_violations: Optional[int] = None
    max_user_language_violations: Optional[int] = None
    max_fourth_wall_triggers: Optional[int] = None
    max_assistant_inline_failures: Optional[int] = None
    holdout_fraction: float = 0.0

    def is_critical(self, axis: str) -> bool:
        return ALL_FLOORS in self.critical_axes or axis in self.critical_axes

    def label(self) -> str:
        return f"{self.name}:{self.version}"


def registry_axis_scorer_map() -> dict[str, Optional[str]]:
    """Map each registry axis to the scorer that owns it (``None`` = judge axis).

    Built by inverting :func:`taxonomy.capabilities.capability_definitions`.
    Used to attribute an errored scorer to the axis it would have scored.
    """
    out: dict[str, Optional[str]] = {}
    for definition in capability_definitions():
        for source in definition.sources:
            for axis in source.axes:
                # Don't clobber a real scorer with a later None mapping.
                if axis not in out or source.scorer is not None:
                    out[axis] = source.scorer
    return out


def registry_axis_capability_map() -> dict[str, str]:
    """Map each registry axis to the (first) capability id that consumes it."""
    out: dict[str, str] = {}
    for definition in capability_definitions():
        for source in definition.sources:
            for axis in source.axes:
                out.setdefault(axis, definition.id)
    return out


_AXIS_SCORER = registry_axis_scorer_map()
_AXIS_CAPABILITY = registry_axis_capability_map()


def _axis_present(eval_cell: Mapping, axis: str) -> bool:
    """True if a gate for ``axis`` should apply to this cell.

    An axis is "present" if it appears among judge axes, in any scorer's
    scores, OR its registry-owning scorer ran at all (even with an error —
    so a floored scorer-axis on an errored scorer still produces a gate,
    which the engine records as ``<axis>:scorer_error`` rather than
    silently ignoring).
    """
    if axis in (eval_cell.get("axes") or {}):
        return True
    scorers = eval_cell.get("scorers") or {}
    owning = _AXIS_SCORER.get(axis)
    if owning is not None and owning in scorers:
        return True
    for block in scorers.values():
        if isinstance(block, dict) and axis in (block.get("scores") or {}):
            return True
    return False


def applicable_gates(profile: SelectionProfile, eval_cell: Mapping) -> list[AxisGate]:
    """Resolve this profile's floors to the axes actually present in ``eval_cell``.

    Registry-unknown axes are treated as universal judge axes (scorer
    ``None``) — they gate wherever the cell carries them.
    """
    gates: list[AxisGate] = []
    for axis, floor in profile.axis_floors.items():
        if not _axis_present(eval_cell, axis):
            continue
        gates.append(
            AxisGate(
                axis=axis,
                floor=float(floor),
                critical=profile.is_critical(axis),
                scale=axis_scale(axis),
                scorer=_AXIS_SCORER.get(axis),
                capability=_AXIS_CAPABILITY.get(axis),
            )
        )
    return gates


def axis_owning_scorer(axis: str) -> Optional[str]:
    """Scorer that owns ``axis`` per the registry, or ``None`` for judge axes."""
    return _AXIS_SCORER.get(axis)


# ---------------------------------------------------------------------------
# Shipped profiles
# ---------------------------------------------------------------------------

# Default: throw out only clear violations / errors. Grounded in the
# hi_Deva_IN run distribution (accuracy mean ~3.1, response_conciseness
# mean ~2.2), so there are NO hard >=4 gates — a 1-5 floor of 2.0 rejects
# only an outright score==1. Rate floors of 0.5 reject gross
# language/empty-response failures. critical_axes=ALL_FLOORS: every floor
# is a hard gate. response_conciseness / helpfulness / personalization are
# deliberately NOT floored (known-weak / subjective; would over-prune).
GENEROUS = SelectionProfile(
    name="generous",
    version="v1.0",
    axis_floors={
        "safety": 2.0,
        "coherence": 2.0,
        "role_adherence": 2.0,
        "language_appropriateness": 2.0,
        "accuracy": 2.0,
        "language.requested_language_match_rate": 0.5,
        "shape.empty_or_trivial_rate": 0.5,
    },
    critical_axes=(ALL_FLOORS,),
)

# Illustrative high-bar profile (not the default): all primary judge axes
# must clear 4/5 and language/shape compliance must be near-perfect.
STRICT = SelectionProfile(
    name="strict",
    version="v1.0",
    axis_floors={
        "helpfulness": 4.0,
        "accuracy": 4.0,
        "coherence": 4.0,
        "safety": 4.0,
        "role_adherence": 4.0,
        "language_appropriateness": 4.0,
        "language.requested_language_match_rate": 0.95,
        "language.script_compliance_rate": 0.9,
        "shape.empty_or_trivial_rate": 0.95,
    },
    critical_axes=(ALL_FLOORS,),
    max_user_role_violations=0,
    max_user_language_violations=0,
    max_fourth_wall_triggers=0,
)


_PROFILES: dict[str, SelectionProfile] = {
    GENEROUS.name: GENEROUS,
    STRICT.name: STRICT,
}


def get_profile(name: str) -> SelectionProfile:
    """Look up a shipped profile by name (case-insensitive)."""
    key = name.strip().lower()
    if key not in _PROFILES:
        raise KeyError(
            f"unknown selection profile {name!r}; available: {sorted(_PROFILES)}"
        )
    return _PROFILES[key]


def available_profiles() -> tuple[str, ...]:
    return tuple(sorted(_PROFILES))


# Re-export so callers can probe-scope without importing capabilities directly.
__all__ = [
    "ALL_FLOORS",
    "ALL_PROBES",
    "AxisGate",
    "SelectionProfile",
    "GENEROUS",
    "STRICT",
    "applicable_gates",
    "available_profiles",
    "axis_owning_scorer",
    "get_profile",
    "registry_axis_capability_map",
    "registry_axis_scorer_map",
]
