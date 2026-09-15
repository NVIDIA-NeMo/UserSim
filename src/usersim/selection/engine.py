# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure selection engine: turn a trajectory + eval join into keep/drop verdicts.

No I/O. :func:`select_trajectories` takes the trajectory DataFrame and the
evaluation DataFrame, applies a :class:`~selection.profiles.SelectionProfile`,
and returns a :class:`SelectionResult` carrying the curated (passing) subset,
the fully-annotated frame, and a funnel summary.

The join is **trajectory-left** so trajectories that were never evaluated
(e.g. a partial ``usersim eval --num-records`` run) show up as an explicit
``no_eval`` drop rather than silently vanishing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from usersim.taxonomy.eval_cell import (
    decode_cell,
    normalize_axis_score,
    score_from_axis,
    score_from_eval_cell,
    scorer_state,
)
from usersim.selection.profiles import GENEROUS, SelectionProfile, applicable_gates

# Added columns on the annotated / curated frames.
COL_PASSED = "selection_passed"
COL_QUALITY = "selection_quality_score"
COL_AXIS_SCORES = "selection_axis_scores"
COL_FAILED_GATES = "selection_failed_gates"
COL_DROP_REASON = "selection_drop_reason"
COL_PROFILE = "selection_profile"
COL_SPLIT = "selection_split"

ADDED_COLUMNS = (
    COL_PASSED,
    COL_QUALITY,
    COL_AXIS_SCORES,
    COL_FAILED_GATES,
    COL_DROP_REASON,
    COL_PROFILE,
    COL_SPLIT,
)


@dataclass(frozen=True)
class SelectionVerdict:
    trajectory_id: str
    passed: bool
    quality_score: Optional[float]
    axis_scores: dict[str, float]
    failed_gates: list[str]
    drop_reason: Optional[str]
    split: Optional[str]


@dataclass
class SelectionSummary:
    profile: str
    total: int
    passed: int
    dropped: int
    selected: int = 0
    max_records: Optional[int] = None
    max_per_locale: Optional[int] = None
    stratify_by: Optional[tuple[str, ...]] = None
    drop_reasons: dict[str, int] = field(default_factory=dict)
    kept_by_group: dict[str, int] = field(default_factory=dict)
    holdout: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "total": self.total,
            "passed": self.passed,
            "dropped": self.dropped,
            "selected": self.selected,
            "max_records": self.max_records,
            "max_per_locale": self.max_per_locale,
            "stratify_by": list(self.stratify_by) if self.stratify_by else None,
            "drop_reasons": dict(self.drop_reasons),
            "kept_by_group": dict(self.kept_by_group),
            "holdout": dict(self.holdout),
        }


@dataclass
class SelectionResult:
    curated: Any  # pandas.DataFrame of passing rows + ADDED_COLUMNS
    all: Any  # pandas.DataFrame of every row + ADDED_COLUMNS
    summary: SelectionSummary


def _split_for(persona_uuid: Any, fraction: float) -> Optional[str]:
    """Deterministically bucket a persona into train/holdout by uuid hash."""
    if fraction <= 0.0:
        return None
    digest = hashlib.md5(str(persona_uuid).encode("utf-8")).hexdigest()
    bucket = (int(digest, 16) % 10_000) / 10_000.0
    return "holdout" if bucket < fraction else "train"


def _quality_score(cell: dict, profile: SelectionProfile) -> Optional[float]:
    """Unweighted mean of normalized (0-1) applicable 1-5 judge axes."""
    from usersim.taxonomy.eval_cell import axis_scale

    axes = cell.get("axes") or {}
    wanted = profile.quality_score_axes
    candidates = wanted if wanted is not None else tuple(axes.keys())
    vals: list[float] = []
    for axis in candidates:
        if axis_scale(axis) != "score_1_5":
            continue
        score = score_from_eval_cell(cell, axis)
        if score is not None:
            vals.append(normalize_axis_score(axis, score))
    return sum(vals) / len(vals) if vals else None


def _judge_disagreement_reasons(cell: dict, threshold: float) -> list[str]:
    """Axes whose per-judge score spread exceeds ``threshold``."""
    out: list[str] = []
    for axis, judge_block in (cell.get("axes") or {}).items():
        if not isinstance(judge_block, dict):
            continue
        scores = [
            score_from_axis(v) for v in judge_block.values() if isinstance(v, dict)
        ]
        scores = [s for s in scores if s is not None]
        if len(scores) >= 2 and (max(scores) - min(scores)) > threshold:
            out.append(f"judge_disagreement:{axis}")
    return out


def _health_reasons(outcome: dict, profile: SelectionProfile) -> list[str]:
    """Simulator-health gate violations (opt-in; each ``None`` cap is off)."""
    out: list[str] = []
    checks = (
        ("n_user_role_violations", profile.max_user_role_violations),
        ("n_user_language_violations", profile.max_user_language_violations),
        ("n_fourth_wall_triggers", profile.max_fourth_wall_triggers),
        ("n_assistant_inline_failures", profile.max_assistant_inline_failures),
    )
    for counter, cap in checks:
        if cap is None:
            continue
        value = outcome.get(counter)
        if isinstance(value, (int, float)) and value > cap:
            out.append(f"health:{counter}")
    return out


def evaluate_row(
    *,
    outcome_raw: Any,
    eval_cell_raw: Any,
    persona_uuid: Any,
    trajectory_id: Any,
    num_turns_fallback: Any,
    profile: SelectionProfile,
) -> SelectionVerdict:
    """Apply ``profile`` to a single trajectory + eval cell."""
    outcome = decode_cell(outcome_raw)
    cell = decode_cell(eval_cell_raw)
    have_eval = bool(cell)
    skipped = bool(cell.get("skipped"))

    hard: list[str] = []  # drop reasons, in precedence order
    soft: list[str] = []  # recorded-only (do not drop)
    axis_scores: dict[str, float] = {}

    # --- universal gates -------------------------------------------------
    if profile.require_eval_present and not have_eval:
        hard.append("no_eval")
    elif profile.require_eval_present and skipped:
        hard.append("skipped")

    status = outcome.get("status")
    if status not in profile.allowed_statuses:
        hard.append("status")

    n_turns = outcome.get("n_turns")
    if not isinstance(n_turns, (int, float)):
        n_turns = num_turns_fallback if isinstance(num_turns_fallback, (int, float)) else 0
    if n_turns < profile.min_turns:
        hard.append("min_turns")

    hard.extend(_health_reasons(outcome, profile))

    # --- eval-cell gates (only when there is a usable cell) --------------
    if have_eval and not skipped:
        if profile.max_judge_disagreement is not None:
            hard.extend(
                _judge_disagreement_reasons(cell, profile.max_judge_disagreement)
            )
        for gate in applicable_gates(profile, cell):
            score = score_from_eval_cell(cell, gate.axis)
            if score is None:
                # Missing score: only worth recording if an owning scorer
                # erred (a not-applicable / absent scorer is silent — the
                # generous default does not penalize it).
                if gate.scorer:
                    block = (cell.get("scorers") or {}).get(gate.scorer)
                    if scorer_state(block) == "error":
                        soft.append(f"{gate.axis}:scorer_error")
                continue
            axis_scores[gate.axis] = score
            if score < gate.floor:
                (hard if gate.critical else soft).append(gate.axis)
        quality = _quality_score(cell, profile)
    else:
        quality = None

    passed = not hard
    return SelectionVerdict(
        trajectory_id=str(trajectory_id),
        passed=passed,
        quality_score=quality,
        axis_scores=axis_scores,
        failed_gates=hard + soft,
        drop_reason=hard[0] if hard else None,
        split=_split_for(persona_uuid, profile.holdout_fraction) if passed else None,
    )


def _allocate_quota(counts: dict[Any, int], total: int) -> dict[Any, int]:
    """Distribute ``total`` slots across strata as evenly as capacity allows.

    Water-filling: hand out equal shares each round, capped at each stratum's
    available count; the remainder from strata that ran out is redistributed to
    those with capacity left. Deterministic (strata processed in sorted order).
    Returns ``{stratum: quota}`` summing to ``min(total, sum(counts))``.
    """
    quotas: dict[Any, int] = {k: 0 for k in counts}
    remaining = min(total, sum(counts.values()))
    active = sorted((k for k in counts if counts[k] > 0), key=lambda k: repr(k))

    while remaining > 0 and active:
        share = remaining // len(active)
        if share == 0:
            # Fewer slots than active strata: hand out one each, in order.
            for k in active:
                if remaining == 0:
                    break
                if quotas[k] < counts[k]:
                    quotas[k] += 1
                    remaining -= 1
            break
        for k in active:
            give = min(share, counts[k] - quotas[k])
            quotas[k] += give
            remaining -= give
        active = [k for k in active if quotas[k] < counts[k]]
    return quotas


def _sort_by_quality(df: Any) -> Any:
    return df.sort_values(
        [COL_QUALITY, "trajectory_id"], ascending=[False, True], na_position="last"
    )


def _apply_cap(
    curated: Any,
    max_records: Optional[int],
    stratify_by: Optional[Sequence[str]],
) -> Any:
    """Trim ``curated`` to ``max_records`` highest-quality rows.

    With ``stratify_by`` set, the cap is distributed across the strata defined
    by those columns (e.g. ``probe_family``) and the highest-quality rows are
    taken *within* each stratum — so a generously-scored probe cannot crowd out
    a harder one. Without it, a single global top-K is taken.
    """
    if max_records is None or len(curated) <= max_records:
        return _sort_by_quality(curated).reset_index(drop=True)

    if not stratify_by:
        return _sort_by_quality(curated).head(max_records).reset_index(drop=True)

    keys = [c for c in stratify_by if c in curated.columns]
    if not keys:
        return _sort_by_quality(curated).head(max_records).reset_index(drop=True)

    groups = {key: g for key, g in curated.groupby(keys, sort=True, dropna=False)}
    quotas = _allocate_quota({k: len(g) for k, g in groups.items()}, max_records)

    import pandas as pd

    parts = [
        _sort_by_quality(groups[k]).head(quotas[k])
        for k in groups
        if quotas[k] > 0
    ]
    capped = pd.concat(parts) if parts else curated.iloc[0:0]
    return _sort_by_quality(capped).reset_index(drop=True)


def _cap_per_group(
    curated: Any, n: Optional[int], group_cols: Sequence[str]
) -> Any:
    """Keep at most ``n`` highest-quality rows within each group.

    Groups are defined by ``group_cols`` (e.g. ``("locale",)`` for a per-locale
    cap): within each group the top-``n`` rows by ``selection_quality_score``
    are kept (ties broken by ``trajectory_id``). No-op when ``n`` is ``None`` or
    none of ``group_cols`` are present. Applied *before* the global
    ``max_records`` cap so "at most N per locale" and "at most M overall"
    compose cleanly.
    """
    if n is None:
        return curated
    keys = [c for c in group_cols if c in curated.columns]
    if not keys:
        return curated
    import pandas as pd

    parts = [
        _sort_by_quality(g).head(n)
        for _, g in curated.groupby(keys, sort=True, dropna=False)
    ]
    capped = pd.concat(parts) if parts else curated.iloc[0:0]
    return _sort_by_quality(capped).reset_index(drop=True)


def select_trajectories(
    traj_df: Any,
    eval_df: Any,
    *,
    profile: SelectionProfile = GENEROUS,
    eval_column: str = "assistant_eval",
    max_records: Optional[int] = None,
    max_per_locale: Optional[int] = None,
    stratify_by: Optional[Sequence[str]] = ("probe_family",),
) -> SelectionResult:
    """Apply ``profile`` to every trajectory and return the curated subset.

    ``traj_df`` is the full population (left side of the join); ``eval_df``
    (may be ``None``) carries the wide ``eval_column``. Trajectories with no
    matching eval row are kept in the population and reported as ``no_eval``.

    ``max_per_locale`` (optional) keeps at most N highest-quality passing rows
    **per locale** — applied first, so a busy locale can't crowd out a sparse
    one. ``max_records`` then caps the (already per-locale-capped) subset to the
    highest ``selection_quality_score`` rows overall (ties broken by
    ``trajectory_id`` for determinism); ``stratify_by`` (default
    ``("probe_family",)``) distributes that global cap evenly across the named
    strata and takes the best *within* each, so a generously-scored probe
    cannot dominate — pass ``None`` for a single global top-K. Either cap set to
    ``None`` means "no cap" for that axis; both can be combined (e.g.
    ``max_per_locale=10`` alone gives 10 valid per locale across the whole run).
    """

    if "trajectory_id" not in traj_df.columns:
        raise ValueError("traj_df must carry a trajectory_id column")

    joined = traj_df.copy().reset_index(drop=True)
    if eval_df is not None and eval_column in getattr(eval_df, "columns", []):
        # Drop any pre-existing eval column on the trajectory frame so the
        # left-join brings in the authoritative one (no _x/_y suffix collision).
        if eval_column in joined.columns:
            joined = joined.drop(columns=[eval_column])
        right = eval_df[["trajectory_id", eval_column]].drop_duplicates("trajectory_id")
        joined = joined.merge(right, on="trajectory_id", how="left")
    elif eval_column not in joined.columns:
        joined[eval_column] = None

    verdicts: list[SelectionVerdict] = []
    for _, row in joined.iterrows():
        verdicts.append(
            evaluate_row(
                outcome_raw=row.get("simulation_outcome"),
                eval_cell_raw=row.get(eval_column),
                persona_uuid=row.get("persona_uuid"),
                trajectory_id=row.get("trajectory_id"),
                num_turns_fallback=row.get("num_turns"),
                profile=profile,
            )
        )

    annotated = joined.copy()
    annotated[COL_PASSED] = [v.passed for v in verdicts]
    annotated[COL_QUALITY] = [v.quality_score for v in verdicts]
    annotated[COL_AXIS_SCORES] = [json.dumps(v.axis_scores) for v in verdicts]
    annotated[COL_FAILED_GATES] = [json.dumps(v.failed_gates) for v in verdicts]
    annotated[COL_DROP_REASON] = [v.drop_reason for v in verdicts]
    annotated[COL_PROFILE] = profile.label()
    annotated[COL_SPLIT] = [v.split for v in verdicts]

    curated = annotated[annotated[COL_PASSED]].reset_index(drop=True)
    # Per-locale cap first (bound each locale independently), then the global
    # max_records cap over what remains — so the two compose.
    curated = _cap_per_group(curated, max_per_locale, ("locale",))
    curated = _apply_cap(curated, max_records, stratify_by)

    summary = _summarize(verdicts, curated, profile)
    summary.selected = len(curated)
    summary.max_records = max_records
    summary.max_per_locale = max_per_locale
    summary.stratify_by = tuple(stratify_by) if stratify_by else None
    return SelectionResult(curated=curated, all=annotated, summary=summary)


def _summarize(
    verdicts: list[SelectionVerdict], curated: Any, profile: SelectionProfile
) -> SelectionSummary:
    total = len(verdicts)
    passed = sum(1 for v in verdicts if v.passed)
    drop_reasons: dict[str, int] = {}
    for v in verdicts:
        if v.drop_reason is not None:
            drop_reasons[v.drop_reason] = drop_reasons.get(v.drop_reason, 0) + 1

    kept_by_group: dict[str, int] = {}
    if {"locale", "probe_family"}.issubset(getattr(curated, "columns", [])):
        for _, row in curated.iterrows():
            key = f"{row.get('locale')}/{row.get('probe_family')}"
            kept_by_group[key] = kept_by_group.get(key, 0) + 1

    holdout: dict[str, int] = {}
    if profile.holdout_fraction > 0.0:
        for v in verdicts:
            if v.split is not None:
                holdout[v.split] = holdout.get(v.split, 0) + 1

    return SelectionSummary(
        profile=profile.label(),
        total=total,
        passed=passed,
        dropped=total - passed,
        drop_reasons=drop_reasons,
        kept_by_group=kept_by_group,
        holdout=holdout,
    )
