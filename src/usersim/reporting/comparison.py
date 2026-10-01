# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Multi-model comparison report — pure data layer.

Reads N per-run report manifests (the JSON output from
``build_capability_report`` + ``write_capability_dashboard_artifacts``),
pivots them into a comparable ``(capability x locale x model)`` cube,
and assembles the data spine that drives the comparison dashboard
(``reporting.comparison_dashboard``).

Score normalization is single-source-of-truth: every ``CapabilityCell``
on a per-run manifest is *already* normalized to 0-1 (see
``reporting/capability_report.py:_capability_cell``). The
``ComparisonCell`` mirror just renames ``score`` to ``normalized_score``
for clarity at the cross-model layer.

Score rollup discipline (see plan doc — Score Rollup Mathematics):

- Skip ``missing`` and ``insufficient_evidence`` cells from the rollup mean.
- Exclude ``simulation_reliability`` (process-health, not assistant quality;
  surfaced separately in Layer 0).
- Equal weighting across remaining capabilities.

Edge-case discipline (also per plan):

- Zero manifests -> ``ValueError("no runs to compare; ...")``.
- Exactly one manifest -> ``ValueError("comparison requires >=2 runs; ...")``.
- Mixed ``eval_column`` across manifests -> ``ValueError`` (fail closed —
  the cells aren't drawn from the same evaluator output column, so a join
  would silently align mismatched data).

Other drift dimensions (sample_mode, judge prompt version, n_trajectories,
persona panel, missing sim_health) are surfaced as warnings on
``ComparisonReport.apples_to_apples_warnings`` rather than errors.

This module performs no I/O of HTML — that is the dashboard module's
job. ``ComparisonReport.to_dict()`` is JSON-serializable.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from usersim.taxonomy.capabilities import capability_definitions

# Capabilities excluded from the quality rollup. ``simulation_reliability``
# is a process-health metric (success rate of the conversation loop) and
# does not reflect assistant quality — it lives in the Sim Health panel
# (Layer 0), never in the leaderboard.
ROLLUP_EXCLUDED_CAPABILITIES: tuple[str, ...] = ("simulation_reliability",)

# A cell is considered "measured" iff its state is one of these. Other
# states (``missing``, ``insufficient_evidence``) are skipped from the
# rollup mean so coverage gaps don't masquerade as quality failures.
MEASURED_STATES: tuple[str, ...] = ("ready", "blocked")


# ---------------------------------------------------------------------------
# Dataclasses — the public schema embedded into comparison_manifest.json
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ComparisonEntry:
    """One per included run; populates the Models-Compared audit table.

    Token bookkeeping:

    - ``assistant_total_tokens``: gross billing for the model under
      test on this run -- input + output across every assistant call.
      Kept on the entry for back-compat (older comparison manifests
      embed it) but no longer surfaced in the audit table -- the
      verbosity story is fully told by output / reasoning instead.
    - ``assistant_output_tokens``: output-only tally for the same
      assistant calls. The verbosity-driving metric (Layer 1b chart,
      Pareto chart) -- input tokens reflect prompt context size, not
      the model's verbosity, so we exclude them.
    - ``assistant_reasoning_tokens``: invisible-thinking subset of
      ``assistant_output_tokens``. Estimated post-hoc via tiktoken on
      the trajectory's visible content (see ``reporting/_reasoning.py``)
      for runs whose providers don't surface reasoning telemetry.
    - ``user_output_tokens`` / ``user_reasoning_tokens``: the same
      breakdown for the user simulator. Surfacing it in the audit
      table lets reviewers spot when one side of the conversation
      is dominating verbosity (e.g., over-eager probe prompts).

    Conversation shape:

    - ``mean_turns_per_conversation``: average turn count per
      trajectory (sourced from ``sim_health.counters_means.n_turns``).
      Drives the "Avg turns/conversation" hero card.
    """

    run_id: str
    model_id: str | None
    display_label: str
    eval_column: str
    n_trajectories: int
    n_evaluated: int
    assistant_total_tokens: int
    assistant_output_tokens: int
    assistant_reasoning_tokens: int
    user_output_tokens: int
    user_reasoning_tokens: int
    mean_turns_per_conversation: float
    sample_mode: str
    manifest_path: str
    evaluator_prompt_versions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ComparisonCell:
    """One per (capability, locale, run); the spine of every chart.

    ``normalized_score`` mirrors the per-run dashboard's already-0-1
    ``CapabilityCell.score`` — no re-normalization needed.

    NOTE: deliberately carries no ``evidence_snippets`` / per-turn data.
    Drill-down to specific bad conversations lives in the per-run
    dashboard, accessed via the deep-link in the dashboard's tooltip.
    Keeps the comparison HTML small even with 10+ runs.
    """

    capability: str
    capability_label: str
    locale: str
    run_id: str
    display_label: str
    state: str
    normalized_score: float | None
    threshold: float | None
    n: int
    n_total: int
    failed_critical_axes: list[str] = field(default_factory=list)
    aggregation_policy: str = ""
    is_missing: bool = False


@dataclass(slots=True)
class ModelRollup:
    """Per-model summary; data spine for Layers 0-3 of the dashboard.

    ``mean_normalized_score`` follows the rollup discipline at the top
    of this module: skip ``missing`` / ``insufficient_evidence`` cells,
    exclude ``simulation_reliability``, equal weighting otherwise.
    """

    run_id: str
    display_label: str
    model_id: str | None

    # Coverage / state distribution (denominator = all cells INCLUDING
    # simulation_reliability and missing — gives a faithful picture of
    # how full the matrix is for this model).
    n_cells_eligible: int = 0  # excludes simulation_reliability
    n_cells_measured: int = 0  # eligible AND state in MEASURED_STATES
    n_cells_ready: int = 0
    n_cells_blocked: int = 0
    n_cells_missing: int = 0
    n_cells_insufficient: int = 0
    # Eligible-only ready count (excludes simulation_reliability) --
    # the numerator for the "Cleared capabilities" headline chart in
    # the Performance section. ``n_cells_ready`` above includes the
    # simulation_reliability cell, which would inflate the cleared
    # rate by ~1/15th. Pair this with ``n_cells_eligible`` (also
    # exclusion-aware, denominator = 112 in an 8-locale × 14-quality
    # capability panel) for the cleared percentage.
    n_cells_ready_eligible: int = 0
    pct_ready: float = 0.0
    pct_blocked: float = 0.0
    pct_missing: float = 0.0
    pct_insufficient: float = 0.0
    pct_measured: float = 0.0  # n_cells_measured / n_cells_eligible
    # Gap count for Layer 1c: blocked cells with quality-bearing capabilities
    # only (excludes ROLLUP_EXCLUDED_CAPABILITIES, e.g. simulation_reliability).
    # n_cells_blocked above includes ALL blocked cells; n_gaps is the
    # assistant-quality subset for the leaderboard's third chart.
    n_gaps: int = 0

    # Quality (skips missing/insufficient + excludes simulation_reliability).
    mean_normalized_score: float | None = None

    # Verbosity (Layers 1b, 2, 3).
    #
    # ``assistant_total_tokens``: input + output, kept on the rollup so
    # the audit table can show the gross billing figure. NOT used by
    # the verbosity chart -- input tokens dominate the total but reflect
    # prompt context size, not the model's verbosity.
    # ``assistant_output_tokens_per_conversation``: the verbosity metric
    # for Layer 1b's headline bar AND Layer 2's Pareto x-axis. Output
    # tokens INCLUDE any reasoning content (gross billing convention);
    # the reasoning portion is broken out below.
    # ``assistant_reasoning_tokens_per_conversation``: invisible-thinking
    # subset of the per-conversation output. Drives the stacked layer
    # of the Layer 1b chart so reviewers can see at a glance how much
    # of the verbosity is reasoning vs visible response text.
    n_trajectories: int = 0
    assistant_total_tokens: int = 0
    assistant_output_tokens_per_conversation: float = 0.0
    assistant_reasoning_tokens_per_conversation: float = 0.0
    assistant_tokens_per_turn_by_locale: dict[str, float] = field(default_factory=dict)

    # Sim health. May be all-None on very old manifests predating
    # the Sim Health bundle.
    #
    # ``sim_status_ok_rate`` is the fraction of trajectories whose
    # final status was ``ok`` or ``completed_with_warnings``;
    # ``sim_status_failure_rate`` is the complement (= 1 - ok rate).
    # Both are surfaced so the dashboard can pick whichever the
    # reviewer cares about -- failure rates draw the eye to outliers
    # better than success rates that cluster around 100%.
    sim_status_ok_rate: float | None = None
    sim_status_failure_rate: float | None = None
    persona_grounding_rate: float | None = None
    early_stop_rate: float | None = None
    has_sim_health: bool = True
    # Per-class failure breakdown lifted directly from
    # ``sim_health.failure_taxonomy``: a list of dicts shaped
    # ``{failure_class, failure_attribution, n, proportion}``.
    # Drives the failure-mode stacked bar in the Sim Health panel
    # (answers "is this a *model* bug or a *sim* bug?"). Empty list
    # for runs that completed without failures or for legacy manifests
    # that don't carry the taxonomy.
    failure_taxonomy: list[dict[str, Any]] = field(default_factory=list)

    # Pareto frontier (Layer 2). Computed in build_comparison_report
    # via a single O(n^2) sweep over all rollups — strictly dominated
    # iff some other model has BOTH higher score AND lower tokens/conv.
    is_pareto_frontier: bool = False


@dataclass(slots=True)
class ComparisonReport:
    """Top-level report — mirrors :class:`CapabilityReport`'s role for the comparison surface.

    The render layer (``reporting/comparison_dashboard.py``) calls
    ``to_dict()`` and embeds the result in ``index.html``.
    """

    comparison_id: str
    entries: list[ComparisonEntry]
    cells: list[ComparisonCell]
    locales: list[str]
    capabilities: list[tuple[str, str]]  # (id, label) preserving registry order
    model_summary: list[ModelRollup]
    apples_to_apples_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "entries": [asdict(e) for e in self.entries],
            "cells": [asdict(c) for c in self.cells],
            "locales": list(self.locales),
            "capabilities": [list(c) for c in self.capabilities],
            "model_summary": [asdict(m) for m in self.model_summary],
            "apples_to_apples_warnings": list(self.apples_to_apples_warnings),
        }


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_comparison_runs(report_root: Path | str) -> list[Path]:
    """Find every report manifest under ``report_root``, sorted by run id descending.

    Globs ``<report_root>/run=*/report_manifest.json`` and sorts by the
    epoch-second run id (latest first). Run ids in this codebase are
    fixed-length integer strings, so a simple lex sort gives correct
    descending epoch order.
    """
    root = Path(report_root)
    if not root.exists():
        return []
    paths = list(root.glob("run=*/report_manifest.json"))

    def _key(p: Path) -> str:
        return p.parent.name.removeprefix("run=")

    return sorted(paths, key=_key, reverse=True)


def select_runs_for_comparison(
    report_root: Path | str,
    *,
    compare_run_ids: list[str] | None = None,
) -> list[Path]:
    """Discover + (optionally) filter / reorder report manifests for comparison.

    With ``compare_run_ids=None`` returns every manifest under
    ``report_root`` (discovery order — descending run id).

    With ``compare_run_ids=[...]`` returns the matching manifests in the
    user-specified order so the dashboard's bar / Pareto labels render
    in the order the caller asked for. Raises ``FileNotFoundError`` if
    any pinned id has no manifest on disk — the caller never silently
    gets a shorter list than they asked for.

    This is the helper the notebook's discovery cell uses; the validated
    selection then flows into both ``print_comparison_runs`` and
    ``build_comparison_report``.
    """
    discovered = discover_comparison_runs(report_root)
    if not compare_run_ids:
        return discovered
    by_id = {p.parent.name.removeprefix("run="): p for p in discovered}
    missing = [rid for rid in compare_run_ids if rid not in by_id]
    if missing:
        raise FileNotFoundError(
            f"COMPARE_RUN_IDS references run id(s) with no report on disk: {missing}. "
            f"Available under {Path(report_root)}: {sorted(by_id)}"
        )
    return [by_id[rid] for rid in compare_run_ids]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _short_label(model_id: str | None) -> str:
    """Strip provider prefix from a ``model_id`` for the chart label.

    ``"nvidia/google/gemma-4-31b-it"`` -> ``"gemma-4-31b-it"``
    ``"openai/openai/gpt-5.4"`` -> ``"gpt-5.4"``
    Falls back to ``"<unknown>"`` for None / empty.
    """
    if not model_id:
        return "<unknown>"
    parts = str(model_id).split("/")
    return parts[-1] if parts and parts[-1] else str(model_id)


def _disambiguate_labels(entries: list[ComparisonEntry]) -> None:
    """Suffix duplicate ``display_label``s with their ``run_id`` (in place)."""
    from collections import Counter

    counts = Counter(e.display_label for e in entries)
    for entry in entries:
        if counts[entry.display_label] > 1:
            entry.display_label = f"{entry.display_label} (run={entry.run_id})"


def _compute_pareto_frontier(rollups: list[ModelRollup]) -> None:
    """Mark ``is_pareto_frontier=True`` on every non-dominated rollup (in place).

    Dominated iff some other model has BOTH:

    - strictly higher ``mean_normalized_score``, AND
    - strictly lower ``assistant_output_tokens_per_conversation``.

    The verbosity axis is OUTPUT tokens per conversation, not the
    total billing figure (input + output): input is dominated by
    prompt context size and would mask the verbosity signal a thinking
    model produces.

    Strict comparisons mean ties keep both models on the frontier (a
    duplicate pair is not "dominated" by its twin).

    Models with no measurable score or zero output/conv are excluded
    from the frontier (they can't be plotted on the Pareto chart).
    """
    for rollup in rollups:
        if rollup.mean_normalized_score is None or rollup.assistant_output_tokens_per_conversation <= 0:
            rollup.is_pareto_frontier = False
            continue
        dominated = False
        for other in rollups:
            if other is rollup:
                continue
            if other.mean_normalized_score is None or other.assistant_output_tokens_per_conversation <= 0:
                continue
            if (
                other.mean_normalized_score > rollup.mean_normalized_score
                and other.assistant_output_tokens_per_conversation < rollup.assistant_output_tokens_per_conversation
            ):
                dominated = True
                break
        rollup.is_pareto_frontier = not dominated


def _build_rollup(
    entry: ComparisonEntry,
    cells_for_run: list[ComparisonCell],
    manifest: dict[str, Any],
) -> ModelRollup:
    """Compute the per-model rollup from cells + the manifest's sim_health block.

    ``mean_normalized_score`` uses **per-locale mean, then mean across
    locales** (rather than a flat mean over all measured cells). With even
    coverage (every locale × capability cell is measured for every model)
    this gives the same number as a flat mean. With UNEVEN coverage —
    e.g. a model evaluated on 8 cells in en_US but only 3 in fr_FR — it
    equal-weights each locale rather than letting the locale with more
    cells dominate. ``simulation_reliability`` stays excluded from the
    rollup (process-health metric, surfaced in Layer 0 instead).
    """
    n_eligible = sum(1 for c in cells_for_run if c.capability not in ROLLUP_EXCLUDED_CAPABILITIES)
    n_ready = 0
    n_blocked = 0
    n_missing = 0
    n_insufficient = 0
    n_gaps = 0
    # Eligible-only ready cells: same exclusion discipline as
    # ``n_gaps`` so the cleared-rate denominator (n_cells_eligible)
    # and numerator agree on what's in scope.
    n_ready_eligible = 0

    # Per-locale aggregation: each locale contributes one mean to the
    # final score, weighted equally. Cells with simulation_reliability
    # capability or non-measured states are skipped.
    per_locale_scores: dict[str, list[float]] = {}
    n_measured = 0

    for cell in cells_for_run:
        if cell.state == "ready":
            n_ready += 1
        elif cell.state == "blocked":
            n_blocked += 1
        elif cell.state == "missing":
            n_missing += 1
        elif cell.state == "insufficient_evidence":
            n_insufficient += 1
        if cell.capability in ROLLUP_EXCLUDED_CAPABILITIES:
            continue
        if cell.state == "ready":
            n_ready_eligible += 1
        if cell.state == "blocked":
            n_gaps += 1
        if cell.state in MEASURED_STATES and cell.normalized_score is not None:
            n_measured += 1
            per_locale_scores.setdefault(cell.locale, []).append(cell.normalized_score)

    locale_means = [sum(scores) / len(scores) for scores in per_locale_scores.values() if scores]
    mean_score = sum(locale_means) / len(locale_means) if locale_means else None

    # Sim health (Layer 0) — may be absent on very old manifests.
    sim_health = manifest.get("sim_health") or {}
    has_sim_health = bool(sim_health.get("n_trajectories"))
    sim_status_ok_rate: float | None = None
    sim_status_failure_rate: float | None = None
    persona_grounding_rate: float | None = None
    early_stop_rate: float | None = None
    if has_sim_health:
        status_counts = sim_health.get("status_counts") or {}
        n_traj_sh = sim_health.get("n_trajectories", 0) or 0
        if n_traj_sh > 0:
            ok_count = status_counts.get("ok", 0) + status_counts.get("completed_with_warnings", 0)
            sim_status_ok_rate = ok_count / n_traj_sh
            # Failure rate is the complement; clamp at 0 in case of
            # arithmetic precision creep where ok+warnings > traj.
            sim_status_failure_rate = max(0.0, 1.0 - sim_status_ok_rate)
        pg = sim_health.get("persona_grounding_rate")
        if isinstance(pg, (int, float)):
            persona_grounding_rate = float(pg)
        es = sim_health.get("early_stop_rate")
        if isinstance(es, (int, float)):
            early_stop_rate = float(es)

    # Verbosity. Output (not input + output) is the right scope: input
    # tokens reflect prompt context size, while verbosity is "how much
    # does the model SAY per conversation". Including input would
    # dilute the signal -- a model with thinking enabled doubles its
    # output but its total only grows ~15% because input dominates.
    # Use manifest n_trajectories (simulator-side count) as the
    # denominator — locale-stable, independent of any eval-side sampling.
    n_traj = entry.n_trajectories
    output_per_conv = entry.assistant_output_tokens / n_traj if n_traj > 0 else 0.0
    reasoning_per_conv = entry.assistant_reasoning_tokens / n_traj if n_traj > 0 else 0.0

    # Per-locale assistant tokens per turn, lifted directly from
    # sim_health.response_length_by_locale (already aggregated in
    # reporting/diagnostics.py). Drives Layer 3's right-pane heatmap.
    tokens_per_turn: dict[str, float] = {}
    for row in sim_health.get("response_length_by_locale") or []:
        if not isinstance(row, dict):
            continue
        loc = row.get("locale")
        v = row.get("mean_tokens_per_turn")
        if isinstance(loc, str) and isinstance(v, (int, float)):
            tokens_per_turn[loc] = float(v)

    n_total = len(cells_for_run) or 1
    return ModelRollup(
        run_id=entry.run_id,
        display_label=entry.display_label,
        model_id=entry.model_id,
        n_cells_eligible=n_eligible,
        n_cells_measured=n_measured,
        n_cells_ready=n_ready,
        n_cells_blocked=n_blocked,
        n_cells_missing=n_missing,
        n_cells_insufficient=n_insufficient,
        n_cells_ready_eligible=n_ready_eligible,
        n_gaps=n_gaps,
        pct_ready=n_ready / n_total,
        pct_blocked=n_blocked / n_total,
        pct_missing=n_missing / n_total,
        pct_insufficient=n_insufficient / n_total,
        pct_measured=(n_measured / n_eligible) if n_eligible > 0 else 0.0,
        mean_normalized_score=mean_score,
        n_trajectories=n_traj,
        assistant_total_tokens=entry.assistant_total_tokens,
        assistant_output_tokens_per_conversation=output_per_conv,
        assistant_reasoning_tokens_per_conversation=reasoning_per_conv,
        assistant_tokens_per_turn_by_locale=tokens_per_turn,
        sim_status_ok_rate=sim_status_ok_rate,
        sim_status_failure_rate=sim_status_failure_rate,
        persona_grounding_rate=persona_grounding_rate,
        early_stop_rate=early_stop_rate,
        has_sim_health=has_sim_health,
        # Defensive copy of the manifest list -- sim_health is just
        # ``manifest.get("sim_health") or {}`` so the source list is
        # mutable and shared with the caller. List of dicts; we don't
        # validate field names here (chart layer drops malformed rows).
        failure_taxonomy=list(sim_health.get("failure_taxonomy") or []),
    )


def _detect_apples_to_apples_warnings(
    entries: list[ComparisonEntry],
    manifests: list[dict[str, Any]],
) -> list[str]:
    """Surface cross-run drift as warnings (eval_column mismatch is fail-closed elsewhere)."""
    warnings: list[str] = []

    sample_modes = {e.sample_mode for e in entries if e.sample_mode and e.sample_mode != "unknown"}
    if len(sample_modes) > 1:
        warnings.append(
            f"Mixed sample_mode across runs ({sorted(sample_modes)}); per-cell n "
            "values differ, so leaderboard rankings carry different statistical power."
        )

    prompt_versions = sorted({v for e in entries for v in e.evaluator_prompt_versions})
    if len(prompt_versions) > 1:
        warnings.append(
            f"Judge prompt versions differ across runs ({prompt_versions}); judge "
            "scores from different prompt versions are not comparable. Re-run eval "
            "so every run uses the same version."
        )

    n_trajs = [e.n_trajectories for e in entries if e.n_trajectories > 0]
    if len(n_trajs) >= 2:
        lo, hi = min(n_trajs), max(n_trajs)
        if hi > 0 and (hi - lo) / hi > 0.10:
            warnings.append(
                f"n_trajectories deltas exceed 10% across runs (min={lo}, max={hi}); "
                "rollup denominators differ across models."
            )

    personas = []
    for m in manifests:
        ps = (m.get("persona_summary") or {}).get("personas")
        if isinstance(ps, int) and ps > 0:
            personas.append(ps)
    if len(set(personas)) > 1:
        warnings.append(
            f"persona_summary.personas counts differ across runs "
            f"({sorted(set(personas))}); models were evaluated against different "
            "persona panels — cells are still per-cell comparable, rollups use "
            "different denominators."
        )

    no_sh_runs = [e.run_id for e, m in zip(entries, manifests) if not (m.get("sim_health") or {}).get("n_trajectories")]
    if no_sh_runs:
        warnings.append(
            f"No sim_health captured for run(s) {no_sh_runs}; Layer 0 will "
            "render empty bars for those runs (predates Sim Health bundle)."
        )

    return warnings


def _materialize_cells(
    entries: list[ComparisonEntry],
    manifests: list[dict[str, Any]],
    locales: list[str],
    capabilities: list[tuple[str, str]],
) -> list[ComparisonCell]:
    """Build the full ``(capability x locale x run)`` cube.

    Absent ``(run, capability, locale)`` combinations are emitted as
    ``state="missing"`` cells so the chart axis stays consistent across
    models — a model that skipped a locale renders as a dim "?" bar
    rather than a hole that the eye has to compensate for.
    """
    cells: list[ComparisonCell] = []
    for entry, manifest in zip(entries, manifests):
        per_cell: dict[tuple[str, str], dict[str, Any]] = {}
        for cell in manifest.get("capability_cells") or []:
            key = (cell.get("capability"), cell.get("locale"))
            per_cell[key] = cell

        for cap_id, cap_label in capabilities:
            for locale in locales:
                src = per_cell.get((cap_id, locale))
                if src is None:
                    cells.append(
                        ComparisonCell(
                            capability=cap_id,
                            capability_label=cap_label,
                            locale=locale,
                            run_id=entry.run_id,
                            display_label=entry.display_label,
                            state="missing",
                            normalized_score=None,
                            threshold=None,
                            n=0,
                            n_total=0,
                            failed_critical_axes=[],
                            aggregation_policy="",
                            is_missing=True,
                        )
                    )
                    continue
                cells.append(
                    ComparisonCell(
                        capability=cap_id,
                        capability_label=cap_label,
                        locale=locale,
                        run_id=entry.run_id,
                        display_label=entry.display_label,
                        state=src.get("state", "missing"),
                        normalized_score=src.get("score"),
                        threshold=src.get("threshold"),
                        n=int(src.get("n") or 0),
                        n_total=int(src.get("n_total") or 0),
                        failed_critical_axes=list(src.get("failed_critical_axes") or []),
                        aggregation_policy=src.get("aggregation_policy", ""),
                        is_missing=src.get("state") == "missing",
                    )
                )
    return cells


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _build_entry(
    manifest: dict[str, Any],
    manifest_path: Path,
    labels: dict[str, str] | None = None,
) -> ComparisonEntry:
    run_id = str(manifest.get("run_id") or manifest_path.parent.name.removeprefix("run="))
    model_id = manifest.get("model_id")
    custom_label = (labels or {}).get(run_id)
    display_label = custom_label or _short_label(model_id)
    sample_mode = manifest.get("sample_mode") or "unknown"
    # Older manifests (pre-three-row Sim Health redesign) lack
    # ``assistant_output_tokens`` and the user-side token fields;
    # default to 0 so legacy reports still load without crashing.
    # ``mean_turns_per_conversation`` is sourced from sim_health,
    # which itself can be missing on very old runs (default 0.0).
    sim_health = manifest.get("sim_health") or {}
    counters_means = sim_health.get("counters_means") or {}
    mean_turns = counters_means.get("n_turns")
    return ComparisonEntry(
        run_id=run_id,
        model_id=model_id,
        display_label=display_label,
        eval_column=manifest.get("eval_column", ""),
        n_trajectories=int(manifest.get("n_trajectories") or 0),
        n_evaluated=int(manifest.get("n_evaluated") or 0),
        assistant_total_tokens=int(manifest.get("assistant_total_tokens") or 0),
        assistant_output_tokens=int(manifest.get("assistant_output_tokens") or 0),
        assistant_reasoning_tokens=int(manifest.get("assistant_reasoning_tokens") or 0),
        user_output_tokens=int(manifest.get("user_output_tokens") or 0),
        user_reasoning_tokens=int(manifest.get("user_reasoning_tokens") or 0),
        mean_turns_per_conversation=float(mean_turns) if isinstance(mean_turns, (int, float)) else 0.0,
        sample_mode=sample_mode,
        manifest_path=str(manifest_path),
        evaluator_prompt_versions=[str(v) for v in manifest.get("evaluator_prompt_versions") or []],
    )


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_comparison_report(
    manifest_paths: Iterable[Path | str],
    *,
    comparison_id: str | None = None,
    labels: dict[str, str] | None = None,
    locales: Iterable[str] | None = None,
) -> ComparisonReport:
    """Build a :class:`ComparisonReport` from N per-run manifest paths.

    ``locales`` subsets and orders the locale axis; ``None`` uses every locale
    any selected run reports. Note that this is not merely cosmetic: the
    leaderboard score is a mean over per-locale means (see :func:`_build_rollup`),
    so restricting the locales restricts what each model is being scored ON.
    That is usually the point -- "who is best on the Indic locales" is a
    different question from "who is best overall" -- but it does mean two
    comparisons built over different locale sets are not comparable to each
    other.

    Edge cases (per the plan doc):

    - 0 manifests -> ``ValueError`` ("no runs to compare; ...").
    - 1 manifest  -> ``ValueError`` ("comparison requires >=2 runs; ...").
    - Mixed ``eval_column`` -> ``ValueError`` (fail closed; the cells
      aren't drawn from the same evaluator output column).
    - A requested locale no run reports -> ``ValueError``, rather than a
      silently blank column that reads as "every model failed here".

    All other drift dimensions (sample_mode, persona panel, missing
    sim_health) surface as warnings on
    ``ComparisonReport.apples_to_apples_warnings``.

    ``comparison_id`` defaults to ``f"cmp_{int(time.time())}"`` —
    distinct from the trajectory ``run=<epoch>`` format so a comparison
    artifact directory is never confused with a single-run artifact.
    """
    paths = [Path(p) for p in manifest_paths]
    if not paths:
        raise ValueError(
            "no runs to compare; build a per-run report from cell 14 of notebooks/02_evaluate_simulation.ipynb first"
        )
    if len(paths) == 1:
        raise ValueError(
            f"comparison requires >=2 runs; found {len(paths)}. The per-run "
            "dashboard already exists for the single-model case."
        )

    manifests = [_load_manifest(p) for p in paths]
    entries = [_build_entry(m, p, labels=labels) for m, p in zip(manifests, paths)]

    eval_columns = {e.eval_column for e in entries if e.eval_column}
    if len(eval_columns) > 1:
        raise ValueError(
            f"Mixed eval_column across runs ({sorted(eval_columns)}); the cells "
            "are not drawn from the same evaluator output column, so a join "
            "would silently align mismatched data. Re-evaluate the runs under "
            "a consistent eval_column before comparing."
        )

    _disambiguate_labels(entries)

    capabilities: list[tuple[str, str]] = [(d.id, d.label) for d in capability_definitions()]

    locale_set: set[str] = set()
    for m in manifests:
        for loc in m.get("locales") or []:
            if isinstance(loc, str) and loc:
                locale_set.add(loc)
        for cell in m.get("capability_cells") or []:
            loc = cell.get("locale")
            if isinstance(loc, str) and loc:
                locale_set.add(loc)

    if locales is None:
        selected_locales = sorted(locale_set)
    else:
        # Caller order is preserved (it is a display axis), duplicates dropped.
        requested = list(dict.fromkeys(locales))
        unknown = [loc for loc in requested if loc not in locale_set]
        if unknown:
            raise ValueError(f"locales not present in any selected run: {unknown}. Available: {sorted(locale_set)}")
        selected_locales = requested
        if not selected_locales:
            raise ValueError("locales was empty; pass None to compare every locale")
    locales = selected_locales

    cells = _materialize_cells(entries, manifests, locales, capabilities)

    cells_by_run: dict[str, list[ComparisonCell]] = {}
    for cell in cells:
        cells_by_run.setdefault(cell.run_id, []).append(cell)
    rollups = [
        _build_rollup(entry, cells_by_run.get(entry.run_id, []), manifest)
        for entry, manifest in zip(entries, manifests)
    ]
    _compute_pareto_frontier(rollups)

    warnings = _detect_apples_to_apples_warnings(entries, manifests)

    if comparison_id is None:
        comparison_id = f"cmp_{int(time.time())}"

    return ComparisonReport(
        comparison_id=comparison_id,
        entries=entries,
        cells=cells,
        locales=locales,
        capabilities=capabilities,
        model_summary=rollups,
        apples_to_apples_warnings=warnings,
    )
