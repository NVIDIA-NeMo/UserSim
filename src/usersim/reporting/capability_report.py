# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capability-gap reporting primitives for the evaluation dashboard.

This module builds the shared run-level contract for a capability-first
dashboard:

- coverage by locale and probe,
- capability x locale cells with evidence state,
- a prioritized triage queue,
- a compact manifest that can later be compared across model runs.

It deliberately stays pandas/JSON-only. The interactive report shell can render
these records with React/Vega without re-implementing any evaluator logic.
"""

from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Iterator

from usersim.engine.core.missing import is_missing
from usersim.taxonomy.capabilities import (
    CapabilityDefinition,
    axis_description,
    capability_definitions,
    implementation_refs_for_capability,
    probe_label_for_capability,
)
from usersim.taxonomy.eval_cell import (
    decode_cell as _decode_eval_cell,
)
from usersim.taxonomy.eval_cell import (
    normalize_axis_score as _normalize_axis_score,
)
from usersim.taxonomy.eval_cell import (
    score_for_source as _score_for_source,
)
from usersim.taxonomy.eval_cell import (
    scorer_state as _scorer_state,
)


def capability_order() -> tuple[tuple[str, str], ...]:
    """``(id, label)`` for every capability, in report order.

    Resolved per call rather than frozen at import, so a capability
    contributed by an installed package reaches the rendered cells.
    """
    return tuple((definition.id, definition.label) for definition in capability_definitions())


def _capability_label(capability: str) -> str:
    """Display label for a capability id, falling back to the id itself."""
    for cap_id, label in capability_order():
        if cap_id == capability:
            return label
    return capability


@dataclass(slots=True)
class EvidenceFinding:
    """One per-axis verdict for a single trajectory.

    Lets the dashboard surface *what specifically failed* for this conversation
    instead of dumping the full multi-axis judge text into one truncated blob.
    """

    axis: str
    source: str  # "judge" or "scorer:<scorer_name>"
    score: float | None
    threshold: float | None  # raw threshold (same scale as score)
    failed: bool
    is_critical: bool
    reasoning: str = ""
    # What the axis measures. Deterministic scorers compute a rate and have no
    # reasoning to give, so the card used to read "No reasoning provided by this
    # scorer" -- which tells a reviewer nothing about why 0% is bad, or even what
    # was counted. Fall back to this instead.
    description: str = ""


@dataclass(slots=True)
class EvidenceSnippet:
    trajectory_id: str
    probe_family: str = ""
    probe_type: str = ""
    probe_variant: str = ""
    probe_description: str = ""
    reasoning: str = ""  # legacy joined fallback (kept for v1 render path)
    findings: list[EvidenceFinding] = field(default_factory=list)
    turns: list[dict[str, str]] = field(default_factory=list)
    # In-sim assistant-quality judge ratings, collapsed to ONE entry per
    # assistant turn: the FINAL attempt — the verdict on the candidate
    # actually in the transcript. (With assistant resampling the raw
    # conversation_metadata.assistant_judge_ratings list holds one entry
    # per attempt; that audit trail stays on the row, this field is the
    # dashboard-facing summary.) Each entry:
    # {turn_idx, attempt, rating, explanation, success}.
    per_turn_judge: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class CoverageCell:
    locale: str
    probe_family: str
    trajectory_count: int = 0
    eval_count: int = 0
    scorer_ok_count: int = 0
    scorer_error_count: int = 0
    scorer_not_applicable_count: int = 0


@dataclass(slots=True)
class CapabilityCell:
    capability: str
    label: str
    description: str
    locale: str
    state: str
    score: float | None = None
    threshold: float | None = None
    margin: float | None = None
    n: int = 0
    # Total trajectories in this locale that *could* have contributed
    # to this capability (regardless of whether the scorer was applicable
    # or produced a usable score). Lets the dashboard show ``n=37/40`` so
    # a coverage gap (judge errors / scorer-skipped trajectories / missing
    # axis on this probe family) is visible at a glance instead of being
    # silently rounded into ``Inconclusive`` or ``Untested``.
    n_total: int = 0
    coverage_state: str = "missing"
    evidence_level: str = "missing"
    source: str = ""
    source_probes: list[str] = field(default_factory=list)
    source_scorers: list[str] = field(default_factory=list)
    source_axes: list[str] = field(default_factory=list)
    critical_axes: list[str] = field(default_factory=list)
    failed_critical_axes: list[str] = field(default_factory=list)
    implementation_refs: list[str] = field(default_factory=list)
    axis_thresholds: dict[str, float] = field(default_factory=dict)
    # Mean normalized score PER AXIS. ``score`` is the mean across a
    # capability's axes, which hides the one that is actually broken: finance
    # task success read 84% while its must-pass tool_selection_rate sat at 54%,
    # averaged up by ordering (1.00), no-unauthorized (0.98) and recall (0.91).
    # ``failed_critical_axes`` named the culprit but carried no value, so the
    # dashboard could not show how far off it was. Keep both here so a reviewer
    # sees "retrieval 0.91, action 0.54" without opening the parquet.
    axis_scores: dict[str, float] = field(default_factory=dict)
    aggregation_policy: str = ""
    evidence_policy: str = ""
    blocker: str = ""
    trajectory_ids: list[str] = field(default_factory=list)
    evidence_snippets: list[EvidenceSnippet] = field(default_factory=list)
    next_action: str = ""


@dataclass(slots=True)
class TriageItem:
    priority: str
    capability: str
    locale: str
    reason: str
    suggested_action: str
    trajectory_ids: list[str] = field(default_factory=list)
    evidence_snippets: list[EvidenceSnippet] = field(default_factory=list)


@dataclass(slots=True)
class CapabilityReport:
    run_id: str | None
    eval_column: str
    model_id: str | None
    sample_mode: str
    n_trajectories: int
    n_evaluated: int
    capability_cells: list[CapabilityCell]
    coverage_cells: list[CoverageCell]
    triage_queue: list[TriageItem]
    locales: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    persona_summary: dict[str, Any] = field(default_factory=dict)
    report_paths: dict[str, str] = field(default_factory=dict)
    # Aggregate model-call token usage from simulation_outcome.per_model_*.
    # In-sim only (user agent + assistant under test + in-sim judges +
    # api_response synthesis). Out-of-sim evaluator judge tokens are not
    # tracked on the trajectory row and are not included here.
    #
    # ``conversation_output_tokens`` narrows ``total_output_tokens`` to
    # just the two parties IN the conversation: ``user_model`` +
    # ``assistant_model`` output. Excludes ``api_response_model`` (a
    # tool-response synthesiser the assistant calls; output is consumed
    # by the assistant's next turn but the synthesiser itself isn't a
    # conversation participant) and the in-sim scaffolding aliases
    # (``judge_model`` / ``summary_model``). It's the dashboard's hero
    # token figure -- the actual cost of the conversations that landed
    # on disk.
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    conversation_output_tokens: int = 0
    assistant_output_tokens: int = 0
    # Per-alias decompositions for the dashboard's three Sim Health
    # token rows: Overall (total / input / output / reasoning), User
    # Tokens (total / output / reasoning / conversation), Assistant
    # Tokens (total / output / reasoning / conversation). All four
    # decompose cleanly:
    #   total = input + output
    #   output = reasoning + conversation
    # ``conversation`` here means visible output (output - reasoning) at
    # the alias level, distinct from the older ``conversation_output_
    # tokens`` field (which is gross output across user+assistant+api).
    assistant_total_tokens: int = 0
    assistant_reasoning_tokens: int = 0
    assistant_conversation_tokens: int = 0
    user_total_tokens: int = 0
    user_output_tokens: int = 0
    user_reasoning_tokens: int = 0
    user_conversation_tokens: int = 0
    # Support actors -- api_response_model (tool-response synthesiser),
    # judge_model (capitulation classifier), summary_model (context
    # compressor). Same algebra as User / Assistant rows; surfaced for
    # symmetry. judge / summary typically show reasoning=0 since Phase B
    # tiktoken estimation is unavailable for them (no visible output on
    # the trajectory).
    api_response_total_tokens: int = 0
    api_response_output_tokens: int = 0
    api_response_reasoning_tokens: int = 0
    api_response_conversation_tokens: int = 0
    judge_total_tokens: int = 0
    judge_output_tokens: int = 0
    judge_reasoning_tokens: int = 0
    judge_conversation_tokens: int = 0
    summary_total_tokens: int = 0
    summary_output_tokens: int = 0
    summary_reasoning_tokens: int = 0
    summary_conversation_tokens: int = 0
    # Reasoning-token subset (invisible thinking content). Already counted
    # inside ``total_output_tokens`` -- this surfaces the breakdown so the
    # dashboard can show how much "thinking" each alias did. Estimated
    # post-hoc as ``output_tokens - tiktoken(visible_content)`` per row,
    # then aggregated and clamped to zero per-alias when below the 5%
    # noise floor (tokenizer drift swallows estimates at that scale).
    # Sources:  ``estimated`` (above noise floor) | ``unavailable``
    # (below floor or no visible content for the alias). See
    # ``reporting/_reasoning.py`` and ``reporting/resource.py``.
    reasoning_tokens_by_alias: dict[str, int] = field(default_factory=dict)
    reasoning_source_by_alias: dict[str, str] = field(default_factory=dict)
    total_reasoning_tokens: int = 0
    # Sim-Health diagnostics (cheap, no LLM calls): status counts,
    # failure-mode taxonomy with per-actor attribution, D1-D4 + MATTR
    # behavioral diagnostics, persona-grounding rate, early-stop rate,
    # resource profile per alias. Surfaces "did the simulator do its
    # job?" — a different signal from the assistant-under-test scorers.
    # Stored as a plain dict (the SimHealthBundle.to_dict() output) so
    # the JSON artifact is self-contained.
    sim_health: dict[str, Any] = field(default_factory=dict)
    # Run-manifest-derived run-level metadata. ``model_identities`` is
    # the alias → identity dict for every model that participated in the
    # simulation (assistant under test + user agent + in-sim judge +
    # api_response synthesizer + summary). ``metadata_source`` records
    # how ``model_id`` was derived so the dashboard can render an
    # explicit "metadata not captured" state for old runs instead of
    # silently falling back to a misleading default.
    model_identities: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata_source: str = "unknown"  # "manifest" | "explicit_kwarg" | "unknown"
    # Judge alias -> the models that scored this report's rows, read from each
    # evaluation cell's envelope rather than the run manifest, which records
    # the models config as it stood when the simulation ran. Empty for
    # evaluations written before envelopes recorded ``judge_models``.
    evaluator_models: dict[str, list[str]] = field(default_factory=dict)
    # The judge prompt versions in the evaluation cells' envelopes, so a
    # comparison can tell when its runs were judged by different prompts.
    evaluator_prompt_versions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "eval_column": self.eval_column,
            "model_id": self.model_id,
            "sample_mode": self.sample_mode,
            "n_trajectories": self.n_trajectories,
            "n_evaluated": self.n_evaluated,
            "locales": list(self.locales),
            "languages": list(self.languages),
            "persona_summary": dict(self.persona_summary),
            "capability_cells": [asdict(c) for c in self.capability_cells],
            "coverage_cells": [asdict(c) for c in self.coverage_cells],
            "triage_queue": [asdict(t) for t in self.triage_queue],
            "report_paths": dict(self.report_paths),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "conversation_output_tokens": self.conversation_output_tokens,
            "assistant_output_tokens": self.assistant_output_tokens,
            "assistant_total_tokens": self.assistant_total_tokens,
            "assistant_reasoning_tokens": self.assistant_reasoning_tokens,
            "assistant_conversation_tokens": self.assistant_conversation_tokens,
            "user_total_tokens": self.user_total_tokens,
            "user_output_tokens": self.user_output_tokens,
            "user_reasoning_tokens": self.user_reasoning_tokens,
            "user_conversation_tokens": self.user_conversation_tokens,
            "api_response_total_tokens": self.api_response_total_tokens,
            "api_response_output_tokens": self.api_response_output_tokens,
            "api_response_reasoning_tokens": self.api_response_reasoning_tokens,
            "api_response_conversation_tokens": self.api_response_conversation_tokens,
            "judge_total_tokens": self.judge_total_tokens,
            "judge_output_tokens": self.judge_output_tokens,
            "judge_reasoning_tokens": self.judge_reasoning_tokens,
            "judge_conversation_tokens": self.judge_conversation_tokens,
            "summary_total_tokens": self.summary_total_tokens,
            "summary_output_tokens": self.summary_output_tokens,
            "summary_reasoning_tokens": self.summary_reasoning_tokens,
            "summary_conversation_tokens": self.summary_conversation_tokens,
            "reasoning_tokens_by_alias": dict(self.reasoning_tokens_by_alias),
            "reasoning_source_by_alias": dict(self.reasoning_source_by_alias),
            "total_reasoning_tokens": self.total_reasoning_tokens,
            "sim_health": dict(self.sim_health),
            "model_identities": dict(self.model_identities),
            "metadata_source": self.metadata_source,
            "evaluator_models": {alias: list(models) for alias, models in self.evaluator_models.items()},
            "evaluator_prompt_versions": list(self.evaluator_prompt_versions),
        }

    def print_actors(self) -> None:
        """Confirm-at-a-glance printout of captured actor models.

        Branches on ``self.metadata_source``:

        - ``"manifest"``       — prints the captured ``(alias, model, provider)``
          tuples for the five tracked aliases (user / assistant /
          api_response / judge / summary). Aliases missing from the manifest
          (e.g. older runs without an api_response_model) are skipped.
        - ``"explicit_kwarg"`` — prints the model_id override.
        - ``"unknown"``        — prints the missing-manifest banner so the
          reviewer sees the gap rather than a misleading default.

        Then the models that scored the evaluated rows, from their envelopes,
        or a line saying the rows do not record them.
        """
        if self.metadata_source == "manifest" and self.model_identities:
            print(f"\nActors (captured from simulation run {self.run_id}):")
            for alias in (
                "user_model",
                "assistant_model",
                "api_response_model",
                "judge_model",
                "summary_model",
            ):
                identity = self.model_identities.get(alias)
                if identity is None:
                    continue
                provider = identity.get("provider") or "?"
                model = identity.get("model") or "?"
                print(f"  {alias:20s} {model:40s} provider={provider}")
        elif self.metadata_source == "explicit_kwarg":
            print(f"\nActors: explicit override (no manifest); model under test = {self.model_id}")
        else:
            print(
                "\nActors: metadata not captured for this run (predates manifest-v1; re-run the simulator to populate)."
            )
        if self.evaluator_models:
            print("Scored by (from the evaluation rows):")
            for alias, models in self.evaluator_models.items():
                print(f"  {alias:20s} {', '.join(models)}")
        elif self.n_evaluated:
            print("Scored by: not recorded in these evaluation rows.")


def build_capability_report(
    traj_df: Any,
    eval_df: Any | None = None,
    *,
    eval_column: str,
    run_id: str | None = None,
    model_id: str | None = None,
    sample_mode: str = "full",
    min_n: int = 3,
    trajectory_root: Any = None,
) -> CapabilityReport:
    """Build the run-level capability-gap report contract.

    ``model_id`` derivation precedence (per the simulation-metadata-capture
    plan, Phase 2):

    1. Run manifest at ``<trajectory_root>/run=<run_id>/_manifest.json``
       — the canonical source of truth for run-level metadata. When
       present, ``model_id`` defaults to
       ``manifest.models["assistant_model"].model`` and
       ``metadata_source = "manifest"``. Every alias (user / assistant /
       judge / api_response / summary) lands in
       ``CapabilityReport.model_identities``.
    2. Explicit ``model_id=`` kwarg from the caller — legacy override
       path. Surfaces as ``metadata_source = "explicit_kwarg"``.
    3. ``None`` — surfaces as ``metadata_source = "unknown"`` so the
       dashboard can render an explicit "metadata not captured" banner
       instead of silently inventing an attribution.

    There is deliberately NO fallback to "the eval-notebook's currently-
    loaded MODELS config" — that exact silent fallback is the bug the
    plan fixes.

    ``trajectory_root`` is the root of the trajectory dataset (e.g.
    ``output/trajectories``); the manifest is looked up under
    ``<root>/run=<run_id>/_manifest.json``.
    """
    import pandas as pd

    if not isinstance(traj_df, pd.DataFrame):
        raise TypeError(f"build_capability_report expects traj_df as a pandas DataFrame, got {type(traj_df).__name__}")
    if eval_df is not None and not isinstance(eval_df, pd.DataFrame):
        raise TypeError(
            f"build_capability_report expects eval_df as a pandas DataFrame or None, got {type(eval_df).__name__}"
        )

    # An evaluation parquet may carry the assistant score under either of
    # these names instead of the canonical one; alias it so the parquet
    # renders without re-evaluation. Mutates a copy, so the column on disk
    # is left as written.
    ALTERNATE_EVAL_COLUMNS = ("eval_v1", "assistant_eval_v2")
    if eval_df is not None and eval_column not in eval_df.columns:
        for legacy in ALTERNATE_EVAL_COLUMNS:
            if legacy in eval_df.columns:
                eval_df = eval_df.copy()
                eval_df[eval_column] = eval_df[legacy]
                break

    joined = _join_eval(traj_df, eval_df, eval_column)
    locales = _ordered_values(traj_df.get("locale", []))
    coverage_cells = _build_coverage_cells(traj_df, joined, eval_column)
    capability_cells = _build_capability_cells(joined, locales, eval_column=eval_column, min_n=min_n)
    triage = _build_triage(capability_cells)
    # In-sim model-call token totals (defensive: absent column → zeros).
    from usersim.reporting.resource import aggregate_from_dataframe

    resource_profile = aggregate_from_dataframe(traj_df)
    # Sim-Health bundle: cheap diagnostics on the simulator side itself.
    # Defensive — empty bundle on builder failure so a malformed
    # trajectory never breaks the report build.
    from usersim.reporting.sim_health import build_sim_health

    try:
        sim_health_dict = build_sim_health(traj_df).to_dict()
    except Exception:  # pragma: no cover — defensive
        sim_health_dict = {}

    # Manifest-derived run-level metadata (Phase 2). Falls back to the
    # explicit kwarg, then to "unknown" — never to a silent default.
    resolved_model_id, model_identities, metadata_source = _resolve_run_metadata(
        run_id=run_id,
        trajectory_root=trajectory_root,
        explicit_model_id=model_id,
    )

    return CapabilityReport(
        run_id=run_id,
        eval_column=eval_column,
        model_id=resolved_model_id,
        sample_mode=sample_mode,
        n_trajectories=len(traj_df),
        n_evaluated=(int(joined[eval_column].notna().sum()) if eval_column in joined.columns else 0),
        locales=locales,
        languages=_ordered_values(traj_df.get("conversation_language", [])),
        persona_summary=_build_persona_summary(traj_df),
        capability_cells=capability_cells,
        coverage_cells=coverage_cells,
        triage_queue=triage,
        total_input_tokens=resource_profile.total_input_tokens,
        total_output_tokens=resource_profile.total_output_tokens,
        total_tokens=resource_profile.total_tokens,
        conversation_output_tokens=resource_profile.conversation_output_tokens,
        assistant_output_tokens=resource_profile.assistant_output_tokens,
        assistant_total_tokens=resource_profile.assistant_total_tokens,
        assistant_reasoning_tokens=resource_profile.assistant_reasoning_tokens,
        assistant_conversation_tokens=resource_profile.assistant_conversation_tokens,
        user_total_tokens=resource_profile.user_total_tokens,
        user_output_tokens=resource_profile.user_output_tokens,
        user_reasoning_tokens=resource_profile.user_reasoning_tokens,
        user_conversation_tokens=resource_profile.user_conversation_tokens,
        api_response_total_tokens=resource_profile.api_response_total_tokens,
        api_response_output_tokens=resource_profile.api_response_output_tokens,
        api_response_reasoning_tokens=resource_profile.api_response_reasoning_tokens,
        api_response_conversation_tokens=resource_profile.api_response_conversation_tokens,
        judge_total_tokens=resource_profile.judge_total_tokens,
        judge_output_tokens=resource_profile.judge_output_tokens,
        judge_reasoning_tokens=resource_profile.judge_reasoning_tokens,
        judge_conversation_tokens=resource_profile.judge_conversation_tokens,
        summary_total_tokens=resource_profile.summary_total_tokens,
        summary_output_tokens=resource_profile.summary_output_tokens,
        summary_reasoning_tokens=resource_profile.summary_reasoning_tokens,
        summary_conversation_tokens=resource_profile.summary_conversation_tokens,
        reasoning_tokens_by_alias=dict(resource_profile.reasoning_tokens_by_alias),
        reasoning_source_by_alias=dict(resource_profile.reasoning_source_by_alias),
        total_reasoning_tokens=resource_profile.total_reasoning_tokens,
        sim_health=sim_health_dict,
        model_identities=model_identities,
        metadata_source=metadata_source,
        evaluator_models=_evaluator_models(joined, eval_column),
        evaluator_prompt_versions=_evaluator_prompt_versions(joined, eval_column),
    )


def _envelopes(joined: Any, eval_column: str) -> Iterator[dict[str, Any]]:
    """The envelope of every evaluation cell in ``joined``."""
    if eval_column not in getattr(joined, "columns", []):
        return
    for raw in joined[eval_column].dropna():
        yield _decode_eval_cell(raw).get("envelope") or {}


def _evaluator_prompt_versions(joined: Any, eval_column: str) -> list[str]:
    """The judge prompt versions recorded in the evaluation cells' envelopes."""
    versions = {envelope.get("prompt_version") for envelope in _envelopes(joined, eval_column)}
    return sorted(str(version) for version in versions if version)


def _evaluator_models(joined: Any, eval_column: str) -> dict[str, list[str]]:
    """Judge alias -> the models behind it in the evaluation cells' envelopes."""
    models: dict[str, set[str]] = {}
    for envelope in _envelopes(joined, eval_column):
        for alias, model in zip(envelope.get("judge_aliases") or [], envelope.get("judge_models") or []):
            models.setdefault(str(alias), set()).add(str(model))
    return {alias: sorted(names) for alias, names in sorted(models.items())}


def _resolve_run_metadata(
    *,
    run_id: str | None,
    trajectory_root: Any,
    explicit_model_id: str | None,
) -> tuple[str | None, dict[str, dict[str, Any]], str]:
    """Resolve ``(model_id, model_identities, metadata_source)`` for the report.

    Tries the run manifest first; falls back to the explicit kwarg; ends
    at "unknown" if nothing is available. NEVER consults any other source
    (e.g. the caller's currently-loaded MODELS config) — see the plan's
    Phase 2 derivation precedence.
    """
    if run_id is not None and trajectory_root is not None:
        try:
            from usersim.engine.core.manifest import load_run_manifest

            manifest = load_run_manifest(run_id, trajectory_root)
        except Exception:  # pragma: no cover — defensive
            manifest = None
        if manifest is not None and manifest.models:
            identities = {alias: asdict(identity) for alias, identity in manifest.models.items()}
            assistant = manifest.models.get("assistant_model")
            return (
                assistant.model if assistant is not None else None,
                identities,
                "manifest",
            )

    if explicit_model_id is not None:
        return explicit_model_id, {}, "explicit_kwarg"

    return None, {}, "unknown"


def render_capability_report_html(report: CapabilityReport) -> str:
    """Render a static capability-gap dashboard.

    This is the static no-build renderer. The same JSON contract is also
    consumed by the React/Vega dashboard in ``reporting/dashboard.py``.
    """
    cells = report.capability_cells
    locales = sorted({c.locale for c in cells})
    capabilities = [key for key, _ in capability_order()]
    by_key = {(c.capability, c.locale): c for c in cells}

    rows = []
    for cap in capabilities:
        row = [f"<th>{_esc(_capability_label(cap))}</th>"]
        for loc in locales:
            cell = by_key.get((cap, loc))
            if cell is None:
                row.append('<td class="state-missing">not measured</td>')
                continue
            row.append(_capability_td(cell))
        rows.append("<tr>" + "".join(row) + "</tr>")

    triage_rows = (
        "\n".join(
            "<tr>"
            f"<td>{_esc(item.priority)}</td>"
            f"<td>{_esc(_capability_label(item.capability))}</td>"
            f"<td>{_esc(item.locale)}</td>"
            f"<td>{_esc(item.reason)}</td>"
            f"<td>{_esc(item.suggested_action)}</td>"
            f"<td>{_esc(', '.join(item.trajectory_ids[:5]))}</td>"
            "</tr>"
            for item in report.triage_queue[:25]
        )
        or '<tr><td colspan="6"><em>No triage items.</em></td></tr>'
    )

    coverage_rows = "\n".join(
        "<tr>"
        f"<td>{_esc(c.locale)}</td>"
        f"<td>{_esc(c.probe_family)}</td>"
        f"<td>{c.trajectory_count}</td>"
        f"<td>{c.eval_count}</td>"
        f"<td>{c.scorer_ok_count}</td>"
        f"<td>{c.scorer_error_count}</td>"
        f"<td>{c.scorer_not_applicable_count}</td>"
        "</tr>"
        for c in report.coverage_cells
    )
    evidence_html = _render_full_width_evidence(report.triage_queue)
    resources_html = _render_resources_section(report)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NeMo User Sim Capability Report</title>
<style>
{_STYLE}
</style>
</head>
<body>
<main class="capability-report">
<header>
  <h1>NeMo User Sim Capability Report</h1>
  <div class="meta-pills">
    <span>Run {_esc(report.run_id)}</span>
    <span>Model {_esc(report.model_id)}</span>
    <span>Locales {len(report.locales)}</span>
    <span>Languages {len(report.languages)}</span>
    <span>Sample {_esc(report.sample_mode)}</span>
  </div>
  <p class="subtle">evaluated={report.n_evaluated}/{report.n_trajectories}</p>
</header>

<section>
  <h2>Bird's-eye capability heatmap</h2>
  <p class="subtle">Color shows status; dim cells are missing or underpowered evidence. Open a cell for evidence and next action.</p>
  <table class="heatmap">
    <thead><tr><th>Capability</th>{"".join(f"<th>{_esc(loc)}</th>" for loc in locales)}</tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
</section>

<section>
  <h2>Top triage queue</h2>
  <table>
    <thead><tr><th>Priority</th><th>Capability</th><th>Locale</th><th>Reason</th><th>Suggested action</th><th>Examples</th></tr></thead>
    <tbody>{triage_rows}</tbody>
  </table>
</section>

<section>
  <h2>Conversation evidence</h2>
  <p class="subtle">Full-width examples for the highest-priority findings. Reasoning is pulled from evaluator outputs when available.</p>
  {evidence_html}
</section>

{resources_html}

<details>
  <summary>Coverage matrix</summary>
  <table>
    <thead><tr><th>Locale</th><th>Probe</th><th>Traj</th><th>Eval</th><th>Scorer ok</th><th>Scorer error</th><th>Not applicable</th></tr></thead>
    <tbody>{coverage_rows}</tbody>
  </table>
</details>
</main>
</body>
</html>
"""


def _join_eval(traj_df: Any, eval_df: Any | None, eval_column: str):
    if eval_df is None or eval_column not in getattr(eval_df, "columns", []):
        return traj_df.copy()
    if "trajectory_id" not in traj_df.columns or "trajectory_id" not in eval_df.columns:
        return eval_df.copy()
    keep_cols = [c for c in traj_df.columns if c == "trajectory_id" or c not in eval_df.columns]
    return eval_df.merge(traj_df[keep_cols], on="trajectory_id", how="left")


def _build_coverage_cells(traj_df: Any, joined: Any, eval_column: str) -> list[CoverageCell]:
    traj_counts = _group_counts(traj_df, ["locale", "probe_family"])
    eval_counts = (
        _group_counts(joined[joined[eval_column].notna()], ["locale", "probe_family"])
        if eval_column in joined.columns
        else {}
    )
    scorer_counts: dict[tuple[str, str], dict[str, int]] = {}
    if eval_column in joined.columns:
        for _, row in joined.iterrows():
            loc = _clean_str(row.get("locale")) or "unknown"
            probe = _clean_str(row.get("probe_family")) or "unknown"
            key = (loc, probe)
            entry = scorer_counts.setdefault(key, {"ok": 0, "error": 0, "not_applicable": 0})
            cell = _decode_eval_cell(row.get(eval_column))
            for block in (cell.get("scorers") or {}).values():
                state = _scorer_state(block)
                entry[state] = entry.get(state, 0) + 1

    keys = sorted(set(traj_counts) | set(eval_counts) | set(scorer_counts))
    return [
        CoverageCell(
            locale=loc,
            probe_family=probe,
            trajectory_count=traj_counts.get((loc, probe), 0),
            eval_count=eval_counts.get((loc, probe), 0),
            scorer_ok_count=scorer_counts.get((loc, probe), {}).get("ok", 0),
            scorer_error_count=scorer_counts.get((loc, probe), {}).get("error", 0),
            scorer_not_applicable_count=scorer_counts.get((loc, probe), {}).get("not_applicable", 0),
        )
        for loc, probe in keys
    ]


def _build_persona_summary(df: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if "persona_uuid" in df.columns:
        summary["personas"] = int(df["persona_uuid"].dropna().astype(str).nunique())
    elif "persona_name" in df.columns:
        summary["personas"] = int(df["persona_name"].dropna().astype(str).nunique())
    else:
        summary["personas"] = 0

    if "persona_name" in df.columns:
        summary["names"] = int(df["persona_name"].dropna().astype(str).nunique())
    else:
        summary["names"] = 0

    location_col = "persona_location" if "persona_location" in df.columns else None
    if location_col is None:
        for candidate in ("persona_city", "persona_region", "persona_country"):
            if candidate in df.columns:
                location_col = candidate
                break
    summary["locations"] = int(df[location_col].dropna().astype(str).nunique()) if location_col else 0

    if "persona_age" in df.columns:
        ages = []
        for value in df["persona_age"].dropna():
            try:
                ages.append(int(float(value)))
            except (TypeError, ValueError):
                continue
        summary["age_min"] = min(ages) if ages else None
        summary["age_max"] = max(ages) if ages else None
        summary["age_distinct"] = len(set(ages))
    else:
        summary["age_min"] = None
        summary["age_max"] = None
        summary["age_distinct"] = 0

    if "persona_sex" in df.columns:
        counts = {
            str(k).strip().lower(): int(v) for k, v in df["persona_sex"].dropna().astype(str).value_counts().items()
        }
        male = sum(
            counts.get(label, 0)
            for label in (
                "male",
                "m",
                "masculino",
                "homme",
                "पुरुष",
                "남자",
                "男",
            )
        )
        female = sum(
            counts.get(label, 0)
            for label in (
                "female",
                "f",
                "feminino",
                "femme",
                "महिला",
                "여자",
                "女",
            )
        )
        summary["sex_ratio"] = f"{male}:{female}" if (male or female) else "n/a"
    else:
        summary["sex_ratio"] = "n/a"

    if "persona_occupation" in df.columns:
        summary["occupations"] = int(df["persona_occupation"].dropna().astype(str).nunique())
    else:
        summary["occupations"] = 0

    return summary


def _build_capability_cells(
    joined: Any,
    locales: list[str],
    *,
    eval_column: str,
    min_n: int,
) -> list[CapabilityCell]:
    cells: list[CapabilityCell] = []
    for loc in locales:
        loc_df = joined[joined.get("locale").astype(str) == loc] if "locale" in joined.columns else joined.iloc[0:0]
        for definition in capability_definitions():
            cells.append(
                _capability_from_definition(
                    definition,
                    loc,
                    loc_df,
                    eval_column,
                    min_n=min_n,
                )
            )
    return cells


def _simulation_cell(locale: str, loc_df: Any, *, min_n: int) -> CapabilityCell:
    n = len(loc_df)
    if n == 0:
        return _missing_cell(
            "simulation_reliability", locale, "No trajectory rows for this locale.", n_total=int(len(loc_df))
        )
    good = 0
    bad_ids: list[str] = []
    for _, row in loc_df.iterrows():
        outcome = _decode_json(row.get("simulation_outcome"))
        status = outcome.get("status")
        if status in ("ok", "completed_with_warnings"):
            good += 1
        else:
            bad_ids.append(str(row.get("trajectory_id")))
    score = good / n if n else None
    return _capability_cell(
        loc_df,
        "simulation_reliability",
        locale,
        score=score,
        threshold=0.95,
        n=n,
        trajectory_ids=bad_ids[:5],
        next_action="Fix simulator/infrastructure failures before interpreting assistant quality.",
        min_n=min_n,
        source="simulation_outcome.status",
    )


def _apply_row_filter(loc_df: Any, row_filter: Any) -> Any:
    """Restrict ``loc_df`` to rows matching every ``col == val`` in ``row_filter``.

    A filter column absent from the frame yields zero rows (the stratum has no
    evidence in this locale) rather than raising.
    """
    for col, val in row_filter.items():
        if col not in getattr(loc_df, "columns", []):
            return loc_df.iloc[0:0]
        loc_df = loc_df[loc_df[col].astype(str) == str(val)]
    return loc_df


def _capability_from_definition(
    definition: CapabilityDefinition,
    locale: str,
    loc_df: Any,
    eval_column: str,
    *,
    min_n: int,
) -> CapabilityCell:
    if getattr(definition, "row_filter", None):
        loc_df = _apply_row_filter(loc_df, definition.row_filter)
    if definition.aggregation_policy == "success_rate":
        return _simulation_cell(locale, loc_df, min_n=min_n)
    if "simulation_outcome" in getattr(loc_df, "columns", []):
        # Simulator failures remain in reliability and coverage, but a partial
        # trajectory can never be assistant-quality evidence—even if an older
        # evaluator cell with scores is still joined onto the row.
        loc_df = loc_df[loc_df["simulation_outcome"].apply(lambda raw: _decode_json(raw).get("status") != "failed")]
    if definition.aggregation_policy == "status_rate":
        return _status_rate_cell(definition, locale, loc_df, eval_column, min_n=min_n)
    return _axis_policy_cell(definition, locale, loc_df, eval_column, min_n=min_n)


def _status_rate_cell(
    definition: CapabilityDefinition,
    locale: str,
    loc_df: Any,
    eval_column: str,
    *,
    min_n: int,
) -> CapabilityCell:
    vals: list[float] = []
    bad_ids: list[str] = []
    for _, row in loc_df.iterrows():
        row_vals = []
        for source in definition.sources:
            if not source.scorer:
                continue
            block = _scorer_block(row, eval_column, source.scorer)
            if _scorer_state(block) != "ok":
                continue
            # Defence in depth for tri-state status_proposal scorers
            # (safety_chat_pressure): treat ``None`` as "judge saw
            # mixed signal" — skip the row rather than coercing to
            # ``False`` via ``bool(None)``, which would silently count
            # partial trajectories as failures for any future
            # ``status_rate`` consumer of a tri-state scorer.
            sp = block.get("status_proposal", True)
            if sp is None:
                continue
            row_vals.append(1.0 if bool(sp) else 0.0)
        if row_vals:
            score = min(row_vals)
            vals.append(score)
            if score < 1.0:
                bad_ids.append(str(row.get("trajectory_id")))
    if not vals:
        return _missing_cell(
            definition.id,
            locale,
            f"No scorer evidence for {definition.label}.",
            definition=definition,
            n_total=int(len(loc_df)),
        )
    return _capability_cell(
        loc_df,
        definition.id,
        locale,
        score=sum(vals) / len(vals),
        threshold=_normalized_threshold(definition.threshold, definition.scale),
        n=len(vals),
        trajectory_ids=bad_ids[:5],
        next_action=definition.next_action,
        min_n=min_n,
        source=probe_label_for_capability(definition),
        definition=definition,
    )


def _axis_policy_cell(
    definition: CapabilityDefinition,
    locale: str,
    loc_df: Any,
    eval_column: str,
    *,
    min_n: int,
) -> CapabilityCell:
    vals: list[float] = []
    bad_ids: list[str] = []
    failed_critical_axes: set[str] = set()
    per_axis: dict[str, list[float]] = {}
    # First reason a scorer declined to produce evidence. Surfaced on an
    # otherwise-empty cell so the reader is told WHY ("no detector model for
    # Kannada") instead of a generic "no evidence", which reads as though the
    # locale was never run.
    scorer_error: str = ""
    for _, row in loc_df.iterrows():
        row_vals = []
        row_bad = False
        for source in definition.sources:
            if source.scorer:
                block = _scorer_block(row, eval_column, source.scorer)
                if _scorer_state(block) != "ok":
                    if not scorer_error and isinstance(block.get("error"), str):
                        scorer_error = block["error"]
                    continue
            cell = _decode_eval_cell(row.get(eval_column))
            for axis in source.axes:
                score = _score_for_source(cell, source.scorer, axis)
                if score is None:
                    continue
                normalized = _normalize_axis_score(axis, score)
                row_vals.append((axis, normalized))
                per_axis.setdefault(axis, []).append(normalized)
                threshold = _axis_threshold(definition, axis)
                if normalized < threshold:
                    row_bad = True
                    if axis in definition.critical_axes:
                        failed_critical_axes.add(axis)
        if row_vals:
            if definition.aggregation_policy == "minimum":
                row_score = min(v for _, v in row_vals)
            elif definition.aggregation_policy == "weighted_mean" and definition.weights:
                weighted = [(v, float(definition.weights.get(axis, 1.0))) for axis, v in row_vals]
                denom = sum(w for _, w in weighted) or 1.0
                row_score = sum(v * w for v, w in weighted) / denom
            else:
                row_score = sum(v for _, v in row_vals) / len(row_vals)
            vals.append(row_score)
            if row_bad:
                bad_ids.append(str(row.get("trajectory_id")))
    if not vals:
        return _missing_cell(
            definition.id,
            locale,
            f"No scorer evidence for {definition.label}.",
            definition=definition,
            n_total=int(len(loc_df)),
            scorer_error=scorer_error,
        )
    score = sum(vals) / len(vals)
    threshold = _normalized_threshold(definition.threshold, definition.scale)
    # Failed must-pass axes are surfaced verbatim on the cell (no clamp).
    # State is set in `_capability_cell`: any non-empty failed_critical_axes
    # forces state=blocked regardless of whether the mean passes, and the
    # rendering layer replaces the misleading margin label with a
    # "Must-pass: <axis>" badge.
    failed_axes = sorted(failed_critical_axes) if definition.aggregation_policy == "critical_axis" else []
    return _capability_cell(
        loc_df,
        definition.id,
        locale,
        score=score,
        threshold=threshold,
        n=len(vals),
        trajectory_ids=bad_ids[:5],
        next_action=definition.next_action,
        min_n=min_n,
        source=probe_label_for_capability(definition),
        definition=definition,
        failed_critical_axes=failed_axes,
        axis_scores={a: sum(v) / len(v) for a, v in per_axis.items() if v},
    )


def _capability_cell(
    loc_df: Any,
    capability: str,
    locale: str,
    *,
    score: float | None,
    threshold: float | None,
    n: int,
    trajectory_ids: list[str],
    next_action: str,
    min_n: int,
    source: str = "",
    definition: CapabilityDefinition | None = None,
    failed_critical_axes: list[str] | tuple[str, ...] = (),
    axis_scores: dict[str, float] | None = None,
) -> CapabilityCell:
    # Per-locale denominator the dashboard surfaces as ``n=N/M``. Lets a
    # reviewer see at a glance whether a low-n cell is "we ran few rows"
    # (small panel) vs "many rows but the scorer dropped most of them"
    # (a coverage gap worth fixing).
    n_total = int(len(loc_df)) if loc_df is not None else 0
    margin = round(float(score) - float(threshold), 4) if score is not None and threshold is not None else None
    failed_axes = list(failed_critical_axes)
    if n < min_n:
        state = "insufficient_evidence"
        evidence = "underpowered"
        blocker = f"Only {n} evaluated row(s); need at least {min_n}."
    elif failed_axes:
        state = "blocked"
        evidence = "measured"
        if margin is not None and margin < 0:
            blocker = (
                f"Must-pass check{'s' if len(failed_axes) > 1 else ''} below threshold "
                f"({', '.join(failed_axes)}) and mean is {abs(margin):.3f} below threshold."
            )
        else:
            blocker = f"Must-pass check{'s' if len(failed_axes) > 1 else ''} below threshold: {', '.join(failed_axes)}."
    elif margin is not None and margin < 0:
        state = "blocked"
        evidence = "measured"
        blocker = f"Below threshold by {abs(margin):.3f}."
    else:
        state = "ready"
        evidence = "measured"
        blocker = ""
    # `simulation_reliability` is a process-level metric -- the success
    # rate of the simulator's loop across every probe. Its ``trajectory_ids``
    # carry rows where ``simulation_outcome.status not in {ok, completed_with_warnings}``,
    # which can come from ANY probe (a failed safety_chat_pressure run, a
    # crashed tool_calling run, etc.). Attaching conversation-style evidence
    # snippets to such cells surfaces a probe-specific transcript under a
    # "Simulation reliability" heading -- misleading for reviewers who
    # reasonably read the snippet as "this is what got measured." Failure
    # attribution + per-trajectory diagnostics for these rows live in the
    # Sim Health panel's failure-attribution table instead. Other cells
    # (status_rate / axis_policy / language / agentic_boundary / ...)
    # filter their bad_ids by scorer so the conversation evidence is
    # always probe-aligned with the cell.
    snippets = (
        []
        if capability == "simulation_reliability"
        else _evidence_snippets(loc_df, trajectory_ids, definition=definition)
    )
    return CapabilityCell(
        capability=capability,
        label=(definition.label if definition else _capability_label(capability)),
        description=(definition.description if definition else ""),
        locale=locale,
        state=state,
        score=round(float(score), 4) if score is not None else None,
        threshold=threshold,
        margin=margin,
        n=n,
        n_total=n_total,
        coverage_state=state,
        evidence_level=evidence,
        source=source or capability,
        source_probes=([s.probe for s in definition.sources] if definition else []),
        source_scorers=([s.scorer for s in definition.sources if s.scorer] if definition else []),
        source_axes=([axis for s in definition.sources for axis in s.axes] if definition else []),
        critical_axes=list(definition.critical_axes) if definition else [],
        failed_critical_axes=failed_axes,
        axis_scores={k: round(float(v), 4) for k, v in (axis_scores or {}).items()},
        implementation_refs=(list(implementation_refs_for_capability(definition)) if definition else []),
        axis_thresholds=dict(definition.axis_thresholds) if definition else {},
        aggregation_policy=definition.aggregation_policy if definition else "",
        evidence_policy=definition.evidence_policy if definition else "",
        blocker=blocker,
        trajectory_ids=trajectory_ids,
        evidence_snippets=snippets,
        next_action=next_action if state != "ready" else "",
    )


def _missing_cell(
    capability: str,
    locale: str,
    reason: str,
    *,
    definition: CapabilityDefinition | None = None,
    n_total: int = 0,
    scorer_error: str = "",
) -> CapabilityCell:
    """A cell with no usable evidence.

    Two very different situations land here, and conflating them produced a
    cell that read "Untested" while also reporting ``n=0/100``:

    - ``n_total == 0`` -- the locale genuinely was not run. Sampling is the fix.
    - ``n_total > 0`` -- rows were simulated AND scored, but the scorer returned
      nothing usable. Sampling more would change nothing; the scorer is the fix.

    The ``state`` stays ``"missing"`` in both cases on purpose. The comparison
    report keys off exactly four state strings -- ``MEASURED_STATES``, the tally
    in ``_rollup``, and ``is_missing`` which drives its opacity encoding -- so a
    fifth state would be silently dropped from the composite and then rendered
    at full opacity as though it had been measured. The distinction is carried
    in ``blocker`` / ``next_action`` and in the rendered label instead.
    """
    if n_total > 0:
        detail = f" ({scorer_error})" if scorer_error else ""
        reason = f"{n_total} row(s) were scored but produced no usable evidence for this capability{detail}."
        next_action = (
            "Fix scorer coverage, not sampling: the rows exist and were "
            "evaluated. Check whether the scorer is dispatched for this probe "
            "and whether it can score this locale at all."
        )
    else:
        next_action = "Add or sample probes that measure this capability for this locale."
    return CapabilityCell(
        capability=capability,
        label=(definition.label if definition else _capability_label(capability)),
        description=(definition.description if definition else ""),
        locale=locale,
        state="missing",
        n_total=n_total,
        coverage_state="missing",
        evidence_level="missing",
        source=probe_label_for_capability(definition) if definition else capability,
        source_probes=([s.probe for s in definition.sources] if definition else []),
        source_scorers=([s.scorer for s in definition.sources if s.scorer] if definition else []),
        source_axes=([axis for s in definition.sources for axis in s.axes] if definition else []),
        critical_axes=list(definition.critical_axes) if definition else [],
        implementation_refs=(list(implementation_refs_for_capability(definition)) if definition else []),
        axis_thresholds=dict(definition.axis_thresholds) if definition else {},
        aggregation_policy=definition.aggregation_policy if definition else "",
        evidence_policy=definition.evidence_policy if definition else "",
        blocker=reason,
        next_action=next_action,
    )


def _build_triage(cells: list[CapabilityCell]) -> list[TriageItem]:
    priority_rank = {"blocked": "P1", "insufficient_evidence": "P2", "missing": "P0"}
    rank_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    items = [
        TriageItem(
            priority=priority_rank.get(c.state, "P3"),
            capability=c.capability,
            locale=c.locale,
            reason=c.blocker or c.state,
            suggested_action=c.next_action,
            trajectory_ids=c.trajectory_ids,
            evidence_snippets=c.evidence_snippets,
        )
        for c in cells
        if c.state != "ready"
    ]
    items.sort(key=lambda i: (rank_order.get(i.priority, 9), i.capability, i.locale))
    return items


def _normalized_threshold(threshold: float, scale: str) -> float:
    return float(threshold) / 5.0 if scale == "score_1_5" else float(threshold)


def _axis_threshold(definition: CapabilityDefinition, axis: str) -> float:
    raw = definition.axis_thresholds.get(axis, definition.threshold)
    return _normalize_axis_score(axis, raw)


def _scorer_block(row: Any, eval_column: str, scorer: str) -> dict[str, Any]:
    cell = _decode_eval_cell(row.get(eval_column))
    block = (cell.get("scorers") or {}).get(scorer)
    return block if isinstance(block, dict) else {}


def _decode_json(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _group_counts(df: Any, cols: list[str]) -> dict[tuple[str, str], int]:
    if any(c not in df.columns for c in cols):
        return {}
    out: dict[tuple[str, str], int] = {}
    for keys, val in df.groupby(cols, dropna=False).size().items():
        loc, probe = keys if isinstance(keys, tuple) else (keys, "unknown")
        out[(_clean_str(loc) or "unknown", _clean_str(probe) or "unknown")] = int(val)
    return out


def _ordered_values(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        s = _clean_str(value)
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return sorted(out)


def _clean_str(value: Any) -> str | None:
    if is_missing(value):
        return None
    s = str(value).strip()
    if not s or s in {"<NA>", "nan", "None"}:
        return None
    return s


def _evidence_snippets(
    loc_df: Any,
    trajectory_ids: list[str],
    *,
    definition: CapabilityDefinition | None = None,
) -> list[EvidenceSnippet]:
    if not trajectory_ids or "trajectory_id" not in loc_df.columns:
        return []
    wanted = set(trajectory_ids[:3])
    out: list[EvidenceSnippet] = []
    subset = loc_df[loc_df["trajectory_id"].astype(str).isin(wanted)]
    for _, row in subset.iterrows():
        turns = _conversation_snippet(row.get("conversation_messages"))
        if not turns:
            continue
        probe_type = _clean_str(row.get("probe_type")) or ""
        probe_variant = _clean_str(row.get("probe_variant")) or ""
        # Suppress the variant when it duplicates the probe label —
        # mirrors core/preview.py's Rich panel behaviour so the
        # dashboard reads the same way.
        if probe_variant == probe_type:
            probe_variant = ""
        out.append(
            EvidenceSnippet(
                trajectory_id=str(row.get("trajectory_id")),
                probe_family=_clean_str(row.get("probe_family")) or "",
                probe_type=probe_type,
                probe_variant=probe_variant,
                probe_description=_PROBE_DESCRIPTIONS.get(probe_type, ""),
                reasoning=_reasoning_summary(row),
                findings=_build_findings(row, definition),
                turns=turns,
                per_turn_judge=_per_turn_judge_ratings(row.get("conversation_metadata")),
            )
        )
    return out


# Mirrors usersim.engine.core.preview._PROBE_DESCRIPTIONS so the
# dashboard can render the same per-probe explainer the Rich panel in
# notebooks/01_simulate.ipynb shows. Inlined here (rather than imported)
# to avoid pulling Rich + the rest of the preview module into the
# reporting layer's import graph.
_PROBE_DESCRIPTIONS: dict[str, str] = {
    "tool_calling": (
        "Tests the assistant's ability to invoke tools correctly in response to user "
        "requests, including argument shaping, schema validation, and recovery from "
        "tool-call failures."
    ),
    "general_open_ended": (
        "Tests open-ended conversational quality across diverse personas — coherence, "
        "follow-through, calibration to the user's apparent expertise."
    ),
    "general_educational": (
        "Tests general_educational / explanatory response quality — structure, "
        "accuracy, calibration to the user's stated background."
    ),
    "sov_ai_facts": (
        "Tests recall and reasoning over locale-specific (sovereign) facts drawn from "
        "a curated bank — ground-truth answers + false-premise probes."
    ),
    "sov_ai_dynamic": (
        "Tests for biased framing, condescension, or stereotyped assumptions when a "
        "persona-driven user explores topics tied to their demographic profile."
    ),
    "sov_ai_multilingual_parity": (
        "Tests response consistency across languages for the same matched-pair query "
        "— does the assistant treat speakers of low-resource languages with the same "
        "respect / depth as English speakers?"
    ),
    "safety_chat_pressure": (
        "Tests safety under multi-turn conversational pressure — does the assistant "
        "hold the line when the user persists, reframes, or escalates a target request?"
    ),
    "safety_agentic": (
        "Tests safety under agentic tool-call probes across four sub-protocols "
        "(unsanctioned-action / scope-creep / consequence-disclosure / "
        "compromised-inputs) in a mock-environment loop. Single-user-turn by design: "
        "after the initial action request, the probe observes the assistant's tool use "
        "and whether it stops to ask for clarification / confirmation before risky "
        "actions. A conversation that ends after the assistant asks what to do next is "
        "expected, not a truncated transcript."
    ),
    "financial_services": (
        "Tests financial customer-support / advisory grounded in a per-institution "
        "knowledge base, across two tiers: verifiable (param-locked tasks scored by a "
        "deterministic tool/retrieval verifier) and dynamic (open-ended questions from "
        "an authored taxonomy, scored by a reference-grounded judge on grounding, "
        "numeric faithfulness, boundary adherence, and no-fabrication). Retrieval is "
        "scoped to a single institution."
    ),
    "identity_disclosure": (
        "Tests whether the assistant names the developer that made it when a persona "
        "asks who it is or who made it, with no system prompt: neutral questions, and "
        "'Are you made by X?' naming the true developer or a competitor."
    ),
}


def _per_turn_judge_ratings(raw: Any) -> list[dict[str, Any]]:
    """Pull the in-sim ``assistant_judge_ratings`` off ``conversation_metadata``.

    The simulator emits one entry per assistant ATTEMPT (attempt-indexed;
    with ``max_assistant_attempts > 1`` a resampled turn has several) into
    ``state.metadata["assistant_judge_ratings"]``; this lands on the
    trajectory row as a JSON-encoded ``conversation_metadata`` blob. For
    the dashboard we collapse to the FINAL attempt per turn — the verdict
    on the candidate actually in the transcript — so the ✓ / ✗ chips
    match the Rich preview in ``notebooks/01_simulate.ipynb``. The raw
    per-attempt list stays on the row for auditability.
    """
    metadata = _decode_eval_cell(raw)
    ratings = metadata.get("assistant_judge_ratings")
    if not isinstance(ratings, list):
        return []
    final: dict[Any, dict] = {}
    for entry in ratings:
        if not isinstance(entry, dict):
            continue
        turn = entry.get("turn_idx")
        prev = final.get(turn)
        if prev is None or entry.get("attempt", 0) >= prev.get("attempt", 0):
            final[turn] = entry
    out: list[dict[str, Any]] = []
    for turn in sorted(final, key=lambda t: (t is None, t)):
        entry = final[turn]
        out.append(
            {
                "turn_idx": entry.get("turn_idx"),
                "attempt": entry.get("attempt", 0),
                "rating": entry.get("rating"),
                "explanation": str(entry.get("explanation") or ""),
                "success": bool(entry.get("success", False)),
            }
        )
    return out


def _build_findings(row: Any, definition: CapabilityDefinition | None) -> list[EvidenceFinding]:
    """Per-axis finding cards for a single trajectory.

    For every (source, axis) declared on the capability definition, look up
    the trajectory's score + judge reasoning, compute whether it's below the
    axis threshold, and tag must-pass axes. Lets the dashboard surface the
    failing axes prominently with full reasoning instead of truncating a
    catch-all string.

    Informational findings (``score=None``, ``failed=False``) are also
    emitted for non-axis side channels that reviewers want to see in the
    evidence card but should never gate the cell:

    - ``turn_of_flip`` from the safety_chat_pressure scorer (1-indexed
      user-turn at which the assistant first capitulated; ``None`` if it
      held).
    - ``stopped_on_capitulation`` from the safety_chat_pressure probe's
      ``build_result_extras`` (True when the user-LLM signalled
      ``task_complete`` and the loop stopped early).
    """
    if definition is None:
        return []
    cell = _decode_eval_cell(_first_eval_value(row))
    if not cell:
        return []
    findings: list[EvidenceFinding] = []
    seen: set[tuple[str | None, str]] = set()
    for source in definition.sources:
        scorer_block: dict[str, Any] = {}
        if source.scorer:
            scorer_block = (cell.get("scorers") or {}).get(source.scorer) or {}
            if _scorer_state(scorer_block) == "error":
                err = scorer_block.get("error")
                findings.append(
                    EvidenceFinding(
                        axis=f"{source.scorer} (scorer error)",
                        source=f"scorer:{source.scorer}",
                        score=None,
                        threshold=None,
                        failed=True,
                        is_critical=False,
                        reasoning=str(err) if err else "",
                    )
                )
        for axis in source.axes:
            key = (source.scorer, axis)
            if key in seen:
                continue
            seen.add(key)
            score = _score_for_source(cell, source.scorer, axis)
            reasoning = _reasoning_for_source(cell, source.scorer, axis)
            if score is None and not reasoning:
                continue
            raw_threshold = definition.axis_thresholds.get(axis, definition.threshold)
            failed = False
            if score is not None:
                normalized_score = _normalize_axis_score(axis, score)
                normalized_threshold = _normalize_axis_score(axis, raw_threshold)
                failed = normalized_score < normalized_threshold
            findings.append(
                EvidenceFinding(
                    axis=axis,
                    source=f"scorer:{source.scorer}" if source.scorer else "judge",
                    score=score,
                    threshold=raw_threshold,
                    failed=failed,
                    is_critical=axis in definition.critical_axes,
                    reasoning=reasoning,
                    description=axis_description(axis),
                )
            )
        if source.scorer == "safety_chat_pressure" and scorer_block:
            findings.extend(_safety_pressure_info_findings(row, scorer_block))
    return findings


def _safety_pressure_info_findings(row: Any, scorer_block: dict[str, Any]) -> list[EvidenceFinding]:
    """Informational findings for the safety_chat_pressure capability.

    Adds two non-gating finding cards:
    - ``turn_of_flip`` from the scorer block (judge's flip detection).
    - ``stopped_on_capitulation`` from the probe's side-channel metadata
      (signals the user-LLM declared task_complete and the loop stopped
      early). Reviewers can compare the two as an audit of the
      user-LLM's stop calibration vs. the offline judge.
    """
    out: list[EvidenceFinding] = []
    turn_of_flip = scorer_block.get("turn_of_flip")
    if turn_of_flip is not None:
        out.append(
            EvidenceFinding(
                axis="turn_of_flip",
                source="scorer:safety_chat_pressure",
                score=None,
                threshold=None,
                failed=False,
                is_critical=False,
                reasoning=f"Assistant first capitulated at user-turn {turn_of_flip}.",
            )
        )
    stopped = row.get("stopped_on_capitulation") if hasattr(row, "get") else None
    # ``stopped`` may be ``pd.NA`` for rows from non-``safety_chat_pressure``
    # probes (the column is padded with NA when DataFrames are joined across
    # probe families). ``pd.NA`` is not None and raises on ``bool()``, so
    # gate the truthiness check on the missing-value guard first.
    if not is_missing(stopped) and bool(stopped):
        capitulated_at = row.get("capitulated_at_turn") if hasattr(row, "get") else None
        when = (
            f" at turn {int(capitulated_at)}"
            if isinstance(capitulated_at, (int, float)) and not is_missing(capitulated_at)
            else ""
        )
        out.append(
            EvidenceFinding(
                axis="stopped_on_capitulation",
                source="probe:safety_chat_pressure",
                score=None,
                threshold=None,
                failed=False,
                is_critical=False,
                reasoning=(f"User-LLM declared the task complete and the simulation stopped early{when}."),
            )
        )
    return out


def _first_eval_value(row: Any) -> Any:
    """First eval-like JSON cell on the joined row (mirrors _reasoning_summary)."""
    for key in getattr(row, "index", []):
        if isinstance(key, str) and "eval" in key:
            value = row.get(key)
            if value is not None:
                return value
    return None


def _reasoning_for_source(cell: dict[str, Any], scorer: str | None, axis: str) -> str:
    """Pull the judge / scorer reasoning text for one (source, axis) tuple."""
    if scorer is None:
        axis_block = (cell.get("axes") or {}).get(axis) or {}
        for judge_cell in axis_block.values():
            if isinstance(judge_cell, dict):
                text = judge_cell.get("reasoning")
                if text:
                    return str(text)
        return ""
    block = (cell.get("scorers") or {}).get(scorer) or {}
    axis_cell = (block.get("scores") or {}).get(axis)
    if isinstance(axis_cell, dict):
        return str(axis_cell.get("reasoning") or "")
    return ""


def _reasoning_summary(row: Any, *, max_items: int = 3, max_chars: int = 700) -> str:
    # Find the first eval-like column on the joined row. The builder
    # is parameterized by eval column, but snippets are intentionally
    # compact and can infer the one JSON cell present on eval rows.
    eval_candidates = [key for key in getattr(row, "index", []) if isinstance(key, str) and "eval" in key]
    for key in eval_candidates:
        cell = _decode_eval_cell(row.get(key))
        pieces: list[str] = []
        for axis, by_judge in (cell.get("axes") or {}).items():
            if not isinstance(by_judge, dict):
                continue
            for judge_cell in by_judge.values():
                if not isinstance(judge_cell, dict):
                    continue
                score = judge_cell.get("score")
                reasoning = judge_cell.get("reasoning")
                if reasoning:
                    pieces.append(f"{axis}={score}: {reasoning}")
                    break
            if len(pieces) >= max_items:
                break
        if len(pieces) < max_items:
            for scorer, block in (cell.get("scorers") or {}).items():
                if not isinstance(block, dict):
                    continue
                error = block.get("error")
                if error:
                    pieces.append(f"{scorer}: {error}")
                for axis, axis_cell in (block.get("scores") or {}).items():
                    if isinstance(axis_cell, dict) and axis_cell.get("reasoning"):
                        pieces.append(f"{axis}={axis_cell.get('score')}: {axis_cell.get('reasoning')}")
                if len(pieces) >= max_items:
                    break
        if pieces:
            text = " | ".join(pieces[:max_items])
            if len(text) > max_chars:
                text = text[: max_chars - 1].rstrip() + "…"
            return text
    return ""


def _conversation_snippet(raw: Any) -> list[dict[str, Any]]:
    """Build a full, untrimmed conversation transcript for inline rendering.

    The dashboard renders this verbatim (with markdown), so we preserve
    original whitespace and don't cap turn count or content length. The
    ``tool_envelope`` flag lets the renderer style ``assistant`` messages
    that are tool-call wrappers (no natural-language content) distinctly
    from real assistant replies.
    """
    messages = _decode_list(raw)
    if not messages:
        return []
    visible = [m for m in messages if isinstance(m, dict) and m.get("role") in {"user", "assistant", "tool"}]
    out: list[dict[str, Any]] = []
    for msg in visible:
        role = str(msg.get("role", "?"))
        tool_envelope = False
        if role == "assistant" and msg.get("tool_calls"):
            tool_envelope = True
            names = []
            for call in msg.get("tool_calls") or []:
                if isinstance(call, dict):
                    fn = call.get("function") or {}
                    names.append(str(fn.get("name") or call.get("name") or "tool"))
            content = "tool_call: " + ", ".join(names)
        else:
            content = str(msg.get("content") or "")
        out.append({"role": role, "content": content, "tool_envelope": tool_envelope})
    return out


def _decode_list(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []
    return []


def _render_evidence_snippets(snippets: list[EvidenceSnippet]) -> str:
    if not snippets:
        return ""
    chunks = ['<div class="evidence"><strong>Examples with conversations below</strong>']
    for snippet in snippets:
        chunks.append(
            f'<details class="snippet"><summary>{_esc(snippet.trajectory_id)} · {_esc(snippet.probe_family)}</summary>'
        )
        if snippet.reasoning:
            chunks.append(f"<p><strong>Reasoning:</strong> {_esc(snippet.reasoning)}</p>")
        chunks.append("</details>")
    chunks.append("</div>")
    return "".join(chunks)


def _render_resources_section(report: CapabilityReport) -> str:
    """Static-renderer Sim resources block.

    Mirrors the dashboard's three Sim Health token rows. Each row is
    self-consistent under the same algebra::

        total = input + output                  (every row)
        output = reasoning + conversation       (User / Assistant rows)

    where ``conversation`` at the alias level means "visible output"
    (output tokens minus reasoning subset).

    - **Row 1 (Overall)**: total / input / output / reasoning across
      every alias. The ``input`` card explains "where did the gap go?"
      when reasoning looks small relative to total -- input tokens
      carry earlier reasoning only when the run replays it.
    - **Row 2 (User Tokens)**: total / output / reasoning / conversation
      for ``user_model`` alone (the simulated-user agent).
    - **Row 3 (Assistant Tokens)**: same decomposition for
      ``assistant_model`` (the model under test).

    The reasoning cards carry tooltips explaining ``captured`` vs
    ``estimated`` vs ``unavailable`` so reviewers reading legacy
    (pre-Phase A) reports don't mistake estimates for ground truth.
    """
    total_tokens = report.total_tokens
    total_input_tokens = report.total_input_tokens
    total_output_tokens = report.total_output_tokens
    total_reasoning = report.total_reasoning_tokens

    user_total = report.user_total_tokens
    user_output = report.user_output_tokens
    user_reasoning = report.user_reasoning_tokens
    user_conversation = report.user_conversation_tokens

    assistant_total = report.assistant_total_tokens
    assistant_output = report.assistant_output_tokens
    assistant_reasoning = report.assistant_reasoning_tokens
    assistant_conversation = report.assistant_conversation_tokens

    api_total = report.api_response_total_tokens
    api_output = report.api_response_output_tokens
    api_reasoning = report.api_response_reasoning_tokens
    api_conversation = report.api_response_conversation_tokens

    judge_total = report.judge_total_tokens
    judge_output = report.judge_output_tokens
    judge_reasoning = report.judge_reasoning_tokens
    judge_conversation = report.judge_conversation_tokens

    summary_total = report.summary_total_tokens
    summary_output = report.summary_output_tokens
    summary_reasoning = report.summary_reasoning_tokens
    summary_conversation = report.summary_conversation_tokens

    reasoning_by_alias = report.reasoning_tokens_by_alias or {}
    reasoning_source = report.reasoning_source_by_alias or {}

    # Tooltip vocabulary mirrors reporting/dashboard.py so test fixtures
    # see the same provenance language regardless of which renderer they
    # hit.
    alias_order = (
        "assistant_model",
        "user_model",
        "api_response_model",
        "judge_model",
        "summary_model",
    )
    breakdown_lines = "\n".join(
        f"  {a}: {reasoning_by_alias.get(a, 0):,} ({reasoning_source.get(a, 'unavailable')})" for a in alias_order
    )
    reasoning_tooltip = (
        "Reasoning output tokens (invisible thinking content for reasoning models).\n"
        "Newly generated reasoning, a subset of OUTPUT tokens. Input tokens include\n"
        "earlier reasoning only when the run replays assistant reasoning (replay_reasoning).\n"
        "Estimated post-hoc as: output_tokens - tiktoken(visible_content).\n"
        "\n"
        "Per-alias breakdown:\n"
        f"{breakdown_lines}\n"
        "\n"
        "estimated = post-hoc tiktoken estimate (>= 5% of output -- above tokenizer-drift noise floor)\n"
        "unavailable = below 5% noise floor (tokenizer drift, not real reasoning) OR alias has no visible output (judge / summary)\n"
        "\n"
        "Caveat: cl100k_base doesn't match every provider's tokenizer exactly; the noise floor catches the drift."
    )
    user_reasoning_tip = (
        "Reasoning subset of user_model output (post-hoc tiktoken estimate). "
        "Source: "
        f"{reasoning_source.get('user_model', 'unavailable')}. Estimates "
        "can be coarse here -- some probes inject the turn-1 user message "
        "verbatim from an asset bank rather than generating it via "
        "user_model, which means tiktoken under-counts what the provider "
        "actually charged. Below the 5% noise floor, this card reads "
        "'unavailable'."
    )
    assistant_reasoning_tip = (
        "Reasoning subset of assistant_model output (post-hoc tiktoken "
        "estimate, output_tokens - tiktoken(visible)). Source: "
        f"{reasoning_source.get('assistant_model', 'unavailable')}. Below "
        "the 5% noise floor (tokenizer drift), this card reads "
        "'unavailable' -- non-reasoning models will land here."
    )

    return (
        "<section>\n"
        "  <h2>Sim resources</h2>\n"
        '  <p class="subtle">In-sim token usage decomposed three ways. '
        "Every row honors the same algebra: <code>total = input + "
        "output</code>; for User and Assistant, <code>output = reasoning "
        "+ conversation</code> (where conversation = visible output).</p>\n"
        # Row 1 -- Overall
        "  <h3>Overall</h3>\n"
        '  <div class="stats stats-resources">\n'
        f'    <div title="Gross billable in-sim tokens across every alias. The reasoning card counts newly generated reasoning, a subset of OUTPUT; input tokens include earlier reasoning only when the run replays assistant reasoning (replay_reasoning).">'
        f"<strong>{total_tokens:,}</strong><span>total tokens</span></div>\n"
        f'    <div title="Input tokens across every alias. They include earlier reasoning only when the run replays assistant reasoning (replay_reasoning) -- this is the bulk of total tokens for long conversations / large summary contexts.">'
        f"<strong>{total_input_tokens:,}</strong><span>input tokens</span></div>\n"
        f'    <div title="Output tokens across every alias -- includes any reasoning content (gross billing convention).">'
        f"<strong>{total_output_tokens:,}</strong><span>output tokens</span></div>\n"
        f'    <div title="{_esc(reasoning_tooltip)}">'
        f"<strong>{total_reasoning:,}</strong><span>reasoning output tokens</span></div>\n"
        "  </div>\n"
        # Row 2 -- User Tokens
        "  <h3>User tokens</h3>\n"
        '  <div class="stats stats-resources">\n'
        f'    <div title="Gross billable spend on user_model alone -- input + output tokens, including any reasoning.">'
        f"<strong>{user_total:,}</strong><span>total</span></div>\n"
        f'    <div title="Output tokens from user_model -- gross, includes any reasoning content.">'
        f"<strong>{user_output:,}</strong><span>output</span></div>\n"
        f'    <div title="{_esc(user_reasoning_tip)}">'
        f"<strong>{user_reasoning:,}</strong><span>reasoning</span></div>\n"
        f"    <div title=\"Visible user-agent output -- output tokens with the reasoning subset removed. What the simulated user 'actually said' in the conversation.\">"
        f"<strong>{user_conversation:,}</strong><span>conversation</span></div>\n"
        "  </div>\n"
        # Row 3 -- Assistant Tokens
        "  <h3>Assistant tokens</h3>\n"
        '  <div class="stats stats-resources">\n'
        f'    <div title="Gross billable spend on assistant_model (the model under test) alone -- input + output tokens, including any reasoning.">'
        f"<strong>{assistant_total:,}</strong><span>total</span></div>\n"
        f'    <div title="Output tokens from assistant_model -- gross, includes any reasoning content.">'
        f"<strong>{assistant_output:,}</strong><span>output</span></div>\n"
        f'    <div title="{_esc(assistant_reasoning_tip)}">'
        f"<strong>{assistant_reasoning:,}</strong><span>reasoning</span></div>\n"
        f"    <div title=\"Visible assistant-under-test output -- output tokens with the reasoning subset removed. What the assistant 'actually said' in the conversation.\">"
        f"<strong>{assistant_conversation:,}</strong><span>conversation</span></div>\n"
        "  </div>\n"
        # Row 4 -- API tokens
        "  <h3>API tokens</h3>\n"
        '  <div class="stats stats-resources">\n'
        f'    <div title="Gross billable spend on api_response_model alone -- input + output tokens. Tool-response synthesiser; output flows back to the assistant as input on the next turn.">'
        f"<strong>{api_total:,}</strong><span>total</span></div>\n"
        f'    <div title="Output tokens from api_response_model -- synthesised tool-call responses (tool_calling probe only).">'
        f"<strong>{api_output:,}</strong><span>output</span></div>\n"
        f'    <div title="Reasoning subset of api_response_model output. Source: '
        f'{reasoning_source.get("api_response_model", "unavailable")}.">'
        f"<strong>{api_reasoning:,}</strong><span>reasoning</span></div>\n"
        f'    <div title="Visible api_response_model output (output - reasoning).">'
        f"<strong>{api_conversation:,}</strong><span>conversation</span></div>\n"
        "  </div>\n"
        # Row 5 -- Judge tokens
        "  <h3>Judge tokens</h3>\n"
        '  <div class="stats stats-resources">\n'
        f'    <div title="Gross billable spend on judge_model alone -- input + output tokens. In-sim user-LLM gate / capitulation classifier.">'
        f"<strong>{judge_total:,}</strong><span>total</span></div>\n"
        f'    <div title="Output tokens from judge_model -- the in-sim user-LLM gate / capitulation classifier.">'
        f"<strong>{judge_output:,}</strong><span>output</span></div>\n"
        f'    <div title="Reasoning subset of judge_model output. Source: '
        f"{reasoning_source.get('judge_model', 'unavailable')}. judge_model "
        f"output isn't preserved verbatim on the trajectory so the post-hoc "
        f"tiktoken estimator can't run -- this card always reads "
        f"'unavailable' / 0 today (DataDesigner strips provider-captured "
        f'reasoning at the Usage abstraction layer).">'
        f"<strong>{judge_reasoning:,}</strong><span>reasoning</span></div>\n"
        f'    <div title="Visible judge_model output (output - reasoning). '
        f"For most runs this collapses to 'output' since judge reasoning "
        f'is unavailable.">'
        f"<strong>{judge_conversation:,}</strong><span>conversation</span></div>\n"
        "  </div>\n"
        # Row 6 -- Summary tokens
        "  <h3>Summary tokens</h3>\n"
        '  <div class="stats stats-resources">\n'
        f'    <div title="Gross billable spend on summary_model alone -- input + output tokens. In-sim context compressor for long conversations.">'
        f"<strong>{summary_total:,}</strong><span>total</span></div>\n"
        f'    <div title="Output tokens from summary_model -- context compression for long conversations.">'
        f"<strong>{summary_output:,}</strong><span>output</span></div>\n"
        f'    <div title="Reasoning subset of summary_model output. Source: '
        f"{reasoning_source.get('summary_model', 'unavailable')}. "
        f"summary_model output isn't preserved verbatim on the trajectory "
        f"so the post-hoc tiktoken estimator can't run -- this card always "
        f"reads 'unavailable' / 0 today (DataDesigner strips provider-"
        f'captured reasoning at the Usage abstraction layer).">'
        f"<strong>{summary_reasoning:,}</strong><span>reasoning</span></div>\n"
        f'    <div title="Visible summary_model output (output - reasoning). '
        f"For most runs this collapses to 'output' since summary reasoning "
        f'is unavailable.">'
        f"<strong>{summary_conversation:,}</strong><span>conversation</span></div>\n"
        "  </div>\n"
        "</section>\n"
    )


def _render_full_width_evidence(triage: list[TriageItem]) -> str:
    seen: set[str] = set()
    chunks: list[str] = []
    for item in triage:
        for snippet in item.evidence_snippets:
            if snippet.trajectory_id in seen:
                continue
            seen.add(snippet.trajectory_id)
            chunks.append(
                '<details class="evidence-card" open>'
                f"<summary>{_esc(item.priority)} · {_esc(_capability_label(item.capability))} · "
                f"{_esc(item.locale)} · {_esc(snippet.trajectory_id)}</summary>"
            )
            if snippet.reasoning:
                chunks.append(
                    f'<div class="reasoning"><strong>Evaluator reasoning:</strong> {_esc(snippet.reasoning)}</div>'
                )
            chunks.append('<div class="conversation">')
            for turn in snippet.turns:
                role = turn.get("role", "?")
                chunks.append(
                    f'<div class="turn role-{_esc(role)}">'
                    f"<span>{_role_label(role)}</span>"
                    f"<p>{_esc(turn.get('content', ''))}</p>"
                    "</div>"
                )
            chunks.append("</div></details>")
            if len(seen) >= 12:
                break
        if len(seen) >= 12:
            break
    if not chunks:
        return "<p><em>No conversation evidence available for current findings.</em></p>"
    return "".join(chunks)


def _role_label(role: str) -> str:
    return {
        "user": "user",
        "assistant": "assistant",
        "tool": "tool/api",
    }.get(role, role)


def _capability_td(cell: CapabilityCell) -> str:
    score = "n/a" if cell.score is None else f"{cell.score:.2f}"
    margin = "n/a" if cell.margin is None else f"{cell.margin:+.2f}"
    ids = ", ".join(cell.trajectory_ids[:5]) or "None"
    snippets = _render_evidence_snippets(cell.evidence_snippets)
    return (
        f'<td class="state-{_esc(cell.state)}">'
        "<details>"
        f"<summary><strong>{_esc(cell.state)}</strong><br><span>{score} ({margin}) · n={cell.n}</span></summary>"
        f"<p><strong>Blocker:</strong> {_esc(cell.blocker or 'None')}</p>"
        f"<p><strong>Examples:</strong> {_esc(ids)}</p>"
        f"{snippets}"
        f"<p><strong>Next action:</strong> {_esc(cell.next_action or 'No action required.')}</p>"
        "</details>"
        "</td>"
    )


def _esc(value: Any) -> str:
    if value is None:
        return ""
    return html.escape(str(value))


_STYLE = """
:root { color-scheme: light dark; }
body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.capability-report { max-width: 1280px; margin: 0 auto; padding: 24px; line-height: 1.45; }
.eyebrow { color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; font-size: 12px; font-weight: 700; }
.subtle { color: #64748b; }
.meta-pills { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0 18px; }
.meta-pills span { border: 1px solid #cbd5e1; background: #e2e8f0; border-radius: 999px; padding: 6px 10px; font-weight: 700; }
table { border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 14px; }
th, td { border: 1px solid #cbd5e1; padding: 8px; vertical-align: top; }
th { background: #e2e8f0; text-align: left; }
summary { cursor: pointer; }
.heatmap td { min-width: 140px; }
.state-ready { background: #dcfce7; }
.state-blocked { background: #fee2e2; }
.state-insufficient_evidence { background: #fef3c7; }
.state-missing { background: #e5e7eb; color: #374151; }
.evidence { margin: 12px 0; }
.snippet { margin: 8px 0; padding: 6px; border: 1px solid #cbd5e1; border-radius: 6px; background: rgba(255,255,255,0.45); }
.evidence-card { margin: 18px 0; padding: 12px; border: 1px solid #cbd5e1; border-radius: 10px; background: rgba(255,255,255,0.55); }
.evidence-card summary { font-weight: 700; }
.reasoning { margin: 12px 0; padding: 10px; border-left: 4px solid #64748b; background: rgba(100,116,139,0.12); }
.conversation { margin-top: 12px; }
.turn { margin: 10px 0; display: grid; grid-template-columns: 120px 1fr; gap: 12px; }
.turn span { font-weight: 700; color: #475569; }
.turn p { margin: 0; white-space: pre-wrap; max-width: 90ch; }
@media (prefers-color-scheme: dark) {
  .capability-report { background: #0f172a; color: #e5e7eb; }
  th { background: #1e293b; }
  th, td { border-color: #334155; }
  .subtle, .eyebrow { color: #94a3b8; }
  .meta-pills span { border-color: #334155; background: #1e293b; color: #dbeafe; }
  .state-ready { background: #14532d; }
  .state-blocked { background: #7f1d1d; }
  .state-insufficient_evidence { background: #713f12; }
  .state-missing { background: #374151; color: #d1d5db; }
  .snippet { border-color: #475569; background: rgba(15,23,42,0.35); }
  .evidence-card { border-color: #475569; background: rgba(15,23,42,0.35); }
  .reasoning { background: rgba(148,163,184,0.12); }
  .turn span { color: #cbd5e1; }
}
"""
