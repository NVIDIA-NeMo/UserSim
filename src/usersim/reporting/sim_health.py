# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SimHealthBundle — answers "did the simulator do its job?".

Cheap, no model calls. Renders in seconds against a stored trajectory
parquet. Surfaces:

- Failure-mode taxonomy: ``failure_class × failure_attribution``
  groupby with proportions, plus per-locale and per-probe_family
  breakdowns.
- Behavioral fidelity: D1–D4 + per-trajectory MATTR, stratified by
  ``user_interaction_style`` and ``disclosure_style`` (the fields the
  simulator already writes, so the OCEAN-driven design choices are
  inspectable).
- Trajectory yield: counts of ``ok`` / ``completed_with_warnings`` /
  ``failed`` from ``simulation_outcome.status``.
- Persona-grounding rate, fourth-wall trigger rate, gate-retry
  distributions — the sim-side knobs from
  ``simulation_outcome.n_*`` counters.
- Resource profile by model alias (calls / tokens / wall clock).
- Provenance: the set of distinct ``code_sha`` /
  ``nemotron_personas_version`` / ``scenario_prompt_version`` values
  observed in the bundle. Mixing in unexpected ways = drift.

The bundle is a frozen-after-build dataclass; the renderer in
``render.py`` consumes it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from usersim.engine.core.missing import none_for_missing
from usersim.reporting.diagnostics import (
    compute_diagnostics_frame,
    compute_response_length_by_locale,
)
from usersim.reporting.resource import (
    ResourceProfile,
    aggregate_from_dataframe,
)


def _decode_outcome(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


@dataclass
class SimHealthBundle:
    """Structured bundle of simulator-side diagnostics for one trajectory parquet."""

    n_trajectories: int = 0
    status_counts: dict[str, int] = field(default_factory=dict)
    failure_taxonomy: list[dict[str, Any]] = field(default_factory=list)
    failure_taxonomy_by_locale: list[dict[str, Any]] = field(default_factory=list)

    diagnostics_means: dict[str, float | None] = field(default_factory=dict)
    diagnostics_by_interaction_style: list[dict[str, Any]] = field(default_factory=list)

    persona_grounding_rate: float | None = None
    early_stop_rate: float | None = None

    counters_means: dict[str, float] = field(default_factory=dict)
    counters_totals: dict[str, int] = field(default_factory=dict)

    resource_profile: ResourceProfile = field(default_factory=ResourceProfile)

    # Per-locale assistant-response length aggregates. Each row carries:
    # ``locale``, ``n_trajectories``, mean / median / p95 of both
    # ``chars_per_turn`` (computed from conversation_messages) and
    # ``tokens_per_turn`` (computed from
    # ``simulation_outcome.per_model_output_tokens["assistant_model"]`` /
    # ``per_model_calls["assistant_model"]``), plus
    # ``mean_total_tokens_per_trajectory``. The equity diagnostic this
    # surfaces: are non-English locales getting systematically shorter
    # responses for matched workloads?
    response_length_by_locale: list[dict[str, Any]] = field(default_factory=list)

    provenance_observed: dict[str, list[str]] = field(default_factory=dict)
    probe_family_counts: dict[str, int] = field(default_factory=dict)
    locale_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_trajectories": self.n_trajectories,
            "status_counts": dict(self.status_counts),
            "failure_taxonomy": list(self.failure_taxonomy),
            "failure_taxonomy_by_locale": list(self.failure_taxonomy_by_locale),
            "diagnostics_means": dict(self.diagnostics_means),
            "diagnostics_by_interaction_style": list(self.diagnostics_by_interaction_style),
            "persona_grounding_rate": self.persona_grounding_rate,
            "early_stop_rate": self.early_stop_rate,
            "counters_means": dict(self.counters_means),
            "counters_totals": dict(self.counters_totals),
            "resource_profile": self.resource_profile.to_dict(),
            "response_length_by_locale": list(self.response_length_by_locale),
            "provenance_observed": {k: list(v) for k, v in self.provenance_observed.items()},
            "probe_family_counts": dict(self.probe_family_counts),
            "locale_counts": dict(self.locale_counts),
        }


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------


def _explode_outcome_columns(df: Any) -> Any:
    """Decode ``simulation_outcome`` into per-field columns for groupby."""

    out = df.copy()
    if "simulation_outcome" not in out.columns:
        # No structured outcome column — nothing to explode.
        return out
    decoded = out["simulation_outcome"].apply(_decode_outcome)
    out["_status"] = decoded.apply(lambda d: d.get("status"))
    out["_failure_class"] = decoded.apply(lambda d: d.get("failure_class"))
    out["_failure_attribution"] = decoded.apply(lambda d: d.get("failure_attribution"))
    out["_early_stop"] = decoded.apply(lambda d: bool(d.get("early_stop", False)))
    counter_keys = [
        "n_turns",
        "n_tool_calls",
        "n_user_query_attempts",
        "n_user_followup_retries",
        "n_assistant_inline_failures",
        "n_assistant_retries",
        "n_api_response_rerolls",
        "n_fourth_wall_triggers",
        "n_user_role_violations",
    ]
    for k in counter_keys:
        out[f"_{k}"] = decoded.apply(lambda d, _k=k: d.get(_k))
    out["_provenance"] = decoded.apply(lambda d: d.get("provenance") or {})
    return out


def _build_failure_taxonomy(df_exp: Any) -> list[dict[str, Any]]:
    """``groupby(failure_class × failure_attribution).size()`` with proportions."""

    failed = df_exp[df_exp["_failure_class"].notna()].copy()
    if failed.empty:
        return []
    n = len(df_exp) or 1
    grp = failed.groupby(["_failure_class", "_failure_attribution"], dropna=False).size().reset_index(name="n")
    grp["proportion"] = (grp["n"] / n).round(4)
    grp = grp.sort_values("n", ascending=False)
    out: list[dict[str, Any]] = []
    for row in map(none_for_missing, grp.to_dict(orient="records")):
        out.append(
            {
                "failure_class": row["_failure_class"],
                "failure_attribution": row["_failure_attribution"],
                "n": int(row["n"]),
                "proportion": float(row["proportion"]),
            }
        )
    return out


def _build_failure_taxonomy_by_locale(df_exp: Any) -> list[dict[str, Any]]:
    if "locale" not in df_exp.columns:
        return []
    failed = df_exp[df_exp["_failure_class"].notna()].copy()
    if failed.empty:
        return []
    grp = (
        failed.groupby(["locale", "_failure_class", "_failure_attribution"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["locale", "n"], ascending=[True, False])
    )
    return [
        {
            "locale": row["locale"],
            "failure_class": row["_failure_class"],
            "failure_attribution": row["_failure_attribution"],
            "n": int(row["n"]),
        }
        for row in map(none_for_missing, grp.to_dict(orient="records"))
    ]


def _build_diagnostics_means(df_diag: Any) -> dict[str, float | None]:
    cols = ["d1_front_loading", "d2_polite_fraction", "d3_verbosity_cv", "d4_frustration_markers", "mattr_assistant"]
    out: dict[str, float | None] = {}
    for c in cols:
        if c not in df_diag.columns:
            out[c] = None
            continue
        s = df_diag[c].dropna()
        out[c] = round(float(s.mean()), 4) if len(s) > 0 else None
    return out


def _build_diagnostics_by_interaction_style(df_diag: Any) -> list[dict[str, Any]]:
    if "user_interaction_style" not in df_diag.columns:
        return []
    cols = ["d1_front_loading", "d2_polite_fraction", "d3_verbosity_cv", "d4_frustration_markers", "mattr_assistant"]
    available = [c for c in cols if c in df_diag.columns]
    if not available:
        return []
    grp = df_diag.groupby("user_interaction_style", dropna=False)[available].mean(numeric_only=True).reset_index()
    # ``mean(numeric_only=True)`` silently drops columns whose values are
    # all-None (object dtype), so the surviving metrics are exactly
    # ``grp.columns - {"user_interaction_style"}``. Iterate over those
    # rather than ``available`` to avoid KeyError on the dropped ones.
    metric_cols = [c for c in grp.columns if c != "user_interaction_style"]
    out: list[dict[str, Any]] = []
    for row in map(none_for_missing, grp.to_dict(orient="records")):
        entry: dict[str, Any] = {
            "user_interaction_style": row["user_interaction_style"],
        }
        for c in metric_cols:
            v = row[c]
            entry[c] = None if v is None else round(float(v), 4)
        out.append(entry)
    return out


def _provenance_observed(df_exp: Any) -> dict[str, list[str]]:
    """Distinct (sorted) values seen for each provenance field."""
    if "_provenance" not in df_exp.columns:
        return {}
    keys = ["nemotron_personas_version", "scenario_prompt_version", "code_sha"]
    seen: dict[str, set] = {k: set() for k in keys}
    for prov in df_exp["_provenance"]:
        if not isinstance(prov, dict):
            continue
        for k in keys:
            v = prov.get(k)
            if v is not None:
                seen[k].add(str(v))
    return {k: sorted(v) for k, v in seen.items()}


def _value_counts_dict(series: Any) -> dict[str, int]:
    if series is None or len(series) == 0:
        return {}
    vc = series.fillna("__null__").value_counts()
    return {str(k): int(v) for k, v in vc.items()}


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_sim_health(df: Any) -> SimHealthBundle:
    """Build the simulator-side health bundle from a trajectory frame.

    ``df`` is the trajectory parquet read into a pandas DataFrame.
    Missing columns are tolerated — affected aggregates default to
    empty / None rather than raising.
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"build_sim_health expects a pandas DataFrame, got {type(df).__name__}")

    bundle = SimHealthBundle()
    bundle.n_trajectories = len(df)
    if bundle.n_trajectories == 0:
        return bundle

    # Diagnostics frame is independent of outcome decoding.
    df_diag = compute_diagnostics_frame(df) if "conversation_messages" in df.columns else df.copy()
    bundle.diagnostics_means = _build_diagnostics_means(df_diag)
    bundle.diagnostics_by_interaction_style = _build_diagnostics_by_interaction_style(df_diag)

    # Outcome-derived aggregates.
    df_exp = _explode_outcome_columns(df)
    if "_status" in df_exp.columns:
        bundle.status_counts = _value_counts_dict(df_exp["_status"])
        bundle.failure_taxonomy = _build_failure_taxonomy(df_exp)
        bundle.failure_taxonomy_by_locale = _build_failure_taxonomy_by_locale(df_exp)
        bundle.early_stop_rate = round(float(df_exp["_early_stop"].mean()), 4)
        counter_keys = [
            "n_turns",
            "n_tool_calls",
            "n_user_query_attempts",
            "n_user_followup_retries",
            "n_assistant_inline_failures",
            "n_assistant_retries",
            "n_api_response_rerolls",
            "n_fourth_wall_triggers",
            "n_user_role_violations",
        ]
        for k in counter_keys:
            col = f"_{k}"
            if col not in df_exp.columns:
                continue
            s = df_exp[col].dropna()
            if len(s) == 0:
                continue
            bundle.counters_means[k] = round(float(s.mean()), 3)
            bundle.counters_totals[k] = int(s.sum())
        bundle.provenance_observed = _provenance_observed(df_exp)

    # Persona-grounding rate from the simulator's side-effect column.
    if "persona_grounding" in df.columns:
        s = df["persona_grounding"].dropna()
        if len(s) > 0:
            bundle.persona_grounding_rate = round(float(s.astype(bool).mean()), 4)

    # Scenario / locale breakdowns.
    if "probe_family" in df.columns:
        bundle.probe_family_counts = _value_counts_dict(df["probe_family"])
    elif "probe_type" in df.columns:
        bundle.probe_family_counts = _value_counts_dict(df["probe_type"])
    if "locale" in df.columns:
        bundle.locale_counts = _value_counts_dict(df["locale"])

    bundle.resource_profile = aggregate_from_dataframe(df)

    # Equity diagnostic: per-locale response length. Requires all three
    # columns (locale + conversation_messages + simulation_outcome);
    # silently skip if any are missing rather than raise.
    if all(c in df.columns for c in ("locale", "conversation_messages", "simulation_outcome")):
        bundle.response_length_by_locale = compute_response_length_by_locale(df)

    return bundle
