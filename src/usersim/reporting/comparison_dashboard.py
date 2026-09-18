# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-model comparison dashboard — Vega-Lite + React render layer.

Mirrors the shape of :func:`reporting.dashboard.write_capability_dashboard_artifacts`
but for the cross-model comparison surface. Reads a
:class:`reporting.comparison.ComparisonReport` and writes three
artifacts under ``out_dir``:

- ``comparison_manifest.json`` — full ``ComparisonReport.to_dict()``.
- ``comparison_matrix.json`` — long-form ``cells`` list for downstream tools.
- ``index.html`` — self-contained: embeds JSON + Vega-Lite specs in
  ``<script type="application/json">`` blocks, ships a static-fallback
  DOM (so the verdict reads offline), then a module script that imports
  React, react-dom, and vega-embed at exact pinned versions from
  ``https://esm.sh`` (see ``reporting.dashboard``) and
  renders the layered narrative.

Six layers, top-down:

- Layer 0 — Sim Health Comparison (3 mini bars per model)
- Layer 1 — The Verdict (leaderboard + verbosity headline)
- Layer 2 — Quality vs Verbosity (Pareto scatter, frontier overlay)
- Layer 3 — Verbosity Profile (bar + per-locale token heatmap)
- Layer 4 — Per-Locale Capability Comparison (locale-tabbed grouped bars)
- Layer 5 — Per-Capability Locale Comparison (collapsible flipped facet)
- Layer 6 — Coverage Matrix (collapsible)

Cross-cutting design discipline (see plan doc):

- Locked color + shape per model (via :func:`assign_palette`).
- Capabilities in registry order (never sorted by score).
- Reference rules at thresholds; quadrant medians on the Pareto plot.
- Identical tooltip schema across charts; deep-link to the per-run
  dashboard (``../report/run=<id>/index.html``) on every per-cell tooltip.
- Shared :data:`_VEGA_THEME` config merged into every spec.
"""

from __future__ import annotations

import html
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Optional

from usersim.reporting.comparison import (
    MEASURED_STATES,
    ROLLUP_EXCLUDED_CAPABILITIES,
    ComparisonCell,
    ComparisonReport,
    ModelRollup,
)
from usersim.reporting.dashboard import (
    _CSP,
    _DASHBOARD_CSS,
    _ESM_REACT,
    _ESM_REACT_DOM,
    _ESM_VEGA_EMBED,
    _locale_label,
)

# ---------------------------------------------------------------------------
# Locked palette + shape (cross-cutting design rule 1)
# ---------------------------------------------------------------------------

# Tableau 10 categorical palette — picked for dark-theme legibility and
# colorblind safety. 8 entries cap the comfortable scale; comparisons
# with >8 distinct models will wrap and need shape disambiguation.
_PALETTE: tuple[str, ...] = (
    "#4e79a7",  # blue
    "#f28e2c",  # orange
    "#e15759",  # red
    "#76b7b2",  # teal
    "#59a14f",  # green
    "#edc949",  # yellow
    "#af7aa1",  # purple
    "#ff9da7",  # pink
)

# Locale palette — deliberately distinct from _PALETTE (model colors) so
# Layer 7's per-model breakdown (where bars are grouped BY LOCALE within
# each capability) can't be visually confused with the per-locale layers
# above (where bars are grouped BY MODEL).
#
# We landed on mid-saturation jewel tones (L≈55%) rather than the prior
# pastel set (L≈85%) — the pastels washed out badly against the dashboard's
# dark background (#111827) and were near-impossible to tell apart side
# by side in a grouped bar chart. Each color sits in a different hue
# region from the Tableau-10 model palette, so a viewer never has to
# ask "is this row colored by model or by locale?". 8 entries match
# the standard locale set; wrap on overflow.
_LOCALE_PALETTE: tuple[str, ...] = (
    "#5b8bbf",  # muted blue (en_US)
    "#d97757",  # warm terracotta (en_IN)
    "#c25c7d",  # rose (en_SG)
    "#5fae8a",  # sage green (fr_FR)
    "#d99c4c",  # burnished gold (hi_Deva_IN)
    "#a772c0",  # muted plum (ja_JP)
    "#7d72c9",  # periwinkle (ko_KR)
    "#b1a44c",  # olive (pt_BR)
)

# Per-model color overrides applied AFTER the alphabetical palette
# assignment. Branding intent: nemotron family in NVIDIA-green tones —
# Ultra in the canonical NVIDIA green, Super in a paler shade so the two
# read as related but distinguishable. Match is by substring on the
# display_label (so disambiguated labels like "nemotron-3-ultra-preview
# (run=...)" still resolve). When an override fires, the palette slot
# the model would have taken is freed up for other models.
_MODEL_COLOR_OVERRIDES: tuple[tuple[str, str], ...] = (
    ("nemotron-3-ultra", "#76b900"),  # canonical NVIDIA green
    ("nemotron-3-super", "#a8d63d"),  # paler NVIDIA green
)

# Hex colors that read as "NVIDIA-ish green" and must therefore stay
# reserved for branded models only. Without this guard, the Tableau-10
# slot ``#59a14f`` (chartreuse green) would get handed to any non-NVIDIA
# model whose alphabetical sort lands at index 4 -- e.g. qwen3.6 in
# 6-model panels alongside two gemmas and two nemotrons. That created
# a misleading "this third-party model is also NVIDIA-branded" signal.
_RESTRICTED_GREEN_HEX_COLORS: frozenset[str] = frozenset(
    {
        "#59a14f",  # Tableau-10 chartreuse green (palette slot 4)
        "#76b900",  # canonical NVIDIA green (override target)
        "#a8d63d",  # paler NVIDIA green (override target)
    }
)

# Substrings that grant a label "green eligibility" -- i.e., the model
# is allowed to land on a restricted green hex. Match is case-insensitive
# on the display_label so e.g. "nemotron-3-Nano (run=...)" still passes.
# Mirrors the NVIDIA naming convention: nano / super / ultra are the
# tier suffixes that brand the model as part of the family.
_GREEN_ELIGIBLE_SUBSTRINGS: tuple[str, ...] = ("nano", "super", "ultra")


def _is_green_eligible(label: str) -> bool:
    """True iff ``label`` may be assigned a NVIDIA-green hex."""
    haystack = label.lower()
    return any(needle in haystack for needle in _GREEN_ELIGIBLE_SUBSTRINGS)


# Vega-Lite shape vocabulary — paired with palette for dual-channel
# encoding on the Pareto scatter.
_SHAPES: tuple[str, ...] = (
    "circle",
    "square",
    "triangle-up",
    "diamond",
    "cross",
    "triangle-down",
    "triangle-right",
    "triangle-left",
)


def assign_palette(
    labels: Iterable[str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Stable color + shape assignment for a set of model labels.

    Process:

    1. Apply :data:`_MODEL_COLOR_OVERRIDES` first (substring match) so
       branded models get their fixed hue regardless of how many other
       models are in the set.
    2. The remaining labels are sorted alphabetically and assigned
       colors from :data:`_PALETTE`. Two filters apply per-label:
       (a) palette slots already claimed by a branded override are
       always skipped (so we never visually clash a branded NVIDIA-green
       model against another model that happened to land on a similar
       palette slot); (b) green slots in
       :data:`_RESTRICTED_GREEN_HEX_COLORS` are only available to
       labels that pass :func:`_is_green_eligible` (substring match
       on ``nano`` / ``super`` / ``ultra``). Without (b), e.g. qwen
       could pick up Tableau chartreuse green and read as
       NVIDIA-branded.
    3. Shapes are always assigned from :data:`_SHAPES` in alphabetical
       order — the shape encoding is purely for colorblind disambiguation
       on the Pareto plot and doesn't carry brand meaning.
    """
    unique = sorted({str(label) for label in labels if label})

    # Pass 1: branded overrides.
    colors: dict[str, str] = {}
    used_hexes: set[str] = set()
    for label in unique:
        for needle, hex_color in _MODEL_COLOR_OVERRIDES:
            if needle in label:
                colors[label] = hex_color
                used_hexes.add(hex_color)
                break

    # Pass 2: per-label palette walk. We can't pre-compute one
    # ``available`` list because each label may have a different set
    # of allowed slots (green-eligible labels see all colors;
    # non-eligible labels see the palette minus
    # ``_RESTRICTED_GREEN_HEX_COLORS``). Walk a per-label rotation
    # cursor through ``_PALETTE`` until we find an allowed unused
    # slot, falling back to the first allowed slot (with reuse) only
    # if the panel grew beyond the palette size.
    fill_idx = 0
    for label in unique:
        if label in colors:
            continue
        green_ok = _is_green_eligible(label)
        # First try: distinct slot. Walk the palette starting at
        # ``fill_idx`` and skip slots that are either claimed by
        # branded overrides or restricted-green for an ineligible
        # label.
        chosen: str | None = None
        for offset in range(len(_PALETTE)):
            candidate = _PALETTE[(fill_idx + offset) % len(_PALETTE)]
            if candidate in used_hexes:
                continue
            if not green_ok and candidate in _RESTRICTED_GREEN_HEX_COLORS:
                continue
            chosen = candidate
            break
        if chosen is None:
            # Wrap-around: every slot is either branded or restricted-
            # for-this-label. Reuse one we've already given out, but
            # still respect the green ban so we don't color a non-NVIDIA
            # model green even when the palette is exhausted.
            for candidate in _PALETTE:
                if not green_ok and candidate in _RESTRICTED_GREEN_HEX_COLORS:
                    continue
                chosen = candidate
                break
            if chosen is None:
                # Pathological: every palette slot is restricted-green
                # AND the label is ineligible. Fall back to the first
                # palette slot to avoid raising; in practice this
                # branch is unreachable because ``_PALETTE`` carries
                # 7 non-green slots.
                chosen = _PALETTE[0]
        colors[label] = chosen
        used_hexes.add(chosen)
        fill_idx += 1

    shapes = {label: _SHAPES[i % len(_SHAPES)] for i, label in enumerate(unique)}
    return colors, shapes


# ---------------------------------------------------------------------------
# Vega-Lite theme — merged into every spec via ``config:``
# ---------------------------------------------------------------------------


_VEGA_THEME: dict[str, Any] = {
    "background": "#111827",
    "title": {"color": "#e5e7eb", "fontSize": 14, "fontWeight": 600},
    "view": {"stroke": "transparent"},
    "axis": {
        "labelColor": "#cbd5e1",
        "titleColor": "#94a3b8",
        "gridColor": "#1f2937",
        "domainColor": "#475569",
        "tickColor": "#475569",
        "labelFontSize": 11,
        "titleFontSize": 12,
        "titleFontWeight": 600,
    },
    "legend": {
        "labelColor": "#cbd5e1",
        "titleColor": "#94a3b8",
        "labelFontSize": 11,
        "titleFontSize": 12,
        "symbolSize": 80,
    },
    "header": {
        "labelColor": "#94a3b8",
        "titleColor": "#94a3b8",
        "labelFontSize": 11,
    },
    "range": {"category": list(_PALETTE)},
}


def _spec_with_theme(spec: dict[str, Any]) -> dict[str, Any]:
    """Merge ``_VEGA_THEME`` into ``spec.config`` (creating it if absent)."""
    out = dict(spec)
    cfg = dict(out.get("config") or {})
    for k, v in _VEGA_THEME.items():
        if k in cfg and isinstance(cfg[k], dict) and isinstance(v, dict):
            merged = dict(v)
            merged.update(cfg[k])
            cfg[k] = merged
        else:
            cfg.setdefault(k, v)
    out["config"] = cfg
    out["$schema"] = "https://vega.github.io/schema/vega-lite/v5.json"
    return out


# ---------------------------------------------------------------------------
# Helpers shared across spec builders
# ---------------------------------------------------------------------------


def _models_sorted_by_score(report: ComparisonReport) -> list[str]:
    """Sort ``display_label``s by mean_normalized_score descending; None last."""
    rollups = list(report.model_summary)
    rollups.sort(
        key=lambda r: (
            -1 if r.mean_normalized_score is None else 1,
            -(r.mean_normalized_score or 0.0),
        )
    )
    return [r.display_label for r in rollups]


def _models_sorted_alpha(report: ComparisonReport) -> list[str]:
    return sorted({r.display_label for r in report.model_summary})


def _color_scale(colors: dict[str, str], domain: list[str]) -> dict[str, Any]:
    return {
        "domain": list(domain),
        "range": [colors.get(d, _PALETTE[0]) for d in domain],
    }


def _shape_scale(shapes: dict[str, str], domain: list[str]) -> dict[str, Any]:
    return {
        "domain": list(domain),
        "range": [shapes.get(d, _SHAPES[0]) for d in domain],
    }


def _deeplink_for(run_id: str) -> str:
    """Relative URL from output/comparison/run=<cmp>/ to per-run dashboard."""
    return f"../../report/run={run_id}/index.html"


# ---------------------------------------------------------------------------
# Chart title helper
# ---------------------------------------------------------------------------


def _chart_title(text: str) -> dict[str, Any]:
    """Standard chart-title config rendered above the chart's plot area.

    Vega-Lite's default ``orient: top`` + ``anchor: middle`` puts the
    title centred above the *plot frame* (the bars themselves, not
    the bounding box that includes the y-axis label gutter). With
    ``anchor: start``, Vega anchors the title to the LEFT EDGE of the
    plot frame -- which, on charts with long y-axis labels, sits to the
    right of the panel's left edge and reads like a row label rather
    than a chart title. ``anchor: middle`` (default) gives the standard
    "title on top of the chart" look with no special positioning.

    Layer numbers belong to the section ``<h2>`` headers, not to the
    chart title, which carries only the chart's own descriptor.
    """
    return {"text": text, "anchor": "middle"}


# ---------------------------------------------------------------------------
# Layer 0 — Sim Health Comparison
# ---------------------------------------------------------------------------


def _spec_sim_health_metric(
    report: ComparisonReport,
    colors: dict[str, str],
    *,
    metric_field: str,
    metric_title: str,
    show_y_labels: bool,
    x_domain: tuple[float, float] | None = None,
    x_axis_format: str = ".1f",
) -> dict[str, Any]:
    """One sim-health metric (status_failure / persona_grounding / early_stop).

    Three of these are rendered side by side in a CSS grid in the
    Sim Health row, so each chart can reflow independently when the
    window resizes. Y-axis labels are shown only on the first chart
    in the row — the others share the same model order via the
    locked palette + locked sort.

    ``x_domain`` defaults to None, which scales each chart to its
    own data extent (with a 1.2× ceiling). A fixed 0-1 domain suits the
    OK-rate / persona / early-stop metrics, which typically span [0, 1];
    the auto-scaled domain is for the failure-rate flavour, where values
    cluster below 10% and would otherwise collapse into invisibility.
    """
    label_order = _models_sorted_by_score(report)
    values = []
    for r in report.model_summary:
        v = getattr(r, metric_field)
        values.append(
            {
                "display_label": r.display_label,
                "value": v,
                "value_str": (None if v is None else f"{v:.2%}"),
                "has_sim_health": r.has_sim_health,
                "deeplink": _deeplink_for(r.run_id),
            }
        )
    y_axis: dict[str, Any] | None = {"labelLimit": 280, "labelFontWeight": 600} if show_y_labels else None
    if x_domain is None:
        max_v = max(
            (v["value"] for v in values if isinstance(v["value"], (int, float))),
            default=1.0,
        )
        # Pad the data extent by 20% so the longest bar doesn't kiss
        # the right edge; floor at 0.05 (5%) to keep an empty panel
        # from rendering as a hairline.
        x_max = max(max_v * 1.2, 0.05) if max_v > 0 else 0.05
        x_domain_value = [0, x_max]
    else:
        x_domain_value = list(x_domain)
    spec = {
        "title": _chart_title(metric_title),
        # Match the Performance row's bar height ({"step": 32}) so the
        # Sim Health and Performance rows are visually consistent.
        "width": 280,
        "height": {"step": 32},
        "data": {"values": values},
        "mark": {"type": "bar", "tooltip": True, "cornerRadiusEnd": 3},
        "encoding": {
            "y": {
                "field": "display_label",
                "type": "nominal",
                "sort": label_order,
                "title": None,
                "axis": y_axis,
            },
            "x": {
                "field": "value",
                "type": "quantitative",
                "scale": {"domain": x_domain_value},
                "axis": {"format": x_axis_format, "title": None},
            },
            "color": {
                "field": "display_label",
                "type": "nominal",
                "scale": _color_scale(colors, label_order),
                "legend": None,
            },
            "opacity": {
                "condition": {"test": "datum.has_sim_health", "value": 1.0},
                "value": 0.15,
            },
            "tooltip": [
                {"field": "display_label", "title": "Model"},
                {"field": "value_str", "title": metric_title},
                {"field": "deeplink", "title": "Open per-run dashboard"},
            ],
        },
    }
    return _spec_with_theme(spec)


# Categorical palette for the failure-mode stacked bar. The classes
# themselves are open-ended (the trajectory aggregator emits whatever
# failure_class strings the simulator records, e.g.
# ``infrastructure_error``, ``user_followup_gate_exhausted``,
# ``user_query_gate_exhausted``, ``max_turns_reached``,
# ``persona_break``, ``parse_error``...). We pin the most common ones
# to specific hues so the legend reads consistently across reports;
# unfamiliar classes fall through to a neutral cycle. Palette is
# intentionally diagnostic-flavoured (warmer reds/ambers for sim
# breakage, cooler purples for assistant misbehaviour) so a reviewer
# can read the bar's *colour* before its *legend*.
_FAILURE_CLASS_COLORS: dict[str, str] = {
    "infrastructure_error": "#d04848",  # red — provider / network
    "max_turns_reached": "#d49a3a",  # amber — ran on
    "user_query_gate_exhausted": "#c5814b",  # warm tan — sim couldn't open
    "user_followup_gate_exhausted": "#c2b13a",  # mustard — sim gave up
    "persona_break": "#9b5cb8",  # violet — assistant stepped out
    "parse_error": "#c9617a",  # rose — assistant produced garbage
    "tool_call_error": "#a07acc",  # lavender — tool plumbing
    "fourth_wall_break": "#8077c2",  # periwinkle — meta failure
}
# Fallback palette for classes not in the pinned dict above. Keeps
# the chart deterministic without forcing an exhaustive lookup table.
_FAILURE_CLASS_FALLBACK_PALETTE: tuple[str, ...] = (
    "#7a8d9c",
    "#b3835c",
    "#c0937c",
    "#9c7d6f",
    "#857a90",
    "#a0a07a",
    "#7d9082",
    "#a09080",
)


def _failure_class_color_scale(classes: list[str]) -> dict[str, Any]:
    """Vega-Lite color scale mapping each failure_class to a stable hue.

    Pinned classes always resolve to the same colour regardless of
    panel composition; unknown classes wrap through the fallback
    palette in registration order so the same panel always renders
    the same scale.
    """
    fallback_idx = 0
    range_: list[str] = []
    for cls in classes:
        if cls in _FAILURE_CLASS_COLORS:
            range_.append(_FAILURE_CLASS_COLORS[cls])
        else:
            range_.append(_FAILURE_CLASS_FALLBACK_PALETTE[fallback_idx % len(_FAILURE_CLASS_FALLBACK_PALETTE)])
            fallback_idx += 1
    return {"domain": classes, "range": range_}


def _spec_failure_taxonomy(report: ComparisonReport) -> dict[str, Any]:
    """Stacked horizontal bar of failure proportions per model.

    The total bar length encodes ``1 - sim_status_ok_rate`` (the
    failure fraction); each stacked segment is the proportion of
    trajectories that hit a specific failure_class. Same y-axis
    ordering as the rate charts above so the eye reads cleanly
    across the row.

    Why proportions, not counts: trajectory budgets vary slightly
    across runs and across locales, so absolute counts would dilute
    the comparison. Proportion-of-trajectories normalises out panel
    size while preserving the "is this a *model* bug or a *sim* bug?"
    triage signal.
    """
    label_order = _models_sorted_by_score(report)

    # Walk every (model, failure_class) pair so the legend covers the
    # full panel, not just the first model's classes. Stable insertion
    # order = order classes appear in the manifests (they're already
    # sorted descending by count in the per-run aggregator).
    classes_seen: list[str] = []
    seen: set[str] = set()
    for r in report.model_summary:
        for entry in r.failure_taxonomy:
            cls = entry.get("failure_class")
            if isinstance(cls, str) and cls not in seen:
                seen.add(cls)
                classes_seen.append(cls)

    values: list[dict[str, Any]] = []
    for r in report.model_summary:
        for entry in r.failure_taxonomy:
            cls = entry.get("failure_class")
            if not isinstance(cls, str):
                continue
            proportion = entry.get("proportion")
            if not isinstance(proportion, (int, float)):
                # Defensive: derive proportion from n if missing.
                n = entry.get("n") or 0
                proportion = (n / r.n_trajectories) if r.n_trajectories else 0.0
            n_count = entry.get("n", 0)
            attribution = entry.get("failure_attribution", "—")
            values.append(
                {
                    "display_label": r.display_label,
                    "failure_class": cls,
                    "proportion": float(proportion),
                    "proportion_pct": f"{100 * float(proportion):.2f}%",
                    "n": int(n_count) if isinstance(n_count, (int, float)) else 0,
                    "failure_attribution": attribution,
                    "deeplink": _deeplink_for(r.run_id),
                }
            )

    # Cap the x-axis at the panel-wide max failure fraction with a
    # 20% padding so the longest bar doesn't kiss the right edge.
    # Avoids a fixed [0, 1] domain that would shrink the bars to
    # invisibility at the typical 1-5% failure rates.
    max_total = 0.0
    by_label_total: dict[str, float] = {}
    for v in values:
        by_label_total[v["display_label"]] = by_label_total.get(v["display_label"], 0.0) + v["proportion"]
    if by_label_total:
        max_total = max(by_label_total.values())
    x_max = max(max_total * 1.2, 0.01)

    spec = {
        "title": _chart_title("Failure breakdown by class"),
        # Match the rate charts' bar height ({"step": 32}) so the
        # 2x2 Sim Health grid reads as a coherent set; matching width
        # (280) keeps the bars sized consistently with the rate
        # charts that pair beside / above it.
        "width": 280,
        "height": {"step": 32},
        "data": {"values": values},
        "mark": {"type": "bar", "tooltip": True, "cornerRadiusEnd": 3},
        "encoding": {
            "y": {
                "field": "display_label",
                "type": "nominal",
                "sort": label_order,
                "title": None,
                # Keep y-labels visible on this chart -- it's the
                # second row, not the first; without labels the
                # stacked bars would float disconnected from the
                # rate charts above.
                "axis": {"labelLimit": 280, "labelFontWeight": 600},
            },
            "x": {
                "field": "proportion",
                "type": "quantitative",
                "aggregate": "sum",
                "stack": "zero",
                "scale": {"domain": [0, x_max]},
                "axis": {"format": ".0%", "title": "Failure rate"},
            },
            "color": {
                "field": "failure_class",
                "type": "nominal",
                "scale": _failure_class_color_scale(classes_seen),
                "legend": {
                    "title": "Failure class",
                    "orient": "right",
                    "labelLimit": 220,
                    "symbolType": "square",
                },
            },
            "order": {"field": "failure_class", "type": "nominal"},
            "tooltip": [
                {"field": "display_label", "title": "Model"},
                {"field": "failure_class", "title": "Class"},
                {"field": "failure_attribution", "title": "Attribution"},
                {"field": "n", "title": "Trajectories", "format": ","},
                {"field": "proportion_pct", "title": "Share of run"},
                {"field": "deeplink", "title": "Open per-run dashboard"},
            ],
        },
    }
    return _spec_with_theme(spec)


# ---------------------------------------------------------------------------
# Layer 1 — The Verdict (leaderboard + verbosity headline)
# ---------------------------------------------------------------------------


def _leaderboard_annotation(r: ModelRollup) -> str:
    if r.mean_normalized_score is None:
        return "no measured cells"
    return f"{r.mean_normalized_score:.2f}"


def _spec_leaderboard(report: ComparisonReport, colors: dict[str, str]) -> dict[str, Any]:
    label_order = _models_sorted_by_score(report)
    values = []
    for r in report.model_summary:
        values.append(
            {
                "display_label": r.display_label,
                "score": r.mean_normalized_score,
                "score_pct": (None if r.mean_normalized_score is None else round(r.mean_normalized_score * 100, 1)),
                "annotation": _leaderboard_annotation(r),
                "deeplink": _deeplink_for(r.run_id),
                "n_measured": r.n_cells_measured,
                "n_eligible": r.n_cells_eligible,
                "n_ready": r.n_cells_ready,
            }
        )
    spec = {
        "title": _chart_title("Overall capability score"),
        "width": 280,
        "height": {"step": 32},
        "data": {"values": values},
        "layer": [
            {
                "mark": {"type": "bar", "tooltip": True, "cornerRadiusEnd": 3},
                "encoding": {
                    "y": {
                        "field": "display_label",
                        "type": "nominal",
                        "sort": label_order,
                        "title": None,
                        # Labels live on the cleared chart (first in
                        # the Performance row); the leaderboard hides
                        # them so the row reads as a coherent triplet
                        # instead of three duplicated y-axis gutters.
                        "axis": None,
                    },
                    "x": {
                        "field": "score",
                        "type": "quantitative",
                        "scale": {"domain": [0, 1]},
                        "axis": {"format": ".1f", "title": "Mean normalized score"},
                    },
                    "color": {
                        "field": "display_label",
                        "type": "nominal",
                        "scale": _color_scale(colors, label_order),
                        "legend": None,
                    },
                    "tooltip": [
                        {"field": "display_label", "title": "Model"},
                        {"field": "score_pct", "title": "Score (%)"},
                        {"field": "n_measured", "title": "Measured"},
                        {"field": "n_eligible", "title": "Eligible"},
                        {"field": "n_ready", "title": "Passing"},
                        {"field": "deeplink", "title": "Open per-run dashboard"},
                    ],
                },
            },
            {
                "mark": {
                    "type": "text",
                    "align": "left",
                    "dx": 6,
                    "color": "#cbd5e1",
                    "fontSize": 11,
                },
                "encoding": {
                    "y": {"field": "display_label", "type": "nominal", "sort": label_order},
                    "x": {"field": "score", "type": "quantitative"},
                    "text": {"field": "annotation", "type": "nominal"},
                },
            },
        ],
    }
    return _spec_with_theme(spec)


def _spec_verbosity_headline(report: ComparisonReport, colors: dict[str, str]) -> dict[str, Any]:
    """Layer 1b — assistant output tokens / conv, stacked by reasoning subset.

    The bar's full length encodes ``assistant_output_tokens / n_traj``
    (the verbosity metric). Inside each bar, the reasoning portion
    (``assistant_reasoning_tokens / n_traj``) is rendered in a paler
    shade so reviewers can see at a glance how much of the verbosity
    comes from invisible thinking vs visible response text.

    Why not include input tokens here? Input is dominated by prompt /
    persona / probe context, not by the model's verbosity. Including
    them dilutes the signal a thinking model produces (e.g., enabling
    Gemma's reasoning channel doubled output but only nudged total
    by ~15% on one A/B run, which would have looked like noise).
    """
    label_order = _models_sorted_by_score(report)
    by_output = sorted(
        report.model_summary,
        key=lambda r: -r.assistant_output_tokens_per_conversation,
    )
    rank_by_label = {r.display_label: i + 1 for i, r in enumerate(by_output)}
    values = []
    for r in report.model_summary:
        rank = rank_by_label.get(r.display_label, 0)
        output_per_conv = r.assistant_output_tokens_per_conversation
        reasoning_per_conv = r.assistant_reasoning_tokens_per_conversation
        # Visible output = output minus reasoning. Clamp at 0 in case
        # of pathological rounding (the noise floor in resource.py
        # zeroes alias-level reasoning when the ratio is implausible,
        # so this should never go negative in practice).
        visible_per_conv = max(output_per_conv - reasoning_per_conv, 0.0)
        # Annotation reads e.g. "8,495 (5,417 reasoning)" -- the bare
        # number is output / conv (the bar's full length); the
        # parenthetical is the reasoning subset, only shown when
        # non-zero so non-thinking models stay uncluttered.
        if reasoning_per_conv > 0 and output_per_conv > 0:
            annotation = f"{int(round(output_per_conv)):,} ({int(round(reasoning_per_conv)):,} reasoning)"
        else:
            annotation = f"{int(round(output_per_conv)):,}"
        # Two stack rows per model: visible bottom, reasoning on top.
        # ``order`` controls draw order so reasoning sits to the
        # right of the visible portion in the stacked horizontal bar.
        values.append(
            {
                "display_label": r.display_label,
                "segment": "visible",
                "value": visible_per_conv,
                "order": 0,
                "output_per_conv": output_per_conv,
                "reasoning_per_conv": reasoning_per_conv,
                "annotation": annotation,
                "rank": rank,
                "deeplink": _deeplink_for(r.run_id),
            }
        )
        values.append(
            {
                "display_label": r.display_label,
                "segment": "reasoning",
                "value": reasoning_per_conv,
                "order": 1,
                "output_per_conv": output_per_conv,
                "reasoning_per_conv": reasoning_per_conv,
                "annotation": annotation,
                "rank": rank,
                "deeplink": _deeplink_for(r.run_id),
            }
        )
    x_max = max(
        (r.assistant_output_tokens_per_conversation for r in report.model_summary),
        default=1.0,
    )
    x_max = max(x_max * 1.2, 1.0)
    # Per-model color drives the bar fill; the reasoning segment is
    # shown in a contrasting pale tint so reviewers can read the
    # subset visually without a second hue per model. The stacked
    # bar uses ``opacity`` (mark-level binding to ``segment``) for
    # the same reason -- one pass for visible, one for reasoning.
    spec = {
        "title": _chart_title("Output tokens per conversation"),
        "width": 280,
        "height": {"step": 32},
        "data": {"values": values},
        "layer": [
            {
                "mark": {"type": "bar", "tooltip": True, "cornerRadiusEnd": 3},
                "encoding": {
                    "y": {
                        "field": "display_label",
                        "type": "nominal",
                        "sort": label_order,
                        "title": None,
                        "axis": None,
                    },
                    "x": {
                        "field": "value",
                        "type": "quantitative",
                        "aggregate": "sum",
                        "stack": "zero",
                        "scale": {"domain": [0, x_max]},
                        "axis": {"title": "Output tokens / conversation"},
                    },
                    "order": {"field": "order", "type": "ordinal"},
                    "color": {
                        "field": "display_label",
                        "type": "nominal",
                        "scale": _color_scale(colors, label_order),
                        "legend": None,
                    },
                    "opacity": {
                        "field": "segment",
                        "type": "nominal",
                        # Visible segment full saturation, reasoning segment paled.
                        "scale": {"domain": ["visible", "reasoning"], "range": [1.0, 0.45]},
                        "legend": None,
                    },
                    "tooltip": [
                        {"field": "display_label", "title": "Model"},
                        {"field": "segment", "title": "Segment"},
                        {"field": "value", "title": "Tokens (this segment)", "format": ",.0f"},
                        {"field": "output_per_conv", "title": "Output / conv", "format": ",.0f"},
                        {"field": "reasoning_per_conv", "title": "Reasoning / conv", "format": ",.0f"},
                        {"field": "rank", "title": "Verbosity rank"},
                        {"field": "deeplink", "title": "Open per-run dashboard"},
                    ],
                },
            },
            {
                # Single text label per model, anchored at the total
                # output (sum of segments). Filter to ``segment ==
                # 'reasoning'`` so we don't double-print the annotation
                # (both rows carry the same ``annotation`` string).
                "transform": [{"filter": "datum.segment === 'reasoning'"}],
                "mark": {
                    "type": "text",
                    "align": "left",
                    "dx": 6,
                    "color": "#cbd5e1",
                    "fontSize": 11,
                },
                "encoding": {
                    "y": {"field": "display_label", "type": "nominal", "sort": label_order},
                    "x": {"field": "output_per_conv", "type": "quantitative"},
                    "text": {"field": "annotation", "type": "nominal"},
                },
            },
        ],
    }
    return _spec_with_theme(spec)


def _cleared_counts_for_locale(report: ComparisonReport, locale: str) -> dict[str, tuple[int, int]]:
    """Map ``run_id -> (n_ready, n_eligible)`` filtered to one locale.

    Eligible cells exclude :data:`ROLLUP_EXCLUDED_CAPABILITIES`
    (so the denominator matches the headline cleared chart). Cells
    in ``missing`` / ``insufficient_evidence`` count toward eligible
    but not ready, mirroring the per-run dashboard's discipline.
    """
    by_run: dict[str, list[int]] = {}
    for cell in report.cells:
        if cell.locale != locale:
            continue
        if cell.capability in ROLLUP_EXCLUDED_CAPABILITIES:
            continue
        bucket = by_run.setdefault(cell.run_id, [0, 0])
        bucket[1] += 1
        if cell.state == "ready":
            bucket[0] += 1
    return {run_id: (b[0], b[1]) for run_id, b in by_run.items()}


def _cleared_counts_for_capability(report: ComparisonReport, capability_id: str) -> dict[str, tuple[int, int]]:
    """Map ``run_id -> (n_ready, n_eligible)`` filtered to one capability.

    No exclusion of simulation_reliability here -- it's the caller's
    job to skip that capability id (the per-capability tab strip
    drops it via :func:`_plotting_capabilities`). Denominator =
    number of locales that scored this capability per model.
    """
    by_run: dict[str, list[int]] = {}
    for cell in report.cells:
        if cell.capability != capability_id:
            continue
        bucket = by_run.setdefault(cell.run_id, [0, 0])
        bucket[1] += 1
        if cell.state == "ready":
            bucket[0] += 1
    return {run_id: (b[0], b[1]) for run_id, b in by_run.items()}


def _cleared_counts_per_locale_for_model(report: ComparisonReport, run_id: str) -> dict[str, tuple[int, int]]:
    """Map ``locale -> (n_ready, n_eligible)`` filtered to one model.

    Same ROLLUP_EXCLUDED_CAPABILITIES discipline as the headline. Each
    locale's denominator is the count of quality capabilities that
    appear in the cell spine (typically 14 in a complete panel).
    """
    by_locale: dict[str, list[int]] = {}
    for cell in report.cells:
        if cell.run_id != run_id:
            continue
        if cell.capability in ROLLUP_EXCLUDED_CAPABILITIES:
            continue
        bucket = by_locale.setdefault(cell.locale, [0, 0])
        bucket[1] += 1
        if cell.state == "ready":
            bucket[0] += 1
    return {locale: (b[0], b[1]) for locale, b in by_locale.items()}


def _build_cleared_bar_spec(
    *,
    title: str,
    values: list[dict[str, Any]],
    sort_order: list[str],
    color_field: str,
    color_scale: dict[str, Any],
    bar_field_label: str,
) -> dict[str, Any]:
    """Shared shape: horizontal cleared-rate bar chart.

    Used by the per-locale, per-capability, and per-model tab
    sections. Each row is one bar; ``values`` carries
    ``display_label`` / ``pct_cleared`` / ``n_ready`` / ``n_eligible``
    / ``annotation`` / ``deeplink`` (matching the headline chart).
    """
    return _spec_with_theme(
        {
            "title": _chart_title(title),
            # 800px matches the per-tab grouped-bar chart width
            # (_spec_per_locale_grouped_bars / _spec_per_capability_facet
            # / _spec_per_model_locale_breakdown all use 800), so the
            # cleared chart sits flush below its main chart in each tab
            # section instead of looking visually orphaned.
            "width": 800,
            "height": {"step": 32},
            "data": {"values": values},
            "layer": [
                {
                    "mark": {"type": "bar", "tooltip": True, "cornerRadiusEnd": 3},
                    "encoding": {
                        "y": {
                            "field": "display_label",
                            "type": "nominal",
                            "sort": sort_order,
                            "title": None,
                            "axis": {"labelLimit": 280, "labelFontWeight": 600},
                        },
                        "x": {
                            "field": "pct_cleared",
                            "type": "quantitative",
                            "scale": {"domain": [0, 1]},
                            "axis": {"title": "Cleared %", "format": ".0%"},
                        },
                        "color": {
                            "field": color_field,
                            "type": "nominal",
                            "scale": color_scale,
                            "legend": None,
                        },
                        "tooltip": [
                            {"field": "display_label", "title": bar_field_label},
                            {"field": "pct_cleared_label", "title": "Cleared"},
                            {"field": "n_ready", "title": "Cleared cells"},
                            {"field": "n_eligible", "title": "Eligible cells"},
                            {"field": "deeplink", "title": "Open per-run dashboard"},
                        ],
                    },
                },
                {
                    "mark": {
                        "type": "text",
                        "align": "left",
                        "dx": 6,
                        "color": "#cbd5e1",
                        "fontSize": 11,
                    },
                    "encoding": {
                        "y": {"field": "display_label", "type": "nominal", "sort": sort_order},
                        "x": {"field": "pct_cleared", "type": "quantitative"},
                        "text": {"field": "annotation", "type": "nominal"},
                    },
                },
            ],
        }
    )


def _spec_cleared_for_locale(
    report: ComparisonReport,
    locale: str,
    colors: dict[str, str],
    *,
    is_overall: bool,
) -> dict[str, Any]:
    """Cleared % per model within a single locale (or panel-wide on Overall).

    Locale labels with flag emoji are produced by :func:`_locale_label`
    so the chart title reads e.g. "Cleared capabilities · 🇺🇸  en_US".
    """
    label_order = _models_sorted_by_score(report)
    values: list[dict[str, Any]] = []
    if is_overall:
        # Reuse the global cleared rate -- already on every rollup.
        for r in report.model_summary:
            denom = r.n_cells_eligible or 1
            pct = r.n_cells_ready_eligible / denom
            values.append(
                {
                    "display_label": r.display_label,
                    "pct_cleared": pct,
                    "n_ready": r.n_cells_ready_eligible,
                    "n_eligible": r.n_cells_eligible,
                    "annotation": (f"{round(100 * pct)}% ({r.n_cells_ready_eligible}/{r.n_cells_eligible})"),
                    "pct_cleared_label": f"{round(100 * pct)}%",
                    "deeplink": _deeplink_for(r.run_id),
                }
            )
        title = "Cleared capabilities · Overall (across all locales)"
    else:
        counts = _cleared_counts_for_locale(report, locale)
        for r in report.model_summary:
            n_ready, n_eligible = counts.get(r.run_id, (0, 0))
            denom = n_eligible or 1
            pct = n_ready / denom
            values.append(
                {
                    "display_label": r.display_label,
                    "pct_cleared": pct,
                    "n_ready": n_ready,
                    "n_eligible": n_eligible,
                    "annotation": f"{round(100 * pct)}% ({n_ready}/{n_eligible})",
                    "pct_cleared_label": f"{round(100 * pct)}%",
                    "deeplink": _deeplink_for(r.run_id),
                }
            )
        title = f"Cleared capabilities · {_locale_label(locale)}"
    return _build_cleared_bar_spec(
        title=title,
        values=values,
        sort_order=label_order,
        color_field="display_label",
        color_scale=_color_scale(colors, label_order),
        bar_field_label="Model",
    )


def _spec_cleared_for_capability(
    report: ComparisonReport,
    capability_id: str,
    capability_label: str,
    colors: dict[str, str],
) -> dict[str, Any]:
    """Cleared % per model for a single capability across all locales.

    Numerator: ready cells where ``capability == capability_id``.
    Denominator: per model, the number of locales that scored this
    capability (typically the full locale count).
    """
    label_order = _models_sorted_by_score(report)
    counts = _cleared_counts_for_capability(report, capability_id)
    values: list[dict[str, Any]] = []
    for r in report.model_summary:
        n_ready, n_eligible = counts.get(r.run_id, (0, 0))
        denom = n_eligible or 1
        pct = n_ready / denom
        values.append(
            {
                "display_label": r.display_label,
                "pct_cleared": pct,
                "n_ready": n_ready,
                "n_eligible": n_eligible,
                "annotation": f"{round(100 * pct)}% ({n_ready}/{n_eligible})",
                "pct_cleared_label": f"{round(100 * pct)}%",
                "deeplink": _deeplink_for(r.run_id),
            }
        )
    return _build_cleared_bar_spec(
        title=f"Cleared {capability_label} · across locales",
        values=values,
        sort_order=label_order,
        color_field="display_label",
        color_scale=_color_scale(colors, label_order),
        bar_field_label="Model",
    )


def _spec_cleared_for_model_per_locale(
    report: ComparisonReport,
    run_id: str,
    display_label: str,
) -> dict[str, Any]:
    """Cleared % per locale within a single model.

    Bars are coloured with the locale palette so the chart reads
    visually as "this model's locale ranking" -- distinct from the
    Pareto / leaderboard colour-by-model views above.
    """
    counts = _cleared_counts_per_locale_for_model(report, run_id)
    locales = list(report.locales)  # preserves original report order
    values: list[dict[str, Any]] = []
    for loc in locales:
        n_ready, n_eligible = counts.get(loc, (0, 0))
        denom = n_eligible or 1
        pct = n_ready / denom
        values.append(
            {
                # ``display_label`` is the y-axis field name expected by
                # the shared bar-spec helper. We populate it with the
                # locale label (flag + code) so the bars read "🇺🇸 en_US".
                "display_label": _locale_label(loc),
                "pct_cleared": pct,
                "n_ready": n_ready,
                "n_eligible": n_eligible,
                "annotation": f"{round(100 * pct)}% ({n_ready}/{n_eligible})",
                "pct_cleared_label": f"{round(100 * pct)}%",
                "deeplink": _deeplink_for(run_id),
            }
        )
    sort_order = [_locale_label(loc) for loc in locales]
    color_scale = _locale_color_scale(locales)
    # The locale color scale is keyed by raw locale code; the chart's
    # color field is the labelled string. Re-key the scale so colors
    # line up with the y-axis labels.
    color_scale = {
        "domain": [_locale_label(loc) for loc in color_scale["domain"]],
        "range": list(color_scale["range"]),
    }
    return _build_cleared_bar_spec(
        title=f"Cleared capabilities · {display_label} (by locale)",
        values=values,
        sort_order=sort_order,
        color_field="display_label",
        color_scale=color_scale,
        bar_field_label="Locale",
    )


def _spec_cleared_headline(report: ComparisonReport, colors: dict[str, str]) -> dict[str, Any]:
    """Cleared-capability percentage per model — Performance section's
    headline chart.

    Numerator: ``n_cells_ready_eligible`` (ready cells excluding the
    simulation_reliability capability).
    Denominator: ``n_cells_eligible`` (total non-reliability cells in
    scope, typically 14 capabilities × 8 locales = 112 per model).

    This replaces the older "Capability gaps" chart -- showing how
    much of the matrix a model has *cleared* (passed) is more
    motivating than counting what it failed.
    """
    label_order = _models_sorted_by_score(report)
    values = []
    for r in report.model_summary:
        denom = r.n_cells_eligible or 1  # avoid div-by-zero on degenerate runs
        pct_cleared = r.n_cells_ready_eligible / denom
        values.append(
            {
                "display_label": r.display_label,
                "pct_cleared": pct_cleared,
                "n_ready": r.n_cells_ready_eligible,
                "n_eligible": r.n_cells_eligible,
                # Annotation reads e.g. "73% (82/112)" so reviewers
                # see both the proportion and the raw count without
                # needing the tooltip.
                "annotation": (f"{round(100 * pct_cleared)}% ({r.n_cells_ready_eligible}/{r.n_cells_eligible})"),
                "pct_cleared_label": f"{round(100 * pct_cleared)}%",
                "deeplink": _deeplink_for(r.run_id),
            }
        )
    spec = {
        "title": _chart_title("Cleared capabilities"),
        "width": 280,
        "height": {"step": 32},
        "data": {"values": values},
        "layer": [
            {
                "mark": {"type": "bar", "tooltip": True, "cornerRadiusEnd": 3},
                "encoding": {
                    "y": {
                        "field": "display_label",
                        "type": "nominal",
                        "sort": label_order,
                        "title": None,
                        # Cleared headline is now the FIRST chart in
                        # the Performance row, so it carries the model
                        # labels for the whole row -- the leaderboard
                        # and verbosity charts to its right hide them.
                        "axis": {"labelLimit": 280, "labelFontWeight": 600},
                    },
                    "x": {
                        "field": "pct_cleared",
                        "type": "quantitative",
                        # Lock the domain at 0-1: cleared rate is a
                        # bounded percentage, and a fixed scale lets
                        # the eye compare directly against the OK /
                        # persona-grounding charts in the Sim Health
                        # row above.
                        "scale": {"domain": [0, 1]},
                        "axis": {"title": "Cleared %", "format": ".0%"},
                    },
                    "color": {
                        "field": "display_label",
                        "type": "nominal",
                        "scale": _color_scale(colors, label_order),
                        "legend": None,
                    },
                    "tooltip": [
                        {"field": "display_label", "title": "Model"},
                        {"field": "pct_cleared_label", "title": "Cleared"},
                        {"field": "n_ready", "title": "Cleared cells"},
                        {"field": "n_eligible", "title": "Eligible cells (excl. sim_reliability)"},
                        {"field": "deeplink", "title": "Open per-run dashboard"},
                    ],
                },
            },
            {
                "mark": {
                    "type": "text",
                    "align": "left",
                    "dx": 6,
                    "color": "#cbd5e1",
                    "fontSize": 11,
                },
                "encoding": {
                    "y": {"field": "display_label", "type": "nominal", "sort": label_order},
                    "x": {"field": "pct_cleared", "type": "quantitative"},
                    "text": {"field": "annotation", "type": "nominal"},
                },
            },
        ],
    }
    return _spec_with_theme(spec)


# ---------------------------------------------------------------------------
# Layer 2 — Quality vs Verbosity (Pareto)
# ---------------------------------------------------------------------------


def _per_locale_mean_score(report: ComparisonReport, locale: str) -> dict[str, Optional[float]]:
    """Map run_id -> mean normalized score for ``locale`` only.

    Same rollup discipline as :func:`reporting.comparison._build_rollup`:

    - Skips ``missing`` / ``insufficient_evidence`` cells (we don't
      penalise gaps, we exclude them).
    - Excludes ``simulation_reliability`` (process metric, not quality).
    - Returns ``None`` for runs that have zero measurable cells in this
      locale (so the Pareto chart can drop them rather than plot a
      misleading 0).
    """
    by_run: dict[str, list[float]] = {}
    for cell in report.cells:
        if cell.locale != locale:
            continue
        if cell.capability in ROLLUP_EXCLUDED_CAPABILITIES:
            continue
        if cell.state not in MEASURED_STATES:
            continue
        if cell.normalized_score is None:
            continue
        by_run.setdefault(cell.run_id, []).append(cell.normalized_score)
    return {run_id: (sum(scores) / len(scores) if scores else None) for run_id, scores in by_run.items()}


def _pareto_score_lookup_overall(report: ComparisonReport):
    """Score lookup for the Overall tab: pre-computed run-level rollup."""
    by_run = {r.run_id: r.mean_normalized_score for r in report.model_summary}
    return lambda rollup: by_run.get(rollup.run_id)


def _pareto_score_lookup_for_locale(report: ComparisonReport, locale: str):
    """Score lookup for a per-locale tab."""
    by_run = _per_locale_mean_score(report, locale)
    return lambda rollup: by_run.get(rollup.run_id)


def _pareto_frontier_for(
    plottable: list[tuple[ModelRollup, float]],
) -> dict[str, bool]:
    """Pareto frontier among ``(rollup, score)`` pairs for one tab.

    Per-locale tabs need their own frontier flag -- a model that
    dominates globally may be Pareto-dominated in one locale (lower
    score there with the same verbosity). Mirrors the global
    dominance rule from ``_compute_pareto_frontier``: dominated iff
    some other point has BOTH strictly higher score AND strictly
    lower output_per_conv.
    """
    by_run: dict[str, bool] = {}
    for rollup, score in plottable:
        dominated = False
        for other, other_score in plottable:
            if other is rollup:
                continue
            if (
                other_score > score
                and other.assistant_output_tokens_per_conversation < rollup.assistant_output_tokens_per_conversation
            ):
                dominated = True
                break
        by_run[rollup.run_id] = not dominated
    return by_run


def _spec_pareto(
    report: ComparisonReport,
    colors: dict[str, str],
    shapes: dict[str, str],
    *,
    score_lookup=None,
    locale_label: str = "Overall",
) -> dict[str, Any]:
    """Quality-vs-verbosity scatter, optionally filtered by locale.

    ``score_lookup`` is a callable ``ModelRollup -> Optional[float]``
    that returns the score to plot on the y-axis. Defaults to the
    rollup's pre-computed ``mean_normalized_score`` (the Overall tab).
    Per-locale tabs pass a lookup that returns the locale-specific
    mean score; the x-axis stays at run-level
    ``assistant_output_tokens_per_conversation`` so all tabs share a
    consistent verbosity axis.
    """
    if score_lookup is None:
        score_lookup = _pareto_score_lookup_overall(report)
    label_order = _models_sorted_by_score(report)

    # ``plottable`` keeps a (rollup, score) pair so we can run the
    # tab-local Pareto-frontier sweep without re-walking score_lookup.
    plottable: list[tuple[ModelRollup, float]] = []
    for r in report.model_summary:
        score = score_lookup(r)
        if score is None or r.assistant_output_tokens_per_conversation <= 0:
            continue
        plottable.append((r, float(score)))

    frontier = _pareto_frontier_for(plottable)
    values = []
    for r, score in plottable:
        passing_denom = max(r.n_cells_measured, 1)
        pct_passing = r.n_cells_ready / passing_denom
        output_per_conv = r.assistant_output_tokens_per_conversation
        reasoning_per_conv = r.assistant_reasoning_tokens_per_conversation
        values.append(
            {
                "display_label": r.display_label,
                "score": score,
                "score_pct": round(100 * score, 1),
                # ``tokens_per_conv`` keeps its name on the chart for
                # the existing client-side wiring, but the value is now
                # output-only (matches the Layer 1b headline metric).
                "tokens_per_conv": output_per_conv,
                "tokens_per_conv_label": f"{int(round(output_per_conv)):,}",
                "reasoning_per_conv": reasoning_per_conv,
                "reasoning_per_conv_label": f"{int(round(reasoning_per_conv)):,}",
                "pct_passing": pct_passing,
                "pct_passing_label": f"{round(100 * pct_passing)}%",
                "n_ready": r.n_cells_ready,
                "n_measured": r.n_cells_measured,
                "is_pareto_frontier": frontier.get(r.run_id, False),
                "deeplink": _deeplink_for(r.run_id),
                "model_id": r.model_id or "",
                "run_id": r.run_id,
            }
        )

    # The y-scale must clamp identically across all three layers
    # (line, point, text). When only the points layer set
    # ``scale.domain``, Vega-Lite's shared-scale resolver merged it
    # with the line/text layers' default (zero-anchored) and the chart
    # silently expanded back to [0, 1]. Hoisting the same scale block
    # to every layer keeps the resolved domain at [0.5, 1.0].
    y_scale = {"domain": [0.5, 1.0], "zero": False}
    # Lock tick positions to explicit 0.1 increments. With auto-tick
    # picking, Vega-Lite was stepping at 0.05 and the ``.1f`` label
    # formatter then rounded each pair to the same string (0.5, 0.5,
    # 0.6, 0.6, ...). Using ``values`` as a hard list bypasses the
    # auto-picker and the resulting labels read cleanly in 0.1 steps.
    y_axis = {
        "format": ".1f",
        "values": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    }
    title_text = "Quality vs Verbosity (target zone: top-left)"
    if locale_label != "Overall":
        title_text = f"Quality vs Verbosity · {locale_label}"
    spec = {
        "title": _chart_title(title_text),
        # Pareto gets a generous width so the model labels don't
        # collide and the eye reads "quality vs verbosity" as the
        # section's headline visualization. Height matches the new
        # width's aspect ratio (3:2) so the trend line reads cleanly.
        "width": 720,
        "height": 480,
        "data": {"values": values},
        "layer": [
            {
                "transform": [
                    {"filter": "datum.is_pareto_frontier"},
                    {"calculate": "datum.tokens_per_conv", "as": "tokens_sort"},
                ],
                "mark": {
                    "type": "line",
                    "color": "#76b900",
                    "strokeDash": [3, 3],
                    "opacity": 0.55,
                    "interpolate": "linear",
                },
                "encoding": {
                    "x": {"field": "tokens_per_conv", "type": "quantitative", "sort": "ascending"},
                    "y": {"field": "score", "type": "quantitative", "scale": y_scale},
                    "order": {"field": "tokens_per_conv", "type": "quantitative"},
                },
            },
            {
                "mark": {
                    "type": "point",
                    "filled": True,
                    "tooltip": True,
                    "stroke": "#0f172a",
                    "strokeWidth": 1.5,
                    "opacity": 0.95,
                },
                "encoding": {
                    "x": {
                        "field": "tokens_per_conv",
                        "type": "quantitative",
                        # Output-only (input excluded). See
                        # ``_compute_pareto_frontier`` and
                        # ``_spec_verbosity_headline`` for the rationale.
                        "title": "Mean assistant output tokens per conversation",
                        "axis": {"format": ",.0f"},
                    },
                    "y": {
                        "field": "score",
                        "type": "quantitative",
                        "title": "Mean normalized score",
                        # Real measured scores cluster in the 0.6-0.9
                        # band; zooming the y-axis from 0.5-1.0 lets
                        # reviewers actually see the gaps between
                        # models. ``zero: false`` is required because
                        # quantitative scales auto-include 0 by default,
                        # which would otherwise re-expand the domain.
                        "scale": y_scale,
                        "axis": y_axis,
                    },
                    "color": {
                        "field": "display_label",
                        "type": "nominal",
                        "scale": _color_scale(colors, label_order),
                        "legend": None,
                    },
                    "shape": {
                        "field": "display_label",
                        "type": "nominal",
                        "scale": _shape_scale(shapes, label_order),
                        "legend": None,
                    },
                    "size": {
                        "field": "pct_passing",
                        "type": "quantitative",
                        "scale": {"range": [180, 480]},
                        "legend": None,
                    },
                    "tooltip": [
                        {"field": "display_label", "title": "Model"},
                        {"field": "model_id", "title": "Identity"},
                        {"field": "run_id", "title": "Run"},
                        {"field": "score_pct", "title": "Score (%)"},
                        {"field": "tokens_per_conv_label", "title": "Output / conv"},
                        {"field": "reasoning_per_conv_label", "title": "Reasoning / conv"},
                        {"field": "pct_passing_label", "title": "% passing"},
                        {"field": "n_ready", "title": "Cells passing"},
                        {"field": "n_measured", "title": "Cells measured"},
                        {"field": "is_pareto_frontier", "title": "On Pareto frontier"},
                        {"field": "deeplink", "title": "Open per-run dashboard"},
                    ],
                },
            },
            {
                "mark": {
                    "type": "text",
                    "align": "left",
                    "dx": 12,
                    "fontSize": 12,
                    "color": "#e5e7eb",
                    "fontWeight": 600,
                },
                "encoding": {
                    "x": {"field": "tokens_per_conv", "type": "quantitative"},
                    "y": {"field": "score", "type": "quantitative", "scale": y_scale},
                    "text": {"field": "display_label", "type": "nominal"},
                },
            },
        ],
    }
    return _spec_with_theme(spec)


# ---------------------------------------------------------------------------
# Layer 3 — Verbosity Profile (per-locale token heatmap)
# ---------------------------------------------------------------------------


def _spec_verbosity_heatmap(report: ComparisonReport, colors: dict[str, str]) -> dict[str, Any]:
    label_order = _models_sorted_by_score(report)
    values = []
    for r in report.model_summary:
        for locale in report.locales:
            v = r.assistant_tokens_per_turn_by_locale.get(locale)
            values.append(
                {
                    "display_label": r.display_label,
                    "locale": locale,
                    "tokens_per_turn": v,
                    "tokens_per_turn_label": ("—" if v is None else f"{int(round(v)):,}"),
                    "deeplink": _deeplink_for(r.run_id),
                }
            )
    spec = {
        "title": _chart_title("Mean assistant tokens per turn (model × locale)"),
        "width": 480,
        "height": {"step": 28},
        "data": {"values": values},
        "mark": {"type": "rect", "tooltip": True, "stroke": "#0f172a", "strokeWidth": 1},
        "encoding": {
            "y": {
                "field": "display_label",
                "type": "nominal",
                "sort": label_order,
                "title": None,
                "axis": {"labelFontWeight": 600, "labelLimit": 240},
            },
            "x": {
                "field": "locale",
                "type": "nominal",
                "sort": list(report.locales),
                "title": None,
                "axis": {"labelAngle": -25},
            },
            "color": {
                "field": "tokens_per_turn",
                "type": "quantitative",
                "scale": {"scheme": "tealblues"},
                "legend": {"title": "tokens/turn", "format": ",.0f"},
            },
            "tooltip": [
                {"field": "display_label", "title": "Model"},
                {"field": "locale", "title": "Locale"},
                {"field": "tokens_per_turn_label", "title": "Tokens/turn"},
                {"field": "deeplink", "title": "Open per-run dashboard"},
            ],
        },
    }
    return _spec_with_theme(spec)


# ---------------------------------------------------------------------------
# Layer 4 — Per-Locale Capability Comparison (locale-tabbed grouped bars)
# ---------------------------------------------------------------------------

# simulation_reliability is a process-health metric (success rate of the
# conversation loop) rather than an assistant-quality signal. It is
# surfaced in Layer 0 (Sim Health Comparison) and deliberately excluded
# from the per-locale + per-capability comparison surfaces.
_PLOT_EXCLUDED_CAPABILITIES: tuple[str, ...] = ("simulation_reliability",)


def _plotting_capabilities(report: ComparisonReport) -> list[tuple[str, str]]:
    """Capabilities surfaced in Layers 4 + 5 (drops simulation_reliability)."""
    return [
        (cap_id, cap_label) for cap_id, cap_label in report.capabilities if cap_id not in _PLOT_EXCLUDED_CAPABILITIES
    ]


def _split_capabilities_evenly(capabilities: list[tuple[str, str]], n_rows: int = 2) -> list[list[tuple[str, str]]]:
    """Split capabilities into ``n_rows`` roughly-equal contiguous chunks.

    Preserves registry order within each row so a reviewer's eye learns
    consistent positions across locales (cross-cutting design rule 2).
    """
    if not capabilities:
        return [[] for _ in range(n_rows)]
    per_row = (len(capabilities) + n_rows - 1) // n_rows
    return [capabilities[i : i + per_row] for i in range(0, len(capabilities), per_row)]


def _cells_for_locale(report: ComparisonReport, locale: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (chart_rows, threshold_rows) for the locale.

    Filters out simulation_reliability per the plotting policy above, so
    Layer 4 always reflects assistant-quality capabilities only.
    """
    cap_order = [label for _, label in _plotting_capabilities(report)]
    plotted_ids = {cap_id for cap_id, _ in _plotting_capabilities(report)}
    deeplink_by_run = {e.run_id: _deeplink_for(e.run_id) for e in report.entries}
    chart: list[dict[str, Any]] = []
    threshold_by_cap: dict[str, float] = {}
    for cell in report.cells:
        if cell.locale != locale or cell.capability not in plotted_ids:
            continue
        if cell.threshold is not None and cell.capability_label not in threshold_by_cap:
            threshold_by_cap[cell.capability_label] = cell.threshold
        chart.append(
            {
                "capability": cell.capability,
                "capability_label": cell.capability_label,
                "display_label": cell.display_label,
                "run_id": cell.run_id,
                "state": cell.state,
                "is_missing": cell.is_missing,
                "has_must_pass_failure": bool(cell.failed_critical_axes),
                "failed_critical_axes": ", ".join(cell.failed_critical_axes) or "—",
                "normalized_score": cell.normalized_score if cell.normalized_score is not None else 0.0,
                "score_pct": (None if cell.normalized_score is None else round(100 * cell.normalized_score, 1)),
                "threshold": cell.threshold,
                "threshold_pct": (None if cell.threshold is None else round(100 * cell.threshold, 1)),
                "n": cell.n,
                "n_total": cell.n_total,
                "deeplink": deeplink_by_run.get(cell.run_id, ""),
            }
        )
    threshold_rows = [
        {"capability_label": cap_label, "threshold": threshold_by_cap[cap_label]}
        for cap_label in cap_order
        if cap_label in threshold_by_cap
    ]
    return chart, threshold_rows


def _layer4_subspec(
    *,
    chart_rows: list[dict[str, Any]],
    cap_order: list[str],
    label_order: list[str],
    colors: dict[str, str],
    show_legend: bool,
    extra_tooltip_fields: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One row of the Layer 4 vconcat: vertical grouped bars over a capability slice.

    Vertical orientation (capability on x, model as xOffset, score on y).
    No per-capability threshold marker — Vega-Lite's ``tick`` mark on a
    grouped-bar chart with ``xOffset`` doesn't visually align cleanly with
    the bar clusters; the per-cell threshold is in the tooltip and the
    state encoding (opacity for missing/insufficient, dashed red outline
    for must-pass failures) already conveys pass/fail at a glance.
    """
    tooltip = [
        {"field": "display_label", "title": "Model"},
        {"field": "capability_label", "title": "Capability"},
        {"field": "score_pct", "title": "Score (%)"},
        {"field": "threshold_pct", "title": "Threshold (%)"},
        {"field": "state", "title": "State"},
        {"field": "n", "title": "Measured n"},
        {"field": "n_total", "title": "Eligible n"},
        {"field": "failed_critical_axes", "title": "Must-pass failures"},
        {"field": "deeplink", "title": "Open per-run dashboard"},
    ]
    if extra_tooltip_fields:
        # Insert extras after the score fields, before state/n.
        tooltip = tooltip[:4] + list(extra_tooltip_fields) + tooltip[4:]
    return {
        # Fixed width keeps the chart inside the dashboard panel (max ~900 px
        # in practice). With 7 capabilities × 6 models = 42 bars on each
        # row, 800 px gives ~19 px per bar — comfortable for tooltips and
        # the must-pass-failure outline to be visible.
        "width": 800,
        "height": 240,
        "data": {"values": chart_rows},
        "transform": [{"filter": {"field": "capability_label", "oneOf": cap_order}}],
        "encoding": {
            "x": {
                "field": "capability_label",
                "type": "nominal",
                "sort": cap_order,
                "title": None,
                "axis": {"labelAngle": -25, "labelLimit": 160, "labelFontWeight": 600},
            },
        },
        "layer": [
            {
                "mark": {"type": "bar", "tooltip": True, "cornerRadiusEnd": 1},
                "encoding": {
                    "xOffset": {
                        "field": "display_label",
                        "type": "nominal",
                        "sort": label_order,
                    },
                    "y": {
                        "field": "normalized_score",
                        "type": "quantitative",
                        "scale": {"domain": [0, 1]},
                        "axis": {"format": ".1f", "title": "Normalized score"},
                    },
                    "color": {
                        "field": "display_label",
                        "type": "nominal",
                        "scale": _color_scale(colors, label_order),
                        "legend": ({"title": "Model", "orient": "top"} if show_legend else None),
                    },
                    "opacity": {
                        "condition": [
                            {"test": "datum.is_missing", "value": 0.15},
                            {"test": "datum.state == 'insufficient_evidence'", "value": 0.45},
                        ],
                        "value": 1.0,
                    },
                    "stroke": {
                        "condition": {"test": "datum.has_must_pass_failure", "value": "#dc2626"},
                        "value": None,
                    },
                    "strokeWidth": {
                        "condition": {"test": "datum.has_must_pass_failure", "value": 2},
                        "value": 0,
                    },
                    "strokeDash": {
                        "condition": {"test": "datum.has_must_pass_failure", "value": [3, 2]},
                        "value": [0, 0],
                    },
                    "tooltip": tooltip,
                },
            },
        ],
    }


def _cells_for_overall(
    report: ComparisonReport,
) -> list[dict[str, Any]]:
    """Aggregate cells across locales for the 'Overall' tab.

    For each (capability, run) pair we average ``normalized_score`` across
    all locales whose state is in :data:`MEASURED_STATES` (matches the
    leaderboard rollup discipline — ``missing`` / ``insufficient_evidence``
    cells are skipped, not penalized as zero). ``has_must_pass_failure`` is
    the union: if ANY locale tripped a must-pass axis, the aggregated bar
    carries the dashed-red-border state encoding so the failure isn't hidden
    by averaging.
    """
    plotted_ids = {cap_id for cap_id, _ in _plotting_capabilities(report)}
    deeplink_by_run = {e.run_id: _deeplink_for(e.run_id) for e in report.entries}

    by_key: dict[tuple[str, str], list[ComparisonCell]] = {}
    for cell in report.cells:
        if cell.capability not in plotted_ids:
            continue
        if cell.state not in MEASURED_STATES:
            continue
        if cell.normalized_score is None:
            continue
        by_key.setdefault((cell.capability, cell.run_id), []).append(cell)

    chart: list[dict[str, Any]] = []
    for (cap_id, run_id), cells in by_key.items():
        scores = [c.normalized_score for c in cells if c.normalized_score is not None]
        if not scores:
            continue
        first = cells[0]
        mean_score = sum(scores) / len(scores)
        threshold = first.threshold
        all_failed_axes = sorted({a for c in cells for a in c.failed_critical_axes})
        n_locales_failed = sum(1 for c in cells if c.failed_critical_axes)
        # Aggregated state derivation: any must-pass failure across locales
        # forces blocked; otherwise compare mean to threshold.
        if all_failed_axes:
            state = "blocked"
        elif threshold is not None and mean_score < threshold:
            state = "blocked"
        else:
            state = "ready"
        chart.append(
            {
                "capability": cap_id,
                "capability_label": first.capability_label,
                "display_label": first.display_label,
                "run_id": run_id,
                "state": state,
                "is_missing": False,
                "has_must_pass_failure": bool(all_failed_axes),
                "failed_critical_axes": ", ".join(all_failed_axes) or "—",
                "normalized_score": mean_score,
                "score_pct": round(100 * mean_score, 1),
                "threshold": threshold,
                "threshold_pct": (None if threshold is None else round(100 * threshold, 1)),
                "n": sum(c.n for c in cells),
                "n_total": sum(c.n_total for c in cells),
                "deeplink": deeplink_by_run.get(run_id, ""),
                "n_locales_measured": len(cells),
                "n_locales_failed_must_pass": n_locales_failed,
            }
        )
    return chart


def _spec_per_locale_grouped_bars(report: ComparisonReport, locale: str, colors: dict[str, str]) -> dict[str, Any]:
    """Vertical grouped bars per (capability, model), split into 2 rows.

    Splitting the 14 plottable capabilities (15 minus simulation_reliability)
    into two rows of ~7 keeps each capability column wide enough to host
    side-by-side bars for every model without crowding.
    """
    label_order = _models_sorted_alpha(report)
    plottable = _plotting_capabilities(report)
    chart_rows, _ = _cells_for_locale(report, locale)
    cap_rows = _split_capabilities_evenly(plottable, n_rows=2)

    sub_specs: list[dict[str, Any]] = []
    for row_idx, cap_chunk in enumerate(cap_rows):
        if not cap_chunk:
            continue
        cap_labels = [label for _, label in cap_chunk]
        sub_specs.append(
            _layer4_subspec(
                chart_rows=chart_rows,
                cap_order=cap_labels,
                label_order=label_order,
                colors=colors,
                show_legend=(row_idx == 0),
            )
        )

    spec = {
        "title": _chart_title(f"Capability comparison · {_locale_label(locale)}"),
        "vconcat": sub_specs,
        "spacing": 24,
    }
    return _spec_with_theme(spec)


def _spec_overall_grouped_bars(report: ComparisonReport, colors: dict[str, str]) -> dict[str, Any]:
    """Layer 4 Overall tab — locale-averaged grouped bars per (capability, model).

    Same layout as the per-locale charts (vconcat'd two-row vertical bars),
    fed by :func:`_cells_for_overall`. Tooltip carries
    ``n_locales_measured`` so a reviewer can spot the "averaged over only
    2 of 8 locales" coverage pathology.
    """
    label_order = _models_sorted_alpha(report)
    plottable = _plotting_capabilities(report)
    chart_rows = _cells_for_overall(report)
    cap_rows = _split_capabilities_evenly(plottable, n_rows=2)
    extra_tooltips = [
        {"field": "n_locales_measured", "title": "Locales measured"},
        {"field": "n_locales_failed_must_pass", "title": "Locales w/ must-pass fail"},
    ]
    sub_specs: list[dict[str, Any]] = []
    for row_idx, cap_chunk in enumerate(cap_rows):
        if not cap_chunk:
            continue
        cap_labels = [label for _, label in cap_chunk]
        sub_specs.append(
            _layer4_subspec(
                chart_rows=chart_rows,
                cap_order=cap_labels,
                label_order=label_order,
                colors=colors,
                show_legend=(row_idx == 0),
                extra_tooltip_fields=extra_tooltips,
            )
        )
    spec = {
        "title": _chart_title("Capability comparison · Overall (mean across locales)"),
        "vconcat": sub_specs,
        "spacing": 24,
    }
    return _spec_with_theme(spec)


# ---------------------------------------------------------------------------
# Layer 5 — Per-Capability Locale Comparison (flipped facet)
# ---------------------------------------------------------------------------


def _spec_per_capability_facet(
    report: ComparisonReport, capability_id: str, capability_label: str, colors: dict[str, str]
) -> dict[str, Any]:
    """Vertical grouped bars over (locale, model) for a single capability.

    Mirrors Layer 4's vertical orientation: x = locale, xOffset = model,
    y = normalized_score. No threshold rule — bars are read relative to
    each other; per-cell threshold is in the tooltip.
    """
    label_order = _models_sorted_alpha(report)
    locale_order = list(report.locales)
    cells = [c for c in report.cells if c.capability == capability_id]
    chart_rows = []
    deeplink_by_run = {e.run_id: _deeplink_for(e.run_id) for e in report.entries}
    for cell in cells:
        chart_rows.append(
            {
                "locale": cell.locale,
                "display_label": cell.display_label,
                "state": cell.state,
                "is_missing": cell.is_missing,
                "has_must_pass_failure": bool(cell.failed_critical_axes),
                "failed_critical_axes": ", ".join(cell.failed_critical_axes) or "—",
                "normalized_score": cell.normalized_score if cell.normalized_score is not None else 0.0,
                "score_pct": (None if cell.normalized_score is None else round(100 * cell.normalized_score, 1)),
                "threshold_pct": (None if cell.threshold is None else round(100 * cell.threshold, 1)),
                "n": cell.n,
                "deeplink": deeplink_by_run.get(cell.run_id, ""),
            }
        )
    spec = {
        "title": _chart_title(f"{capability_label} (across locales × models)"),
        # Fixed width keeps the chart inside the dashboard panel (max ~900 px
        # in practice). With 8 locales × 6 models = 48 bars, 800 px gives
        # ~16 px per bar — comfortable for tooltips and per-bar state encoding.
        "width": 800,
        "height": 280,
        "data": {"values": chart_rows},
        "encoding": {
            "x": {
                "field": "locale",
                "type": "nominal",
                "sort": locale_order,
                "title": None,
                "axis": {"labelAngle": -25, "labelLimit": 140, "labelFontWeight": 600},
            },
        },
        "layer": [
            {
                "mark": {"type": "bar", "tooltip": True, "cornerRadiusEnd": 1},
                "encoding": {
                    "xOffset": {
                        "field": "display_label",
                        "type": "nominal",
                        "sort": label_order,
                    },
                    "y": {
                        "field": "normalized_score",
                        "type": "quantitative",
                        "scale": {"domain": [0, 1]},
                        "axis": {"format": ".1f", "title": "Normalized score"},
                    },
                    "color": {
                        "field": "display_label",
                        "type": "nominal",
                        "scale": _color_scale(colors, label_order),
                        "legend": {"title": "Model", "orient": "top"},
                    },
                    "opacity": {
                        "condition": [
                            {"test": "datum.is_missing", "value": 0.15},
                            {"test": "datum.state == 'insufficient_evidence'", "value": 0.45},
                        ],
                        "value": 1.0,
                    },
                    "stroke": {
                        "condition": {"test": "datum.has_must_pass_failure", "value": "#dc2626"},
                        "value": None,
                    },
                    "strokeWidth": {
                        "condition": {"test": "datum.has_must_pass_failure", "value": 2},
                        "value": 0,
                    },
                    "strokeDash": {
                        "condition": {"test": "datum.has_must_pass_failure", "value": [3, 2]},
                        "value": [0, 0],
                    },
                    "tooltip": [
                        {"field": "display_label", "title": "Model"},
                        {"field": "locale", "title": "Locale"},
                        {"field": "score_pct", "title": "Score (%)"},
                        {"field": "threshold_pct", "title": "Threshold (%)"},
                        {"field": "state", "title": "State"},
                        {"field": "n", "title": "Measured n"},
                        {"field": "failed_critical_axes", "title": "Must-pass failures"},
                        {"field": "deeplink", "title": "Open per-run dashboard"},
                    ],
                },
            },
        ],
    }
    # No threshold rule: per the dashboard's relative-comparison framing
    # (per-cap thresholds are arbitrary and noisy when overlaid on grouped
    # bars). The exact threshold is still surfaced per-bar in the tooltip.
    return _spec_with_theme(spec)


# ---------------------------------------------------------------------------
# Layer 7 — Per-Model Locale Breakdown (model-tabbed grouped bars)
# ---------------------------------------------------------------------------


def _locale_color_scale(locales: list[str]) -> dict[str, Any]:
    """Build a Vega-Lite color scale mapping locales to the locale palette.

    Distinct from :func:`_color_scale` (which uses the model palette) — the
    color encoding in Layer 7 is on LOCALE, not on model, so it must use a
    visually different categorical scheme.
    """
    return {
        "domain": list(locales),
        "range": [_LOCALE_PALETTE[i % len(_LOCALE_PALETTE)] for i, _ in enumerate(locales)],
    }


def _spec_per_model_locale_breakdown(report: ComparisonReport, run_id: str, display_label: str) -> dict[str, Any]:
    """Vertical grouped bars over (capability, locale) for a single model.

    Mirrors Layer 4's structure but flips the grouping: x = capability,
    xOffset = locale, color = locale (using :data:`_LOCALE_PALETTE`).
    Splits capabilities into two rows for the same breathing-room
    treatment as Layer 4.
    """
    locale_order = list(report.locales)
    plottable = _plotting_capabilities(report)
    plotted_ids = {cap_id for cap_id, _ in plottable}

    # Pull this run's chart rows (skipping simulation_reliability + missing).
    deeplink = _deeplink_for(run_id)
    chart_rows: list[dict[str, Any]] = []
    for cell in report.cells:
        if cell.run_id != run_id or cell.capability not in plotted_ids:
            continue
        chart_rows.append(
            {
                "capability": cell.capability,
                "capability_label": cell.capability_label,
                "locale": cell.locale,
                "state": cell.state,
                "is_missing": cell.is_missing,
                "has_must_pass_failure": bool(cell.failed_critical_axes),
                "failed_critical_axes": ", ".join(cell.failed_critical_axes) or "—",
                "normalized_score": (cell.normalized_score if cell.normalized_score is not None else 0.0),
                "score_pct": (None if cell.normalized_score is None else round(100 * cell.normalized_score, 1)),
                "threshold_pct": (None if cell.threshold is None else round(100 * cell.threshold, 1)),
                "n": cell.n,
                "deeplink": deeplink,
            }
        )

    cap_rows = _split_capabilities_evenly(plottable, n_rows=2)

    def _subspec(cap_chunk: list[tuple[str, str]], show_legend: bool) -> dict[str, Any]:
        cap_labels = [label for _, label in cap_chunk]
        return {
            "width": 800,
            "height": 240,
            "data": {"values": chart_rows},
            "transform": [{"filter": {"field": "capability_label", "oneOf": cap_labels}}],
            "encoding": {
                "x": {
                    "field": "capability_label",
                    "type": "nominal",
                    "sort": cap_labels,
                    "title": None,
                    "axis": {"labelAngle": -25, "labelLimit": 160, "labelFontWeight": 600},
                },
            },
            "layer": [
                {
                    "mark": {"type": "bar", "tooltip": True, "cornerRadiusEnd": 1},
                    "encoding": {
                        "xOffset": {
                            "field": "locale",
                            "type": "nominal",
                            "sort": locale_order,
                        },
                        "y": {
                            "field": "normalized_score",
                            "type": "quantitative",
                            "scale": {"domain": [0, 1]},
                            "axis": {"format": ".1f", "title": "Normalized score"},
                        },
                        "color": {
                            "field": "locale",
                            "type": "nominal",
                            "scale": _locale_color_scale(locale_order),
                            "legend": ({"title": "Locale", "orient": "top"} if show_legend else None),
                        },
                        "opacity": {
                            "condition": [
                                {"test": "datum.is_missing", "value": 0.15},
                                {
                                    "test": "datum.state == 'insufficient_evidence'",
                                    "value": 0.45,
                                },
                            ],
                            "value": 1.0,
                        },
                        "stroke": {
                            "condition": {
                                "test": "datum.has_must_pass_failure",
                                "value": "#dc2626",
                            },
                            "value": None,
                        },
                        "strokeWidth": {
                            "condition": {
                                "test": "datum.has_must_pass_failure",
                                "value": 2,
                            },
                            "value": 0,
                        },
                        "strokeDash": {
                            "condition": {
                                "test": "datum.has_must_pass_failure",
                                "value": [3, 2],
                            },
                            "value": [0, 0],
                        },
                        "tooltip": [
                            {"field": "locale", "title": "Locale"},
                            {"field": "capability_label", "title": "Capability"},
                            {"field": "score_pct", "title": "Score (%)"},
                            {"field": "threshold_pct", "title": "Threshold (%)"},
                            {"field": "state", "title": "State"},
                            {"field": "n", "title": "Measured n"},
                            {"field": "failed_critical_axes", "title": "Must-pass failures"},
                            {"field": "deeplink", "title": "Open per-run dashboard"},
                        ],
                    },
                },
            ],
        }

    sub_specs = [_subspec(chunk, show_legend=(idx == 0)) for idx, chunk in enumerate(cap_rows) if chunk]
    spec = {
        "title": _chart_title(f"Locale breakdown · {display_label}"),
        "vconcat": sub_specs,
        "spacing": 24,
    }
    return _spec_with_theme(spec)


# ---------------------------------------------------------------------------
# Layer 6 — Coverage Matrix
# ---------------------------------------------------------------------------


def _spec_coverage(report: ComparisonReport, colors: dict[str, str]) -> dict[str, Any]:
    """(model × locale) -> mean n across capabilities (excluding simulation_reliability)."""
    label_order = _models_sorted_by_score(report)
    locale_order = list(report.locales)
    by_key: dict[tuple[str, str], list[int]] = {}
    deeplink_by_run = {e.run_id: _deeplink_for(e.run_id) for e in report.entries}
    run_by_label = {e.display_label: e.run_id for e in report.entries}
    for cell in report.cells:
        if cell.capability == "simulation_reliability":
            continue
        if cell.is_missing:
            continue
        by_key.setdefault((cell.display_label, cell.locale), []).append(cell.n)
    values = []
    for label in label_order:
        for locale in locale_order:
            ns = by_key.get((label, locale)) or []
            mean_n = (sum(ns) / len(ns)) if ns else 0
            values.append(
                {
                    "display_label": label,
                    "locale": locale,
                    "mean_n": round(mean_n, 1),
                    "n_capabilities": len(ns),
                    "deeplink": deeplink_by_run.get(run_by_label.get(label, ""), ""),
                }
            )
    spec = {
        "title": _chart_title("Coverage matrix (mean trajectories per capability cell)"),
        "width": 480,
        "height": {"step": 28},
        "data": {"values": values},
        "mark": {"type": "rect", "tooltip": True, "stroke": "#0f172a", "strokeWidth": 1},
        "encoding": {
            "y": {
                "field": "display_label",
                "type": "nominal",
                "sort": label_order,
                "title": None,
                "axis": {"labelFontWeight": 600, "labelLimit": 240},
            },
            "x": {
                "field": "locale",
                "type": "nominal",
                "sort": locale_order,
                "title": None,
                "axis": {"labelAngle": -25},
            },
            "color": {
                "field": "mean_n",
                "type": "quantitative",
                "scale": {"scheme": "greens"},
                "legend": {"title": "trajectories", "format": ",.0f"},
            },
            "tooltip": [
                {"field": "display_label", "title": "Model"},
                {"field": "locale", "title": "Locale"},
                {"field": "mean_n", "title": "Mean n per capability"},
                {"field": "n_capabilities", "title": "Capabilities measured"},
                {"field": "deeplink", "title": "Open per-run dashboard"},
            ],
        },
    }
    return _spec_with_theme(spec)


# ---------------------------------------------------------------------------
# Static fallback HTML (no JS / pre-Vega)
# ---------------------------------------------------------------------------


def _fallback_html(report: ComparisonReport) -> str:
    """Static, no-JS fallback DOM with the verdict + verbosity table."""
    rollups = report.model_summary
    rollups_sorted = sorted(
        rollups,
        key=lambda r: (
            -1 if r.mean_normalized_score is None else 1,
            -(r.mean_normalized_score or 0.0),
        ),
    )
    summary_rows = []
    for r in rollups_sorted:
        score = "n/a" if r.mean_normalized_score is None else f"{r.mean_normalized_score * 100:.1f}%"
        passing_denom = max(r.n_cells_eligible, 1)
        passing = f"{round(100 * r.n_cells_ready / passing_denom)}%"
        coverage = f"{r.n_cells_measured}/{r.n_cells_eligible}"
        output_per_conv = f"{int(round(r.assistant_output_tokens_per_conversation)):,}"
        reasoning_per_conv = f"{int(round(r.assistant_reasoning_tokens_per_conversation)):,}"
        sim_ok = "—" if r.sim_status_ok_rate is None else f"{r.sim_status_ok_rate * 100:.1f}%"
        deeplink = _deeplink_for(r.run_id)
        summary_rows.append(
            "<tr>"
            f"<td><strong>{html.escape(r.display_label)}</strong></td>"
            f"<td>{html.escape(score)}</td>"
            f"<td>{html.escape(coverage)}</td>"
            f"<td>{html.escape(passing)}</td>"
            f"<td>{html.escape(output_per_conv)}</td>"
            f"<td>{html.escape(reasoning_per_conv)}</td>"
            f"<td>{html.escape(sim_ok)}</td>"
            f'<td><a href="{html.escape(deeplink)}">Open per-run</a></td>'
            "</tr>"
        )

    entries_rows = []
    for entry in report.entries:
        entries_rows.append(
            "<tr>"
            f"<td><code>{html.escape(entry.model_id or '—')}</code></td>"
            f"<td><code>{html.escape(entry.run_id)}</code></td>"
            f"<td>{entry.n_trajectories:,}</td>"
            f"<td>{entry.mean_turns_per_conversation:.2f}</td>"
            f"<td>{html.escape(entry.sample_mode)}</td>"
            # User column tracks output only -- user-side reasoning
            # is uncommon for simulator personas so it lives in the
            # per-run Sim Health drill-down, not this audit row.
            # Assistant column shows output PLUS its reasoning subset
            # (the verbosity vs invisible-thinking story per model).
            f"<td>{entry.user_output_tokens:,}</td>"
            f"<td>{entry.assistant_output_tokens:,}</td>"
            f"<td>{entry.assistant_reasoning_tokens:,}</td>"
            "</tr>"
        )

    # The apples-to-apples warning callout used to render here.
    # Removed from the dashboard surface -- at panel sizes (10+ runs
    # × multiple locales) the warning thresholds (10% n_trajectories
    # delta, sample_mode drift, persona panel size mismatch) trip on
    # nearly every comparison and the noise drowns the signal. The
    # warnings list is still populated on ``ComparisonReport`` so
    # CLI / notebook log surfaces can opt back in.

    locale_count = len(report.locales)
    capability_count = len(report.capabilities)
    n_models = len(report.entries)
    n_conv = sum(e.n_trajectories for e in report.entries)
    # Conversation tokens = user output + assistant output across the
    # full panel. Excludes input tokens (those reflect prompt context
    # size, not what was said in the conversation) and excludes
    # api_response/judge/summary aliases (they're tool synthesisers,
    # not conversational participants -- matches the per-run report's
    # ``conversation_output_tokens`` definition).
    conversation_tokens_sum = sum(e.user_output_tokens + e.assistant_output_tokens for e in report.entries)
    assistant_tokens_sum = sum(e.assistant_output_tokens for e in report.entries)
    # Avg turns/conversation, weighted by trajectory count -- so a run
    # with 1024 trajs at 3.5 turns dominates a run with 32 trajs at
    # 5.0 turns (which is what we want for a panel-level summary).
    total_turns = sum(e.mean_turns_per_conversation * e.n_trajectories for e in report.entries)
    avg_turns = total_turns / n_conv if n_conv > 0 else 0.0

    return f"""
<main class="comparison-app">
  <header class="hero">
    <h1>NeMo UserSim Multi-Model Comparison</h1>
    <div class="hero-runid">Comparison ID: <code>{html.escape(report.comparison_id)}</code></div>
    <p class="muted">Static fallback view. React 19 mounts the interactive Vega-Lite dashboard above (leaderboard, Pareto scatter, locale-tabbed grouped bars, etc.) when it can — but it ships ESM-only and modern browsers block ES module imports from <code>file://</code> URLs. If you're seeing this, run <code>python -m http.server 8000</code> from the repo root and reopen via <code>http://localhost:8000/...</code>, or render inline from the notebook (Jupyter serves over HTTP).</p>
    <div class="stats stats-row-1">
      <div><strong>{n_models}</strong><span>models compared</span></div>
      <div><strong>{locale_count}</strong><span>locales</span></div>
      <div><strong>{capability_count}</strong><span>capabilities</span></div>
      <div><strong>{n_conv:,}</strong><span>total conversations</span></div>
      <div><strong>{avg_turns:.2f}</strong><span>avg turns/conversation</span></div>
      <div><strong>{conversation_tokens_sum:,}</strong><span>conversation tokens</span></div>
      <div><strong>{assistant_tokens_sum:,}</strong><span>assistant tokens</span></div>
    </div>
  </header>

  <section class="panel">
    <h2>Models compared</h2>
    <table class="comparison-table">
      <thead><tr>
        <th>Model</th><th>Run</th><th>Trajectories</th>
        <th>Avg turns/conv</th><th>Sample mode</th>
        <th>User tokens</th>
        <th>Asst tokens</th><th>Asst reas tokens</th>
      </tr></thead>
      <tbody>{"".join(entries_rows)}</tbody>
    </table>
  </section>

  <section class="panel">
    <h2>Layer 1 / Layer 2 — Verdict + Verbosity (textual fallback)</h2>
    <p class="muted">Sorted by mean normalized score (highest first). The interactive view adds the leaderboard bar, verbosity headline (output tokens with reasoning subset), and quality-vs-verbosity Pareto scatter (output-token x-axis).</p>
    <table class="comparison-table">
      <thead><tr>
        <th>Model</th><th>Mean score</th><th>Coverage</th><th>% passing</th>
        <th>Output / conv</th><th>Reasoning / conv</th><th>Sim OK rate</th><th>Drill-down</th>
      </tr></thead>
      <tbody>{"".join(summary_rows)}</tbody>
    </table>
  </section>
</main>
"""


# ---------------------------------------------------------------------------
# index.html assembler
# ---------------------------------------------------------------------------


def _safe_json(payload: Any) -> str:
    """JSON-encode for embedding in a <script> tag (escapes ``</``)."""
    return json.dumps(payload, default=str, ensure_ascii=False).replace("</", "<\\/")


_COMPARISON_CSS = """
.usersim-comparison-dashboard .comparison-app { width: 100%; box-sizing: border-box; margin: 0; padding: 28px; }
.usersim-comparison-dashboard .comparison-warnings { background: rgba(202,138,4,.12); border-left: 4px solid #ca8a04; border-radius: 8px; padding: 14px 18px; margin: 0 28px 20px; color: #fde68a; }
.usersim-comparison-dashboard .comparison-warnings strong { color: #facc15; }
.usersim-comparison-dashboard .comparison-warnings ul { margin: 8px 0 0; padding-left: 22px; }
.usersim-comparison-dashboard .comparison-warnings code { background: rgba(15,23,42,.55); padding: 1px 6px; border-radius: 4px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.92em; color: #fde68a; }
.usersim-comparison-dashboard .comparison-warnings pre { background: rgba(15,23,42,.7); border: 1px solid rgba(202,138,4,.3); border-radius: 6px; padding: 10px 14px; margin: 8px 0; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.92em; color: #fef3c7; overflow-x: auto; }
.usersim-comparison-dashboard .comparison-table { width: 100%; border-collapse: collapse; }
.usersim-comparison-dashboard .comparison-table th, .usersim-comparison-dashboard .comparison-table td { padding: 8px 10px; border-bottom: 1px solid #1f2937; text-align: left; vertical-align: top; }
.usersim-comparison-dashboard .comparison-table th { color: #94a3b8; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; font-weight: 800; background: #0b1220; }
.usersim-comparison-dashboard .comparison-table tr:last-child td { border-bottom: none; }
.usersim-comparison-dashboard .comparison-table code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; color: #e5e7eb; }
.usersim-comparison-dashboard .comparison-table a { color: #93c5fd; text-decoration: none; }
.usersim-comparison-dashboard .comparison-table a:hover { text-decoration: underline; }
/* auto-fit lets the row hold 2 OR 3 charts and reflow naturally as the
   window shrinks (3 cols at wide widths -> 2 -> 1 single column). */
.usersim-comparison-dashboard .verdict-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
.usersim-comparison-dashboard .verbosity-row { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
@media (max-width: 1100px) { .usersim-comparison-dashboard .verbosity-row { grid-template-columns: 1fr; } }
/* Sim Health: fixed 2-up grid so each row is a balanced pair
   (Failure rate / Failure breakdown, then Persona / Early stop).
   ``row-gap`` matches column gap for symmetric spacing. */
.usersim-comparison-dashboard .sim-health-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
@media (max-width: 1100px) { .usersim-comparison-dashboard .sim-health-grid { grid-template-columns: 1fr; } }
/* Verbosity section: divider between the Pareto scatter and the
   per-locale heatmap so the two visualisations don't bleed into
   each other. ``hr`` is hidden by Vega-Lite's box-sizing reset, so
   we use a simple bottom-border on a div. */
.usersim-comparison-dashboard .panel-divider { height: 0; margin: 28px 0 18px; border-top: 1px solid #1f2937; }
.usersim-comparison-dashboard .vega-chart { background: #111827; border-radius: 12px; padding: 16px; min-height: 80px; }
.usersim-comparison-dashboard .locale-tabs { display: flex; gap: 8px; flex-wrap: wrap; margin: 0 0 14px; padding-bottom: 8px; border-bottom: 1px solid #334155; }
/* white-space: nowrap is load-bearing — without it, long labels (e.g.
   "Agentic disclosure/confirmation") wrap inside the pill and collapse
   the button down to a tiny dot. line-height keeps the pill height
   stable across labels of different word counts. */
.usersim-comparison-dashboard .locale-tab { background: transparent; border: 1px solid #334155; color: #cbd5e1; padding: 6px 14px; border-radius: 999px; cursor: pointer; font: inherit; font-size: 13px; font-weight: 600; white-space: nowrap; line-height: 1.2; }
.usersim-comparison-dashboard .locale-tab:hover { border-color: #76b900; color: #e5e7eb; }
.usersim-comparison-dashboard .locale-tab.active { background: #76b900; color: #0f172a; border-color: #76b900; }
.usersim-comparison-dashboard .layer0-grid { display: grid; grid-template-columns: 1fr; gap: 20px; }
.usersim-comparison-dashboard .helper-text { color: #94a3b8; font-size: 13px; margin: 6px 0 14px; }
.usersim-comparison-dashboard details.collapsible-panel summary { cursor: pointer; list-style: none; padding: 0; margin: 0; }
.usersim-comparison-dashboard details.collapsible-panel summary::-webkit-details-marker { display: none; }
.usersim-comparison-dashboard details.collapsible-panel summary h2::before { content: "▸"; display: inline-block; color: #93c5fd; margin-right: 10px; transition: transform .15s ease; }
.usersim-comparison-dashboard details.collapsible-panel[open] summary h2::before { transform: rotate(90deg); }
.usersim-comparison-dashboard details.collapsible-panel[open] > .collapsible-body { margin-top: 18px; }
"""


def _write_index_html(
    path: Path,
    report: ComparisonReport,
    specs: dict[str, Any],
    locale_specs: dict[str, Any],
    locale_cleared_specs: dict[str, Any],
    per_capability_specs: list[dict[str, Any]],
    per_model_specs: list[dict[str, Any]],
) -> None:
    """Assemble the comparison dashboard's index.html."""
    payload = _safe_json(report.to_dict())
    specs_payload = _safe_json(
        {
            "global": specs,
            "per_locale": locale_specs,
            "per_locale_cleared": locale_cleared_specs,
            "per_capability": per_capability_specs,
            "per_model": per_model_specs,
        }
    )
    fallback = _fallback_html(report)
    css = _DASHBOARD_CSS + _COMPARISON_CSS

    # Namespace every DOM id with the comparison_id so multiple dashboards
    # rendered in the same notebook (e.g. a per-run dashboard from cell 14
    # AND this comparison from cell 20, or two consecutive comparison
    # builds) never collide. createRoot(getElementById("root")) only
    # returns the FIRST #root in the document; without namespacing, the
    # later-mounted React app overwrites the earlier output and the user
    # sees the wrong dashboard in the wrong cell. Prefix is sanitized to a
    # CSS-safe identifier (alphanumerics + hyphens / underscores).
    cmp_uid = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(report.comparison_id))
    root_id = f"root-{cmp_uid}"
    data_id = f"comparison-data-{cmp_uid}"
    specs_id = f"comparison-specs-{cmp_uid}"

    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<meta http-equiv="Content-Security-Policy" content="{_CSP}" />
<title>NeMo UserSim Multi-Model Comparison</title>
<style>{css}</style>
</head>
<body>
<!-- The mount point's id is namespaced with the comparison_id so multiple
     dashboards in the same notebook (per-run + comparison + a second
     comparison) never collide on getElementById("root") and swap their
     outputs around. The static fallback inside #root is wiped when React
     mounts; visible only if React fails to load (e.g. file:// origin
     blocking ESM imports), in which case the fallback message itself
     explains how to recover. -->
<div id="{root_id}" class="usersim-capability-dashboard usersim-comparison-dashboard">{fallback}</div>
<script id="{data_id}" type="application/json">{payload}</script>
<script id="{specs_id}" type="application/json">{specs_payload}</script>
<script type="module">
// ESM imports from esm.sh — matches the per-run dashboard pattern that
// reliably renders in JupyterLab cell output. The DOM ids are namespaced
// per comparison so two dashboards in the same page (per-run + comparison,
// or two comparisons) cannot collide on createRoot(getElementById("root")).
// File:// origins block ESM imports for security; the pre-flight banner
// above explains the fix when this happens.
import React, {{useEffect, useMemo, useRef, useState}} from "{_ESM_REACT}";
import {{createRoot}} from "{_ESM_REACT_DOM}";
import vegaEmbed from "{_ESM_VEGA_EMBED}";

const h = React.createElement;
const report = JSON.parse(document.getElementById("{data_id}").textContent);
const specsPayload = JSON.parse(document.getElementById("{specs_id}").textContent);

const formatTokens = (n) => {{
  if (typeof n !== "number" || !Number.isFinite(n) || n <= 0) return "0";
  if (n >= 1_000_000) return `${{(n / 1_000_000).toFixed(1)}}M`;
  if (n >= 1_000) return `${{(n / 1_000).toFixed(1)}}K`;
  return n.toLocaleString();
}};

// Mirrors reporting/dashboard.py::_locale_label (kept in sync by hand —
// changes there should propagate here for visual consistency across the
// per-run and comparison dashboards).
const _LOCALE_FLAGS = {{
  en_IN: "🇮🇳", en_SG: "🇸🇬", en_US: "🇺🇸", fr_FR: "🇫🇷",
  hi_Deva_IN: "🇮🇳", ja_JP: "🇯🇵", ko_KR: "🇰🇷", pt_BR: "🇧🇷",
}};
// India language-variants (ta_Taml_IN, ...) all end in _IN and fly the India flag.
const localeLabel = (locale) => `${{_LOCALE_FLAGS[locale] ?? (locale.endsWith("_IN") ? "🇮🇳" : "🌐")}}  ${{locale}}`;

function VegaChart({{spec}}) {{
  const ref = useRef(null);
  useEffect(() => {{
    if (!ref.current || !spec) return;
    let view;
    vegaEmbed(ref.current, spec, {{actions: false, renderer: "canvas"}}).then((result) => {{
      view = result.view;
      view.addEventListener("click", (event, item) => {{
        if (item && item.datum && item.datum.deeplink) {{
          window.open(item.datum.deeplink, "_blank", "noopener,noreferrer");
        }}
      }});
    }}).catch((err) => {{
      ref.current.innerHTML = '<p class="muted">Vega-Lite render failed: ' + (err && err.message || err) + '</p>';
    }});
    return () => {{ if (view) {{ view.finalize(); }} }};
  }}, [spec]);
  return h("div", {{ref, className: "vega-chart"}});
}}

function HeroPanel() {{
  // Token cards now mirror the Layer 1b / Pareto framing: only output
  // tokens count as "conversation" or "assistant" verbosity. Input
  // tokens reflect prompt context size and would dilute the signal.
  // ``avg turns/conversation`` is weighted by per-run trajectory count
  // so a 1024-traj run dominates a 32-traj run in the panel summary.
  const totals = useMemo(() => {{
    const conv = report.entries.reduce((a, e) => a + (e.n_trajectories || 0), 0);
    const assistantOutput = report.entries.reduce((a, e) => a + (e.assistant_output_tokens || 0), 0);
    const userOutput = report.entries.reduce((a, e) => a + (e.user_output_tokens || 0), 0);
    const conversationTokens = userOutput + assistantOutput;
    const totalTurns = report.entries.reduce(
      (a, e) => a + ((e.mean_turns_per_conversation || 0) * (e.n_trajectories || 0)),
      0,
    );
    const avgTurns = conv > 0 ? totalTurns / conv : 0;
    return {{conv, assistantOutput, conversationTokens, avgTurns}};
  }}, []);
  return h("header", {{className: "hero"}},
    h("h1", null,
      h("span", null, "NeMo UserSim Multi-Model Comparison"),
      h("span", {{className: "hero-model"}}, `${{report.entries.length}} models · ${{report.locales.length}} locales`)
    ),
    h("div", {{className: "hero-runid"}}, `Comparison ID: ${{report.comparison_id}}`),
    h("div", {{className: "stats stats-row-1"}},
      h("div", null, h("strong", null, report.entries.length), h("span", null, "models compared")),
      h("div", null, h("strong", null, report.locales.length), h("span", null, "locales")),
      h("div", null, h("strong", null, report.capabilities.length), h("span", null, "capabilities")),
      h("div", null, h("strong", null, totals.conv.toLocaleString()), h("span", null, "total conversations")),
      h("div", null, h("strong", null, totals.avgTurns.toFixed(2)), h("span", null, "avg turns/conversation")),
      h("div", null, h("strong", null, formatTokens(totals.conversationTokens)), h("span", null, "conversation tokens")),
      h("div", null, h("strong", null, formatTokens(totals.assistantOutput)), h("span", null, "assistant tokens"))
    )
  );
}}

// The WarningsCallout component used to render a yellow aside listing
// every apples-to-apples drift heuristic that tripped. Removed -- at
// panel scale the warnings fire on nearly every comparison and dilute
// the signal. ``report.apples_to_apples_warnings`` is still populated
// on the data model so CLI / notebook log surfaces can surface them.

function ModelsCompared() {{
  // Per-row layout:
  //   Model | Run | Trajectories | Avg turns/conv | Sample mode |
  //   User tokens | Asst tokens | Asst reas tokens | Drill-down
  //
  // Only assistant reasoning is surfaced -- user simulators rarely
  // emit reasoning content, so the column was redundant clutter;
  // anyone needing it can drill into the per-run Sim Health panel.
  // ``Evaluated`` and ``Eval column`` are dropped -- the comparison
  // builder fails closed on eval_column drift, so by construction
  // every row is fully evaluated under the same column.
  return h("section", {{className: "panel"}},
    h("h2", null, "Models compared"),
    h("table", {{className: "comparison-table"}},
      h("thead", null, h("tr", null,
        ["Model", "Run", "Trajectories", "Avg turns/conv", "Sample mode", "User tokens", "Asst tokens", "Asst reas tokens", "Drill-down"]
          .map(x => h("th", {{key: x}}, x))
      )),
      h("tbody", null, report.entries.map(e => h("tr", {{key: e.run_id}},
        h("td", null, h("code", null, e.model_id || "—")),
        h("td", null, h("code", null, e.run_id)),
        h("td", null, e.n_trajectories.toLocaleString()),
        h("td", null, (e.mean_turns_per_conversation || 0).toFixed(2)),
        h("td", null, e.sample_mode),
        h("td", null, (e.user_output_tokens || 0).toLocaleString()),
        h("td", null, (e.assistant_output_tokens || 0).toLocaleString()),
        h("td", null, (e.assistant_reasoning_tokens || 0).toLocaleString()),
        h("td", null, h("a", {{
          href: `../../report/run=${{e.run_id}}/index.html`, target: "_blank", rel: "noopener noreferrer"
        }}, "Open per-run →"))
      )))
    )
  );
}}

function ParetoTabsBody() {{
  // Tab bundle: ["Overall", ...locale codes]. Object.keys preserves
  // insertion order in modern browsers (ES2020+), so "Overall" stays
  // the default active tab regardless of locale list. This component
  // renders ONLY the inner content (helper text + tab strip + chart);
  // the parent <section> + <h2> wrapper lives in the Verbosity
  // section so the heatmap below shares the same panel.
  const bundle = specsPayload.global.pareto_per_locale || {{}};
  const tabs = Object.keys(bundle);
  const [active, setActive] = useState(tabs.length ? tabs[0] : "");
  if (!tabs.length) return null;
  return h(React.Fragment, null,
    h("p", {{className: "helper-text"}}, "Top-left = better quality, fewer tokens. Pareto frontier in green. Dot size = % of measured cells passing. Click a dot to open the per-run dashboard. Tabs swap the Y-axis between the panel-wide score and a single-locale rollup; the verbosity X-axis stays at the run level so all tabs are comparable."),
    h("div", {{className: "locale-tabs"}}, tabs.map(loc =>
      h("button", {{
        key: loc, type: "button",
        className: `locale-tab ${{loc === active ? "active" : ""}}`,
        onClick: () => setActive(loc)
      }}, loc === "Overall" ? loc : localeLabel(loc))
    )),
    h(VegaChart, {{spec: bundle[active]}})
  );
}}

function VerbositySection() {{
  // Single section combining the Pareto tabs + verbosity heatmap.
  // Both charts speak to verbosity (Pareto puts it on the x-axis vs
  // quality; heatmap shows per-locale verbosity), so they read as a
  // single story instead of two adjacent panels. A horizontal-rule
  // divider sits between the two so the visual transition is
  // explicit -- without it the two charts blurred together.
  return h("section", {{className: "panel"}},
    h("h2", null, "Verbosity"),
    h(ParetoTabsBody),
    h("div", {{className: "panel-divider"}}),
    h("p", {{className: "helper-text"}}, "Per-locale assistant tokens per turn. Reveals language-specific verbosity (non-ASCII script expansion, etc.) that the per-conversation rollup hides."),
    h(VegaChart, {{spec: specsPayload.global.verbosity_heatmap}})
  );
}}

function PerLocaleTabs() {{
  // The tabs come from the per_locale dict's key order, which Python
  // inserts as ["Overall", ...locale codes]. So "Overall" is always
  // first and the default active tab. Object.keys preserves insertion
  // order in modern browsers (ES2020+).
  // Each tab renders TWO charts: the grouped-bar capability chart
  // (existing) and a cleared-rate companion underneath, both
  // filtered to the active tab.
  const tabs = Object.keys(specsPayload.per_locale || {{}});
  const [active, setActive] = useState(tabs.length ? tabs[0] : "");
  if (!tabs.length) return null;
  const clearedBundle = specsPayload.per_locale_cleared || {{}};
  const clearedSpec = clearedBundle[active];
  return h("section", {{className: "panel"}},
    h("h2", null, "Per-locale capability comparison"),
    h("div", {{className: "locale-tabs"}}, tabs.map(loc =>
      h("button", {{
        key: loc, type: "button",
        className: `locale-tab ${{loc === active ? "active" : ""}}`,
        onClick: () => setActive(loc)
      }}, loc === "Overall" ? loc : localeLabel(loc))
    )),
    h(VegaChart, {{spec: specsPayload.per_locale[active]}}),
    clearedSpec ? h("p", {{className: "helper-text"}}, "The cleared chart below tallies how many capability cells each model cleared in this view (score \u2265 threshold and no dashed-red must-pass check failed).") : null,
    clearedSpec ? h(VegaChart, {{spec: clearedSpec}}) : null
  );
}}

function PerCapabilityTabs() {{
  // Per-capability section: each tab is one quality capability.
  // Inside the tab we render the across-locales grouped bar chart
  // PLUS a cleared-rate companion (one bar per model, % of locales
  // passing for THIS capability).
  const entries = specsPayload.per_capability || [];
  const [active, setActive] = useState(entries.length ? entries[0].capability_id : "");
  if (!entries.length) return null;
  const activeEntry = entries.find(e => e.capability_id === active) || entries[0];
  return h("section", {{className: "panel"}},
    h("h2", null, "Performance by capability across locales"),
    h("div", {{className: "locale-tabs"}}, entries.map(e =>
      h("button", {{
        key: e.capability_id, type: "button",
        className: `locale-tab ${{e.capability_id === active ? "active" : ""}}`,
        onClick: () => setActive(e.capability_id)
      }}, e.capability_label)
    )),
    h(VegaChart, {{spec: activeEntry.spec}}),
    activeEntry.cleared_spec ? h("p", {{className: "helper-text"}}, "The cleared chart below tallies how many locales each model cleared this capability in (score \u2265 threshold and no dashed-red must-pass check failed).") : null,
    activeEntry.cleared_spec ? h(VegaChart, {{spec: activeEntry.cleared_spec}}) : null
  );
}}

function PerModelTabs() {{
  // Per-model section: each tab is one model. Inside the tab we
  // render the locale-breakdown grouped-bar chart PLUS a cleared-
  // rate companion (one bar per locale, % of capabilities cleared
  // in THIS locale for THIS model).
  const entries = specsPayload.per_model || [];
  const [active, setActive] = useState(entries.length ? entries[0].run_id : "");
  if (!entries.length) return null;
  const activeEntry = entries.find(e => e.run_id === active) || entries[0];
  return h("section", {{className: "panel"}},
    h("h2", null, "Performance by model across locales"),
    h("div", {{className: "locale-tabs"}}, entries.map(e =>
      h("button", {{
        key: e.run_id, type: "button",
        className: `locale-tab ${{e.run_id === active ? "active" : ""}}`,
        onClick: () => setActive(e.run_id)
      }}, e.display_label)
    )),
    h(VegaChart, {{spec: activeEntry.spec}}),
    activeEntry.cleared_spec ? h("p", {{className: "helper-text"}}, "The cleared chart below tallies how many capabilities this model cleared in each locale (score \u2265 threshold and no dashed-red must-pass check failed).") : null,
    activeEntry.cleared_spec ? h(VegaChart, {{spec: activeEntry.cleared_spec}}) : null
  );
}}

function App() {{
  return h("main", {{className: "comparison-app"}},
    h(HeroPanel),
    h(ModelsCompared),
    h("section", {{className: "panel"}},
      h("h2", null, "Simulator health"),
      // 2x2 grid:
      //   row 1: Failure rate | Failure breakdown by class
      //   row 2: Early stop rate | Persona grounding
      // Failure rate + breakdown stay paired because the breakdown
      // is the failure rate's drilldown -- they read together.
      // Early stop / persona pair on the second row as the remaining
      // sim-quality signals; early-stop leads the row because it's
      // a structural metric (did the conversation finish?) while
      // persona grounding is a quality metric (did the persona
      // stay coherent?).
      h("div", {{className: "sim-health-grid"}},
        h(VegaChart, {{spec: specsPayload.global.sim_health_failure}}),
        h(VegaChart, {{spec: specsPayload.global.sim_health_failure_taxonomy}}),
        h(VegaChart, {{spec: specsPayload.global.sim_health_early}}),
        h(VegaChart, {{spec: specsPayload.global.sim_health_persona}})
      )
    ),
    h("section", {{className: "panel"}},
      h("h2", null, "Performance"),
      // Defines "cleared" once for the whole report; per-tab
      // sections below reference the same definition with a tighter
      // one-liner.
      h("p", {{className: "helper-text"}}, "Cleared capabilities (leftmost chart): a (capability, locale) cell clears when the score meets its threshold and no must-pass check fails. Must-pass failures appear as dashed-red bars in the per-locale and per-capability charts below — they count as blocked even when the score is high."),
      h("div", {{className: "verdict-row"}},
        // Cleared % goes first -- "did this model pass?" reads
        // before "what was its score?". Score and verbosity follow.
        h(VegaChart, {{spec: specsPayload.global.cleared_headline}}),
        h(VegaChart, {{spec: specsPayload.global.leaderboard}}),
        h(VegaChart, {{spec: specsPayload.global.verbosity_headline}})
      )
    ),
    h(VerbositySection),
    h(PerLocaleTabs),
    h(PerCapabilityTabs),
    h(PerModelTabs),
    h("details", {{className: "panel collapsible-panel"}},
      h("summary", null, h("h2", null, "Coverage matrix")),
      h("div", {{className: "collapsible-body"}},
        h("p", {{className: "helper-text"}}, "(Model × locale) -> mean trajectories per capability cell (excluding simulation_reliability). Dim cells flag a model that under-evaluated a locale."),
        h(VegaChart, {{spec: specsPayload.global.coverage}})
      )
    )
  );
}}

createRoot(document.getElementById("{root_id}")).render(h(App));
</script>
</body>
</html>
"""
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def write_comparison_dashboard_artifacts(
    report: ComparisonReport,
    out_dir: str | Path,
    *,
    build_react: bool = True,  # kept for API symmetry with the per-run dashboard
) -> Path:
    """Write ``comparison_manifest.json`` + ``comparison_matrix.json`` + ``index.html``.

    Returns the path to ``index.html``.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    (out_path / "comparison_manifest.json").write_text(
        json.dumps(report.to_dict(), indent=2, default=str, ensure_ascii=False)
    )
    (out_path / "comparison_matrix.json").write_text(
        json.dumps(
            [asdict(c) for c in report.cells],
            indent=2,
            default=str,
            ensure_ascii=False,
        )
    )

    labels = [r.display_label for r in report.model_summary]
    colors, shapes = assign_palette(labels)

    global_specs = {
        # Layer 0 — three separate small charts so they reflow with
        # window width (matches the Layer 1 row's float behavior).
        # Failure rate replaced the old OK rate: failure clusters
        # below ~10% so reviewers can spot outliers, while OK rates
        # all sit clamped near 100% and read as visual noise. The
        # same auto-domain reasoning applies to persona-grounding and
        # early-stop, so all three Sim Health charts now scale to
        # their data extent rather than a fixed 0-1 axis.
        #
        # Every chart shows model y-labels: with the 2x2 grid layout,
        # the rate-chart row 2 (early stop / persona) sits in
        # different DOM cells than the row 1 (failure / breakdown), so
        # there's no shared y-axis gutter to inherit -- each chart
        # needs its own labels.
        "sim_health_failure": _spec_sim_health_metric(
            report,
            colors,
            metric_field="sim_status_failure_rate",
            metric_title="Simulator failure rate",
            show_y_labels=True,
            x_axis_format=".0%",
        ),
        "sim_health_persona": _spec_sim_health_metric(
            report,
            colors,
            metric_field="persona_grounding_rate",
            metric_title="Persona grounding",
            show_y_labels=True,
            x_domain=(0, 1),
            x_axis_format=".0%",
        ),
        "sim_health_early": _spec_sim_health_metric(
            report,
            colors,
            metric_field="early_stop_rate",
            metric_title="Early stop rate",
            show_y_labels=True,
            x_domain=(0, 1),
            x_axis_format=".0%",
        ),
        # Failure-mode breakdown sits below the rate row in the same
        # Sim Health section; full-width stacked bar so the legend
        # has room to breathe alongside the bars.
        "sim_health_failure_taxonomy": _spec_failure_taxonomy(report),
        # Performance row: cleared first (the most informative
        # at-a-glance signal), then mean score, then verbosity.
        "cleared_headline": _spec_cleared_headline(report, colors),
        "leaderboard": _spec_leaderboard(report, colors),
        "verbosity_headline": _spec_verbosity_headline(report, colors),
        # Layer 2 carries an "Overall" tab plus one tab per locale.
        # The Overall tab's score is the run-level mean (same as the
        # leaderboard); each per-locale tab swaps in a locale-specific
        # mean computed from the cell spine. The verbosity x-axis is
        # held constant across tabs so the chart's interpretation
        # ("higher = better, leftward = less verbose") doesn't shift.
        "pareto_per_locale": {
            "Overall": _spec_pareto(report, colors, shapes),
            **{
                locale: _spec_pareto(
                    report,
                    colors,
                    shapes,
                    score_lookup=_pareto_score_lookup_for_locale(report, locale),
                    locale_label=locale,
                )
                for locale in report.locales
            },
        },
        "verbosity_heatmap": _spec_verbosity_heatmap(report, colors),
        "coverage": _spec_coverage(report, colors),
    }
    # "Overall" is inserted FIRST so it's the default tab in the React
    # PerLocaleTabs component (which keys on Object.keys() insertion order).
    locale_specs = {
        "Overall": _spec_overall_grouped_bars(report, colors),
        **{locale: _spec_per_locale_grouped_bars(report, locale, colors) for locale in report.locales},
    }
    # Parallel cleared-rate bundle for the per-locale tab section.
    # Same keys as ``locale_specs`` so the React component can index
    # by the active tab name. The Overall key reuses the panel-wide
    # cleared rates already on each rollup.
    locale_cleared_specs = {
        "Overall": _spec_cleared_for_locale(report, "", colors, is_overall=True),
        **{locale: _spec_cleared_for_locale(report, locale, colors, is_overall=False) for locale in report.locales},
    }
    # One entry per plottable capability (drops simulation_reliability —
    # that's a process metric covered by Sim Health). Each entry
    # carries the capability id + label alongside both the main facet
    # spec and a cleared-rate companion so the React tab strip can
    # render labels and toggle through both charts.
    per_capability_specs = [
        {
            "capability_id": cap_id,
            "capability_label": cap_label,
            "spec": _spec_per_capability_facet(report, cap_id, cap_label, colors),
            "cleared_spec": _spec_cleared_for_capability(
                report,
                cap_id,
                cap_label,
                colors,
            ),
        }
        for cap_id, cap_label in _plotting_capabilities(report)
    ]

    # Per-model bundle: one entry per model. Models are plotted
    # alphabetically so the tab order is stable across reruns
    # regardless of which models are in the comparison. Each entry
    # carries the original locale-breakdown spec and a cleared-rate
    # companion (per-locale within this model).
    per_model_specs = [
        {
            "run_id": entry.run_id,
            "display_label": entry.display_label,
            "spec": _spec_per_model_locale_breakdown(report, entry.run_id, entry.display_label),
            "cleared_spec": _spec_cleared_for_model_per_locale(
                report,
                entry.run_id,
                entry.display_label,
            ),
        }
        for entry in sorted(report.entries, key=lambda e: e.display_label)
    ]

    index_path = out_path / "index.html"
    _write_index_html(
        index_path,
        report,
        global_specs,
        locale_specs,
        locale_cleared_specs,
        per_capability_specs,
        per_model_specs,
    )
    return index_path
