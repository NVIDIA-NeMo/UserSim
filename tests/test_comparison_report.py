# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the multi-model comparison report.

Mirrors the structure of ``tests/test_capability_report.py``. Synthetic
manifests are built in-process — no fixtures on disk required.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pytest

from usersim.taxonomy.capabilities import capability_definitions
from usersim.reporting.comparison import (
    ComparisonCell,
    ComparisonReport,
    ModelRollup,
    _compute_pareto_frontier,
    _short_label,
    build_comparison_report,
    discover_comparison_runs,
    select_runs_for_comparison,
)
from usersim.reporting.comparison_dashboard import (
    _PALETTE,
    _SHAPES,
    assign_palette,
    write_comparison_dashboard_artifacts,
)
from usersim.reporting.runs import print_comparison_runs


# Quality-capability count, derived from the registry rather than hardcoded:
# rollups exclude ``simulation_reliability`` (a process-health metric, not
# assistant quality), so "eligible" cells are every OTHER capability. Adding a
# capability row (e.g. a new probe family's) must not require editing counts
# scattered through this file.
N_ELIGIBLE_CAPS = len(
    [d for d in capability_definitions() if d.id != "simulation_reliability"]
)


# ---------------------------------------------------------------------------
# Synthetic manifest builders
# ---------------------------------------------------------------------------


def _capability_cell(
    *,
    capability: str,
    locale: str,
    state: str = "ready",
    score: float | None = 0.9,
    threshold: float | None = 0.8,
    n: int = 100,
    n_total: int = 100,
    failed_critical_axes: list[str] | None = None,
    aggregation_policy: str = "mean",
) -> dict:
    return {
        "capability": capability,
        "label": capability.replace("_", " ").title(),
        "description": "",
        "locale": locale,
        "state": state,
        "score": score,
        "threshold": threshold,
        "margin": None if (score is None or threshold is None) else round(score - threshold, 4),
        "n": n,
        "n_total": n_total,
        "failed_critical_axes": failed_critical_axes or [],
        "aggregation_policy": aggregation_policy,
    }


def _manifest(
    *,
    run_id: str,
    model_id: str = "vendor/family/test-model",
    locales: list[str] | None = None,
    capability_cells: list[dict] | None = None,
    eval_column: str = "assistant_eval",
    sample_mode: str = "full",
    n_trajectories: int = 1024,
    n_evaluated: int = 1024,
    assistant_total_tokens: int = 1_000_000,
    # Output is roughly a quarter of total (input dominates real
    # conversations). Tests that exercise the verbosity / Pareto charts
    # rely on this being non-zero.
    assistant_output_tokens: int | None = None,
    assistant_reasoning_tokens: int = 0,
    user_output_tokens: int | None = None,
    user_reasoning_tokens: int = 0,
    mean_turns_per_conversation: float = 3.5,
    sim_health: dict | None | bool = True,  # True=default, None=omit
    persona_summary: dict | None = None,
    omit_assistant_output_tokens: bool = False,
) -> dict:
    locales = locales or ["en_US"]
    capability_cells = capability_cells or []
    if assistant_output_tokens is None:
        assistant_output_tokens = assistant_total_tokens // 4
    if user_output_tokens is None:
        # User output is typically ~half of assistant output in real
        # runs (assistant is the verbose one). Plenty for the tests
        # that just need a non-zero value for hero / audit cells.
        user_output_tokens = assistant_output_tokens // 2
    out = {
        "run_id": run_id,
        "eval_column": eval_column,
        "model_id": model_id,
        "sample_mode": sample_mode,
        "n_trajectories": n_trajectories,
        "n_evaluated": n_evaluated,
        "locales": locales,
        "capability_cells": capability_cells,
        "assistant_total_tokens": assistant_total_tokens,
        "assistant_reasoning_tokens": assistant_reasoning_tokens,
        "user_output_tokens": user_output_tokens,
        "user_reasoning_tokens": user_reasoning_tokens,
        "persona_summary": persona_summary or {"personas": n_trajectories},
    }
    # Caller can opt out of emitting the new field to simulate the
    # legacy manifest shape (pre-three-row Sim Health redesign). The
    # entry-builder must default missing values to 0 without raising.
    if not omit_assistant_output_tokens:
        out["assistant_output_tokens"] = assistant_output_tokens
    if sim_health is True:
        out["sim_health"] = {
            "n_trajectories": n_trajectories,
            "status_counts": {"ok": n_trajectories, "completed_with_warnings": 0, "failed": 0},
            "persona_grounding_rate": 1.0,
            "early_stop_rate": 0.5,
            # ``counters_means.n_turns`` feeds the new
            # avg-turns/conversation hero card -- always include it
            # so synthetic reports don't accidentally land on 0.0.
            "counters_means": {"n_turns": mean_turns_per_conversation},
            "response_length_by_locale": [
                {"locale": loc, "mean_tokens_per_turn": 200.0 + i * 10}
                for i, loc in enumerate(locales)
            ],
        }
    elif sim_health is None:
        # Don't add sim_health key at all
        pass
    else:
        out["sim_health"] = sim_health
    return out


def _full_capability_cells(
    *,
    locales: list[str],
    base_score: float = 0.9,
    score_overrides: dict[tuple[str, str], dict] | None = None,
) -> list[dict]:
    """One cell per (capability, locale) pair, with optional per-cell overrides."""
    cells: list[dict] = []
    overrides = score_overrides or {}
    for definition in capability_definitions():
        for locale in locales:
            base = {
                "capability": definition.id,
                "locale": locale,
                "score": base_score,
                "threshold": 0.8,
                "state": "ready" if base_score >= 0.8 else "blocked",
            }
            base.update(overrides.get((definition.id, locale), {}))
            cells.append(_capability_cell(**base))
    return cells


def _write_manifest(tmp_path: Path, manifest: dict) -> Path:
    run_dir = tmp_path / f"run={manifest['run_id']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "report_manifest.json"
    path.write_text(json.dumps(manifest))
    return path


# ===========================================================================
# Discovery + materialization
# ===========================================================================


class TestDiscovery:
    def test_discover_comparison_runs_orders_descending(self, tmp_path):
        for run_id in ["1000000001", "1000000003", "1000000002"]:
            _write_manifest(tmp_path, _manifest(run_id=run_id))
        paths = discover_comparison_runs(tmp_path)
        ids = [p.parent.name.removeprefix("run=") for p in paths]
        assert ids == ["1000000003", "1000000002", "1000000001"]

    def test_discover_comparison_runs_returns_empty_for_missing_root(self, tmp_path):
        assert discover_comparison_runs(tmp_path / "does-not-exist") == []

    def test_discover_comparison_runs_returns_empty_for_no_manifests(self, tmp_path):
        # Directory exists but no run=*/report_manifest.json children.
        (tmp_path / "unrelated.txt").write_text("noise")
        assert discover_comparison_runs(tmp_path) == []


class TestMaterialization:
    def test_full_matrix_includes_every_capability_locale_run(self, tmp_path):
        locales = ["en_US", "fr_FR"]
        m1 = _write_manifest(
            tmp_path,
            _manifest(
                run_id="1",
                model_id="v/x/alpha",
                locales=locales,
                capability_cells=_full_capability_cells(locales=locales),
            ),
        )
        m2 = _write_manifest(
            tmp_path,
            _manifest(
                run_id="2",
                model_id="v/x/beta",
                locales=locales,
                capability_cells=_full_capability_cells(locales=locales),
            ),
        )
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        n_caps = len(capability_definitions())
        assert len(report.cells) == n_caps * 2 * 2  # caps × locales × runs
        # Check no missing combos
        keys = {(c.capability, c.locale, c.run_id) for c in report.cells}
        assert len(keys) == len(report.cells)

    def test_missing_combos_marked_state_missing(self, tmp_path):
        # m1 covers en_US only, m2 covers en_US + fr_FR.
        m1 = _write_manifest(
            tmp_path,
            _manifest(
                run_id="1",
                locales=["en_US"],
                capability_cells=_full_capability_cells(locales=["en_US"]),
            ),
        )
        m2 = _write_manifest(
            tmp_path,
            _manifest(
                run_id="2",
                locales=["en_US", "fr_FR"],
                capability_cells=_full_capability_cells(locales=["en_US", "fr_FR"]),
            ),
        )
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        # m1's fr_FR cells should be present but state=missing.
        m1_fr = [
            c for c in report.cells if c.run_id == "1" and c.locale == "fr_FR"
        ]
        assert len(m1_fr) == len(capability_definitions())
        assert all(c.state == "missing" and c.is_missing for c in m1_fr)


class TestEntryTokenFields:
    """ComparisonEntry surfaces both the gross billing total AND the
    output-only figure, with a graceful default for legacy manifests
    that pre-date the three-row Sim Health redesign.
    """

    def test_assistant_output_tokens_is_read_from_manifest(self, tmp_path):
        m1 = _write_manifest(
            tmp_path,
            _manifest(
                run_id="1", model_id="v/x/a",
                assistant_total_tokens=1_000_000,
                assistant_output_tokens=375_000,
                assistant_reasoning_tokens=42_000,
            ),
        )
        m2 = _write_manifest(
            tmp_path,
            _manifest(
                run_id="2", model_id="v/x/b",
                assistant_total_tokens=2_000_000,
                assistant_output_tokens=500_000,
            ),
        )
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        by_run = {e.run_id: e for e in report.entries}
        assert by_run["1"].assistant_output_tokens == 375_000
        assert by_run["2"].assistant_output_tokens == 500_000

    def test_legacy_manifest_without_output_field_defaults_to_zero(self, tmp_path):
        # Legacy manifests (predating the three-row redesign) do not
        # carry ``assistant_output_tokens``. The entry builder must
        # default to 0 instead of raising; the chart will then render
        # those models with an empty verbosity bar rather than crashing.
        m1 = _write_manifest(
            tmp_path,
            _manifest(
                run_id="1", model_id="v/x/legacy",
                assistant_total_tokens=1_000_000,
                omit_assistant_output_tokens=True,
            ),
        )
        m2 = _write_manifest(
            tmp_path,
            _manifest(run_id="2", model_id="v/x/modern"),
        )
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        by_run = {e.run_id: e for e in report.entries}
        assert by_run["1"].assistant_output_tokens == 0
        # And modern runs still surface their output figure intact.
        assert by_run["2"].assistant_output_tokens > 0

    def test_user_tokens_flow_through_to_entry(self, tmp_path):
        # User-side output / reasoning fields drive the new "user
        # tokens" / "user reasoning" columns in the audit table and
        # the "conversation tokens" hero card (user_output + asst_output).
        m1 = _write_manifest(
            tmp_path,
            _manifest(
                run_id="1", model_id="v/x/a",
                assistant_output_tokens=400_000,
                user_output_tokens=180_000,
                user_reasoning_tokens=12_000,
            ),
        )
        m2 = _write_manifest(
            tmp_path,
            _manifest(run_id="2", model_id="v/x/b"),
        )
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        e1 = next(e for e in report.entries if e.run_id == "1")
        assert e1.user_output_tokens == 180_000
        assert e1.user_reasoning_tokens == 12_000

    def test_mean_turns_per_conversation_read_from_sim_health(self, tmp_path):
        # ``sim_health.counters_means.n_turns`` -> entry field. Drives
        # the "avg turns/conversation" hero card.
        m1 = _write_manifest(
            tmp_path, _manifest(run_id="1", model_id="v/x/a", mean_turns_per_conversation=3.7),
        )
        m2 = _write_manifest(
            tmp_path, _manifest(run_id="2", model_id="v/x/b", mean_turns_per_conversation=4.2),
        )
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        by_run = {e.run_id: e for e in report.entries}
        assert by_run["1"].mean_turns_per_conversation == pytest.approx(3.7)
        assert by_run["2"].mean_turns_per_conversation == pytest.approx(4.2)

    def test_mean_turns_defaults_to_zero_when_sim_health_missing(self, tmp_path):
        # Very old runs / synthetic fixtures may omit sim_health
        # entirely. The entry builder must default to 0.0, not crash.
        m1 = _write_manifest(tmp_path, _manifest(run_id="1", sim_health=None))
        m2 = _write_manifest(tmp_path, _manifest(run_id="2"))
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        e1 = next(e for e in report.entries if e.run_id == "1")
        assert e1.mean_turns_per_conversation == 0.0


class TestHeroPanelLabels:
    """Hero stats row: 7 cards in a fixed order, output-only token math."""

    def test_hero_card_order_and_labels(self, tmp_path):
        # Order: models / locales / capabilities / total conversations
        #        / avg turns / conversation tokens / assistant tokens.
        # The dropped cards ("assistant tokens (sum)", "reasoning tokens (sum)")
        # must NOT appear; new ones must be present.
        m1 = _write_manifest(tmp_path, _manifest(run_id="1", model_id="v/x/a"))
        m2 = _write_manifest(tmp_path, _manifest(run_id="2", model_id="v/x/b"))
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        out = tmp_path / "out"
        write_comparison_dashboard_artifacts(report, out)
        html = (out / "index.html").read_text()
        for label in (
            "models compared",
            "locales",
            "capabilities",
            "total conversations",
            "avg turns/conversation",
            "conversation tokens",
            "assistant tokens",
        ):
            assert label in html, f"missing hero label: {label}"
        # Dropped labels:
        assert "assistant tokens (sum)" not in html
        assert "reasoning tokens (sum)" not in html

    def test_avg_turns_computed_with_trajectory_weighting(self, tmp_path):
        # Run A: 1024 trajs at 3.0 turns. Run B: 256 trajs at 7.0 turns.
        # Total turns = 1024*3 + 256*7 = 3072 + 1792 = 4864.
        # Total conv = 1280.  Weighted avg = 4864 / 1280 = 3.8.
        # The static fallback prints with two decimals -> "3.80".
        m1 = _write_manifest(
            tmp_path,
            _manifest(
                run_id="1", model_id="v/x/a",
                n_trajectories=1024, n_evaluated=1024,
                mean_turns_per_conversation=3.0,
            ),
        )
        m2 = _write_manifest(
            tmp_path,
            _manifest(
                run_id="2", model_id="v/x/b",
                n_trajectories=256, n_evaluated=256,
                mean_turns_per_conversation=7.0,
            ),
        )
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        out = tmp_path / "out"
        write_comparison_dashboard_artifacts(report, out)
        html = (out / "index.html").read_text()
        # The computed avg should appear in the hero block as "3.80".
        assert ">3.80<" in html, "expected weighted avg-turns 3.80 in hero stats"


class TestModelsComparedTable:
    """The audit table dropped Identity/Evaluated/Eval column / total
    tokens / user reasoning, renamed Asst reasoning to Asst Reas Tokens,
    and added Avg turns/conv."""

    def test_table_columns_match_redesign(self, tmp_path):
        m1 = _write_manifest(tmp_path, _manifest(run_id="1", model_id="v/x/a"))
        m2 = _write_manifest(tmp_path, _manifest(run_id="2", model_id="v/x/b"))
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        out = tmp_path / "out"
        write_comparison_dashboard_artifacts(report, out)
        html = (out / "index.html").read_text()
        # Static-fallback table headers we KEEP / ADD:
        for header in (
            "<th>Model</th>",
            "<th>Run</th>",
            "<th>Trajectories</th>",
            "<th>Avg turns/conv</th>",
            "<th>Sample mode</th>",
            "<th>User tokens</th>",
            "<th>Asst tokens</th>",
            "<th>Asst reas tokens</th>",
        ):
            assert header in html, f"missing static-fallback header: {header}"
        # Headers the audit table must not carry:
        assert "<th>Identity</th>" not in html
        assert "<th>Evaluated</th>" not in html
        assert "<th>Eval column</th>" not in html
        assert "<th>User reasoning</th>" not in html
        assert "Asst tokens (total)" not in html
        assert "Asst tokens (output)" not in html
        # Same column set must show up in the React-side header array.
        # It's emitted as a plain JS string in the bundle.
        assert '"User tokens", "Asst tokens", "Asst reas tokens"' in html
        # And the React row helper formats avg turns to 2 decimals.
        assert "mean_turns_per_conversation || 0).toFixed(2)" in html

    def test_avg_turns_renders_per_row_in_static_fallback(self, tmp_path):
        # Static fallback emits the per-entry mean_turns_per_conversation
        # formatted to 2 decimals inside a <td>. Two rows -> two cells.
        m1 = _write_manifest(
            tmp_path,
            _manifest(run_id="1", model_id="v/x/a", mean_turns_per_conversation=3.49),
        )
        m2 = _write_manifest(
            tmp_path,
            _manifest(run_id="2", model_id="v/x/b", mean_turns_per_conversation=4.75),
        )
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        out = tmp_path / "out"
        write_comparison_dashboard_artifacts(report, out)
        html = (out / "index.html").read_text()
        assert "<td>3.49</td>" in html
        assert "<td>4.75</td>" in html


class TestApplesToApplesWarningsRemoved:
    """Drift warnings are no longer rendered on the comparison surface
    (they fire on nearly every panel-scale comparison and dilute signal).
    The data still lives on ``ComparisonReport`` for CLI / log surfaces.
    """

    def test_warnings_callout_dropped_from_react_app(self, tmp_path):
        # Build a report with a sample_mode mismatch so the warnings
        # list is non-empty -- if the React component were still wired
        # in, the rendered HTML would carry the callout markup.
        m1 = _write_manifest(tmp_path, _manifest(run_id="1", model_id="v/x/a", sample_mode="full"))
        m2 = _write_manifest(tmp_path, _manifest(run_id="2", model_id="v/x/b", sample_mode="per_locale"))
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        # The data model still surfaces them (CLI / logs can opt in).
        assert any("sample_mode" in w for w in report.apples_to_apples_warnings)
        out = tmp_path / "out"
        write_comparison_dashboard_artifacts(report, out)
        html = (out / "index.html").read_text()
        # Neither the React component nor the static <aside> may render.
        assert "Apples-to-apples warnings:" not in html
        # No live component definition or mount remains (the deleted
        # function used to declare ``function WarningsCallout()`` and
        # mount as ``h(WarningsCallout)`` -- both should be gone; a
        # docstring or comment can still mention the name historically).
        assert "function WarningsCallout()" not in html
        assert "h(WarningsCallout)" not in html
        # The CSS class is no longer instantiated as an element either
        # (the class definition stays in the stylesheet to support
        # other surfaces, so we only check element rendering).
        assert '<aside class="comparison-warnings"' not in html


class TestRollupTokenFields:
    """ModelRollup carries output- and reasoning-per-conv (the chart-
    driving metrics) alongside the gross billing total.
    """

    def test_rollup_per_conv_metrics_use_output_only(self, tmp_path):
        # 1024 trajectories, 512000 output tokens, 128000 reasoning ->
        # output_per_conv = 500, reasoning_per_conv = 125.
        m1 = _write_manifest(
            tmp_path,
            _manifest(
                run_id="1", model_id="v/x/a", n_trajectories=1024,
                assistant_total_tokens=4_000_000,
                assistant_output_tokens=512_000,
                assistant_reasoning_tokens=128_000,
            ),
        )
        m2 = _write_manifest(
            tmp_path,
            _manifest(run_id="2", model_id="v/x/b", n_trajectories=1024),
        )
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        m1_rollup = next(r for r in report.model_summary if r.run_id == "1")
        assert m1_rollup.assistant_output_tokens_per_conversation == pytest.approx(500.0)
        assert m1_rollup.assistant_reasoning_tokens_per_conversation == pytest.approx(125.0)


class TestEvidenceSnippetsDiscipline:
    def test_comparison_cell_carries_no_evidence_snippets(self):
        cell_fields = {f.name for f in fields(ComparisonCell)}
        # The size-control discipline: no per-turn / per-snippet payload.
        assert "evidence_snippets" not in cell_fields
        assert "turns" not in cell_fields
        assert "per_turn_judge" not in cell_fields


class TestLabelCollision:
    def test_label_collision_disambiguates_with_run_id(self, tmp_path):
        m1 = _write_manifest(tmp_path, _manifest(run_id="111", model_id="v/x/twin"))
        m2 = _write_manifest(tmp_path, _manifest(run_id="222", model_id="v/x/twin"))
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        labels = {e.display_label for e in report.entries}
        # Both labels must be distinct AND include run id suffix.
        assert len(labels) == 2
        assert all("(run=" in lbl for lbl in labels)

    def test_distinct_models_keep_short_labels(self, tmp_path):
        m1 = _write_manifest(tmp_path, _manifest(run_id="111", model_id="v/x/alpha"))
        m2 = _write_manifest(tmp_path, _manifest(run_id="222", model_id="v/x/beta"))
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        labels = {e.display_label for e in report.entries}
        assert labels == {"alpha", "beta"}

    def test_short_label_strips_provider_prefix(self):
        assert _short_label("nvidia/google/gemma-4-31b-it") == "gemma-4-31b-it"
        assert _short_label("openai/openai/gpt-5.4") == "gpt-5.4"
        assert _short_label(None) == "<unknown>"
        assert _short_label("") == "<unknown>"


# ===========================================================================
# Score rollup math (the load-bearing arithmetic)
# ===========================================================================


class TestRollupMath:
    def _two_run_report_with_overrides(
        self,
        tmp_path: Path,
        *,
        m1_overrides: dict[tuple[str, str], dict] | None = None,
    ) -> ComparisonReport:
        locales = ["en_US"]
        cells_baseline = _full_capability_cells(locales=locales, base_score=0.9)
        cells_with_overrides = _full_capability_cells(
            locales=locales, base_score=0.9, score_overrides=m1_overrides
        )
        m1 = _write_manifest(
            tmp_path,
            _manifest(run_id="1", model_id="v/x/m1", locales=locales,
                      capability_cells=cells_with_overrides),
        )
        m2 = _write_manifest(
            tmp_path,
            _manifest(run_id="2", model_id="v/x/m2", locales=locales,
                      capability_cells=cells_baseline),
        )
        return build_comparison_report([m1, m2], comparison_id="cmp_test")

    def _rollup(self, report: ComparisonReport, label: str):
        return next(r for r in report.model_summary if r.display_label == label)

    def test_mean_normalized_score_skips_missing_cells(self, tmp_path):
        # Flip the first 5 capability cells to missing on m1; score on
        # measured cells stays identical to m2 -> rollup score stays equal.
        cap_ids = [d.id for d in capability_definitions()][:5]
        overrides = {
            (cap_id, "en_US"): {"state": "missing", "score": None, "threshold": None}
            for cap_id in cap_ids
        }
        report = self._two_run_report_with_overrides(tmp_path, m1_overrides=overrides)
        m1, m2 = self._rollup(report, "m1"), self._rollup(report, "m2")
        assert m1.mean_normalized_score is not None
        assert m2.mean_normalized_score is not None
        assert m1.mean_normalized_score == pytest.approx(m2.mean_normalized_score)

    def test_mean_normalized_score_excludes_simulation_reliability(self, tmp_path):
        # Flip simulation_reliability from 0.99 (default ish) to 0.0 on m1;
        # rollup score must not change.
        baseline = self._two_run_report_with_overrides(tmp_path)
        m1_baseline = self._rollup(baseline, "m1").mean_normalized_score

        overrides = {("simulation_reliability", "en_US"): {"score": 0.0, "state": "blocked"}}
        report = self._two_run_report_with_overrides(tmp_path, m1_overrides=overrides)
        m1 = self._rollup(report, "m1")
        assert m1.mean_normalized_score == pytest.approx(m1_baseline)

    def test_pct_measured_surfaces_coverage_gap(self, tmp_path):
        # 5 missing cells out of (15-1)=14 eligible per locale (1 run, 1 locale)
        # -> measured = 14 - 5 = 9, eligible = 14, pct_measured ~= 9/14 = 0.643
        cap_ids = [d.id for d in capability_definitions() if d.id != "simulation_reliability"][:5]
        overrides = {
            (cap_id, "en_US"): {"state": "missing", "score": None, "threshold": None}
            for cap_id in cap_ids
        }
        report = self._two_run_report_with_overrides(tmp_path, m1_overrides=overrides)
        m1 = self._rollup(report, "m1")
        m2 = self._rollup(report, "m2")
        assert m2.pct_measured == pytest.approx(1.0)
        # Eligible = every cap minus simulation_reliability; 5 of them are
        # overridden to "missing" above.
        assert m1.n_cells_eligible == N_ELIGIBLE_CAPS
        assert m1.n_cells_measured == N_ELIGIBLE_CAPS - 5
        assert m1.pct_measured == pytest.approx((N_ELIGIBLE_CAPS - 5) / N_ELIGIBLE_CAPS)

    def test_mean_normalized_score_uses_per_locale_then_mean_of_means(self, tmp_path):
        # Two locales. For run m1, en_US scores 1.0 on every cell and
        # fr_FR scores 0.5 on every cell. Per-locale-then-mean gives
        # (1.0 + 0.5) / 2 = 0.75. A flat mean would give the same here
        # because both locales have the same cell count, but the test
        # locks the math regardless. For run m2, both locales score 0.75
        # uniformly -> rollup also 0.75. The two runs end up tied at the
        # leaderboard top, validating that per-locale aggregation works.
        locales = ["en_US", "fr_FR"]

        def _cells(per_locale_score):
            cells: list[dict] = []
            for cap_def in capability_definitions():
                for loc in locales:
                    cells.append(_capability_cell(
                        capability=cap_def.id,
                        locale=loc,
                        score=per_locale_score[loc],
                        threshold=0.8,
                        state="ready" if per_locale_score[loc] >= 0.8 else "blocked",
                    ))
            return cells

        m1 = _write_manifest(
            tmp_path,
            _manifest(run_id="1", model_id="v/x/m1", locales=locales,
                      capability_cells=_cells({"en_US": 1.0, "fr_FR": 0.5})),
        )
        m2 = _write_manifest(
            tmp_path,
            _manifest(run_id="2", model_id="v/x/m2", locales=locales,
                      capability_cells=_cells({"en_US": 0.75, "fr_FR": 0.75})),
        )
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        m1 = next(r for r in report.model_summary if r.display_label == "m1")
        m2 = next(r for r in report.model_summary if r.display_label == "m2")
        assert m1.mean_normalized_score == pytest.approx(0.75)
        assert m2.mean_normalized_score == pytest.approx(0.75)

    def test_normalizes_scores_across_scales(self, tmp_path):
        # The per-run dashboard already normalizes to 0-1, so every
        # ComparisonCell.normalized_score should be in [0,1] regardless of
        # whether the source capability was rate or score_1_5. Sanity check.
        m1 = _write_manifest(
            tmp_path,
            _manifest(
                run_id="1", locales=["en_US"],
                capability_cells=_full_capability_cells(locales=["en_US"]),
            ),
        )
        m2 = _write_manifest(
            tmp_path,
            _manifest(
                run_id="2", locales=["en_US"],
                capability_cells=_full_capability_cells(locales=["en_US"]),
            ),
        )
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        for c in report.cells:
            if c.normalized_score is not None:
                assert 0.0 <= c.normalized_score <= 1.0
            if c.threshold is not None:
                assert 0.0 <= c.threshold <= 1.0


# ===========================================================================
# Pareto frontier (Python-side)
# ===========================================================================


class TestParetoFrontier:
    def _rollup(self, label, score, tokens, *, reasoning: float = 0.0):
        # ``tokens`` here is the OUTPUT-only verbosity figure, since
        # that's what the Pareto frontier dominance check now uses.
        # ``reasoning`` is optional and unused by the dominance logic
        # — included so individual tests can populate the new field
        # if they want to assert it surfaces in chart specs.
        from usersim.reporting.comparison import ModelRollup

        return ModelRollup(
            run_id=f"r-{label}",
            display_label=label,
            model_id=label,
            mean_normalized_score=score,
            assistant_output_tokens_per_conversation=tokens,
            assistant_reasoning_tokens_per_conversation=reasoning,
        )

    def test_pareto_frontier_marks_dominant_models(self):
        # 4 models: alpha (high score, low tokens) and beta (low score, low tokens
        # — frontier because no model has both higher score AND lower tokens).
        # gamma is dominated by alpha (lower score, higher tokens).
        # delta is dominated by alpha too.
        rollups = [
            self._rollup("alpha", 0.9, 1000),
            self._rollup("beta", 0.5, 500),
            self._rollup("gamma", 0.7, 2000),
            self._rollup("delta", 0.6, 1500),
        ]
        _compute_pareto_frontier(rollups)
        on_frontier = {r.display_label for r in rollups if r.is_pareto_frontier}
        assert on_frontier == {"alpha", "beta"}

    def test_pareto_frontier_handles_ties(self):
        # Two models with identical (score, tokens) — neither dominates the
        # other (strict comparison required), so both stay on the frontier.
        rollups = [
            self._rollup("alpha", 0.8, 1000),
            self._rollup("twin", 0.8, 1000),
            self._rollup("worse", 0.5, 1500),
        ]
        _compute_pareto_frontier(rollups)
        on_frontier = {r.display_label for r in rollups if r.is_pareto_frontier}
        assert on_frontier == {"alpha", "twin"}

    def test_pareto_frontier_excludes_unscored_or_zero_tokens(self):
        rollups = [
            self._rollup("scored", 0.9, 1000),
            self._rollup("no_score", None, 1000),
            self._rollup("no_tokens", 0.9, 0),
        ]
        _compute_pareto_frontier(rollups)
        assert rollups[0].is_pareto_frontier is True
        assert rollups[1].is_pareto_frontier is False
        assert rollups[2].is_pareto_frontier is False


# ===========================================================================
# Apples-to-apples discipline
# ===========================================================================


class TestApplesToApples:
    def test_fails_closed_on_eval_column_mismatch(self, tmp_path):
        m1 = _write_manifest(tmp_path, _manifest(run_id="1", eval_column="assistant_eval"))
        m2 = _write_manifest(tmp_path, _manifest(run_id="2", eval_column="eval_v1"))
        with pytest.raises(ValueError, match="Mixed eval_column"):
            build_comparison_report([m1, m2], comparison_id="cmp_test")

    def test_warns_on_sample_mode_mismatch(self, tmp_path):
        m1 = _write_manifest(tmp_path, _manifest(run_id="1", sample_mode="full"))
        m2 = _write_manifest(tmp_path, _manifest(run_id="2", sample_mode="per_locale_probe"))
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        assert any("sample_mode" in w for w in report.apples_to_apples_warnings)

    def test_warns_on_persona_panel_mismatch(self, tmp_path):
        m1 = _write_manifest(
            tmp_path, _manifest(run_id="1", persona_summary={"personas": 100})
        )
        m2 = _write_manifest(
            tmp_path, _manifest(run_id="2", persona_summary={"personas": 1000})
        )
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        assert any("persona_summary.personas" in w for w in report.apples_to_apples_warnings)

    def test_warns_on_n_trajectories_delta(self, tmp_path):
        m1 = _write_manifest(tmp_path, _manifest(run_id="1", n_trajectories=1000))
        m2 = _write_manifest(tmp_path, _manifest(run_id="2", n_trajectories=500))
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        assert any("n_trajectories" in w for w in report.apples_to_apples_warnings)

    def test_handles_missing_sim_health(self, tmp_path):
        m1 = _write_manifest(tmp_path, _manifest(run_id="1", sim_health=None))
        m2 = _write_manifest(tmp_path, _manifest(run_id="2"))
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        assert any("No sim_health" in w for w in report.apples_to_apples_warnings)
        m1_rollup = next(r for r in report.model_summary if r.run_id == "1")
        assert m1_rollup.has_sim_health is False
        assert m1_rollup.sim_status_ok_rate is None
        # Still rendered, no exception.

    def test_clean_runs_emit_no_warnings(self, tmp_path):
        m1 = _write_manifest(tmp_path, _manifest(run_id="1", model_id="v/x/alpha"))
        m2 = _write_manifest(tmp_path, _manifest(run_id="2", model_id="v/x/beta"))
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        assert report.apples_to_apples_warnings == []


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_raises_on_empty_input(self):
        with pytest.raises(ValueError, match="no runs to compare"):
            build_comparison_report([], comparison_id="cmp_test")

    def test_raises_on_single_run(self, tmp_path):
        m1 = _write_manifest(tmp_path, _manifest(run_id="1"))
        with pytest.raises(ValueError, match="comparison requires >=2 runs"):
            build_comparison_report([m1], comparison_id="cmp_test")

    def test_default_comparison_id_uses_cmp_prefix(self, tmp_path):
        m1 = _write_manifest(tmp_path, _manifest(run_id="1", model_id="v/x/a"))
        m2 = _write_manifest(tmp_path, _manifest(run_id="2", model_id="v/x/b"))
        report = build_comparison_report([m1, m2])
        assert report.comparison_id.startswith("cmp_")


# ===========================================================================
# Color + shape palette
# ===========================================================================


class TestPalette:
    def test_assign_palette_deterministic_per_label(self):
        labels = ["alpha", "beta", "gamma"]
        c1, s1 = assign_palette(labels)
        c2, s2 = assign_palette(labels)
        assert c1 == c2
        assert s1 == s2

    def test_assign_palette_distinct_for_seven_non_green_models(self):
        # Generic labels (no nano/super/ultra) are denied the
        # restricted-green slot, so the addressable palette shrinks
        # from 8 to 7. Up to 7 distinct labels each get a distinct
        # non-green color; the 8th label wraps and reuses the first
        # palette slot.
        labels = [f"m{i}" for i in range(7)]
        colors, shapes = assign_palette(labels)
        assert len(set(colors.values())) == 7
        assert len(set(shapes.values())) == 7
        # Tableau-10's chartreuse green never lands on a non-NVIDIA
        # label (regression guard for the qwen-shown-as-green bug).
        assert "#59a14f" not in set(colors.values())

    def test_assign_palette_eight_distinct_when_one_label_is_green_eligible(self):
        # If one label includes a green-eligibility substring, the
        # restricted slot opens up for IT and the panel can reach 8
        # distinct colors total.
        labels = [f"m{i}" for i in range(7)] + ["something-ultra-fancy"]
        colors, _ = assign_palette(labels)
        assert len(set(colors.values())) == 8
        # And the green slot is the one assigned to the eligible label.
        assert colors["something-ultra-fancy"] in {"#76b900", "#a8d63d", "#59a14f"}

    def test_assign_palette_wraps_above_seven_for_non_green_panels(self):
        # 9 generic labels: only 7 non-green palette slots are usable,
        # so two slots get reused. Still distinct count == 7.
        labels = [f"m{i}" for i in range(9)]
        colors, _ = assign_palette(labels)
        assert len(set(colors.values())) == 7
        assert "#59a14f" not in set(colors.values())

    def test_qwen_does_not_get_tableau_green(self):
        # Regression guard: in 6-model panels alongside two gemmas
        # and two nemotrons, qwen used to alphabetically land on
        # palette slot 4 (#59a14f, Tableau chartreuse green) which
        # made it visually read as NVIDIA-branded. With the green
        # restriction in place, qwen now picks the next non-green
        # slot instead.
        labels = [
            "gemma-4-31b-it (run=1)",
            "gemma-4-31b-it (run=2)",
            "gpt-5.4",
            "gpt-oss-120b",
            "nemotron-3-super-120b-long-ctx",
            "nemotron-3-ultra-preview",
            "qwen3.6-35b-a3b",
        ]
        colors, _ = assign_palette(labels)
        # Branded NVIDIA models keep their NVIDIA greens.
        assert colors["nemotron-3-ultra-preview"] == "#76b900"
        assert colors["nemotron-3-super-120b-long-ctx"] == "#a8d63d"
        # Non-branded models do NOT land on any NVIDIA-ish green.
        for label in ("gemma-4-31b-it (run=1)", "gemma-4-31b-it (run=2)",
                      "gpt-5.4", "gpt-oss-120b", "qwen3.6-35b-a3b"):
            assert colors[label] not in {"#76b900", "#a8d63d", "#59a14f"}, (
                f"{label} got a green-restricted hex: {colors[label]}"
            )

    def test_palette_uses_locked_tableau_set(self):
        assert len(_PALETTE) == 8
        assert len(_SHAPES) == 8

    def test_nemotron_ultra_gets_nvidia_green(self):
        labels = ["alpha", "beta", "nemotron-3-ultra-preview"]
        colors, _ = assign_palette(labels)
        # Canonical NVIDIA green.
        assert colors["nemotron-3-ultra-preview"] == "#76b900"

    def test_nemotron_super_gets_paler_nvidia_green(self):
        labels = ["alpha", "nemotron-3-super-120b-long-ctx"]
        colors, _ = assign_palette(labels)
        # Paler NVIDIA-derived green.
        assert colors["nemotron-3-super-120b-long-ctx"] == "#a8d63d"

    def test_nemotron_overrides_dont_collide_with_other_models(self):
        # Both nemotron variants + 6 other models = 8 colors total.
        # Overrides shouldn't steal palette slots from the rest.
        labels = [
            "alpha", "beta", "gamma", "delta", "epsilon", "zeta",
            "nemotron-3-ultra-preview",
            "nemotron-3-super-120b-long-ctx",
        ]
        colors, _ = assign_palette(labels)
        assert len(set(colors.values())) == 8
        # The two greens are present.
        assert "#76b900" in colors.values()
        assert "#a8d63d" in colors.values()

    def test_nemotron_override_matches_disambiguated_label(self):
        # Substring match so labels suffixed with " (run=...)" still hit.
        labels = ["alpha", "nemotron-3-ultra-preview (run=123)"]
        colors, _ = assign_palette(labels)
        assert colors["nemotron-3-ultra-preview (run=123)"] == "#76b900"


class TestVerbosityHeadlineAnnotation:
    def test_verbosity_annotation_omits_rank_parenthetical(self, tmp_path):
        import json

        from usersim.reporting.comparison_dashboard import _spec_verbosity_headline

        m1 = _write_manifest(tmp_path, _manifest(run_id="1", model_id="v/x/a"))
        m2 = _write_manifest(tmp_path, _manifest(run_id="2", model_id="v/x/b"))
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        spec = _spec_verbosity_headline(report, colors={"a": "#000", "b": "#fff"})
        rendered = json.dumps(spec)
        # Output-only metric replaced the old "tokens/conv" label.
        assert "Output tokens / conversation" in rendered
        # Rank belongs in the tooltip, not the inline annotation.
        assert "most verbose" not in rendered or "tooltip" in rendered
        # More specific: no annotation row should contain the parenthetical.
        for row in spec.get("data", {}).get("values", []):
            assert "most verbose" not in (row.get("annotation") or "")

    def test_verbosity_chart_is_stacked_visible_plus_reasoning(self, tmp_path):
        # Layer 1b stacks the bar into "visible" (output - reasoning)
        # and "reasoning" segments so reviewers can see what fraction
        # of the verbosity is invisible thinking. The data emits one
        # row per (model, segment), and the bar mark stacks on the
        # x axis.
        from usersim.reporting.comparison_dashboard import _spec_verbosity_headline

        # Two models, both with non-zero reasoning so both produce a
        # 2-row payload and we can verify the stack carries reasoning
        # data through to the spec.
        m1 = _write_manifest(
            tmp_path,
            _manifest(
                run_id="1", model_id="v/x/a",
                assistant_total_tokens=1_000_000,
                assistant_output_tokens=400_000,
                assistant_reasoning_tokens=120_000,
            ),
        )
        m2 = _write_manifest(
            tmp_path,
            _manifest(
                run_id="2", model_id="v/x/b",
                assistant_total_tokens=2_000_000,
                assistant_output_tokens=500_000,
                assistant_reasoning_tokens=80_000,
            ),
        )
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        spec = _spec_verbosity_headline(report, colors={"a": "#000", "b": "#fff"})
        values = spec["data"]["values"]
        # Two segments per model -> 2 models * 2 segments = 4 rows.
        assert len(values) == 4
        segments_per_model: dict[str, set[str]] = {}
        for row in values:
            segments_per_model.setdefault(row["display_label"], set()).add(row["segment"])
        for label, segs in segments_per_model.items():
            assert segs == {"visible", "reasoning"}, (
                f"model {label} missing a stack segment"
            )
        # The bar mark stacks the segments along x.
        bar_layer = spec["layer"][0]
        assert bar_layer["encoding"]["x"]["aggregate"] == "sum"
        assert bar_layer["encoding"]["x"]["stack"] == "zero"

    def test_pareto_x_axis_uses_output_only_tokens(self, tmp_path):
        import json

        from usersim.reporting.comparison_dashboard import _spec_pareto

        m1 = _write_manifest(
            tmp_path,
            _manifest(
                run_id="1", model_id="v/x/a",
                assistant_total_tokens=1_000_000,
                assistant_output_tokens=300_000,
                assistant_reasoning_tokens=50_000,
            ),
        )
        m2 = _write_manifest(
            tmp_path,
            _manifest(
                run_id="2", model_id="v/x/b",
                assistant_total_tokens=2_000_000,
                assistant_output_tokens=600_000,
                assistant_reasoning_tokens=200_000,
            ),
        )
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        spec = _spec_pareto(
            report,
            colors={"a": "#000", "b": "#fff"},
            shapes={"a": "circle", "b": "square"},
        )
        rendered = json.dumps(spec)
        # X-axis title makes "output" explicit so reviewers don't
        # confuse the metric with the gross input + output total.
        assert "output tokens per conversation" in rendered
        # Pareto datum carries both output and reasoning per-conv values
        # so the tooltip can show the breakdown.
        for row in spec["data"]["values"]:
            assert row["tokens_per_conv"] > 0
            assert "reasoning_per_conv" in row

    def test_pareto_y_axis_zooms_to_realistic_score_band(self, tmp_path):
        # Real measured rollups cluster in 0.6-0.9; the y-domain is
        # zoomed to [0.5, 1.0] so reviewers can see gaps between
        # models instead of staring at empty 0.0-0.5 dead-space.
        # The scale must be set on EVERY layer (line, point, text):
        # Vega-Lite's shared-scale resolver merges the layer scales
        # by union, and a layer without an explicit domain pulls the
        # scale back to its data extent (which for quantitative
        # scales defaults to ``zero: true`` and re-includes 0).
        from usersim.reporting.comparison_dashboard import _spec_pareto

        m1 = _write_manifest(tmp_path, _manifest(run_id="1", model_id="v/x/a"))
        m2 = _write_manifest(tmp_path, _manifest(run_id="2", model_id="v/x/b"))
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        spec = _spec_pareto(
            report,
            colors={"a": "#000", "b": "#fff"},
            shapes={"a": "circle", "b": "square"},
        )
        # All three layers (line, point, text) must carry the same
        # zoomed y-scale. Without that, Vega-Lite's resolver expanded
        # the domain back to [0, 1] silently.
        for layer_idx in (0, 1, 2):
            y_enc = spec["layer"][layer_idx]["encoding"]["y"]
            assert y_enc["scale"]["domain"] == [0.5, 1.0], (
                f"layer {layer_idx} y-scale not zoomed: {y_enc.get('scale')}"
            )
            assert y_enc["scale"]["zero"] is False, (
                f"layer {layer_idx} y-scale missing 'zero: false' override"
            )
        # The point layer (index 1) carries the axis tick lock too:
        # explicit ``values`` keep the labels at clean 0.1 increments
        # so .1f rounding never produces duplicate "0.5/0.5/0.6/0.6"
        # tick strings.
        point_axis = spec["layer"][1]["encoding"]["y"]["axis"]
        assert point_axis["values"] == [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        assert point_axis["format"] == ".1f"


class TestParetoLocaleTabs:
    """Layer 2 ships an Overall tab + per-locale tabs."""

    def test_pareto_per_locale_bundle_keys_are_overall_then_locale_codes(self, tmp_path):
        # Bundle key order (and therefore the React tab default-active
        # tab) must be ["Overall", ...locales].
        import json, re

        m1 = _write_manifest(tmp_path, _manifest(run_id="1", model_id="v/x/a", locales=["en_US", "fr_FR"],
                                                  capability_cells=_full_capability_cells(locales=["en_US", "fr_FR"])))
        m2 = _write_manifest(tmp_path, _manifest(run_id="2", model_id="v/x/b", locales=["en_US", "fr_FR"],
                                                  capability_cells=_full_capability_cells(locales=["en_US", "fr_FR"])))
        report = build_comparison_report([m1, m2], comparison_id="cmp_t")
        out = tmp_path / "out"
        write_comparison_dashboard_artifacts(report, out)
        html = (out / "index.html").read_text()
        m = re.search(
            r'<script id="comparison-specs-cmp_t" type="application/json">(.*?)</script>',
            html, re.DOTALL,
        )
        specs = json.loads(m.group(1).replace("<\\/", "</"))
        bundle = specs["global"]["pareto_per_locale"]
        assert list(bundle.keys()) == ["Overall", "en_US", "fr_FR"]
        # And the legacy single ``pareto`` key has been retired so
        # downstream consumers must opt into the bundle explicitly.
        assert "pareto" not in specs["global"]

    def test_per_locale_tab_score_uses_locale_specific_rollup(self, tmp_path):
        # Build two models. m1 scores 1.0 on every cell in en_US and
        # 0.5 on every cell in fr_FR. The Overall tab shows the
        # average (0.75); the en_US tab must show 1.0 and the fr_FR
        # tab must show 0.5 -- proving the y-axis swaps to the
        # locale-specific score, not the global rollup.
        locales = ["en_US", "fr_FR"]
        from usersim.taxonomy.capabilities import capability_definitions

        def _cells(per_locale):
            cells = []
            for cap_def in capability_definitions():
                for loc in locales:
                    cells.append(_capability_cell(
                        capability=cap_def.id, locale=loc,
                        score=per_locale[loc], threshold=0.4,
                        state="ready",
                    ))
            return cells

        m1 = _write_manifest(
            tmp_path,
            _manifest(run_id="1", model_id="v/x/m1", locales=locales,
                      capability_cells=_cells({"en_US": 1.0, "fr_FR": 0.5})),
        )
        m2 = _write_manifest(
            tmp_path,
            _manifest(run_id="2", model_id="v/x/m2", locales=locales,
                      capability_cells=_cells({"en_US": 0.7, "fr_FR": 0.7})),
        )
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        from usersim.reporting.comparison_dashboard import (
            _pareto_score_lookup_for_locale,
            _spec_pareto,
        )
        # Overall: m1 == 0.75 (avg of 1.0 and 0.5).
        overall = _spec_pareto(
            report,
            colors={"m1": "#000", "m2": "#fff"},
            shapes={"m1": "circle", "m2": "square"},
        )
        m1_overall_score = next(
            row["score"] for row in overall["data"]["values"]
            if row["display_label"] == "m1"
        )
        assert m1_overall_score == pytest.approx(0.75)
        # en_US tab: m1 score == 1.0.
        en_us = _spec_pareto(
            report,
            colors={"m1": "#000", "m2": "#fff"},
            shapes={"m1": "circle", "m2": "square"},
            score_lookup=_pareto_score_lookup_for_locale(report, "en_US"),
            locale_label="en_US",
        )
        m1_en_us_score = next(
            row["score"] for row in en_us["data"]["values"]
            if row["display_label"] == "m1"
        )
        assert m1_en_us_score == pytest.approx(1.0)
        # fr_FR tab: m1 score == 0.5.
        fr_fr = _spec_pareto(
            report,
            colors={"m1": "#000", "m2": "#fff"},
            shapes={"m1": "circle", "m2": "square"},
            score_lookup=_pareto_score_lookup_for_locale(report, "fr_FR"),
            locale_label="fr_FR",
        )
        m1_fr_fr_score = next(
            row["score"] for row in fr_fr["data"]["values"]
            if row["display_label"] == "m1"
        )
        assert m1_fr_fr_score == pytest.approx(0.5)
        # Per-locale tab title carries the locale label. Titles are
        # now dict-shaped ({text, anchor}) for left-anchored layout.
        title = fr_fr["title"]
        title_text = title["text"] if isinstance(title, dict) else title
        assert "fr_FR" in title_text
        # And titles drop the "Layer X — " prefix (lives in <h2> now).
        assert not title_text.startswith("Layer ")

    def test_layer1b_annotation_format(self, tmp_path):
        # Annotation should read e.g. "8,495 (5,417 reasoning)" --
        # plain output count, then reasoning subset spelled out (not
        # "reas"), and no " out" suffix on the bare number.
        from usersim.reporting.comparison_dashboard import _spec_verbosity_headline

        m1 = _write_manifest(
            tmp_path,
            _manifest(
                run_id="1", model_id="v/x/with_reasoning",
                # Output ~ 8500 / conv, reasoning ~ 5400 / conv.
                # Pick ratios that round cleanly to 8,495 / 5,417.
                n_trajectories=1000,
                assistant_output_tokens=8_495_000,
                assistant_reasoning_tokens=5_417_000,
            ),
        )
        m2 = _write_manifest(
            tmp_path,
            _manifest(
                run_id="2", model_id="v/x/no_reasoning",
                n_trajectories=1000,
                assistant_output_tokens=2_000_000,
                assistant_reasoning_tokens=0,
            ),
        )
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        spec = _spec_verbosity_headline(
            report, colors={"with_reasoning": "#000", "no_reasoning": "#fff"},
        )
        # Each model emits two rows (visible + reasoning); the
        # annotation field is identical across them, so collect the
        # set of unique annotations.
        annotations = {row["annotation"] for row in spec["data"]["values"]}
        assert "8,495 (5,417 reasoning)" in annotations
        # Models without reasoning omit the parenthetical entirely.
        assert "2,000" in annotations
        # No legacy " out" suffix or "reas" abbreviation anywhere.
        for ann in annotations:
            assert " out" not in ann, f"stale ' out' suffix in: {ann!r}"
            # The substring "reas" must only appear as part of the
            # full word "reasoning" (so " reas)" or " reas " must not
            # appear).
            assert " reas)" not in ann
            assert " reas " not in ann


class TestLeaderboardAnnotation:
    def test_leaderboard_annotation_is_score_only(self):
        from usersim.reporting.comparison_dashboard import _leaderboard_annotation

        rollup = ModelRollup(
            run_id="r1", display_label="m1", model_id="m1",
            n_cells_eligible=120, n_cells_ready=80, n_cells_measured=100,
            mean_normalized_score=0.832,
        )
        # Drops the "·measured·passing" decoration to keep the bar label clean.
        assert _leaderboard_annotation(rollup) == "0.83"

    def test_leaderboard_annotation_handles_no_score(self):
        from usersim.reporting.comparison_dashboard import _leaderboard_annotation

        rollup = ModelRollup(
            run_id="r1", display_label="m1", model_id="m1",
            mean_normalized_score=None,
        )
        assert _leaderboard_annotation(rollup) == "no measured cells"


class TestGapsCount:
    """ModelRollup.n_gaps counts blocked cells excluding simulation_reliability."""

    def test_n_gaps_excludes_simulation_reliability(self, tmp_path):
        # Force two cells blocked: one simulation_reliability + one quality.
        # n_cells_blocked counts both; n_gaps counts only the quality one.
        cap_ids = [d.id for d in capability_definitions()]
        first_quality_cap = next(c for c in cap_ids if c != "simulation_reliability")
        overrides = {
            ("simulation_reliability", "en_US"): {"score": 0.0, "state": "blocked"},
            (first_quality_cap, "en_US"): {"score": 0.0, "state": "blocked"},
        }
        cells_a = _full_capability_cells(locales=["en_US"], score_overrides=overrides)
        cells_b = _full_capability_cells(locales=["en_US"], base_score=0.9)
        m1 = _write_manifest(tmp_path, _manifest(run_id="1", model_id="v/x/m1",
                                                  locales=["en_US"], capability_cells=cells_a))
        m2 = _write_manifest(tmp_path, _manifest(run_id="2", model_id="v/x/m2",
                                                  locales=["en_US"], capability_cells=cells_b))
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        m1_rollup = next(r for r in report.model_summary if r.display_label == "m1")
        m2_rollup = next(r for r in report.model_summary if r.display_label == "m2")
        # m1: simulation_reliability blocked + 1 other -> n_cells_blocked=2,
        #     but n_gaps=1 (excludes simulation_reliability).
        assert m1_rollup.n_cells_blocked == 2
        assert m1_rollup.n_gaps == 1
        # m2: nothing blocked.
        assert m2_rollup.n_gaps == 0


class TestLocaleFlags:
    def test_locale_tabs_use_flag_emoji_in_jupyter_html(self, tmp_path):
        # The JS-side localeLabel function maps locale codes to flag emoji.
        # Verify the helper + flag dict ship in the rendered HTML so the
        # PerLocaleTabs buttons read e.g. "🇺🇸  en_US" instead of "en_US".
        m1 = _write_manifest(tmp_path, _manifest(run_id="1", model_id="v/x/a", locales=["en_US", "ja_JP"],
                                                  capability_cells=_full_capability_cells(locales=["en_US", "ja_JP"])))
        m2 = _write_manifest(tmp_path, _manifest(run_id="2", model_id="v/x/b", locales=["en_US", "ja_JP"],
                                                  capability_cells=_full_capability_cells(locales=["en_US", "ja_JP"])))
        report = build_comparison_report([m1, m2], comparison_id="cmp_t")
        out = tmp_path / "out"
        write_comparison_dashboard_artifacts(report, out)
        html = (out / "index.html").read_text()
        # JS helper present:
        assert "const localeLabel" in html
        assert "_LOCALE_FLAGS" in html
        # Per-locale tab JSX uses localeLabel for non-Overall labels:
        assert 'localeLabel(loc)' in html
        # At least the en_US + ja_JP flags are baked into the JS dict:
        assert '"🇺🇸"' in html or "🇺🇸" in html
        assert '"🇯🇵"' in html or "🇯🇵" in html

    def test_layer4_chart_titles_carry_flags(self, tmp_path):
        import json, re

        m1 = _write_manifest(tmp_path, _manifest(run_id="1", model_id="v/x/a", locales=["en_US"],
                                                  capability_cells=_full_capability_cells(locales=["en_US"])))
        m2 = _write_manifest(tmp_path, _manifest(run_id="2", model_id="v/x/b", locales=["en_US"],
                                                  capability_cells=_full_capability_cells(locales=["en_US"])))
        report = build_comparison_report([m1, m2], comparison_id="cmp_t")
        out = tmp_path / "out"
        write_comparison_dashboard_artifacts(report, out)
        html = (out / "index.html").read_text()
        m = re.search(
            r'<script id="comparison-specs-cmp_t" type="application/json">(.*?)</script>',
            html, re.DOTALL,
        )
        specs = json.loads(m.group(1).replace("<\\/", "</"))
        en_us_spec = specs["per_locale"]["en_US"]
        # Titles are dict-shaped ({text, anchor}) for the left-anchored
        # layout; the flag emoji lives inside ``text``.
        title = en_us_spec["title"]
        title_text = title["text"] if isinstance(title, dict) else title
        assert "🇺🇸" in title_text


class TestPerModelLayer:
    """Layer 7: tab strip across models, x = capability, color = locale."""

    def test_per_model_specs_emitted_with_run_id_and_label(self, tmp_path):
        import json, re

        m1 = _write_manifest(tmp_path, _manifest(run_id="1", model_id="v/x/alpha", locales=["en_US", "fr_FR"],
                                                  capability_cells=_full_capability_cells(locales=["en_US", "fr_FR"])))
        m2 = _write_manifest(tmp_path, _manifest(run_id="2", model_id="v/x/beta", locales=["en_US", "fr_FR"],
                                                  capability_cells=_full_capability_cells(locales=["en_US", "fr_FR"])))
        report = build_comparison_report([m1, m2], comparison_id="cmp_t")
        out = tmp_path / "out"
        write_comparison_dashboard_artifacts(report, out)
        html = (out / "index.html").read_text()
        m = re.search(
            r'<script id="comparison-specs-cmp_t" type="application/json">(.*?)</script>',
            html, re.DOTALL,
        )
        specs = json.loads(m.group(1).replace("<\\/", "</"))
        per_model = specs["per_model"]
        # One entry per model in alphabetical order (alpha < beta).
        assert len(per_model) == 2
        assert [e["display_label"] for e in per_model] == ["alpha", "beta"]
        # Each entry carries run_id + spec.
        for entry in per_model:
            assert "run_id" in entry and "display_label" in entry and "spec" in entry

    def test_per_model_spec_uses_capability_x_locale_color(self, tmp_path):
        import json, re

        m1 = _write_manifest(tmp_path, _manifest(run_id="1", model_id="v/x/a"))
        m2 = _write_manifest(tmp_path, _manifest(run_id="2", model_id="v/x/b"))
        report = build_comparison_report([m1, m2], comparison_id="cmp_t")
        out = tmp_path / "out"
        write_comparison_dashboard_artifacts(report, out)
        html = (out / "index.html").read_text()
        m = re.search(
            r'<script id="comparison-specs-cmp_t" type="application/json">(.*?)</script>',
            html, re.DOTALL,
        )
        specs = json.loads(m.group(1).replace("<\\/", "</"))
        sample = specs["per_model"][0]["spec"]
        # vconcat'd two-row layout (matches Layer 4's structure).
        sub = sample["vconcat"][0]
        # x = capability_label, xOffset = locale, color = locale.
        assert sub["encoding"]["x"]["field"] == "capability_label"
        bar = sub["layer"][0]
        assert bar["encoding"]["xOffset"]["field"] == "locale"
        assert bar["encoding"]["color"]["field"] == "locale"
        # Locale color scale is distinct from the model palette (uses
        # _LOCALE_PALETTE pastels, not _PALETTE saturated tableau-10).
        assert bar["encoding"]["color"]["scale"]["range"][0] != _PALETTE[0]

    def test_per_model_skips_simulation_reliability(self, tmp_path):
        import json, re

        m1 = _write_manifest(tmp_path, _manifest(run_id="1", model_id="v/x/a"))
        m2 = _write_manifest(tmp_path, _manifest(run_id="2", model_id="v/x/b"))
        report = build_comparison_report([m1, m2], comparison_id="cmp_t")
        out = tmp_path / "out"
        write_comparison_dashboard_artifacts(report, out)
        html = (out / "index.html").read_text()
        m = re.search(
            r'<script id="comparison-specs-cmp_t" type="application/json">(.*?)</script>',
            html, re.DOTALL,
        )
        specs = json.loads(m.group(1).replace("<\\/", "</"))
        # No data row references simulation_reliability in the per-model
        # bars (chart_rows are pre-filtered through _plotting_capabilities).
        rendered = json.dumps(specs["per_model"])
        # The capability label "Simulation reliability" must not appear
        # in any chart row data.
        assert "Simulation reliability" not in rendered


class TestVerdictRowSpecs:
    """Layer 0 + Layer 1 each ship 3 charts in the global specs dict."""

    def test_sim_health_emits_three_separate_specs(self, tmp_path):
        # The Sim Health row ships failure / persona / early-stop;
        # the OK-rate flavour was retired because its bars all sat
        # clamped near 100% and read as visual noise. Failure rate
        # auto-scales because failure clusters <10%.
        import json, re

        m1 = _write_manifest(tmp_path, _manifest(run_id="1", model_id="v/x/a"))
        m2 = _write_manifest(tmp_path, _manifest(run_id="2", model_id="v/x/b"))
        report = build_comparison_report([m1, m2], comparison_id="cmp_t")
        out = tmp_path / "out"
        write_comparison_dashboard_artifacts(report, out)
        html = (out / "index.html").read_text()
        m = re.search(
            r'<script id="comparison-specs-cmp_t" type="application/json">(.*?)</script>',
            html, re.DOTALL,
        )
        specs = json.loads(m.group(1).replace("<\\/", "</"))
        for key in ("sim_health_failure", "sim_health_persona", "sim_health_early"):
            assert key in specs["global"], f"missing {key} in global specs"
        # Legacy OK-rate spec is gone.
        assert "sim_health_ok" not in specs["global"]
        # Sim Health axes show percentages (..%, not bare floats).
        for key in ("sim_health_failure", "sim_health_persona", "sim_health_early"):
            assert specs["global"][key]["encoding"]["x"]["axis"]["format"] == ".0%"
        # No threshold rule on the failure-rate spec.
        assert "rule" not in json.dumps(specs["global"]["sim_health_failure"])

    def test_performance_row_emits_cleared_score_verbosity(self, tmp_path):
        # Performance row swaps the old gaps_headline for cleared_headline
        # and reorders to: cleared | leaderboard | verbosity.
        import json, re

        m1 = _write_manifest(tmp_path, _manifest(run_id="1", model_id="v/x/a"))
        m2 = _write_manifest(tmp_path, _manifest(run_id="2", model_id="v/x/b"))
        report = build_comparison_report([m1, m2], comparison_id="cmp_t")
        out = tmp_path / "out"
        write_comparison_dashboard_artifacts(report, out)
        html = (out / "index.html").read_text()
        m = re.search(
            r'<script id="comparison-specs-cmp_t" type="application/json">(.*?)</script>',
            html, re.DOTALL,
        )
        specs = json.loads(m.group(1).replace("<\\/", "</"))
        for key in ("cleared_headline", "leaderboard", "verbosity_headline"):
            assert key in specs["global"], f"missing {key} in global specs"
        # Old gaps chart is gone.
        assert "gaps_headline" not in specs["global"]
        # Cleared chart uses pct_cleared on the x axis with a
        # percentage axis format and a fixed 0-1 domain.
        cleared = specs["global"]["cleared_headline"]
        assert cleared["layer"][0]["encoding"]["x"]["field"] == "pct_cleared"
        assert cleared["layer"][0]["encoding"]["x"]["axis"]["format"] == ".0%"
        assert cleared["layer"][0]["encoding"]["x"]["scale"]["domain"] == [0, 1]
        # Leaderboard still uses .1f for clean score labels.
        lb = specs["global"]["leaderboard"]
        assert lb["layer"][0]["encoding"]["x"]["axis"]["format"] == ".1f"
        # No reference rules in any of the three (relative-comparison
        # framing — no arbitrary threshold lines).
        for key in ("cleared_headline", "leaderboard", "verbosity_headline"):
            assert "rule" not in json.dumps(specs["global"][key])

    def test_performance_row_y_label_distribution(self, tmp_path):
        # Cleared chart leads the row -> shows model labels.
        # Leaderboard + verbosity sit to its right -> hide labels
        # (no duplicate y-axis gutters in the same row).
        import json, re

        m1 = _write_manifest(tmp_path, _manifest(run_id="1", model_id="v/x/a"))
        m2 = _write_manifest(tmp_path, _manifest(run_id="2", model_id="v/x/b"))
        report = build_comparison_report([m1, m2], comparison_id="cmp_t")
        out = tmp_path / "out"
        write_comparison_dashboard_artifacts(report, out)
        html = (out / "index.html").read_text()
        m = re.search(
            r'<script id="comparison-specs-cmp_t" type="application/json">(.*?)</script>',
            html, re.DOTALL,
        )
        specs = json.loads(m.group(1).replace("<\\/", "</"))
        # Cleared shows labels (axis is a dict, not None).
        cleared_axis = specs["global"]["cleared_headline"]["layer"][0]["encoding"]["y"]["axis"]
        assert cleared_axis is not None
        assert cleared_axis.get("labelFontWeight") == 600
        # Leaderboard hides labels (axis: None).
        lb_axis = specs["global"]["leaderboard"]["layer"][0]["encoding"]["y"]["axis"]
        assert lb_axis is None
        # Verbosity also hides labels (was already None).
        verb_axis = specs["global"]["verbosity_headline"]["layer"][0]["encoding"]["y"]["axis"]
        assert verb_axis is None

    def test_react_app_renders_charts_in_correct_order(self, tmp_path):
        # The Performance row's React mount drives display order.
        # Cleared must appear before leaderboard, leaderboard before
        # verbosity. Same Sim Health row uses the new failure spec.

        m1 = _write_manifest(tmp_path, _manifest(run_id="1", model_id="v/x/a"))
        m2 = _write_manifest(tmp_path, _manifest(run_id="2", model_id="v/x/b"))
        report = build_comparison_report([m1, m2], comparison_id="cmp_t")
        out = tmp_path / "out"
        write_comparison_dashboard_artifacts(report, out)
        html = (out / "index.html").read_text()
        # Sim Health row uses the failure flavour.
        assert "specsPayload.global.sim_health_failure" in html
        # Performance row order: cleared first, then leaderboard,
        # then verbosity_headline.
        idx_cleared = html.index("specsPayload.global.cleared_headline")
        idx_lb = html.index("specsPayload.global.leaderboard")
        idx_verb = html.index("specsPayload.global.verbosity_headline")
        assert idx_cleared < idx_lb < idx_verb, (
            f"Performance row out of order: cleared={idx_cleared}, "
            f"leaderboard={idx_lb}, verbosity={idx_verb}"
        )


class TestClearedHeadline:
    """Performance row's headline chart shows the cleared-cell
    percentage with simulation_reliability excluded from the
    denominator.
    """

    def test_n_cells_ready_eligible_excludes_simulation_reliability(self, tmp_path):
        # Force every quality capability to ``ready`` and
        # simulation_reliability to ``blocked`` -- n_cells_ready
        # (legacy, includes reliability) drops by 1 cell vs
        # n_cells_ready_eligible (which excludes it).
        from usersim.taxonomy.capabilities import capability_definitions

        cap_ids = [d.id for d in capability_definitions()]
        # Build cells with simulation_reliability blocked, every other
        # capability ready. Single locale keeps the math clean.
        overrides = {
            ("simulation_reliability", "en_US"): {"score": 0.0, "state": "blocked"},
        }
        cells_with_block = _full_capability_cells(
            locales=["en_US"], score_overrides=overrides,
        )
        cells_clean = _full_capability_cells(locales=["en_US"])
        m1 = _write_manifest(
            tmp_path,
            _manifest(run_id="1", model_id="v/x/a", locales=["en_US"],
                      capability_cells=cells_with_block),
        )
        m2 = _write_manifest(
            tmp_path,
            _manifest(run_id="2", model_id="v/x/b", locales=["en_US"],
                      capability_cells=cells_clean),
        )
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        m1_rollup = next(r for r in report.model_summary if r.run_id == "1")
        m2_rollup = next(r for r in report.model_summary if r.run_id == "2")
        # m1: simulation_reliability blocked, every quality cap ready. The
        # eligible-only ready count is just the quality-ready count, and the
        # eligible total is all capabilities minus simulation_reliability.
        assert m1_rollup.n_cells_eligible == len(cap_ids) - 1
        assert m1_rollup.n_cells_ready_eligible == len(cap_ids) - 1
        # m2: every capability (incl. simulation_reliability) ready -> the
        # legacy count includes it, the eligible-only count excludes it.
        assert m2_rollup.n_cells_ready == len(cap_ids)
        assert m2_rollup.n_cells_ready_eligible == len(cap_ids) - 1

    def test_cleared_chart_uses_eligible_only_denominator(self, tmp_path):
        # 8 locales × the quality capabilities = the eligible cells. If every
        # quality cell passes, cleared = 100%; if every capability for one
        # locale fails, cleared = 7/8 = 87.5%.
        from usersim.taxonomy.capabilities import capability_definitions
        from usersim.reporting.comparison_dashboard import _spec_cleared_headline

        locales = ["en_US", "en_IN", "en_SG", "fr_FR",
                   "ja_JP", "ko_KR", "pt_BR", "hi_Deva_IN"]
        cap_ids = [d.id for d in capability_definitions()]
        # m1: every quality capability blocked in en_US.
        overrides = {
            (cap, "en_US"): {"score": 0.0, "state": "blocked"}
            for cap in cap_ids if cap != "simulation_reliability"
        }
        m1 = _write_manifest(
            tmp_path,
            _manifest(run_id="1", model_id="v/x/a", locales=locales,
                      capability_cells=_full_capability_cells(
                          locales=locales, score_overrides=overrides)),
        )
        m2 = _write_manifest(
            tmp_path,
            _manifest(run_id="2", model_id="v/x/b", locales=locales,
                      capability_cells=_full_capability_cells(locales=locales)),
        )
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        m1_rollup = next(r for r in report.model_summary if r.run_id == "1")
        # Quality capabilities × 8 locales = eligible.
        assert m1_rollup.n_cells_eligible == 8 * N_ELIGIBLE_CAPS
        # Every quality capability blocked in en_US -> one locale's worth of
        # cells drops out of the ready count.
        assert m1_rollup.n_cells_ready_eligible == 7 * N_ELIGIBLE_CAPS
        spec = _spec_cleared_headline(report, colors={
            "a": "#000", "b": "#fff", "v/x/a": "#000", "v/x/b": "#fff",
        })
        rows = {r["display_label"]: r for r in spec["data"]["values"]}
        m1_row = rows["a"]
        assert m1_row["pct_cleared"] == pytest.approx(7 / 8)
        assert m1_row["n_eligible"] == 8 * N_ELIGIBLE_CAPS
        assert m1_row["n_ready"] == 7 * N_ELIGIBLE_CAPS
        # m2: clean run -> every eligible cell ready = 100%.
        assert rows["b"]["pct_cleared"] == pytest.approx(1.0)
        # Annotation surfaces both proportion and raw counts.
        assert f"{7 * N_ELIGIBLE_CAPS}/{8 * N_ELIGIBLE_CAPS}" in m1_row["annotation"]


class TestPerTabClearedCharts:
    """Each tab section (per-locale / per-capability / per-model)
    ships a cleared-rate companion chart filtered to the active tab.
    """

    def _build_two_locale_panel(self, tmp_path) -> ComparisonReport:
        # Two locales, two models, with a deliberate per-locale skew:
        # m1 fails every quality cell in fr_FR, passes in en_US.
        # That gives unambiguous per-locale cleared rates we can
        # assert against.
        from usersim.taxonomy.capabilities import capability_definitions

        cap_ids = [d.id for d in capability_definitions()]
        locales = ["en_US", "fr_FR"]
        overrides = {
            (cap, "fr_FR"): {"score": 0.0, "state": "blocked"}
            for cap in cap_ids if cap != "simulation_reliability"
        }
        m1 = _write_manifest(
            tmp_path,
            _manifest(
                run_id="1", model_id="v/x/m1", locales=locales,
                capability_cells=_full_capability_cells(
                    locales=locales, score_overrides=overrides),
            ),
        )
        m2 = _write_manifest(
            tmp_path,
            _manifest(run_id="2", model_id="v/x/m2", locales=locales,
                      capability_cells=_full_capability_cells(locales=locales)),
        )
        return build_comparison_report([m1, m2], comparison_id="cmp_test")

    def test_per_locale_cleared_filters_to_active_locale(self, tmp_path):
        # m1 cleared everything in en_US (14/14) but nothing in
        # fr_FR (0/14). Per-locale charts must reflect that.
        from usersim.reporting.comparison_dashboard import _spec_cleared_for_locale

        report = self._build_two_locale_panel(tmp_path)
        en_us = _spec_cleared_for_locale(
            report, "en_US",
            colors={"m1": "#000", "m2": "#fff", "v/x/m1": "#000", "v/x/m2": "#fff"},
            is_overall=False,
        )
        rows = {r["display_label"]: r for r in en_us["data"]["values"]}
        assert rows["m1"]["pct_cleared"] == pytest.approx(1.0)
        assert rows["m1"]["n_ready"] == N_ELIGIBLE_CAPS
        assert rows["m1"]["n_eligible"] == N_ELIGIBLE_CAPS
        # Title carries the locale label.
        assert "en_US" in en_us["title"]["text"]

        fr_fr = _spec_cleared_for_locale(
            report, "fr_FR",
            colors={"m1": "#000", "m2": "#fff", "v/x/m1": "#000", "v/x/m2": "#fff"},
            is_overall=False,
        )
        rows = {r["display_label"]: r for r in fr_fr["data"]["values"]}
        assert rows["m1"]["pct_cleared"] == pytest.approx(0.0)
        assert rows["m1"]["n_ready"] == 0
        # m2 still passes everywhere.
        assert rows["m2"]["pct_cleared"] == pytest.approx(1.0)

    def test_per_locale_cleared_overall_matches_headline_metric(self, tmp_path):
        # The Overall tab's cleared chart MUST match the global
        # cleared_headline values -- they're the same numerator /
        # denominator (n_ready_eligible / n_eligible) just rendered
        # in two places.
        from usersim.reporting.comparison_dashboard import _spec_cleared_for_locale

        report = self._build_two_locale_panel(tmp_path)
        overall = _spec_cleared_for_locale(
            report, "",
            colors={"m1": "#000", "m2": "#fff", "v/x/m1": "#000", "v/x/m2": "#fff"},
            is_overall=True,
        )
        rows = {r["display_label"]: r for r in overall["data"]["values"]}
        m1_rollup = next(r for r in report.model_summary if r.run_id == "1")
        # Eligible total = quality caps × 2 locales; m1 is ready in en_US only
        # -> cleared = 50%.
        assert rows["m1"]["n_eligible"] == 2 * N_ELIGIBLE_CAPS
        assert rows["m1"]["n_ready"] == N_ELIGIBLE_CAPS
        assert rows["m1"]["pct_cleared"] == pytest.approx(m1_rollup.n_cells_ready_eligible / m1_rollup.n_cells_eligible)

    def test_per_capability_cleared_one_bar_per_model(self, tmp_path):
        # Per-capability cleared chart: one bar per model, showing
        # what fraction of locales cleared this capability for that
        # model.
        from usersim.reporting.comparison_dashboard import _spec_cleared_for_capability

        report = self._build_two_locale_panel(tmp_path)
        spec = _spec_cleared_for_capability(
            report, "assistant_quality", "Assistant quality",
            colors={"m1": "#000", "m2": "#fff", "v/x/m1": "#000", "v/x/m2": "#fff"},
        )
        rows = {r["display_label"]: r for r in spec["data"]["values"]}
        # m1 cleared assistant_quality only in en_US -> 1/2 locales = 50%.
        assert rows["m1"]["pct_cleared"] == pytest.approx(0.5)
        assert rows["m1"]["n_ready"] == 1
        assert rows["m1"]["n_eligible"] == 2
        # m2 cleared in both -> 100%.
        assert rows["m2"]["pct_cleared"] == pytest.approx(1.0)
        # Title names the capability.
        assert "Assistant quality" in spec["title"]["text"]

    def test_per_model_cleared_one_bar_per_locale(self, tmp_path):
        # Per-model cleared chart filters to a single model and
        # surfaces a bar per locale showing % capabilities cleared
        # in that locale.
        from usersim.reporting.comparison_dashboard import _spec_cleared_for_model_per_locale

        report = self._build_two_locale_panel(tmp_path)
        spec = _spec_cleared_for_model_per_locale(report, "1", "m1")
        # Two locale rows.
        labels = [r["display_label"] for r in spec["data"]["values"]]
        # Labels are flag-emoji formatted; locale code is the suffix.
        assert any("en_US" in lbl for lbl in labels)
        assert any("fr_FR" in lbl for lbl in labels)
        rows = {r["display_label"]: r for r in spec["data"]["values"]}
        en_row = next(v for k, v in rows.items() if "en_US" in k)
        fr_row = next(v for k, v in rows.items() if "fr_FR" in k)
        assert en_row["pct_cleared"] == pytest.approx(1.0)
        assert en_row["n_ready"] == N_ELIGIBLE_CAPS
        assert fr_row["pct_cleared"] == pytest.approx(0.0)
        assert fr_row["n_ready"] == 0
        assert "m1" in spec["title"]["text"]

    def test_per_tab_cleared_charts_match_main_chart_width(self, tmp_path):
        # Per-tab cleared charts ship at width 800 to sit flush with
        # the main grouped-bar chart in each tab section (which is
        # also 800). At the previous 480, the cleared chart looked
        # visually orphaned below the wider main chart.
        from usersim.reporting.comparison_dashboard import (
            _spec_cleared_for_capability,
            _spec_cleared_for_locale,
            _spec_cleared_for_model_per_locale,
        )

        report = self._build_two_locale_panel(tmp_path)
        colors = {"m1": "#000", "m2": "#fff", "v/x/m1": "#000", "v/x/m2": "#fff"}
        per_loc = _spec_cleared_for_locale(report, "en_US", colors, is_overall=False)
        per_cap = _spec_cleared_for_capability(report, "assistant_quality", "Assistant quality", colors)
        per_model = _spec_cleared_for_model_per_locale(report, "1", "m1")
        for spec, name in (
            (per_loc, "per_locale"),
            (per_cap, "per_capability"),
            (per_model, "per_model"),
        ):
            assert spec["width"] == 800, f"{name} cleared spec is {spec['width']}, expected 800"

    def test_dashboard_payload_carries_per_tab_cleared_specs(self, tmp_path):
        # The bundles emitted to index.html include the cleared
        # companion specs alongside the existing per-tab grouped
        # bars. React reads them via specsPayload.per_locale_cleared
        # and the cleared_spec field on per_capability / per_model
        # entries.
        import json, re

        report = self._build_two_locale_panel(tmp_path)
        out = tmp_path / "out"
        report.comparison_id = "cmp_t"
        write_comparison_dashboard_artifacts(report, out)
        html = (out / "out").exists() and (out / "index.html").read_text() or (out / "index.html").read_text()
        m = re.search(
            r'<script id="comparison-specs-cmp_t" type="application/json">(.*?)</script>',
            html, re.DOTALL,
        )
        specs = json.loads(m.group(1).replace("<\\/", "</"))
        # New per_locale_cleared bundle present, same key set as per_locale.
        assert set(specs["per_locale_cleared"].keys()) == set(specs["per_locale"].keys())
        # Each per-capability entry carries a cleared_spec.
        for entry in specs["per_capability"]:
            assert "cleared_spec" in entry, entry["capability_id"]
        # Each per-model entry carries a cleared_spec.
        for entry in specs["per_model"]:
            assert "cleared_spec" in entry, entry["run_id"]


class TestFailureRateChart:
    """Sim Health surfaces failure rate (= 1 - ok_rate), not OK rate."""

    def test_sim_status_failure_rate_is_complement_of_ok_rate(self, tmp_path):
        # Synthetic manifest's default sim_health has ok=n, no
        # warnings or failures -> ok_rate=1.0, failure_rate=0.0.
        m1 = _write_manifest(tmp_path, _manifest(run_id="1", model_id="v/x/a"))
        m2 = _write_manifest(tmp_path, _manifest(run_id="2", model_id="v/x/b"))
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        for r in report.model_summary:
            assert r.sim_status_ok_rate == pytest.approx(1.0)
            assert r.sim_status_failure_rate == pytest.approx(0.0)

    def test_failure_rate_with_mixed_status_counts(self, tmp_path):
        m = _manifest(run_id="1", model_id="v/x/a", n_trajectories=1000)
        m["sim_health"]["status_counts"] = {
            "ok": 700, "completed_with_warnings": 200, "failed": 100,
        }
        m1 = _write_manifest(tmp_path, m)
        m2 = _write_manifest(tmp_path, _manifest(run_id="2", model_id="v/x/b"))
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        m1_rollup = next(r for r in report.model_summary if r.run_id == "1")
        # ok+warnings = 900 / 1000 = 0.9
        assert m1_rollup.sim_status_ok_rate == pytest.approx(0.9)
        # failure = 1 - 0.9 = 0.1
        assert m1_rollup.sim_status_failure_rate == pytest.approx(0.1)


class TestFailureTaxonomyChart:
    """Sim Health panel includes a failure-mode stacked bar per model."""

    def _manifest_with_failures(
        self,
        run_id: str,
        model_id: str,
        failures: list[dict],
        n_trajectories: int = 1024,
    ) -> dict:
        m = _manifest(run_id=run_id, model_id=model_id, n_trajectories=n_trajectories)
        # Inject the failure taxonomy under sim_health.
        m["sim_health"]["failure_taxonomy"] = failures
        return m

    def test_rollup_carries_failure_taxonomy_from_manifest(self, tmp_path):
        m1 = _write_manifest(
            tmp_path,
            self._manifest_with_failures(
                run_id="1", model_id="v/x/a", n_trajectories=1000,
                failures=[
                    {
                        "failure_class": "infrastructure_error",
                        "failure_attribution": "assistant_model",
                        "n": 50, "proportion": 0.05,
                    },
                    {
                        "failure_class": "max_turns_reached",
                        "failure_attribution": "user_model",
                        "n": 10, "proportion": 0.01,
                    },
                ],
            ),
        )
        m2 = _write_manifest(tmp_path, _manifest(run_id="2", model_id="v/x/b"))
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        a_rollup = next(r for r in report.model_summary if r.run_id == "1")
        assert len(a_rollup.failure_taxonomy) == 2
        first = a_rollup.failure_taxonomy[0]
        assert first["failure_class"] == "infrastructure_error"
        assert first["proportion"] == pytest.approx(0.05)
        # Run with no failures -> empty list (not missing key).
        b_rollup = next(r for r in report.model_summary if r.run_id == "2")
        assert b_rollup.failure_taxonomy == []

    def test_spec_failure_taxonomy_renders_one_segment_per_class(self, tmp_path):
        from usersim.reporting.comparison_dashboard import _spec_failure_taxonomy

        m1 = _write_manifest(
            tmp_path,
            self._manifest_with_failures(
                run_id="1", model_id="v/x/a", n_trajectories=1000,
                failures=[
                    {"failure_class": "infrastructure_error",
                     "failure_attribution": "assistant_model",
                     "n": 50, "proportion": 0.05},
                    {"failure_class": "max_turns_reached",
                     "failure_attribution": "user_model",
                     "n": 10, "proportion": 0.01},
                ],
            ),
        )
        m2 = _write_manifest(
            tmp_path,
            self._manifest_with_failures(
                run_id="2", model_id="v/x/b", n_trajectories=1000,
                failures=[
                    {"failure_class": "infrastructure_error",
                     "failure_attribution": "assistant_model",
                     "n": 30, "proportion": 0.03},
                ],
            ),
        )
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        spec = _spec_failure_taxonomy(report)
        # 2 segments for run 1 + 1 segment for run 2 = 3 data rows.
        assert len(spec["data"]["values"]) == 3
        # Stack mode is "zero" so segments append along x; aggregate
        # must be ``sum`` so multiple rows per (model, class) (if any)
        # collapse cleanly.
        x_enc = spec["encoding"]["x"]
        assert x_enc["aggregate"] == "sum"
        assert x_enc["stack"] == "zero"
        assert x_enc["axis"]["format"] == ".0%"
        # Color scale lists every failure_class observed across the
        # panel (not just the first model's).
        color_domain = spec["encoding"]["color"]["scale"]["domain"]
        assert set(color_domain) == {"infrastructure_error", "max_turns_reached"}
        # Pinned classes resolve to their fixed hues.
        for cls, hex_color in zip(color_domain, spec["encoding"]["color"]["scale"]["range"]):
            if cls == "infrastructure_error":
                assert hex_color == "#d04848"

    def test_spec_failure_taxonomy_handles_panel_with_no_failures(self, tmp_path):
        from usersim.reporting.comparison_dashboard import _spec_failure_taxonomy

        m1 = _write_manifest(tmp_path, _manifest(run_id="1", model_id="v/x/a"))
        m2 = _write_manifest(tmp_path, _manifest(run_id="2", model_id="v/x/b"))
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        spec = _spec_failure_taxonomy(report)
        assert spec["data"]["values"] == []
        # Domain falls back to the 1% floor so the chart still
        # renders an axis even when the panel has no failures.
        assert spec["encoding"]["x"]["scale"]["domain"][1] >= 0.01

    def test_failure_taxonomy_spec_emitted_in_global_specs(self, tmp_path):
        # Wired into the dashboard payload so the React component
        # can find it under ``specsPayload.global.sim_health_failure_taxonomy``.
        import json, re

        m1 = _write_manifest(tmp_path, _manifest(run_id="1", model_id="v/x/a"))
        m2 = _write_manifest(tmp_path, _manifest(run_id="2", model_id="v/x/b"))
        report = build_comparison_report([m1, m2], comparison_id="cmp_t")
        out = tmp_path / "out"
        write_comparison_dashboard_artifacts(report, out)
        html = (out / "index.html").read_text()
        m = re.search(
            r'<script id="comparison-specs-cmp_t" type="application/json">(.*?)</script>',
            html, re.DOTALL,
        )
        specs = json.loads(m.group(1).replace("<\\/", "</"))
        assert "sim_health_failure_taxonomy" in specs["global"]
        # And the React mount references it from the Sim Health section.
        assert "specsPayload.global.sim_health_failure_taxonomy" in html


class TestClearedExplainers:
    """Each cleared chart sits behind an explainer that defines
    'cleared' and ties it to the dashed-red must-pass bars in the
    capability-comparison chart above. Lock the copy so a casual
    string change doesn't drift the user-facing definition.
    """

    def _build_panel(self, tmp_path) -> str:
        # Two locales × two models × every quality cap is enough to
        # populate every explainer site.
        locales = ["en_US", "fr_FR"]
        m1 = _write_manifest(
            tmp_path,
            _manifest(run_id="1", model_id="v/x/a", locales=locales,
                      capability_cells=_full_capability_cells(locales=locales)),
        )
        m2 = _write_manifest(
            tmp_path,
            _manifest(run_id="2", model_id="v/x/b", locales=locales,
                      capability_cells=_full_capability_cells(locales=locales)),
        )
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        out = tmp_path / "out"
        write_comparison_dashboard_artifacts(report, out)
        return (out / "index.html").read_text()

    def test_performance_section_defines_cleared(self, tmp_path):
        # The Performance section's explainer is the master
        # definition: it spells out the rule (score ≥ threshold AND
        # no must-pass failure) and points at the dashed-red bars
        # in the per-locale and per-capability charts below.
        html = self._build_panel(tmp_path)
        assert "Cleared capabilities (leftmost chart):" in html
        assert "no must-pass check fails" in html
        assert "dashed-red bars in the per-locale and per-capability charts below" in html

    def test_per_locale_tab_explainer(self, tmp_path):
        html = self._build_panel(tmp_path)
        assert (
            "The cleared chart below tallies how many capability "
            "cells each model cleared in this view "
            "(score \u2265 threshold and no dashed-red must-pass "
            "check failed)."
        ) in html

    def test_per_capability_tab_explainer(self, tmp_path):
        html = self._build_panel(tmp_path)
        assert (
            "The cleared chart below tallies how many locales each "
            "model cleared this capability in "
            "(score \u2265 threshold and no dashed-red must-pass "
            "check failed)."
        ) in html

    def test_per_model_tab_explainer(self, tmp_path):
        html = self._build_panel(tmp_path)
        assert (
            "The cleared chart below tallies how many capabilities "
            "this model cleared in each locale "
            "(score \u2265 threshold and no dashed-red must-pass "
            "check failed)."
        ) in html

    def test_must_pass_check_term_used_consistently(self, tmp_path):
        # All four explainers should agree on the antagonist concept's
        # name. Earlier drafts used "must-pass failure" and "must-pass
        # critical axis" interchangeably; the locked term is
        # "must-pass check". Catches drift if anyone tweaks one
        # explainer in isolation.
        html = self._build_panel(tmp_path)
        # Every explainer mentions "must-pass check" -- four sites.
        assert html.count("must-pass check") >= 4
        # No legacy synonyms in the rendered explainers (allowed
        # elsewhere in code comments / docstrings, but not in the
        # JS string literals that hit the user).
        # The rendered explainer strings are the only place these
        # would appear in a React h("p", "...") call.
        for legacy in ("must-pass failure failed", "must-pass critical axis fails"):
            assert legacy not in html, f"stale phrasing: {legacy!r}"


class TestSimHealthLayout:
    """Sim Health renders as a 2x2 grid (failure / breakdown over
    persona / early stop) with consistent gap spacing."""

    def test_sim_health_grid_class_used_instead_of_verdict_row(self, tmp_path):
        m1 = _write_manifest(tmp_path, _manifest(run_id="1", model_id="v/x/a"))
        m2 = _write_manifest(tmp_path, _manifest(run_id="2", model_id="v/x/b"))
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        out = tmp_path / "out"
        write_comparison_dashboard_artifacts(report, out)
        html = (out / "index.html").read_text()
        # Sim Health uses the dedicated grid class. The CSS rule
        # locks columns to 1fr 1fr (2x2 layout) instead of the
        # 3-column verdict-row used elsewhere.
        assert 'h("h2", null, "Simulator health")' in html
        # The Sim Health React block uses sim-health-grid (verify
        # the substring is in the bundled JS).
        assert '"sim-health-grid"' in html
        # And the CSS rule is injected.
        assert ".sim-health-grid { display: grid; grid-template-columns: 1fr 1fr;" in html

    def test_sim_health_charts_in_2x2_order(self, tmp_path):
        # Order in DOM: failure | failure_taxonomy / early | persona.
        # Failure + taxonomy stay paired (taxonomy is the failure
        # drilldown). Early stop leads the second row as the
        # structural completion metric; persona grounding follows
        # as the quality metric.
        m1 = _write_manifest(tmp_path, _manifest(run_id="1", model_id="v/x/a"))
        m2 = _write_manifest(tmp_path, _manifest(run_id="2", model_id="v/x/b"))
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        out = tmp_path / "out"
        write_comparison_dashboard_artifacts(report, out)
        html = (out / "index.html").read_text()
        idx_failure = html.index("specsPayload.global.sim_health_failure}")
        idx_taxonomy = html.index("specsPayload.global.sim_health_failure_taxonomy")
        idx_persona = html.index("specsPayload.global.sim_health_persona")
        idx_early = html.index("specsPayload.global.sim_health_early")
        assert idx_failure < idx_taxonomy < idx_early < idx_persona, (
            f"Sim Health order broken: failure={idx_failure}, taxonomy={idx_taxonomy}, "
            f"early={idx_early}, persona={idx_persona}"
        )

    def test_sim_health_charts_show_y_labels_on_all_four(self, tmp_path):
        # The 2x2 grid puts each chart in its own DOM cell, so unlike
        # a single-row layout there's no shared y-axis gutter to
        # inherit. Every rate chart must render its own model labels;
        # the failure-taxonomy chart already does (axis with
        # labelLimit/labelFontWeight).
        import json, re

        m1 = _write_manifest(tmp_path, _manifest(run_id="1", model_id="v/x/a"))
        m2 = _write_manifest(tmp_path, _manifest(run_id="2", model_id="v/x/b"))
        report = build_comparison_report([m1, m2], comparison_id="cmp_t")
        out = tmp_path / "out"
        write_comparison_dashboard_artifacts(report, out)
        html = (out / "index.html").read_text()
        m = re.search(
            r'<script id="comparison-specs-cmp_t" type="application/json">(.*?)</script>',
            html, re.DOTALL,
        )
        specs = json.loads(m.group(1).replace("<\\/", "</"))
        for key in ("sim_health_failure", "sim_health_persona", "sim_health_early"):
            y_axis = specs["global"][key]["encoding"]["y"]["axis"]
            assert y_axis is not None, f"{key} hides y-axis labels"
            assert "labelFontWeight" in y_axis, (
                f"{key} y-axis missing labelFontWeight: {y_axis}"
            )


class TestVerbositySectionLayout:
    """Verbosity section: enlarged Pareto + visual divider before heatmap."""

    def test_pareto_chart_is_enlarged(self, tmp_path):
        # Pareto used to ship at 520x360; the user asked for it to
        # match the heatmap's footprint within the section.
        from usersim.reporting.comparison_dashboard import _spec_pareto

        m1 = _write_manifest(tmp_path, _manifest(run_id="1", model_id="v/x/a"))
        m2 = _write_manifest(tmp_path, _manifest(run_id="2", model_id="v/x/b"))
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        spec = _spec_pareto(
            report,
            colors={"a": "#000", "b": "#fff"},
            shapes={"a": "circle", "b": "square"},
        )
        assert spec["width"] >= 720
        assert spec["height"] >= 480

    def test_verbosity_section_has_panel_divider(self, tmp_path):
        m1 = _write_manifest(tmp_path, _manifest(run_id="1", model_id="v/x/a"))
        m2 = _write_manifest(tmp_path, _manifest(run_id="2", model_id="v/x/b"))
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        out = tmp_path / "out"
        write_comparison_dashboard_artifacts(report, out)
        html = (out / "index.html").read_text()
        # CSS rule defining the divider shipped.
        assert ".panel-divider {" in html
        # And the React tree includes the divider element between
        # the Pareto body and the heatmap.
        assert '"panel-divider"' in html


class TestSectionHeadings:
    """Section h2 headings drop the 'Layer X' prefix; the report
    stands on its own and the panel layout reflects the verbosity
    merge + Performance / Per-X renames."""

    def _build_with_two_locales(self, tmp_path) -> str:
        # Build a small panel and return the rendered HTML (caller
        # asserts on substrings).
        locales = ["en_US", "fr_FR"]
        m1 = _write_manifest(tmp_path, _manifest(run_id="1", model_id="v/x/a", locales=locales,
                                                  capability_cells=_full_capability_cells(locales=locales)))
        m2 = _write_manifest(tmp_path, _manifest(run_id="2", model_id="v/x/b", locales=locales,
                                                  capability_cells=_full_capability_cells(locales=locales)))
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        out = tmp_path / "out"
        write_comparison_dashboard_artifacts(report, out)
        return (out / "index.html").read_text()

    def test_no_layer_prefix_in_section_headers(self, tmp_path):
        html = self._build_with_two_locales(tmp_path)
        # Old prefixes that must be gone:
        for needle in (
            'h("h2", null, "Layer 0',
            'h("h2", null, "Layer 1',
            'h("h2", null, "Layer 2',
            'h("h2", null, "Layer 3',
            'h("h2", null, "Layer 4',
            'h("h2", null, "Layer 5',
            'h("h2", null, "Layer 6',
            'h("h2", null, "Layer 7',
        ):
            assert needle not in html, f"layer-prefixed header still in source: {needle!r}"

    def test_new_section_titles_appear_in_react_source(self, tmp_path):
        html = self._build_with_two_locales(tmp_path)
        # React-source mounts (string-equality match against the
        # bundled JS, which is the truth source for the React tree).
        assert 'h("h2", null, "Simulator health")' in html
        assert 'h("h2", null, "Performance")' in html
        assert 'h("h2", null, "Verbosity")' in html
        assert 'h("h2", null, "Per-locale capability comparison")' in html
        assert 'h("h2", null, "Performance by capability across locales")' in html
        assert 'h("h2", null, "Performance by model across locales")' in html
        # Coverage matrix is collapsible so its h2 sits inside a <summary>
        # element; the substring still appears in the source.
        assert 'h("h2", null, "Coverage matrix")' in html

    def test_verbosity_section_combines_pareto_and_heatmap(self, tmp_path):
        # The merged Verbosity section is one <section> containing
        # both the Pareto tabs body AND the heatmap chart. The
        # source carries both mounts inside the VerbositySection
        # function; the stale standalone "Verbosity profile" h2 is
        # gone (its data lives in the same section now).
        html = self._build_with_two_locales(tmp_path)
        assert "function VerbositySection()" in html
        # Both charts are referenced from the same component.
        assert "specsPayload.global.pareto_per_locale" in html
        assert "specsPayload.global.verbosity_heatmap" in html
        # And the legacy standalone heatmap section header is gone.
        assert "Verbosity profile" not in html


class TestChartTitleHygiene:
    """Chart titles are dict-shaped, left-anchored, and Layer-prefix-free."""

    def _all_titles(self, specs: dict) -> list[dict]:
        # Walk every chart spec the dashboard ships and collect their
        # title configs. Lets the assertions below run as a single
        # blanket check across the whole surface area.
        titles: list[dict] = []
        for name, spec in specs["global"].items():
            if name == "pareto_per_locale":
                titles.extend(t for t in (s.get("title") for s in spec.values()) if t)
                continue
            t = spec.get("title")
            if t is not None:
                titles.append(t)
        for spec in specs.get("per_locale", {}).values():
            if "title" in spec:
                titles.append(spec["title"])
            # Layer 4 specs are vconcat'd; recurse one level deep.
            for sub in spec.get("vconcat", []):
                if "title" in sub:
                    titles.append(sub["title"])
        for entry in specs.get("per_capability", []):
            t = entry.get("spec", {}).get("title")
            if t is not None:
                titles.append(t)
        for entry in specs.get("per_model", []):
            t = entry.get("spec", {}).get("title")
            if t is not None:
                titles.append(t)
            for sub in entry.get("spec", {}).get("vconcat", []):
                if "title" in sub:
                    titles.append(sub["title"])
        return titles

    def _load_specs(self, tmp_path) -> dict:
        import json, re

        # Build a tiny multi-locale, multi-capability comparison so
        # the per-locale, per-capability, and per-model spec banks
        # all populate. Then yank the JSON specs blob.
        locales = ["en_US", "fr_FR"]
        m1 = _write_manifest(
            tmp_path,
            _manifest(run_id="1", model_id="v/x/a", locales=locales,
                      capability_cells=_full_capability_cells(locales=locales)),
        )
        m2 = _write_manifest(
            tmp_path,
            _manifest(run_id="2", model_id="v/x/b", locales=locales,
                      capability_cells=_full_capability_cells(locales=locales)),
        )
        report = build_comparison_report([m1, m2], comparison_id="cmp_t")
        out = tmp_path / "out"
        write_comparison_dashboard_artifacts(report, out)
        html = (out / "index.html").read_text()
        m = re.search(
            r'<script id="comparison-specs-cmp_t" type="application/json">(.*?)</script>',
            html, re.DOTALL,
        )
        return json.loads(m.group(1).replace("<\\/", "</"))

    def test_chart_titles_drop_layer_prefix(self, tmp_path):
        # Layer numbers belong to the section <h2> headers, so the chart
        # body title carries only the descriptor.
        specs = self._load_specs(tmp_path)
        for title in self._all_titles(specs):
            text = title["text"] if isinstance(title, dict) else title
            assert not text.startswith("Layer "), (
                f"chart title still carries layer prefix: {text!r}"
            )

    def test_chart_titles_render_above_plot(self, tmp_path):
        # All chart titles ship with ``anchor: middle`` (the
        # Vega-Lite default for top-oriented titles). Earlier versions
        # used ``anchor: start`` to "left-align" titles, but Vega
        # anchored them to the LEFT EDGE of the plot frame -- which,
        # on charts with long y-axis labels, sat to the right of the
        # panel's left edge and visually read like a row label rather
        # than a chart title. ``anchor: middle`` gives the standard
        # "title centred above the chart" look.
        specs = self._load_specs(tmp_path)
        for title in self._all_titles(specs):
            assert isinstance(title, dict), (
                f"expected dict-shaped title, got {type(title).__name__}: {title!r}"
            )
            assert title.get("anchor") == "middle", (
                f"chart title not centred above plot: {title!r}"
            )


# ===========================================================================
# Dashboard artifacts
# ===========================================================================


class TestDashboardArtifacts:
    def _build_report(self, tmp_path: Path, n_runs: int = 5, n_locales: int = 8) -> ComparisonReport:
        locales = ["en_US", "en_IN", "en_SG", "fr_FR", "ja_JP", "ko_KR", "pt_BR", "hi_Deva_IN"]
        locales = locales[:n_locales]
        paths = []
        for i in range(n_runs):
            cells = _full_capability_cells(locales=locales, base_score=0.7 + 0.04 * i)
            m = _write_manifest(
                tmp_path,
                _manifest(
                    run_id=f"100000000{i}",
                    model_id=f"vendor/family/model-{chr(ord('a') + i)}",
                    locales=locales,
                    capability_cells=cells,
                    assistant_total_tokens=1_000_000 + i * 200_000,
                ),
            )
            paths.append(m)
        return build_comparison_report(paths, comparison_id="cmp_test")

    def test_writes_three_files(self, tmp_path):
        report = self._build_report(tmp_path)
        out = tmp_path / "comparison_out"
        index = write_comparison_dashboard_artifacts(report, out)
        assert index.exists() and index.stat().st_size > 0
        for name in ("comparison_manifest.json", "comparison_matrix.json", "index.html"):
            p = out / name
            assert p.exists(), f"{name} missing"
            assert p.stat().st_size > 0, f"{name} empty"

    def test_index_html_below_size_budget(self, tmp_path):
        # The payload scales with (runs × locales × capabilities), so the budget
        # is expressed PER ELIGIBLE CAPABILITY rather than as a flat byte count
        # — otherwise every new probe family's capability rows re-break this
        # test. History: 1.5 MB -> 2 MB when the Layer 2 Pareto chart started
        # shipping an "Overall + per-locale" spec bundle; the per-capability
        # form keeps that headroom while absorbing registry growth. Vega is
        # still loaded from esm.sh ESM imports (no JS libraries embedded).
        budget = 140_000 * N_ELIGIBLE_CAPS
        report = self._build_report(tmp_path, n_runs=5, n_locales=8)
        out = tmp_path / "comparison_out"
        write_comparison_dashboard_artifacts(report, out)
        size = (out / "index.html").stat().st_size
        assert size < budget, (
            f"index.html is {size:,} bytes (budget {budget:,} = "
            f"140 KB × {N_ELIGIBLE_CAPS} eligible capabilities)"
        )

    def test_uses_esm_imports_for_vega(self, tmp_path):
        # Comparison dashboard mirrors the per-run dashboard pattern:
        # ESM imports from esm.sh inside <script type="module">. No UMD
        # <script src=> tags, no inlined bundle.
        report = self._build_report(tmp_path, n_runs=2, n_locales=2)
        out = tmp_path / "comparison_out"
        write_comparison_dashboard_artifacts(report, out)
        html = (out / "index.html").read_text()
        assert "import vegaEmbed from " in html
        assert "import React" in html
        # Exact pins, not floating majors: a generated report is long-lived, so
        # a hijacked or merely broken upstream release must not change what an
        # already-published report executes. Asserted by shape rather than by
        # version so a deliberate bump does not fail here spuriously.
        import re as _re

        floating = _re.findall(r'https://esm\.sh/[a-z0-9-]+@\d+(?:/[a-z]+)?"', html)
        assert not floating, f"floating esm.sh pins: {floating}"
        assert _re.search(r'https://esm\.sh/vega-embed@\d+\.\d+\.\d+', html)
        # No UMD CDN tag, no inline bundle marker.
        assert "cdn.jsdelivr.net/npm/vega" not in html
        assert "// === vega@" not in html

    def test_layer4_includes_overall_tab(self, tmp_path):
        # Layer 4's per_locale spec dict must include "Overall" as the
        # first key so the React tab strip defaults to it. The Overall
        # spec is the locale-averaged grouped-bar chart.
        import json, re

        report = self._build_report(tmp_path, n_runs=3, n_locales=3)
        out = tmp_path / "out"
        write_comparison_dashboard_artifacts(report, out)
        html = (out / "index.html").read_text()
        cmp_uid = report.comparison_id.replace("/", "_")
        m = re.search(
            rf'<script id="comparison-specs-{cmp_uid}" type="application/json">(.*?)</script>',
            html, re.DOTALL,
        )
        specs = json.loads(m.group(1).replace("<\\/", "</"))
        per_locale = specs["per_locale"]
        # "Overall" must be the first key (drives the default tab in React).
        assert next(iter(per_locale)) == "Overall"
        assert "Overall" in per_locale
        # Overall spec is itself a vconcat (two-row layout matches per-locale).
        assert "vconcat" in per_locale["Overall"]
        # Overall tooltip carries the n_locales_measured field.
        rendered_overall = json.dumps(per_locale["Overall"])
        assert "n_locales_measured" in rendered_overall

    def test_dom_ids_are_namespaced_per_comparison(self, tmp_path):
        # Two consecutive comparison renders must produce different DOM IDs
        # so they can coexist in the same Jupyter notebook without
        # createRoot() / getElementById() collisions. Without namespacing,
        # the per-run dashboard's <div id="root"> and the comparison
        # dashboard's <div id="root"> share the same first-match DOM id
        # and React mounts the wrong app into the wrong container.
        report_a = self._build_report(tmp_path, n_runs=2, n_locales=2)
        report_a.comparison_id = "cmp_aaaa"
        report_b = self._build_report(tmp_path, n_runs=2, n_locales=2)
        report_b.comparison_id = "cmp_bbbb"
        out_a = tmp_path / "out_a"
        out_b = tmp_path / "out_b"
        write_comparison_dashboard_artifacts(report_a, out_a)
        write_comparison_dashboard_artifacts(report_b, out_b)
        html_a = (out_a / "index.html").read_text()
        html_b = (out_b / "index.html").read_text()
        # Plain id="root" must NOT appear (would collide with the per-run dashboard).
        assert 'id="root"' not in html_a
        assert 'id="root"' not in html_b
        # Each render must use its own comparison_id-namespaced ID.
        assert 'id="root-cmp_aaaa"' in html_a
        assert 'id="root-cmp_bbbb"' in html_b
        # createRoot() and the data scripts must reference the same namespaced ID.
        assert 'getElementById("root-cmp_aaaa")' in html_a
        assert 'getElementById("comparison-data-cmp_aaaa")' in html_a

    def test_static_fallback_contains_models_table(self, tmp_path):
        report = self._build_report(tmp_path, n_runs=3, n_locales=2)
        out = tmp_path / "comparison_out"
        write_comparison_dashboard_artifacts(report, out)
        html = (out / "index.html").read_text()
        assert "Models compared" in html
        assert 'class="comparison-table"' in html

    def test_static_fallback_contains_layer1_summary(self, tmp_path):
        report = self._build_report(tmp_path, n_runs=3, n_locales=2)
        out = tmp_path / "comparison_out"
        write_comparison_dashboard_artifacts(report, out)
        html = (out / "index.html").read_text()
        # The fallback "Verdict + Verbosity" textual table.
        assert "Verdict + Verbosity" in html or "Layer 1" in html
        assert "Mean score" in html
        # Output / conv replaced the old "Tokens / conv" label after the
        # verbosity metric narrowed from input+output to output-only.
        assert "Output / conv" in html
        assert "Reasoning / conv" in html

    def test_embeds_vega_lite_specs(self, tmp_path):
        report = self._build_report(tmp_path, n_runs=3, n_locales=2)
        out = tmp_path / "comparison_out"
        write_comparison_dashboard_artifacts(report, out)
        html = (out / "index.html").read_text()
        assert "vegaEmbed(" in html
        # Schema reference is escaped in JSON so look for a tolerant marker.
        assert "vega-lite" in html and "v5.json" in html

    def test_tooltip_includes_per_run_deeplink(self, tmp_path):
        report = self._build_report(tmp_path, n_runs=3, n_locales=2)
        out = tmp_path / "comparison_out"
        write_comparison_dashboard_artifacts(report, out)
        html = (out / "index.html").read_text()
        # Every per-cell datum carries `deeplink: ../../report/run=<id>/index.html`.
        # The relative path appears in the embedded spec data and in the
        # static-fallback Drill-down column.
        assert "../../report/run=" in html

    def test_warnings_no_longer_rendered_on_dashboard_surface(self, tmp_path):
        # Drift warnings (sample_mode, n_trajectories delta, etc.) used
        # to render as a yellow callout AND -- on the static fallback
        # -- as a list of <li> items. They've been pulled from the
        # rendered surface entirely; the data lives on the report for
        # CLI / log surfaces, but the index.html no longer mentions
        # the offending field name even when the warning fires.
        m1 = _write_manifest(tmp_path, _manifest(run_id="1", model_id="v/x/a", sample_mode="full"))
        m2 = _write_manifest(tmp_path, _manifest(run_id="2", model_id="v/x/b", sample_mode="per_locale"))
        report = build_comparison_report([m1, m2], comparison_id="cmp_test")
        out = tmp_path / "comparison_out"
        write_comparison_dashboard_artifacts(report, out)
        html = (out / "index.html").read_text()
        # The callout aside element is gone in both static and React.
        assert '<aside class="comparison-warnings"' not in html
        assert "Apples-to-apples warnings:" not in html
        # Spot-check: the data is still on the report itself.
        assert any("sample_mode" in w for w in report.apples_to_apples_warnings)

    def test_to_dict_is_json_serializable(self, tmp_path):
        report = self._build_report(tmp_path, n_runs=2, n_locales=2)
        d = report.to_dict()
        # Round-trip without losing structure.
        restored = json.loads(json.dumps(d, default=str))
        assert restored["comparison_id"] == "cmp_test"
        assert len(restored["entries"]) == 2
        assert len(restored["model_summary"]) == 2


# ===========================================================================
# Notebook copy guardrail
# ===========================================================================


class TestSelectRunsForComparison:
    def test_returns_all_when_no_pin(self, tmp_path):
        for run_id in ["1", "2", "3"]:
            _write_manifest(tmp_path, _manifest(run_id=run_id))
        paths = select_runs_for_comparison(tmp_path, compare_run_ids=None)
        assert len(paths) == 3
        # discover_comparison_runs orders descending; the selector preserves
        # that order when no pin is set.
        ids = [p.parent.name.removeprefix("run=") for p in paths]
        assert ids == ["3", "2", "1"]

    def test_filters_to_pinned_subset(self, tmp_path):
        for run_id in ["1", "2", "3"]:
            _write_manifest(tmp_path, _manifest(run_id=run_id))
        paths = select_runs_for_comparison(tmp_path, compare_run_ids=["3", "1"])
        ids = [p.parent.name.removeprefix("run=") for p in paths]
        assert ids == ["3", "1"]  # preserved pin order, not discovery order

    def test_raises_when_pinned_id_missing(self, tmp_path):
        _write_manifest(tmp_path, _manifest(run_id="1"))
        with pytest.raises(FileNotFoundError, match="no report on disk"):
            select_runs_for_comparison(tmp_path, compare_run_ids=["1", "ghost"])

    def test_returns_empty_for_missing_root(self, tmp_path):
        # Discovery returns []; selector does not raise unless ids are pinned.
        assert select_runs_for_comparison(tmp_path / "ghost") == []


class TestPrintComparisonRuns:
    def _capture(self, fn) -> str:
        from io import StringIO

        from rich.console import Console

        buf = StringIO()
        console = Console(file=buf, width=200, force_terminal=False)
        fn(console)
        return buf.getvalue()

    def test_renders_table_with_per_manifest_metadata(self, tmp_path):
        m1 = _write_manifest(
            tmp_path, _manifest(run_id="1", model_id="vendor/x/alpha", n_trajectories=512, sample_mode="full")
        )
        m2 = _write_manifest(
            tmp_path, _manifest(run_id="2", model_id="vendor/x/beta", n_trajectories=128, sample_mode="per_locale")
        )
        out = self._capture(lambda c: print_comparison_runs([m1, m2], console=c))
        assert "Selected for comparison" in out
        assert "alpha" in out and "beta" in out
        assert "512" in out and "128" in out
        assert "full" in out and "per_locale" in out

    def test_empty_input_renders_friendly_hint(self):
        out = self._capture(lambda c: print_comparison_runs([], console=c))
        assert "No reports selected" in out

    def test_single_run_warns_below_table(self, tmp_path):
        m1 = _write_manifest(tmp_path, _manifest(run_id="1"))
        out = self._capture(lambda c: print_comparison_runs([m1], console=c))
        assert "Selected for comparison" in out
        assert "Comparison requires" in out

    def test_handles_corrupt_manifest_gracefully(self, tmp_path):
        bad = tmp_path / "run=1" / "report_manifest.json"
        bad.parent.mkdir(parents=True)
        bad.write_text("{not valid json")
        out = self._capture(lambda c: print_comparison_runs([bad], console=c))
        assert "load error" in out  # row rendered with error indicator


class TestNotebookComparisonSection:
    """Make sure the eval notebook carries the multi-model comparison section.

    Pins it against silent regressions -- the section getting dropped
    during a cell move or refactor.
    """

    def test_eval_notebook_has_comparison_section(self):
        nb_path = (
            Path(__file__).resolve().parents[1]
            / "notebooks" / "02_evaluate_simulation.ipynb"
        )
        assert nb_path.is_file(), f"evaluation notebook is missing: {nb_path}"
        nb = json.loads(nb_path.read_text())
        sources = ["".join(c.get("source", [])) for c in nb["cells"]]
        joined = "\n".join(sources)
        assert "Multi-Model Comparison" in joined
        assert "build_comparison_report" in joined
        # Discovery cell uses the new helpers (not the raw discover_comparison_runs call).
        assert "select_runs_for_comparison" in joined
        assert "print_comparison_runs" in joined
        assert "write_comparison_dashboard_artifacts" in joined
