# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for reporting/sim_health.py — SimHealthBundle builder."""

from __future__ import annotations

import json

import pytest

from usersim.reporting import build_sim_health


def _with_empty_cells(trajectory_df, *, arrow_backed: bool):
    """The fixture with no interaction style on one row and no attribution on its one failure.

    ``arrow_backed`` converts it the way the run store reads it back, so empty
    cells arrive as ``pd.NA`` rather than None or NaN.
    """
    pd = pytest.importorskip("pandas")
    pa = pytest.importorskip("pyarrow")

    df = trajectory_df.copy()
    df.at[2, "user_interaction_style"] = None
    outcome = json.loads(df.at[1, "simulation_outcome"])
    outcome["failure_attribution"] = None
    df.at[1, "simulation_outcome"] = json.dumps(outcome)
    if arrow_backed:
        df = pa.Table.from_pandas(df, preserve_index=False).to_pandas(types_mapper=pd.ArrowDtype)
    return df


class TestBuildSimHealth:
    def test_basic_shape(self, trajectory_df) -> None:
        b = build_sim_health(trajectory_df)
        assert b.n_trajectories == 6

    def test_status_counts(self, trajectory_df) -> None:
        b = build_sim_health(trajectory_df)
        assert b.status_counts.get("ok", 0) == 4
        assert b.status_counts.get("failed", 0) == 1
        assert b.status_counts.get("completed_with_warnings", 0) == 1

    def test_failure_taxonomy_includes_user_model_failure(self, trajectory_df) -> None:
        b = build_sim_health(trajectory_df)
        rows = [r for r in b.failure_taxonomy if r["failure_class"]]
        assert any(
            r["failure_class"] == "user_query_gate_exhausted"
            and r["failure_attribution"] == "user_model"
            and r["n"] == 1
            for r in rows
        )

    def test_failure_taxonomy_excludes_success_rows(self, trajectory_df) -> None:
        b = build_sim_health(trajectory_df)
        assert all(r["failure_class"] for r in b.failure_taxonomy)
        assert all(r["failure_class"] for r in b.failure_taxonomy_by_locale)

    def test_diagnostics_means_present(self, trajectory_df) -> None:
        b = build_sim_health(trajectory_df)
        # All five diagnostic keys exist (some may be None for trajectories
        # with too few user turns; the means here aggregate non-None).
        for k in [
            "d1_front_loading",
            "d2_polite_fraction",
            "d3_verbosity_cv",
            "d4_frustration_markers",
            "mattr_assistant",
        ]:
            assert k in b.diagnostics_means

    def test_diagnostics_by_interaction_style(self, trajectory_df) -> None:
        b = build_sim_health(trajectory_df)
        styles = {r["user_interaction_style"] for r in b.diagnostics_by_interaction_style}
        # Fixture has cooperative, neutral, confrontational, impatient.
        assert "cooperative" in styles
        assert "confrontational" in styles

    def test_persona_grounding_rate(self, trajectory_df) -> None:
        b = build_sim_health(trajectory_df)
        # 4 of 6 persona_grounding=True.
        assert b.persona_grounding_rate == pytest.approx(4 / 6, abs=0.01)

    def test_early_stop_rate(self, trajectory_df) -> None:
        b = build_sim_health(trajectory_df)
        # 2 of 6 early_stop=True (rows 0 and 3).
        assert b.early_stop_rate == pytest.approx(2 / 6, abs=0.01)

    def test_counters(self, trajectory_df) -> None:
        b = build_sim_health(trajectory_df)
        # n_user_query_attempts: 1+3+1+1+1+1 = 8
        assert b.counters_totals["n_user_query_attempts"] == 8
        # n_api_response_rerolls: 0+0+0+0+0+1 = 1
        assert b.counters_totals["n_api_response_rerolls"] == 1
        # n_user_role_violations: 0+0+1+0+0+0 = 1
        assert b.counters_totals["n_user_role_violations"] == 1

    def test_resource_profile_aggregated(self, trajectory_df) -> None:
        b = build_sim_health(trajectory_df)
        assert b.resource_profile.n_trajectories == 6
        # Every row puts at least user_model + assistant_model in
        # per_model_calls.
        assert "user_model" in b.resource_profile.n_calls_by_alias
        assert "assistant_model" in b.resource_profile.n_calls_by_alias

    def test_provenance_observed(self, trajectory_df) -> None:
        b = build_sim_health(trajectory_df)
        # Two distinct code_sha values across the bundle (drift).
        assert "abc123" in b.provenance_observed.get("code_sha", [])
        assert "def456" in b.provenance_observed.get("code_sha", [])
        # One nemotron version.
        assert b.provenance_observed.get("nemotron_personas_version") == ["2026-04"]

    def test_probe_family_and_locale_counts(self, trajectory_df) -> None:
        b = build_sim_health(trajectory_df)
        assert b.probe_family_counts == {"general_open_ended": 3, "tool_calling": 3}
        # Fixture rows: 0,2,3,4=en_US; 1=pt_BR; 5=ja_JP.
        assert b.locale_counts.get("en_US") == 4
        assert b.locale_counts.get("pt_BR") == 1
        assert b.locale_counts.get("ja_JP") == 1

    def test_empty_frame(self) -> None:
        pd = pytest.importorskip("pandas")
        b = build_sim_health(pd.DataFrame())
        assert b.n_trajectories == 0
        assert b.status_counts == {}
        assert b.resource_profile.n_trajectories == 0

    def test_to_dict_serializable(self, trajectory_df) -> None:
        b = build_sim_health(trajectory_df)
        # Round-trip through JSON to confirm the bundle is serializable.
        json.dumps(b.to_dict())

    @pytest.mark.parametrize("arrow_backed", [False, True], ids=["in_memory", "store_read"])
    def test_empty_style_and_attribution_serialize_as_null(self, trajectory_df, arrow_backed) -> None:
        """The dashboard parses the bundle with JSON.parse, which rejects NaN."""
        b = build_sim_health(_with_empty_cells(trajectory_df, arrow_backed=arrow_backed))

        json.dumps(b.to_dict(), allow_nan=False)
        assert None in {r["user_interaction_style"] for r in b.diagnostics_by_interaction_style}
        assert [r["failure_attribution"] for r in b.failure_taxonomy] == [None]
        assert [r["failure_attribution"] for r in b.failure_taxonomy_by_locale] == [None]

    def test_all_empty_metric_group_is_null(self) -> None:
        """A style group whose Arrow-backed metric has no values reports it as None."""
        pd = pytest.importorskip("pandas")
        pa = pytest.importorskip("pyarrow")

        df = pd.DataFrame(
            {
                "user_interaction_style": ["cooperative", "terse"],
                "d1_front_loading": pd.Series([0.5, None], dtype=pd.ArrowDtype(pa.float64())),
            }
        )

        b = build_sim_health(df)

        assert b.diagnostics_by_interaction_style == [
            {"user_interaction_style": "cooperative", "d1_front_loading": 0.5},
            {"user_interaction_style": "terse", "d1_front_loading": None},
        ]

    def test_response_length_by_locale_populated(self, trajectory_df) -> None:
        """Per-locale response-length aggregation lands on the bundle.

        The fixture has en_US (4 rows), pt_BR (1 row), and ja_JP (1 row);
        each row's simulation_outcome carries per-model token data and
        each row's conversation_messages carries assistant turns, so all
        three locales should appear with non-None length stats.
        """
        b = build_sim_health(trajectory_df)
        locales = {row["locale"] for row in b.response_length_by_locale}
        assert locales == {"en_US", "pt_BR", "ja_JP"}
        # Every locale row should carry the chars side at minimum.
        for row in b.response_length_by_locale:
            assert row["n_trajectories"] >= 1
            assert row["mean_chars_per_turn"] is not None or row["n_trajectories"] == 0
