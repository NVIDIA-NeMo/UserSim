# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Tests for evaluator/axes.py — axis registry by probe family."""

from __future__ import annotations

import pytest

from usersim.engine.evaluator.axes import (
    GUARDRAIL_AXES,
    PROBE_SCORES,
    QUALITY_AXES,
    UNIVERSAL_SCORES,
    AxisCategory,
    all_axis_names,
    axes_for_probe,
    axis_category,
    select_axes,
)


class TestUniversalScores:
    def test_universal_scores_have_required_fields(self) -> None:
        assert len(UNIVERSAL_SCORES) >= 9  # 9 axes shipped today
        for s in UNIVERSAL_SCORES:
            assert s.name and isinstance(s.name, str)
            assert s.description and isinstance(s.description, str)
            assert s.options and len(s.options) >= 2  # at least 2 score levels

    def test_score_options_use_full_5_point_scale(self) -> None:
        # All universal axes use 1-5; downstream reporting bootstrap CIs
        # assume this scale.
        for s in UNIVERSAL_SCORES:
            assert set(s.options.keys()) == {1, 2, 3, 4, 5}, f"Axis {s.name} does not use the full 1-5 scale"


class TestProbeScores:
    def test_known_probes_have_extra_axes(self) -> None:
        assert "tool_calling" in PROBE_SCORES
        assert "general_educational" in PROBE_SCORES
        assert len(PROBE_SCORES["tool_calling"]) >= 2  # leaking + correctness
        assert len(PROBE_SCORES["general_educational"]) >= 1  # pedagogical

    def test_unknown_probe_returns_universal_only(self) -> None:
        result = axes_for_probe("nonexistent_probe")
        assert {s.name for s in result} == {s.name for s in UNIVERSAL_SCORES}


class TestAllAxisNames:
    def test_includes_universal_and_probe_specific(self) -> None:
        names = all_axis_names()
        for s in UNIVERSAL_SCORES:
            assert s.name in names
        for scores in PROBE_SCORES.values():
            for s in scores:
                assert s.name in names

    def test_no_duplicates(self) -> None:
        names = all_axis_names()
        assert len(names) == len(set(names))


class TestAxesForProbe:
    def test_open_ended_returns_universal(self) -> None:
        result = axes_for_probe("general_open_ended")
        assert {s.name for s in result} == {s.name for s in UNIVERSAL_SCORES}

    def test_tool_calling_returns_universal_plus_extras(self) -> None:
        result = axes_for_probe("tool_calling")
        names = {s.name for s in result}
        assert "internal_architecture_leaking" in names
        assert "tool_use_correctness" in names
        # Universal axes still present.
        for s in UNIVERSAL_SCORES:
            assert s.name in names

    def test_educational_includes_pedagogical_quality(self) -> None:
        names = {s.name for s in axes_for_probe("general_educational")}
        assert "pedagogical_quality" in names


class TestAxisCategory:
    @pytest.mark.parametrize("name", sorted(GUARDRAIL_AXES))
    def test_guardrail_axes(self, name: str) -> None:
        assert axis_category(name) == AxisCategory.GUARDRAIL

    @pytest.mark.parametrize("name", sorted(QUALITY_AXES))
    def test_quality_axes(self, name: str) -> None:
        assert axis_category(name) == AxisCategory.QUALITY

    def test_role_adherence_falls_under_outcome(self) -> None:
        # role_adherence is intentionally OUTSIDE the quality/guardrail
        # partition (it scores both participants, not pure assistant).
        assert axis_category("role_adherence") == AxisCategory.PROBE_OUTCOME

    def test_unknown_falls_back_to_outcome(self) -> None:
        assert axis_category("Turn_of_Flip") == AxisCategory.PROBE_OUTCOME


class TestSelectAxes:
    def test_none_returns_full(self) -> None:
        full = axes_for_probe("tool_calling")
        assert select_axes("tool_calling", None) == full

    def test_filter_by_name(self) -> None:
        result = select_axes("tool_calling", ["helpfulness", "tool_use_correctness"])
        names = [s.name for s in result]
        assert names == ["helpfulness", "tool_use_correctness"]

    def test_filter_drops_unknown_silently(self) -> None:
        # select_axes is defensive: unknown names are silently dropped
        # (they may be valid for a different probe family).
        result = select_axes("general_open_ended", ["helpfulness", "tool_use_correctness"])
        names = [s.name for s in result]
        assert names == ["helpfulness"]

    def test_empty_filter_returns_empty(self) -> None:
        assert select_axes("general_open_ended", []) == []
