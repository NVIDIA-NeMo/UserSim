# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Tests for evaluator/config.py — TrajectoryEvaluatorConfig + judge spec."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from usersim.engine.evaluator.config import (
    JudgeSpecConfig,
    TrajectoryEvaluatorConfig,
)
from usersim.engine.evaluator.judges import JudgeFamily


class TestJudgeSpecConfig:
    def test_minimal(self) -> None:
        c = JudgeSpecConfig(alias="user_model")
        spec = c.to_spec()
        assert spec.alias == "user_model"
        assert spec.family is None

    def test_with_family(self) -> None:
        c = JudgeSpecConfig(alias="judge_a", family="openai")
        spec = c.to_spec()
        assert spec.family == JudgeFamily.OPENAI

    def test_unknown_family_rejected(self) -> None:
        c = JudgeSpecConfig(alias="judge_a", family="not-a-real-family")
        with pytest.raises(ValueError, match="Unknown judge family"):
            c.to_spec()


class TestTrajectoryEvaluatorConfig:
    def _ok_judges(self) -> list[JudgeSpecConfig]:
        return [
            JudgeSpecConfig(alias="judge_a", family="openai"),
            JudgeSpecConfig(alias="judge_b", family="nvidia_nemotron"),
        ]

    def test_minimal_valid(self) -> None:
        cfg = TrajectoryEvaluatorConfig(name="eval_scores_v2", judges=self._ok_judges())
        assert cfg.column_type == "trajectory-evaluator"
        assert cfg.skip_if_existing is True
        assert cfg.prompt_version == "v1.0"
        assert cfg.model_alias == "judge_a"

    def test_model_alias_tracks_first_judge_for_dd_health_check(self) -> None:
        cfg = TrajectoryEvaluatorConfig(
            name="x",
            judges=[
                JudgeSpecConfig(alias="primary_judge", family="openai"),
                JudgeSpecConfig(alias="secondary_judge", family="nvidia_nemotron"),
            ],
        )
        # Data Designer's model-health-check path expects this scalar
        # attribute on every model-generated column config.
        assert cfg.model_alias == "primary_judge"

    def test_required_columns_include_only_hard_dependencies(self) -> None:
        cfg = TrajectoryEvaluatorConfig(name="x", judges=self._ok_judges())
        required = cfg.required_columns
        assert "conversation_messages" in required
        assert "probe_family" in required
        # Historical trajectory parquets do not carry the raw persona
        # object, only projected persona_* fields; evaluator prompts
        # degrade gracefully when this optional context is absent.
        assert "persona" not in required

    def test_diversity_validated_at_construct(self) -> None:
        bad = [
            JudgeSpecConfig(alias="judge_a", family="openai"),
            JudgeSpecConfig(alias="judge_b", family="openai"),
        ]
        with pytest.raises(ValidationError) as exc:
            TrajectoryEvaluatorConfig(name="x", judges=bad)
        # Pydantic wraps the validator error; the original message bubbles up.
        assert "distinct families" in str(exc.value).lower()

    def test_single_judge_allowed(self) -> None:
        # Single-judge mode (no κ) is valid; no diversity error.
        cfg = TrajectoryEvaluatorConfig(
            name="x",
            judges=[JudgeSpecConfig(alias="solo", family="openai")],
        )
        assert len(cfg.judges) == 1

    def test_axes_filter(self) -> None:
        cfg = TrajectoryEvaluatorConfig(
            name="x",
            judges=self._ok_judges(),
            axes=["helpfulness", "accuracy"],
        )
        assert cfg.axes == ["helpfulness", "accuracy"]

    def test_scorer_dispatch_default_empty(self) -> None:
        cfg = TrajectoryEvaluatorConfig(name="x", judges=self._ok_judges())
        assert cfg.scorers == []

    def test_no_judges_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TrajectoryEvaluatorConfig(name="x", judges=[])
