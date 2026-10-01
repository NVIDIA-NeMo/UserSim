# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The judge prompt: the rubric it carries and the version it is recorded under."""

from __future__ import annotations

import inspect

import pytest

from usersim.cli import _build_parser
from usersim.cli._pipeline import build_evaluator_config_builder, evaluate_dataframe
from usersim.engine.evaluator.axes import PROBE_SCORES, select_axes
from usersim.engine.evaluator.config import JudgeSpecConfig, TrajectoryEvaluatorConfig
from usersim.engine.evaluator.prompts import EVAL_PROMPT_VERSION, EVAL_USER_PROMPT, render_rubric

_FAMILIES = ["general_open_ended", *sorted(PROBE_SCORES)]


@pytest.mark.parametrize("family", _FAMILIES)
def test_the_rubric_carries_every_axis_the_family_is_scored_on(family: str) -> None:
    """The response schema holds the same text, but a schema constrains the
    judge's output; nothing guarantees a provider shows it to the model."""
    axes = select_axes(family)
    rubric = render_rubric(axes)
    for axis in axes:
        assert f"{axis.name}: {axis.description}" in rubric, f"{family}: {axis.name} is missing from the rubric"
        for score, anchor in axis.options.items():
            assert f"{score}: {anchor}" in rubric, f"{family}: anchor {score} of {axis.name} is missing"


def test_the_prompt_template_has_a_place_for_the_rubric() -> None:
    rendered = EVAL_USER_PROMPT.format(
        persona="",
        language="English",
        locale="en_US",
        locale_rigor_instruction="",
        conversation="",
        rubric="<RUBRIC>",
    )
    assert "<RUBRIC>" in rendered


class TestOneVersion:
    """Every entry point defaults to the same version, so the envelope a cell
    records does not depend on how the evaluator was started."""

    def test_the_config_default(self) -> None:
        cfg = TrajectoryEvaluatorConfig(name="eval", judges=[JudgeSpecConfig(alias="judge_model")])
        assert cfg.prompt_version == EVAL_PROMPT_VERSION

    @pytest.mark.parametrize("fn", [build_evaluator_config_builder, evaluate_dataframe])
    def test_the_pipeline_defaults(self, fn) -> None:
        assert inspect.signature(fn).parameters["prompt_version"].default == EVAL_PROMPT_VERSION

    def test_the_cli_default(self) -> None:
        args = _build_parser().parse_args(["eval", "--trajectories", "t", "--out", "o"])
        assert args.prompt_version == EVAL_PROMPT_VERSION
