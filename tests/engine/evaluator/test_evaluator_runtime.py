# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Tests for direct asynchronous trajectory evaluation."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from usersim.engine.evaluator.runtime import TrajectoryEvaluatorRuntime
from usersim.taxonomy.eval_cell import normalize_axis_score, score_from_eval_cell


class _JudgeFacade:
    model_name = "test-judge"

    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []

    async def acompletion(self, messages: list[dict[str, Any]], **kwargs: Any) -> SimpleNamespace:
        self.calls.append((messages, kwargs))
        content = json.dumps(
            {
                "helpfulness": {"score": 5, "reasoning": "Helpful."},
                "accuracy": {"score": 4, "reasoning": "Accurate."},
                "coherence": {"score": 3, "reasoning": "Coherent."},
            }
        )
        return SimpleNamespace(
            message=SimpleNamespace(content=content, reasoning_content=None, tool_calls=None),
            usage=None,
        )


@pytest.mark.asyncio
async def test_runtime_evaluates_trajectory_with_supplied_model() -> None:
    judge = _JudgeFacade()
    runtime = TrajectoryEvaluatorRuntime(
        models={"judge_model": judge},
        axes=["helpfulness", "accuracy", "coherence"],
    )

    cell = await runtime.evaluate(
        {
            "conversation_messages": [
                {"role": "user", "content": "Where should I eat?"},
                {"role": "assistant", "content": "Try the neighborhood cafe."},
            ],
            "persona": {"first_name": "Morgan"},
            "probe_family": "general_open_ended",
            "locale": "en_US",
            "conversation_language": "English",
            "simulation_outcome": {"status": "ok"},
        }
    )

    assert cell["skipped"] is False
    assert score_from_eval_cell(cell, "helpfulness") == 5.0
    assert normalize_axis_score("accuracy", score_from_eval_cell(cell, "accuracy")) == 0.8
    assert judge.calls[0][1]["response_format"]["type"] == "json_schema"
