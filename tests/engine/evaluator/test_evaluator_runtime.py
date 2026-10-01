# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

import json
from types import SimpleNamespace
from typing import Any

import pytest

from usersim.engine.evaluator.config import JudgeSpecConfig, TrajectoryEvaluatorConfig
from usersim.engine.evaluator.generator import TrajectoryEvaluatorGenerator
from usersim.engine.evaluator.runtime import TrajectoryEvaluatorRuntime, native_scorers_for_trajectory


class _ModelRegistry:
    def __init__(self, model: Any) -> None:
        self._model = model

    def get_model(self, *, model_alias: str) -> Any:
        assert model_alias == "judge_model"
        return self._model


def _row(*, probe_type: str = "general_open_ended") -> dict[str, Any]:
    return {
        "trajectory_id": "trajectory-1",
        "probe_type": probe_type,
        "persona": {"name": "Ari"},
        "locale": "en_US",
        "conversation_language": "English",
        "conversation_messages": [
            {"role": "user", "content": "Can you help?"},
            {"role": "assistant", "content": "Yes."},
        ],
        "simulation_outcome": {"status": "completed"},
    }


async def test_runtime_matches_data_designer_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    model = SimpleNamespace(model_name="judge-model")

    async def judge_result(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"helpfulness": {"score": 4, "reasoning": "Useful answer."}}

    monkeypatch.setattr(TrajectoryEvaluatorGenerator, "_call_judge", judge_result)
    row = _row()
    config = TrajectoryEvaluatorConfig(
        name="assistant_eval",
        judges=[JudgeSpecConfig(alias="judge_model")],
        axes=["helpfulness"],
        scorers=[],
        skip_if_existing=False,
    )
    provider = SimpleNamespace(model_registry=_ModelRegistry(model))
    generated = await TrajectoryEvaluatorGenerator(config, provider).agenerate(row)
    expected = json.loads(generated["assistant_eval"])

    actual = await TrajectoryEvaluatorRuntime(
        models={"judge_model": model},
        axes=["helpfulness"],
        scorers=[],
    ).evaluate(row)

    assert actual == expected
    assert actual["axes"]["helpfulness"]["judge_model"]["score"] == 4


async def test_runtime_dispatches_the_probe_native_scorer(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scorer(data: dict[str, Any], models: dict[str, Any]) -> dict[str, Any]:
        assert data["probe_family"] == "tool_calling"
        assert set(models) == {"judge_model"}
        return {"status_proposal": True}

    monkeypatch.setattr("usersim.engine.evaluator.generator.get_scorer", lambda name: scorer)
    result = await TrajectoryEvaluatorRuntime(
        models={"judge_model": SimpleNamespace(model_name="judge-model")},
        axes=[],
    ).evaluate(_row(probe_type="tool_calling"))

    assert result["scorers"] == {"tool_use": {"status_proposal": True}}


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"probe_type": "tool_calling"}, ["tool_use"]),
        ({"probe_type": "general_open_ended"}, []),
        ({"probe_type": "health_general_disclosure"}, []),
        (
            {"probe_type": "health_general_disclosure", "probe_variant": "guarded"},
            ["health_disclosure_concealment"],
        ),
    ],
)
def test_native_scorer_selection(row: dict[str, Any], expected: list[str]) -> None:
    assert native_scorers_for_trajectory(row) == expected


def test_external_api_exports_runtime() -> None:
    from usersim.engine.external import TrajectoryEvaluatorRuntime as ExportedRuntime

    assert ExportedRuntime is TrajectoryEvaluatorRuntime
