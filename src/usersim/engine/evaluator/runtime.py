# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Hosted-runtime facade for UserSim's native trajectory evaluator."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from usersim.engine.evaluator.config import JudgeSpecConfig, TrajectoryEvaluatorConfig
from usersim.engine.evaluator.generator import TrajectoryEvaluatorGenerator

_PROBE_SCORERS = {
    "financial_services": "financial_services",
    "health_decision_support_disclosure": "health_disclosure_concealment",
    "health_general_disclosure": "health_disclosure_concealment",
    "health_therapy_disclosure": "health_disclosure_concealment",
    "health_triage_disclosure": "health_disclosure_concealment",
    "identity_disclosure": "identity_disclosure",
    "safety_agentic": "safety_agentic",
    "safety_chat_pressure": "safety_chat_pressure",
    "sov_ai_dynamic": "sov_ai_dynamic",
    "sov_ai_facts": "sov_ai_facts",
    "sov_ai_multilingual_parity": "sov_ai_multilingual_parity",
    "tool_calling": "tool_use",
}


def native_scorers_for_trajectory(data: Mapping[str, Any]) -> list[str]:
    """Return the native scorer applicable to one resolved trajectory."""

    probe_family = str(data.get("probe_family") or data.get("probe_type") or "general_open_ended")
    scorer = _PROBE_SCORERS.get(probe_family)
    if scorer == "health_disclosure_concealment" and data.get("probe_variant", "default") != "guarded":
        return []
    return [scorer] if scorer is not None else []


class _RuntimeModelRegistry:
    def __init__(self, models: Mapping[str, Any]) -> None:
        self._models = dict(models)

    def get_model(self, *, model_alias: str) -> Any:
        try:
            return self._models[model_alias]
        except KeyError as error:
            raise KeyError(f"Trajectory evaluator model alias {model_alias!r} is not configured") from error


class TrajectoryEvaluatorRuntime:
    """Evaluate completed trajectories without a Data Designer scheduler."""

    def __init__(
        self,
        *,
        models: Mapping[str, Any],
        judge_aliases: Sequence[str] = ("judge_model",),
        axes: list[str] | None = None,
        scorers: list[str] | None = None,
        prompt_version: str = "v1.0",
    ) -> None:
        if not judge_aliases:
            raise ValueError("TrajectoryEvaluatorRuntime requires at least one judge alias")
        self._models = dict(models)
        self._judge_aliases = tuple(judge_aliases)
        for alias in self._judge_aliases:
            if alias not in self._models:
                raise ValueError(f"Trajectory evaluator model alias {alias!r} is not configured")
        self._axes = axes
        self._scorers = scorers
        self._prompt_version = prompt_version

    async def evaluate(self, data: Mapping[str, Any]) -> dict[str, Any]:
        """Return UserSim's native ``assistant_eval`` envelope for one row."""

        row = dict(data)
        row.setdefault("probe_family", row.get("probe_type", "general_open_ended"))
        scorers = self._scorers if self._scorers is not None else native_scorers_for_trajectory(row)
        config = TrajectoryEvaluatorConfig(
            name="assistant_eval",
            judges=[JudgeSpecConfig(alias=alias) for alias in self._judge_aliases],
            axes=self._axes,
            scorers=scorers,
            skip_if_existing=False,
            prompt_version=self._prompt_version,
        )
        provider = SimpleNamespace(model_registry=_RuntimeModelRegistry(self._models))
        generated = await TrajectoryEvaluatorGenerator(config, provider).agenerate(row)
        evaluation = generated.get(config.name)
        if not isinstance(evaluation, str):
            raise RuntimeError("Trajectory evaluator did not produce its assistant_eval JSON cell")
        decoded = json.loads(evaluation)
        if not isinstance(decoded, dict):
            raise RuntimeError("Trajectory evaluator assistant_eval cell is not a JSON object")
        return decoded
