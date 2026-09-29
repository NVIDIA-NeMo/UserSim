# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Direct asynchronous access to UserSim trajectory evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from usersim.engine.evaluator.config import JudgeSpecConfig, TrajectoryEvaluatorConfig
from usersim.engine.evaluator.generator import TrajectoryEvaluatorGenerator
from usersim.taxonomy.eval_cell import decode_cell


class _DirectTrajectoryEvaluatorGenerator(TrajectoryEvaluatorGenerator):
    """Bind the Data Designer evaluator implementation to supplied model facades."""

    def __init__(
        self,
        config: TrajectoryEvaluatorConfig,
        models: Mapping[str, Any],
    ) -> None:
        self._config = config
        self._models = dict(models)
        self._initialize()

    def get_model(self, model_alias: str) -> Any:
        try:
            return self._models[model_alias]
        except KeyError as error:
            raise KeyError(f"Trajectory evaluator model alias {model_alias!r} is not configured") from error

    def get_models(self) -> dict[str, Any]:
        return dict(self._models)


class TrajectoryEvaluatorRuntime:
    """Evaluate one completed trajectory without constructing a Data Designer run."""

    def __init__(
        self,
        *,
        models: Mapping[str, Any],
        judges: list[JudgeSpecConfig] | None = None,
        axes: list[str] | None = None,
        scorers: list[str] | None = None,
    ) -> None:
        """Create a runtime backed by already-configured asynchronous model facades."""
        if judges is None:
            judges = [JudgeSpecConfig(alias=alias) for alias in models]
        config = TrajectoryEvaluatorConfig(
            name="assistant_eval",
            judges=judges,
            axes=axes,
            scorers=scorers or [],
            skip_if_existing=False,
        )
        self._generator = _DirectTrajectoryEvaluatorGenerator(config, models)

    async def evaluate(self, trajectory: Mapping[str, Any]) -> dict[str, Any]:
        """Return UserSim's canonical decoded ``assistant_eval`` cell."""
        generated = await self._generator.agenerate(dict(trajectory))
        return decode_cell(generated["assistant_eval"])
