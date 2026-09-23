# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""The contract Data Designer relies on when it runs this package.

Both column generators are reached only through the entry points declared
in ``pyproject.toml``, so the shape Data Designer expects is not visible
from any single module. These tests hold the generators to it: the
declared generation strategy, a dispatch method Data Designer can find,
and a row that comes back carrying the columns the config promises.

The generators are driven directly rather than through Data Designer's
scheduler. The scheduler cancels workers with
``gather(..., return_exceptions=True)``, which absorbs the hermetic
network guard along with everything else, so a test routed through it can
pass with an unmocked model call underneath.
"""

from typing import Any
from unittest.mock import MagicMock

from data_designer.config.column_configs import GenerationStrategy
from data_designer.engine.column_generators.generators.base import ColumnGenerator
from test_simulator_pipeline_e2e import (
    _cli_built_simulator_config,
    _patched_call_llm,
    _synthetic_row_data,
)

from usersim.engine.evaluator.generator import TrajectoryEvaluatorGenerator
from usersim.engine.generator import ConversationSimulatorGenerator

PROBE_TYPE = "general_open_ended"
LOCALE = "en_US"


def _resource_provider() -> MagicMock:
    """A provider whose registry answers with sentinel facades.

    ``call_llm`` is patched for these tests, so nothing ever reads a
    facade; the registry only has to resolve an alias without raising.
    """
    provider = MagicMock()
    provider.model_registry.get_model.side_effect = lambda model_alias: MagicMock(name=f"facade_{model_alias}")
    return provider


def _dispatch_methods(cls: type) -> set[str]:
    """Which dispatch methods Data Designer would consider overridden."""
    return {name for name in ("generate", "agenerate") if getattr(cls, name) is not getattr(ColumnGenerator, name)}


class TestDataDesignerContract:
    """Structural promises, independent of running a row."""

    def test_both_generators_are_cell_by_cell(self) -> None:
        for cls in (ConversationSimulatorGenerator, TrajectoryEvaluatorGenerator):
            assert cls.get_generation_strategy() == GenerationStrategy.CELL_BY_CELL, (
                f"{cls.__name__} declares a strategy the scheduler dispatches differently"
            )

    def test_both_generators_expose_a_dispatch_method(self) -> None:
        """Data Designer resolves one of these and raises if it finds neither."""
        for cls in (ConversationSimulatorGenerator, TrajectoryEvaluatorGenerator):
            assert _dispatch_methods(cls), (
                f"{cls.__name__} overrides neither generate nor agenerate, so Data Designer cannot run it"
            )

    def test_configs_resolve_through_the_declared_entry_points(self) -> None:
        """The entry points name dotted paths that are resolved at load time."""
        from usersim.engine.plugin import conversation_simulator, trajectory_evaluator

        for plugin, expected in (
            (conversation_simulator, ConversationSimulatorGenerator),
            (trajectory_evaluator, TrajectoryEvaluatorGenerator),
        ):
            module_path, _, attr = plugin.impl_qualified_name.rpartition(".")
            module = __import__(module_path, fromlist=[attr])
            assert getattr(module, attr) is expected, (
                f"{plugin.impl_qualified_name} does not resolve to {expected.__name__}"
            )


class TestSimulatorRunsARow:
    """The simulator, constructed and driven the way the engine does."""

    @staticmethod
    async def _run_one_row() -> tuple[dict[str, Any], Any]:
        cfg = _cli_built_simulator_config(PROBE_TYPE, LOCALE)
        data = _synthetic_row_data(PROBE_TYPE, LOCALE, cfg)
        generator = ConversationSimulatorGenerator(cfg, _resource_provider())
        with _patched_call_llm():
            return await generator.agenerate(data), cfg

    async def test_returns_the_row_with_the_configured_column(self) -> None:
        row, cfg = await self._run_one_row()
        assert isinstance(row, dict), f"a cell-by-cell generator returns the row mapping, got {type(row).__name__}"
        assert cfg.name in row, f"the configured column {cfg.name!r} is absent from the returned row"

    async def test_adds_no_column_the_config_did_not_declare(self) -> None:
        """Only declared columns survive the write.

        The engine persists the configured column plus the declared
        side-effect columns and drops everything else, so a value written
        under any other name is computed and then silently discarded.
        """
        cfg = _cli_built_simulator_config(PROBE_TYPE, LOCALE)
        data = _synthetic_row_data(PROBE_TYPE, LOCALE, cfg)
        before = set(data)
        generator = ConversationSimulatorGenerator(cfg, _resource_provider())
        with _patched_call_llm():
            row = await generator.agenerate(data)

        declarable = {cfg.name} | set(cfg.side_effect_columns)
        undeclared = (set(row) - before) - declarable
        assert not undeclared, (
            f"row carries columns the config never declared, so they are dropped: {sorted(undeclared)}"
        )

    async def test_carries_the_identity_columns_forward(self) -> None:
        row, _cfg = await self._run_one_row()
        for column in ("trajectory_id", "persona_uuid"):
            assert row.get(column), f"{column} is missing or empty, so the row cannot be identified"


class TestEvaluatorRunsARow:
    """The evaluator, fed a trajectory the simulator actually produced.

    Judge payload handling has its own tests; what matters here is that a
    real generator, constructed the way the engine constructs it, accepts a
    real trajectory row and returns one carrying its configured column.
    """

    @staticmethod
    async def _trajectory_row() -> dict[str, Any]:
        cfg = _cli_built_simulator_config(PROBE_TYPE, LOCALE)
        data = _synthetic_row_data(PROBE_TYPE, LOCALE, cfg)
        generator = ConversationSimulatorGenerator(cfg, _resource_provider())
        with _patched_call_llm():
            return await generator.agenerate(data)

    async def test_scores_a_simulator_row(self) -> None:
        from usersim.engine.evaluator.config import JudgeSpecConfig, TrajectoryEvaluatorConfig

        cfg = TrajectoryEvaluatorConfig(
            name="assistant_eval",
            judges=[
                JudgeSpecConfig(alias="judge_a", family="openai"),
                JudgeSpecConfig(alias="judge_b", family="nvidia_nemotron"),
            ],
            axes=["helpfulness"],
        )
        generator = TrajectoryEvaluatorGenerator(cfg, _resource_provider())

        async def _no_judge_verdict(models, judge_alias, prompt, schema_model, max_tokens):
            return {}

        generator._call_judge = _no_judge_verdict

        row = await generator.agenerate(await self._trajectory_row())

        assert isinstance(row, dict), f"a cell-by-cell generator returns the row mapping, got {type(row).__name__}"
        assert cfg.name in row, f"the configured column {cfg.name!r} is absent from the returned row"
