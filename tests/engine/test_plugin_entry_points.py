# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest
from data_designer.config.column_configs import GenerationStrategy
from data_designer.engine.column_generators.generators.base import ColumnGenerator
from test_simulator_pipeline_e2e import (
    _cli_built_simulator_config,
    _patched_call_llm,
    _synthetic_row_data,
)

from usersim.engine.core.probes import known_probes, resolve_probe
from usersim.engine.evaluator.generator import TrajectoryEvaluatorGenerator
from usersim.engine.generator import ConversationSimulatorGenerator
from usersim.engine.probes.health_disclosure.mixin import GUARDED_RESULT_COLUMNS

PROBE_TYPE = "general_open_ended"
LOCALE = "en_US"


def _probe_variant_pairs() -> list[tuple[str, str]]:
    """Every registered probe paired with every variant it declares."""
    pairs = []
    for probe in sorted(known_probes()):
        module = sys.modules[resolve_probe(probe).__module__]
        pairs.extend((probe, str(v)) for v in module.PROBE_VARIANTS)
    return pairs


@pytest.fixture
def no_moves_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """``USERSIM_DISCLOSURE_MOVES`` overrides ``probe_variant``, so a value left set
    in the shell would decide what the variant tests exercise."""
    monkeypatch.delenv("USERSIM_DISCLOSURE_MOVES", raising=False)


def _resource_provider() -> MagicMock:
    """A provider whose registry answers with sentinel facades.

    ``acall_llm`` is patched for these tests, so nothing ever reads a
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
    async def _run_one_row(
        probe_type: str = PROBE_TYPE, variant: str | None = None
    ) -> tuple[dict[str, Any], Any, set[str]]:
        """Run one row; return it, its config, and the keys the row started with."""
        cfg = _cli_built_simulator_config(probe_type, LOCALE)
        data = _synthetic_row_data(probe_type, LOCALE, cfg)
        if variant is not None:
            data["probe_variant"] = variant
        before = set(data)
        generator = ConversationSimulatorGenerator(cfg, _resource_provider())
        with _patched_call_llm():
            return await generator.agenerate(data), cfg, before

    async def test_returns_the_row_with_the_configured_column(self) -> None:
        row, cfg, _ = await self._run_one_row()
        assert isinstance(row, dict), f"a cell-by-cell generator returns the row mapping, got {type(row).__name__}"
        assert cfg.name in row, f"the configured column {cfg.name!r} is absent from the returned row"

    async def test_adds_no_column_the_config_did_not_declare(self) -> None:
        """Only declared columns survive the write.

        The engine persists the configured column plus the declared
        side-effect columns and drops everything else, so a value written
        under any other name is computed and then silently discarded.
        """
        row, cfg, before = await self._run_one_row()
        declarable = {cfg.name} | set(cfg.side_effect_columns)
        undeclared = (set(row) - before) - declarable
        assert not undeclared, (
            f"row carries columns the config never declared, so they are dropped: {sorted(undeclared)}"
        )

    @pytest.mark.usefixtures("no_moves_override")
    @pytest.mark.parametrize(("probe_type", "variant"), _probe_variant_pairs())
    async def test_no_probe_writes_a_column_the_config_did_not_declare(self, probe_type: str, variant: str) -> None:
        """Every probe and every variant it declares, not just the default one.

        ``build_result_extras`` is per-probe, so the columns a row carries
        depend on which probe produced it. The engine writes the configured
        column plus the declared side-effect columns and nothing else (a
        logged warning is easy to miss), so a probe-specific column that is not declared is
        computed, dropped, and then read back as absent by the scorer that
        needs it -- which reports the trajectory as unscoreable rather than
        failing.

        A variant can write columns its default does not: the guarded
        ``health_*`` variants write the move/Guard ground truth that
        ``health_disclosure_concealment`` scores from, and the default
        variants write none of it. Running only the default variant would
        miss a column that only another variant writes.
        """
        row, cfg, before = await self._run_one_row(probe_type, variant)
        if variant != "default":
            # Some probes relabel the default variant from their own data, so
            # only a requested non-default variant is required to stick.
            assert row.get("probe_variant") == variant, (
                f"{probe_type} ran as {row.get('probe_variant')!r}, not {variant!r}, so this case checked nothing"
            )
        undeclared = (set(row) - before) - ({cfg.name} | set(cfg.side_effect_columns))
        assert not undeclared, (
            f"{probe_type} ({variant}) writes columns the config never declares, so they are dropped "
            f"before anything can read them: {sorted(undeclared)}"
        )

    @pytest.mark.usefixtures("no_moves_override")
    @pytest.mark.parametrize(("probe_type", "variant"), _probe_variant_pairs())
    async def test_every_probe_row_can_be_written_to_parquet(
        self, probe_type: str, variant: str, tmp_path: Path
    ) -> None:
        """Data Designer writes each batch of rows to parquet before ``usersim simulate`` reads it back.

        A value pyarrow cannot store, such as an empty mapping, makes Data
        Designer drop the whole batch, so the run ends with no trajectories.
        The row is written here the way Data Designer writes a batch.
        """
        row, _, _ = await self._run_one_row(probe_type, variant)
        path = tmp_path / "batch.parquet"
        pd.DataFrame([row]).to_parquet(path, index=False)
        assert pd.read_parquet(path)["trajectory_id"].tolist() == [row["trajectory_id"]]

    @pytest.mark.usefixtures("no_moves_override")
    @pytest.mark.parametrize("probe_type", sorted(p for p, v in _probe_variant_pairs() if v == "guarded"))
    async def test_the_guarded_variant_adds_exactly_its_result_columns(self, probe_type: str) -> None:
        """A guarded row carries every name in ``GUARDED_RESULT_COLUMNS`` and nothing else
        a default row lacks.

        Checking a few of them would let the mixin stop writing, say,
        ``turn_budget`` while every test stayed green and the scorer reported a
        null stratum. Comparing against a default row of the same probe also
        rules out the guarded case passing by silently running as default.
        """
        guarded, _, _ = await self._run_one_row(probe_type, "guarded")
        default, _, _ = await self._run_one_row(probe_type, "default")
        assert guarded.get("moves_enabled") is True, f"{probe_type} guarded row carries no move/Guard ground truth"
        assert set(guarded) - set(default) == set(GUARDED_RESULT_COLUMNS)

    def test_the_config_declares_every_guarded_result_column(self) -> None:
        cfg = _cli_built_simulator_config("health_therapy_disclosure", LOCALE)
        missing = set(GUARDED_RESULT_COLUMNS) - set(cfg.side_effect_columns)
        assert not missing, f"guarded columns the engine would drop: {sorted(missing)}"

    async def test_carries_the_identity_columns_forward(self) -> None:
        row, _cfg, _ = await self._run_one_row()
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
