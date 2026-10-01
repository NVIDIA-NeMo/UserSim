# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-row state scoping in the simulator's column generator.

A row installs context that applies for its own duration only. Anything it
sets has to be restored on the way out, including when the row fails part
way through, or whatever the engine runs next inherits it.
"""

from pathlib import Path

import pytest

from usersim.engine.config import ConversationSimulatorConfig
from usersim.engine.core._assets import default_assets_dir
from usersim.engine.generator import ConversationSimulatorGenerator


def _generator_without_resources(cfg: ConversationSimulatorConfig) -> ConversationSimulatorGenerator:
    """Build a generator with no resource provider behind it.

    The row then fails on the first model lookup, which is what these tests
    want: the teardown path is the subject, not the simulation.
    """
    gen = ConversationSimulatorGenerator.__new__(ConversationSimulatorGenerator)
    gen._config = cfg
    return gen


class TestRuntimeAssetsRoot:
    """The per-row asset root must not outlive the row that set it."""

    async def test_restored_after_a_row_fails(self, tmp_path: Path) -> None:
        baseline = default_assets_dir()
        gen = _generator_without_resources(ConversationSimulatorConfig(name="conversation", assets_dir=str(tmp_path)))

        with pytest.raises(Exception):
            await gen.agenerate({})

        assert default_assets_dir() == baseline, (
            "the row's asset root is still installed, so banks for whatever "
            "runs next resolve against it instead of the configured root"
        )

    async def test_a_nested_scope_puts_the_row_builder_back(self) -> None:
        """A probe that installs its own builder must not detach the row's.

        Every model call feeds the installed builder, so a scope that clears
        instead of restoring leaves the rest of the row unattributed and its
        token totals short.
        """
        import types

        from usersim.engine.core.llm import get_current_outcome_builder, set_current_outcome_builder
        from usersim.engine.core.outcomes import OutcomeBuilder, Provenance
        from usersim.engine.probes.safety_agentic import generator as safety_agentic

        row_builder = OutcomeBuilder(provenance=Provenance())
        probe = safety_agentic.SafetyAgenticProbe.__new__(safety_agentic.SafetyAgenticProbe)
        probe._provenance = Provenance()
        probe._outcome_builder = OutcomeBuilder(provenance=Provenance())
        probe._task = types.SimpleNamespace(
            action_request=types.SimpleNamespace(
                id="ar-1",
                sub_protocol="sp",
                sanctioned_action_name="a",
                tempted_action_name="b",
                bank_id="bank",
                bank_version="v1",
            )
        )

        def _explode(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("stop once the nested builder is installed")

        set_current_outcome_builder(row_builder)
        try:
            with pytest.raises(RuntimeError):
                with pytest.MonkeyPatch.context() as mp:
                    mp.setattr(safety_agentic, "ConversationState", _explode)
                    await probe.run_dispatch(models={}, data={}, cfg=types.SimpleNamespace())
            assert get_current_outcome_builder() is row_builder, (
                "the nested scope cleared the row's outcome builder instead of "
                "restoring it, so later calls in this row record nowhere"
            )
        finally:
            set_current_outcome_builder(None)

    async def test_a_row_does_not_change_the_process_log_level(self) -> None:
        """Verbosity belongs to the run, so a row must not raise it globally."""
        import logging

        engine_logger = logging.getLogger("usersim.engine")
        before = engine_logger.level
        engine_logger.setLevel(logging.WARNING)
        try:
            gen = _generator_without_resources(ConversationSimulatorConfig(name="conversation", verbosity=2))
            # Resolve models so the row reaches the body; it then fails on the
            # absent persona column, which is past the point of interest.
            gen.get_model = lambda *a, **k: object()
            with pytest.raises(Exception):
                await gen.agenerate({})
            assert engine_logger.level == logging.WARNING, (
                "one row raised the process-wide log level, so every other row "
                "in this process inherits it with no way back"
            )
        finally:
            engine_logger.setLevel(before)

    def test_the_code_sha_is_resolved_before_the_first_row(self) -> None:
        """Resolving the SHA shells out to git, so it happens once, up front.

        Left to the first row, that subprocess runs while a trajectory is in
        flight, where it costs every row sharing the runtime rather than one.
        """
        from usersim.engine.core.provenance import get_code_sha

        get_code_sha.cache_clear()
        gen = _generator_without_resources(ConversationSimulatorConfig(name="conversation"))

        gen._initialize()

        assert get_code_sha.cache_info().currsize == 1, (
            "the code SHA is still unresolved after setup, so the first row pays for the git call"
        )

    async def test_the_row_does_see_its_own_root(self, tmp_path: Path) -> None:
        """Guards the test above: it must fail for the right reason."""
        seen: list[Path] = []
        gen = _generator_without_resources(ConversationSimulatorConfig(name="conversation", assets_dir=str(tmp_path)))
        gen.get_model = lambda *a, **k: seen.append(default_assets_dir()) or (_ for _ in ()).throw(RuntimeError("stop"))

        with pytest.raises(Exception):
            await gen.agenerate({})

        assert seen and seen[0] == tmp_path, f"row did not see its configured root, saw {seen}"
