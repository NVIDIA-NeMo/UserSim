# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The guarded health variants write their move/Guard columns, and the simulator config keeps them."""

from __future__ import annotations

import engine.test_plugin_entry_points as plugin_tests
import pytest
from engine.test_plugin_entry_points import no_moves_override  # noqa: F401  (a fixture used by name)
from test_simulator_pipeline_e2e import _cli_built_simulator_config

from usersim.engine.probes.health_disclosure.mixin import GUARDED_RESULT_COLUMNS

LOCALE = plugin_tests.LOCALE
#: The health probes that declare a guarded variant.
GUARDED_PROBES = sorted(p for p, v in plugin_tests._probe_variant_pairs() if v == "guarded")


@pytest.mark.usefixtures("no_moves_override")
@pytest.mark.parametrize("probe_type", GUARDED_PROBES)
async def test_the_guarded_variant_adds_exactly_its_result_columns(probe_type: str) -> None:
    """A guarded row carries every name in ``GUARDED_RESULT_COLUMNS`` and nothing else
    a default row lacks.

    Checking a few of them would let the mixin stop writing, say,
    ``turn_budget`` while every test stayed green and the scorer reported a
    null stratum. Comparing against a default row of the same probe also
    rules out the guarded case passing by silently running as default.
    """
    guarded, _, _ = await plugin_tests.TestSimulatorRunsARow._run_one_row(probe_type, "guarded")
    default, _, _ = await plugin_tests.TestSimulatorRunsARow._run_one_row(probe_type, "default")
    assert guarded.get("moves_enabled") is True, f"{probe_type} guarded row carries no move/Guard ground truth"
    assert set(guarded) - set(default) == set(GUARDED_RESULT_COLUMNS)


def test_the_config_declares_every_guarded_result_column() -> None:
    cfg = _cli_built_simulator_config("health_therapy_disclosure", LOCALE)
    missing = set(GUARDED_RESULT_COLUMNS) - set(cfg.side_effect_columns)
    assert not missing, f"guarded columns the engine would drop: {sorted(missing)}"
