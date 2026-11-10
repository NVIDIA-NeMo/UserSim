# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A guarded health variant run by a host reproduces a standalone run, including the user's move tool calls."""

from __future__ import annotations

import pytest
from engine.test_hosted_parity_contract import _PERSONA, _PROFILE, _config, _data

from usersim.testing import assert_hosted_parity

_GUARDED = (
    "health_general_disclosure",
    "health_therapy_disclosure",
    "health_triage_disclosure",
    "health_decision_support_disclosure",
)


@pytest.mark.parametrize("probe_type", _GUARDED)
async def test_a_probe_whose_simulated_user_calls_tools_reproduces_a_standalone_run_when_hosted(
    probe_type: str,
) -> None:
    """The guarded variants' simulated user commits moves with a tool call, so the kit drives that call too.

    Three turns is the smallest budget with a turn where the user proposes a move.
    """
    await assert_hosted_parity(
        probe_type,
        config=_config(probe_type).model_copy(update={"max_turns": 3}),
        persona=_PERSONA,
        data={**_data(probe_type), "probe_variant": "guarded"},
        profile=_PROFILE,
    )
