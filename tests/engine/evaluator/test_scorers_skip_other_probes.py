# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A probe's scorer judges its own probe's rows and skips every other row.

``usersim eval --scorers auto`` runs every scorer the run's probes ask for on
every row, so a run that mixes probes hands each probe's scorer the rows of
the others. Each skips such a row with the "no ... scorer skipped" sentinel,
before any model call, and the capability report counts it as not applicable
rather than as evidence. Every registered scorer but the generic ones, which
apply to every row, is checked, whichever probe asks for it.
"""

from __future__ import annotations

import pytest

from usersim.engine.core.probes import default_scorers_by_family
from usersim.engine.evaluator.scorers import GENERIC_SCORERS, get_scorer, list_scorers
from usersim.taxonomy.eval_cell import scorer_state

_PROBE_SCORERS = sorted(set(list_scorers()) - set(GENERIC_SCORERS))


def _general_open_ended_row() -> dict:
    """A row of a probe that asks only for the generic scorers, as a mixed run stores it.

    The columns other probes write are empty on it.
    """
    return {
        "conversation_messages": [
            {"role": "user", "content": "Can you explain how compound interest works?"},
            {"role": "assistant", "content": "It is interest earned on the interest already added to the balance."},
        ],
        "persona": {"first_name": "Ana", "last_name": "Souza", "age": 35},
        "probe_type": "general_open_ended",
        "probe_family": "general_open_ended",
        "probe_variant": "default",
        "simulation_outcome": None,
        "locale": "en_US",
        "language": "English",
        "tool_subset": None,
    }


def test_every_scorer_a_probe_asks_for_is_checked() -> None:
    asked_for = {name for names in default_scorers_by_family().values() for name in names}
    assert asked_for, "no probe asks for a scorer"
    assert asked_for - set(GENERIC_SCORERS) <= set(_PROBE_SCORERS)
    # The row the scorers are given belongs to general_open_ended, which asks for none of them.
    assert not default_scorers_by_family().get("general_open_ended")


@pytest.mark.parametrize("scorer", _PROBE_SCORERS)
async def test_a_probes_scorer_skips_another_probes_row(scorer: str) -> None:
    # No models: a scorer that tried to judge the row would fail, not skip.
    result = await get_scorer(scorer)(_general_open_ended_row(), {})

    assert scorer_state(result) == "not_applicable", f"{scorer} judged a general_open_ended row: {result}"
