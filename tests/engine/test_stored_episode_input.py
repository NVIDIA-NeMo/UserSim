# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A row can carry its stored episode whole, as one JSON cell.

The invariant: the probe is built from the stored values exactly, types
included, whatever a seed table would have made of them as separate columns.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from typing import Any
from unittest.mock import patch

import pytest

from usersim.engine import generator as generator_module
from usersim.engine.config import ConversationSimulatorConfig
from usersim.engine.core.episode_input import EPISODE_INPUT_COLUMN
from usersim.engine.generator import ConversationSimulatorGenerator

_LLM_SITES = (
    "usersim.engine.core.llm",
    "usersim.engine.core.simulation",
    "usersim.engine.core.judges",
    "usersim.engine.core.context",
    "usersim.engine.core.translation",
)

_STORED = {
    "persona": {"first_name": "Sarah", "last_name": "Johnson", "age": 35, "city": "Seattle", "state": "WA"},
    "probe_type": "general_open_ended",
    "theme": "cooking at home",
    "trajectory_id": "stored-a1",
    "ext_big": 9007199254740993,
    "ext_ratio": 0.5,
    "ext_flag": False,
    "ext_nested": {"memory": [{"week": 1, "note": "slept badly"}], "score": 2},
}


async def _stub_llm(models, alias, msgs, **kwargs):
    if alias == "judge_model":
        return {"role": "assistant", "content": "<explanation>fine</explanation>\n<rating>success</rating>"}
    if alias == "summary_model":
        return {"role": "assistant", "content": "no"}
    return {"role": "assistant", "content": "Thanks, that is helpful."}


class _Unused:
    async def acompletion(self, *args, **kwargs):
        raise AssertionError("acall_llm is patched")


async def _build_from(row: dict[str, Any]) -> dict[str, Any]:
    """Run ``row`` and return the row the probe was built from."""
    seen: list[dict[str, Any]] = []
    construct = generator_module.construct_probe_episode

    def spy(data, **kwargs):
        seen.append(dict(data))
        return construct(data, **kwargs)

    config = ConversationSimulatorConfig(name="conversation_messages", locale="en_US", max_turns=1, random_seed=1)
    aliases = ("user_model", "assistant_model", "judge_model", "summary_model", "api_response_model")
    generator = ConversationSimulatorGenerator.with_models(config, {alias: _Unused() for alias in aliases})
    with ExitStack() as stack:
        stack.enter_context(patch.object(generator_module, "construct_probe_episode", spy))
        for site in _LLM_SITES:
            stack.enter_context(patch(f"{site}.acall_llm", side_effect=_stub_llm))
        await generator.agenerate(row)
    (built,) = seen
    return built


def _seed_row(stored: dict[str, Any]) -> dict[str, Any]:
    """The row as a seed table hands it over: the required columns, plus the stored episode."""
    return {
        "persona": json.dumps(stored["persona"]),
        "probe_type": stored["probe_type"],
        "theme": stored["theme"],
        EPISODE_INPUT_COLUMN: json.dumps(stored),
    }


async def test_the_probe_is_built_from_the_stored_values_exactly():
    built = await _build_from(_seed_row(_STORED))

    for key, value in _STORED.items():
        assert built[key] == value and type(built[key]) is type(value), key


async def test_the_stored_episode_wins_over_the_seed_columns():
    seed = {**_seed_row(_STORED), "theme": "a theme the seed table rewrote"}

    built = await _build_from(seed)

    assert built["theme"] == "cooking at home"


async def test_the_carrier_column_does_not_reach_the_probe():
    built = await _build_from(_seed_row(_STORED))

    assert EPISODE_INPUT_COLUMN not in built


@pytest.mark.parametrize("cell", ["[1, 2]", '"text"'])
async def test_a_carrier_that_is_not_a_row_is_refused(cell):
    with pytest.raises(ValueError, match=EPISODE_INPUT_COLUMN):
        await _build_from({**_seed_row(_STORED), EPISODE_INPUT_COLUMN: cell})


async def test_a_row_without_a_carrier_is_built_as_before():
    built = await _build_from({"persona": dict(_STORED["persona"]), "probe_type": "general_open_ended", "theme": "x"})

    assert built["theme"] == "x"
