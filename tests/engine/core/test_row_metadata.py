# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``recorded_value`` finds what a probe recorded, whether the row carries it top-level or in ``conversation_metadata``."""

from __future__ import annotations

import json
from typing import Any, Callable

import pandas as pd
import pytest

from usersim.engine.core.row_metadata import recorded_value


@pytest.mark.parametrize("encode", [json.dumps, dict], ids=["json", "dict"])
def test_reads_conversation_metadata_when_the_row_has_no_value(encode: Callable[[dict[str, Any]], Any]) -> None:
    row = {"conversation_metadata": encode({"capitulated_at_turn": 2})}
    assert recorded_value(row, "capitulated_at_turn") == 2


def test_a_top_level_value_wins_even_when_empty() -> None:
    row = {"attempted_actions": [], "conversation_metadata": json.dumps({"attempted_actions": [{"name": "x"}]})}
    assert recorded_value(row, "attempted_actions") == []


def test_an_empty_cell_falls_back_to_conversation_metadata() -> None:
    row = pd.Series(
        {"stopped_on_capitulation": pd.NA, "conversation_metadata": json.dumps({"stopped_on_capitulation": True})}
    )
    assert recorded_value(row, "stopped_on_capitulation") is True


@pytest.mark.parametrize(
    "metadata",
    [None, "", "not json", json.dumps(["a list"]), json.dumps({"other_key": 1})],
    ids=["none", "empty", "invalid_json", "not_an_object", "key_absent"],
)
def test_returns_none_when_the_row_does_not_record_the_key(metadata: Any) -> None:
    assert recorded_value({"conversation_metadata": metadata}, "stopped_on_capitulation") is None
