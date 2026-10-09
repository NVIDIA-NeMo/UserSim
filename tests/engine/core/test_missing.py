# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for recognising empty table cells."""

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from usersim.engine.core.missing import is_missing, none_for_missing

_FRESH_INTERPRETER = """
import json
import sys

from usersim.engine.core.missing import is_missing, none_for_missing

print(json.dumps({
    "pandas_imported": "pandas" in sys.modules,
    "missing": [is_missing(None), is_missing(float("nan"))],
    "row": none_for_missing({"a": float("nan"), "b": "x"}),
}))
"""


def test_the_helpers_need_no_pandas():
    """Importing them must not import pandas, which takes seconds, and the empty
    cells that can exist without pandas are still recognised."""
    proc = subprocess.run([sys.executable, "-c", _FRESH_INTERPRETER], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == {
        "pandas_imported": False,
        "missing": [True, True],
        "row": {"a": None, "b": "x"},
    }


@pytest.mark.parametrize(
    "value",
    [None, float("nan"), np.nan, pd.NA, pd.NaT],
    ids=["none", "nan", "numpy-nan", "na", "nat"],
)
def test_empty_cells_are_missing(value):
    assert is_missing(value)


@pytest.mark.parametrize(
    "value",
    ["", "nan", 0, 0.0, False, [], {}, [float("nan")], np.array([np.nan, np.nan])],
    ids=["blank", "nan-text", "zero", "zero-float", "false", "empty-list", "empty-dict", "list-of-nan", "array-of-nan"],
)
def test_values_and_containers_are_not_missing(value):
    assert not is_missing(value)


def test_only_empty_cells_become_none():
    nested = [float("nan")]
    row = {"nan": float("nan"), "na": pd.NA, "text": "x", "zero": 0, "nested": nested}
    out = none_for_missing(row)
    assert out["nan"] is None
    assert out["na"] is None
    assert out["text"] == "x"
    assert out["zero"] == 0
    assert out["nested"] is nested
    assert is_missing(row["nan"]), "the input row must not be modified"


def test_a_string_columns_empty_cell_becomes_none():
    """A row taken from a DataFrame holds a string column's empty cell as None
    or NaN, depending on the pandas version."""
    records = pd.DataFrame({"s": ["a", None]}).to_dict(orient="records")
    assert [none_for_missing(record) for record in records] == [{"s": "a"}, {"s": None}]
