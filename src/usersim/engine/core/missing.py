# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Empty table cells, however pandas represents them.

pandas holds an empty cell as None, a float NaN or ``pd.NA``, depending on the
column's dtype, how the frame was built and the installed pandas version. Code
that falls back on a falsy value (``row.get(key) or default``) behaves the same
for all three only once they are None.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from typing import Any


def is_missing(value: Any) -> bool:
    """True for an empty cell: None, a float NaN, ``pd.NA`` or ``pd.NaT``.

    A container is never missing, whatever it holds, so a list or array cell is
    never truth-tested element by element.
    """
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    # pandas' own markers (NA, NaT) exist only once pandas has been imported, and
    # importing it here would make every importer of this module pay for pandas.
    pandas = sys.modules.get("pandas")
    return pandas is not None and pandas.api.types.is_scalar(value) and bool(pandas.isna(value))


def none_for_missing(row: Mapping[str, Any]) -> dict[str, Any]:
    """Copy ``row`` with every empty cell replaced by None."""
    return {key: None if is_missing(value) else value for key, value in row.items()}
