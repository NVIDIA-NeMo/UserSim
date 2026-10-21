# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reading what a probe recorded for a row.

A probe keeps per-row values in ``state.metadata``, and a run stores that dict
with the row as ``conversation_metadata``: a JSON string in parquet, a dict from
the hosted runtime. A row built by hand can carry the same key as a top-level
field instead.
"""

from __future__ import annotations

import json
from typing import Any

from usersim.engine.core.missing import is_missing


def recorded_value(row: Any, key: str) -> Any:
    """Return ``key`` as ``row`` records it, or None when the row doesn't.

    ``row`` is a dict or a pandas Series. A top-level value wins; an empty cell
    falls back to the value in ``conversation_metadata``.
    """
    value = row.get(key)
    if not is_missing(value):
        return value
    metadata = row.get("conversation_metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            return None
    return metadata.get(key) if isinstance(metadata, dict) else None
