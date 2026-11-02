# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from usersim.extension import register_scorer


async def score(row, models):
    """Whether the assistant used the tool. Replace with what your probe measures."""
    if not row.get("my_tool_calls"):
        return {}
    return {"my_tool_used": float(bool(json.loads(row["my_tool_calls"])))}


register_scorer("my_probe", score, deterministic=True)
