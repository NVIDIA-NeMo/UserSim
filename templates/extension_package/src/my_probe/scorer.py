# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from usersim.extension import register_scorer


async def score(row, models):
    """Whether the assistant used the tool. Replace with what your probe measures.

    Each number goes under ``scores``, where the capability report reads it; an
    axis named ``..._rate`` is read as a share from 0 to 1.
    """
    if not row.get("my_tool_calls"):
        return {}
    return {"scores": {"my_probe.tool_used_rate": {"score": float(bool(json.loads(row["my_tool_calls"])))}}}


register_scorer("my_probe", score, deterministic=True)
