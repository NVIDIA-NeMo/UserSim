# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""tool_calling's entries in the shared tests (see ``tests/_probe_contract.py``)."""

from __future__ import annotations

from _probe_contract import ParityCase, ProbeContract

#: The tool the parity episode offers the assistant.
_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}

CONTRACTS = {
    "tool_calling": ProbeContract(
        trajectory_id="4e24fe802886ac50",
        # A conversation with no tool calls completed its protocol, and the evaluator scores it.
        succeeds_on=None,
        parity=(ParityCase(1003, {"tools": [_WEATHER_TOOL]}),),
    ),
}
