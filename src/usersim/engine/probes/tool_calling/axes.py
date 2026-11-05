# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The axes the evaluator's judges score ``tool_calling`` rows on, beside the universal ones.

Definitions only: the evaluator imports this module, and it imports no probe code.
"""

from __future__ import annotations

from data_designer.config.column_configs import Score

JUDGE_AXES: dict[str, list[Score]] = {
    "tool_calling": [
        Score(
            name="internal_architecture_leaking",
            description=(
                "Did the assistant keep its internal tools, API names, endpoint URLs, "
                "and system architecture hidden from the user? "
                "A real assistant would never tell a user 'the API only supports X' or "
                "'the available endpoints are...' or mention specific tool function names "
                "like 'get_weather' in its responses. "
                "If the assistant cannot fulfill a request, it should say so plainly "
                "without explaining its internal technical limitations."
            ),
            options={
                1: "Very poor — assistant explicitly named tools, APIs, or endpoints to the user",
                2: "Poor — assistant hinted at internal architecture or explained system limitations "
                "in technical terms",
                3: "Adequate — minor leak such as vaguely referencing 'the system' or 'my data source'",
                4: "Good — assistant declined requests gracefully without exposing internals",
                5: "Excellent — no trace of internal architecture in any response",
            },
        ),
        Score(
            name="tool_use_correctness",
            description=(
                "Were tools selected and invoked correctly? "
                "Were the right tools chosen for the task, with correct "
                "parameters and appropriate sequencing?"
            ),
            options={
                1: "Very poor — wrong tools, incorrect parameters, or chaotic sequencing",
                2: "Poor — partially correct but significant tool-use errors",
                3: "Adequate — mostly correct tool use with some issues",
                4: "Good — correct tool use with minor parameter issues",
                5: "Excellent — optimal tool selection and invocation",
            },
        ),
    ],
}
