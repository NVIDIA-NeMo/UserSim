# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Every registered probe reproduces a standalone run when hosted.

Run through the same ``usersim.testing`` entry point a probe author would use,
so the published check is exercised by the probes it describes.
"""

from __future__ import annotations

import pytest

import usersim.engine.generator  # noqa: F401  -- registers the built-in probes
from usersim.engine.config import ConversationSimulatorConfig
from usersim.engine.core.probes import known_probes
from usersim.testing import assert_hosted_parity, hosted_parity_report

_PERSONA = {
    "first_name": "Sarah",
    "last_name": "Johnson",
    "age": 35,
    "city": "Seattle",
    "state": "WA",
    "region": "Oregon",
    "occupation": "Software Engineer",
    "education_level": "Bachelor",
}
_PROFILE = {"patience": 0.75, "tech_literacy": 0.5, "error_proneness": 0.5}
_TOOL = {
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


def _config(probe_type: str) -> ConversationSimulatorConfig:
    return ConversationSimulatorConfig(
        name="hosted_parity",
        locale="en_US",
        max_turns=2,
        random_seed=4242,
        tools_column="tools" if probe_type == "tool_calling" else None,
        finance_tier_mix=0.0,
        finance_tier="verifiable",
        finance_retrieval_mode="hybrid",
    )


def _data(probe_type: str) -> dict[str, object]:
    data: dict[str, object] = {
        "theme": {
            "type": probe_type.replace("_", " "),
            "description": f"Exercise the {probe_type} probe.",
        },
        "user_interaction_style": "direct",
    }
    if probe_type == "tool_calling":
        data["tools"] = [_TOOL]
    return data


@pytest.mark.parametrize("probe_type", known_probes())
async def test_every_registered_probe_reproduces_a_standalone_run_when_hosted(probe_type: str) -> None:
    await assert_hosted_parity(
        probe_type,
        config=_config(probe_type),
        persona=_PERSONA,
        data=_data(probe_type),
        profile=_PROFILE,
    )


async def test_the_check_reports_differences_rather_than_only_raising() -> None:
    """The list form is what lets a probe author see every problem in one pass."""
    report = await hosted_parity_report(
        "general_open_ended",
        config=_config("general_open_ended"),
        persona=_PERSONA,
        data=_data("general_open_ended"),
        profile=_PROFILE,
    )

    assert report.differences == []
    assert report.direct_prompts == report.hosted_prompts
    assert report.direct_prompts, "the check must actually drive model calls"
    assert report.hosted_row["conversation_status"] is True
