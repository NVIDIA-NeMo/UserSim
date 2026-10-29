# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The episode runners in ``usersim.testing``, run against every registered probe.

``run_episode`` runs a probe offline with scripted models, and every registered
probe reproduces that standalone run when hosted. Both go through the same
``usersim.testing`` entry points a probe author would use, so the published
checks are exercised by the probes they describe.
"""

from __future__ import annotations

import json

import pytest

import usersim.engine.generator  # noqa: F401  -- registers the built-in probes
from usersim.engine.config import ConversationSimulatorConfig
from usersim.engine.core.probes import known_probes
from usersim.testing import ScriptedModel, assert_hosted_parity, hosted_parity_report, run_episode

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
async def test_every_registered_probe_runs_to_completion_with_scripted_models(probe_type: str) -> None:
    run = await run_episode(
        probe_type, config=_config(probe_type), persona=_PERSONA, data=_data(probe_type), profile=_PROFILE
    )

    assert run.row["conversation_status"] is True
    assert json.loads(run.row["simulation_outcome"])["status"] != "failed"
    assert run.row["trajectory_id"]
    assert run.prompts and all(prompt["messages"] for prompt in run.prompts)


async def test_a_model_factory_scripts_the_replies() -> None:
    """A ``ScriptedModel`` subclass answers in place of the default, with nothing patched."""

    class _Wordy(ScriptedModel):
        CONTENT = {**ScriptedModel.CONTENT, "assistant": "A reply only this subclass gives."}

    run = await run_episode(
        "general_open_ended",
        config=_config("general_open_ended"),
        persona=_PERSONA,
        data=_data("general_open_ended"),
        model_factory=_Wordy,
    )

    replies = [m["content"] for m in json.loads(run.row["conversation_messages"]) if m["role"] == "assistant"]
    assert replies and set(replies) == {"A reply only this subclass gives."}
    assert {prompt["role"] for prompt in run.prompts} >= {"assistant", "user", "judge"}


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


_GUARDED = (
    "health_general_disclosure",
    "health_therapy_disclosure",
    "health_triage_disclosure",
    "health_decision_support_disclosure",
)


@pytest.mark.parametrize("probe_type", _GUARDED)
async def test_a_probe_whose_simulated_user_calls_tools_reproduces_a_standalone_run_when_hosted(
    probe_type: str,
) -> None:
    """The guarded variants' simulated user commits moves with a tool call, so the kit drives that call too.

    Three turns is the smallest budget with a turn where the user proposes a move.
    """
    await assert_hosted_parity(
        probe_type,
        config=_config(probe_type).model_copy(update={"max_turns": 3}),
        persona=_PERSONA,
        data={**_data(probe_type), "probe_variant": "guarded"},
        profile=_PROFILE,
    )


@pytest.mark.parametrize("role", ["assistant", "user"])
async def test_the_scripted_model_calls_a_tool_it_is_offered_on_either_side(role: str) -> None:
    """A probe whose simulated user acts through a tool is driven the way a model would drive it."""
    tool = {"type": "function", "function": {"name": "commit_move", "parameters": {"type": "object"}}}
    reply = await ScriptedModel(role).acompletion([{"role": "user", "content": "hi"}], tools=[tool])

    assert [call["function"]["name"] for call in reply.message.tool_calls] == ["commit_move"]


@pytest.mark.parametrize(("role", "calls"), [("user", True), ("assistant", False)])
async def test_with_a_tool_reply_in_view_only_the_user_calls_again(role: str, calls: bool) -> None:
    """The user re-commits after a rejection, as a model would; the assistant's tool loop ends at a result."""
    tool = {"type": "function", "function": {"name": "commit_move", "parameters": {"type": "object"}}}
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "commit_move", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": '{"accepted": false}'},
    ]
    reply = await ScriptedModel(role).acompletion(messages, tools=[tool])

    assert bool(reply.message.tool_calls) is calls
