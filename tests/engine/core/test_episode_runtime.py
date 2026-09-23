# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Tests for externally hosted, episode-scoped probe execution."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from usersim.engine.core.episode_runtime import ProbeEpisodeRuntime


@pytest.fixture
def safety_runtime() -> ProbeEpisodeRuntime:
    return ProbeEpisodeRuntime(
        probe_type="safety_agentic",
        persona={
            "first_name": "Sarah",
            "last_name": "Johnson",
            "age": 35,
            "city": "Seattle",
            "education_level": "Bachelor",
            "occupation": "Software Engineer",
        },
        locale="en_US",
        language="English",
        models={},
        config=SimpleNamespace(random_seed=42, max_turns=3),
        data={},
    )


def test_runtime_exposes_only_resolved_episode_tools(safety_runtime: ProbeEpisodeRuntime) -> None:
    schemas = safety_runtime.assistant_tools
    assert schemas
    assert safety_runtime.allowed_tool_names == {schema["function"]["name"] for schema in schemas}

    schemas.clear()
    assert safety_runtime.assistant_tools


async def test_runtime_simulates_call_and_finalizes_native_evidence(
    safety_runtime: ProbeEpisodeRuntime,
) -> None:
    tool_name = next(iter(safety_runtime.allowed_tool_names))
    opening = await safety_runtime.initial_user_message()
    assert opening
    safety_runtime.append_message({"role": "user", "content": opening})
    safety_runtime.append_message(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps({"value": "example"}),
                    },
                }
            ],
        }
    )

    payload = await safety_runtime.simulate_tool_call(
        tool_name,
        {"value": "example"},
        tool_call_id="call-1",
    )
    safety_runtime.append_message({"role": "assistant", "content": "Done."})
    result = await safety_runtime.finalize()

    assert payload
    assert result["conversation_status"] is True
    assert result["num_tool_calls"] == 1
    metadata = json.loads(result["conversation_metadata"])
    assert metadata["attempted_actions"][0]["tool_name"] == tool_name
    assert json.loads(result["simulation_traces"])[0]["kind"] == "tool_call_verifier"


async def test_runtime_rejects_cross_episode_tool_without_side_effects(
    safety_runtime: ProbeEpisodeRuntime,
) -> None:
    before = await safety_runtime.evidence()

    with pytest.raises(ValueError, match="not available"):
        await safety_runtime.simulate_tool_call("not_this_episode", {})

    assert await safety_runtime.evidence() == before
