# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""financial_services run by a host: its packaged stateful tools, and parallel calls numbered by its own loop."""

from __future__ import annotations

import json

from engine.core.test_episode_runtime import _drive, _text, _tool_call

from usersim.engine.config import ConversationSimulatorConfig
from usersim.engine.core.episode_runtime import (
    ActivationRequest,
    EpisodeLifecycleComplete,
    ProbeEpisodeRuntime,
)


def _financial_runtime() -> ProbeEpisodeRuntime:
    return ProbeEpisodeRuntime(
        probe_type="financial_services",
        persona={
            "first_name": "Yumi",
            "last_name": "Tanaka",
            "age": 35,
            "region": "Oregon",
            "occupation": "designer",
        },
        locale="en_US",
        language="English",
        models={},
        config=ConversationSimulatorConfig(
            name="episode_runtime_test",
            random_seed=1,
            max_turns=1,
            context_compression=False,
            probe_options={"financial_services": {"tier": "verifiable", "retrieval_mode": "golden"}},
        ),
        data={},
    )


async def test_financial_runtime_uses_packaged_stateful_tools() -> None:
    runtime = _financial_runtime()
    issued = False

    def respond(activation: ActivationRequest) -> dict[str, object]:
        nonlocal issued
        if activation.role == "assistant" and activation.tools_enabled and not issued:
            issued = True
            return _tool_call("kb_search", {"query": "account procedure"}, "call-search")
        return _text(activation)

    event, _seen = await _drive(runtime, respond)

    assert isinstance(event, EpisodeLifecycleComplete)
    executed = await runtime.executed_tool_calls()
    assert [call.tool_name for call in executed] == ["kb_search"]
    assert json.loads(executed[0].payload)["results"]
    assert json.loads(event.result["kb_search_queries"]) == ["account procedure"]


async def test_parallel_calls_in_one_response_get_the_loops_own_indices() -> None:
    """Indices come from UserSim's loop, never from a host-side counter."""
    runtime = _financial_runtime()

    def respond(activation: ActivationRequest) -> dict[str, object]:
        if activation.role == "assistant" and activation.tools_enabled:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "batch-1",
                        "type": "function",
                        "function": {"name": "kb_search", "arguments": json.dumps({"query": "first"})},
                    },
                    {
                        "id": "batch-2",
                        "type": "function",
                        "function": {"name": "kb_search", "arguments": json.dumps({"query": "second"})},
                    },
                ],
            }
        return _text(activation)

    event, _seen = await _drive(runtime, respond)

    assert isinstance(event, EpisodeLifecycleComplete)
    executed = await runtime.executed_tool_calls()
    assert [(call.turn_idx, call.call_idx) for call in executed[:2]] == [(1, 0), (1, 1)]
    assert json.loads(event.result["kb_search_queries"])[:2] == ["first", "second"]
