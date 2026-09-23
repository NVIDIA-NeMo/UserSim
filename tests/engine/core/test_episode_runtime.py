# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Tests for externally hosted, episode-scoped probe execution."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from usersim.engine.core.episode_runtime import ProbeEpisodeRuntime


class _JSONModel:
    model_name = "json-model"

    async def acompletion(self, messages, **kwargs):
        del messages, kwargs
        return SimpleNamespace(
            message=SimpleNamespace(
                content='{"temperature_c": 22}',
                reasoning_content=None,
                tool_calls=None,
            ),
            usage=None,
        )


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


async def test_tool_calling_runtime_reuses_native_verifier_and_response_model() -> None:
    runtime = ProbeEpisodeRuntime(
        probe_type="tool_calling",
        persona={"first_name": "A", "last_name": "User", "age": 35},
        locale="en_US",
        language="English",
        models={"api_response_model": _JSONModel()},
        config=SimpleNamespace(
            tools_column="tools",
            max_tools=1,
            theme_column="theme",
            random_seed=7,
        ),
        data={
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather.",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                        },
                    },
                }
            ],
            "theme": {"type": "weather", "description": "Look up weather."},
        },
    )

    payload = await runtime.simulate_tool_call("get_weather", {"location": "Tokyo"})

    assert json.loads(payload) == {"temperature_c": 22}
    evidence = await runtime.evidence()
    assert evidence["conversation_metadata"]["tools_called"] == ["get_weather"]


async def test_financial_runtime_uses_packaged_stateful_tools() -> None:
    runtime = ProbeEpisodeRuntime(
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
        config=SimpleNamespace(
            random_seed=1,
            finance_tier_mix=0.0,
            finance_tier="verifiable",
            finance_retrieval_mode="golden",
            finance_embedding_model_alias="embedding_model",
        ),
        data={},
    )

    payload = await runtime.simulate_tool_call("kb_search", {"query": "account procedure"})

    assert json.loads(payload)["results"]
    evidence = await runtime.evidence()
    assert evidence["conversation_metadata"]["kb_search_queries"] == ["account procedure"]
    assert evidence["result_extras"]["finance_task_id"]
