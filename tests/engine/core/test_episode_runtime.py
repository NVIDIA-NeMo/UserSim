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


def _tool_calling_runtime() -> ProbeEpisodeRuntime:
    return ProbeEpisodeRuntime(
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
        profile={"tech_literacy": 0.5, "error_proneness": 0.5},
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
        config=SimpleNamespace(
            random_seed=1,
            finance_tier_mix=0.0,
            finance_tier="verifiable",
            finance_retrieval_mode="golden",
            finance_embedding_model_alias="embedding_model",
        ),
        data={},
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
        data={"user_interaction_style": "direct"},
        profile={"patience": 0.75},
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
    assistant_call = {
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
    await safety_runtime.synchronize_transcript(
        [
            {"role": "user", "content": opening},
            assistant_call,
        ]
    )

    payload = await safety_runtime.simulate_tool_call(
        tool_name,
        {"value": "example"},
        tool_call_id="call-1",
        turn_idx=0,
        call_idx=0,
    )
    await safety_runtime.synchronize_transcript(
        [
            {"role": "user", "content": opening},
            assistant_call,
            {
                "role": "tool",
                "content": payload,
                "tool_call_id": "call-1",
            },
            {"role": "assistant", "content": "Done."},
        ]
    )
    result = await safety_runtime.finalize()

    assert payload
    assert result["conversation_status"] is True
    assert result["num_tool_calls"] == 1
    metadata = json.loads(result["conversation_metadata"])
    assert metadata["attempted_actions"][0]["tool_name"] == tool_name
    assert json.loads(result["simulation_traces"])[0]["kind"] == "tool_call_verifier"


async def test_synchronize_transcript_rejects_regression_and_divergence(
    safety_runtime: ProbeEpisodeRuntime,
) -> None:
    opening = await safety_runtime.initial_user_message()
    transcript = [{"role": "user", "content": opening}]
    await safety_runtime.synchronize_transcript(transcript)

    with pytest.raises(ValueError, match="regress"):
        await safety_runtime.synchronize_transcript([])
    with pytest.raises(ValueError, match="diverges"):
        await safety_runtime.synchronize_transcript([{"role": "user", "content": f"{opening} changed"}])


async def test_synchronize_transcript_validates_assistant_call_and_tool_result(
    safety_runtime: ProbeEpisodeRuntime,
) -> None:
    tool_name = next(iter(safety_runtime.allowed_tool_names))
    assistant_call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps({"value": "expected"}),
                },
            }
        ],
    }
    await safety_runtime.synchronize_transcript([assistant_call])
    payload = await safety_runtime.simulate_tool_call(
        tool_name,
        {"value": "different"},
        tool_call_id="call-1",
        turn_idx=0,
        call_idx=0,
    )

    with pytest.raises(ValueError, match="assistant calls"):
        await safety_runtime.synchronize_transcript(
            [
                assistant_call,
                {
                    "role": "tool",
                    "content": payload,
                    "tool_call_id": "call-1",
                },
            ]
        )


async def test_synchronize_transcript_rejects_changed_tool_result(
    safety_runtime: ProbeEpisodeRuntime,
) -> None:
    tool_name = next(iter(safety_runtime.allowed_tool_names))
    assistant_call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": "{}",
                },
            }
        ],
    }
    await safety_runtime.synchronize_transcript([assistant_call])
    payload = await safety_runtime.simulate_tool_call(
        tool_name,
        {},
        tool_call_id="call-1",
        turn_idx=0,
        call_idx=0,
    )

    with pytest.raises(ValueError, match="tool messages"):
        await safety_runtime.synchronize_transcript(
            [
                assistant_call,
                {
                    "role": "tool",
                    "content": payload + "changed",
                    "tool_call_id": "call-1",
                },
            ]
        )


async def test_runtime_rejects_cross_episode_tool_without_side_effects(
    safety_runtime: ProbeEpisodeRuntime,
) -> None:
    before = await safety_runtime.evidence()

    with pytest.raises(ValueError, match="not available"):
        await safety_runtime.simulate_tool_call(
            "not_this_episode",
            {},
            tool_call_id="call-1",
            turn_idx=0,
            call_idx=0,
        )

    assert await safety_runtime.evidence() == before


async def test_tool_calling_runtime_reuses_native_verifier_and_response_model() -> None:
    runtime = _tool_calling_runtime()

    payload = await runtime.simulate_tool_call(
        "get_weather",
        {"location": "Tokyo"},
        tool_call_id="call-weather",
        turn_idx=0,
        call_idx=0,
    )

    assert json.loads(payload) == {"temperature_c": 22}
    evidence = await runtime.evidence()
    assert evidence["conversation_metadata"]["tools_called"] == ["get_weather"]


async def test_financial_runtime_uses_packaged_stateful_tools() -> None:
    runtime = _financial_runtime()

    payload = await runtime.simulate_tool_call(
        "kb_search",
        {"query": "account procedure"},
        tool_call_id="call-search",
        turn_idx=0,
        call_idx=0,
    )

    assert json.loads(payload)["results"]
    evidence = await runtime.evidence()
    assert evidence["conversation_metadata"]["kb_search_queries"] == ["account procedure"]
    assert evidence["result_extras"]["finance_task_id"]


async def test_runtime_policies_match_native_probe_rules(safety_runtime: ProbeEpisodeRuntime) -> None:
    tool_runtime = _tool_calling_runtime()
    financial_runtime = _financial_runtime()

    assert tool_runtime.loop_policy.to_dict() == {
        "tool_round_mode": tool_runtime.probe.tool_loop_mode,
        "max_assistant_activations": 2,
        "final_synthesis_without_tools": tool_runtime.probe.final_synthesis_without_tools,
        "single_user_turn": False,
        "assistant_error_behavior": "fail_episode",
        "tool_error_behavior": "return_error_payload",
        "max_tool_calls_per_turn": tool_runtime.probe.tool_max_calls_per_turn,
        "max_tool_response_attempts": tool_runtime.probe.max_tool_response_attempts,
        "assistant_resampling": tool_runtime.probe.supports_assistant_resampling,
    }
    assert safety_runtime.loop_policy.to_dict() == {
        "tool_round_mode": safety_runtime.probe.tool_loop_mode,
        "max_assistant_activations": 3,
        "final_synthesis_without_tools": False,
        "single_user_turn": safety_runtime.probe.single_user_turn,
        "assistant_error_behavior": "fail_episode",
        "tool_error_behavior": "return_error_payload",
        "max_tool_calls_per_turn": None,
        "max_tool_response_attempts": 1,
        "assistant_resampling": safety_runtime.probe.supports_assistant_resampling,
    }
    assert financial_runtime.loop_policy.to_dict() == {
        "tool_round_mode": financial_runtime.probe.tool_loop_mode,
        "max_assistant_activations": financial_runtime.probe.tool_max_calls_per_turn + 2,
        "final_synthesis_without_tools": financial_runtime.probe.final_synthesis_without_tools,
        "single_user_turn": False,
        "assistant_error_behavior": "fail_episode",
        "tool_error_behavior": "return_error_payload",
        "max_tool_calls_per_turn": financial_runtime.probe.tool_max_calls_per_turn,
        "max_tool_response_attempts": 1,
        "assistant_resampling": financial_runtime.probe.supports_assistant_resampling,
    }


async def test_descriptor_is_deterministic_and_json_serializable(
    safety_runtime: ProbeEpisodeRuntime,
) -> None:
    first = (await safety_runtime.descriptor()).to_dict()
    second = (await safety_runtime.descriptor()).to_dict()
    tool_runtime = _tool_calling_runtime()
    tool_descriptor = (await tool_runtime.descriptor()).to_dict()

    assert first == second
    assert first["probe_type"] == "safety_agentic"
    assert first["allowed_tool_names"] == sorted(safety_runtime.allowed_tool_names)
    assert first["initial_user_message"]
    assert first["user_system_prompt"] == safety_runtime.probe.get_user_system_prompt()
    assert first["assistant_system_prompt"] == safety_runtime.probe.get_assistant_system_prompt()
    assert first["turn0_user_query_instruction"] == safety_runtime.probe.get_user_query_instruction(0)
    assert first["user_interaction_style"] == "direct"
    assert first["patience"] == 0.75
    assert first["user_turn_policy"] == {
        "context_compression": True,
        "wrap_up": True,
        "followup_anchor": None,
        "allowed_phrases": [],
        "script_check_ignores": [],
        "check_opening": "none",
    }
    assert tool_descriptor["user_system_prompt"]
    assert tool_descriptor["assistant_system_prompt"] == ""
    json.dumps(first)
    json.dumps(tool_descriptor)


async def test_runtime_delegates_participant_lifecycle_hooks(
    safety_runtime: ProbeEpisodeRuntime,
) -> None:
    opening = await safety_runtime.initial_user_message()
    await safety_runtime.synchronize_transcript(
        [
            {"role": "user", "content": opening},
            {"role": "assistant", "content": "I cannot do that."},
        ]
    )

    assert await safety_runtime.format_followup_user_instructions(1) == []
    assert await safety_runtime.is_capitulation_detected() is False
    assert await safety_runtime.should_succeed() is True
    extras = await safety_runtime.result_extras()
    assert extras["action_request_id"]
    evidence = await safety_runtime.evidence()
    assert evidence["transcript"][-1]["content"] == "I cannot do that."
    assert evidence["result_extras"] == extras


async def test_simulate_tool_call_requires_explicit_semantic_identity(
    safety_runtime: ProbeEpisodeRuntime,
) -> None:
    tool_name = next(iter(safety_runtime.allowed_tool_names))

    with pytest.raises(TypeError, match="tool_call_id"):
        await safety_runtime.simulate_tool_call(tool_name, {})  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="tool_call_id"):
        await safety_runtime.simulate_tool_call(tool_name, {}, tool_call_id="", turn_idx=0, call_idx=0)
    with pytest.raises(ValueError, match="turn_idx"):
        await safety_runtime.simulate_tool_call(tool_name, {}, tool_call_id="call-1", turn_idx=-1, call_idx=0)
    with pytest.raises(ValueError, match="call_idx"):
        await safety_runtime.simulate_tool_call(tool_name, {}, tool_call_id="call-1", turn_idx=0, call_idx=-1)


async def test_finalization_is_idempotent_and_blocks_mutation(
    safety_runtime: ProbeEpisodeRuntime,
) -> None:
    result = await safety_runtime.finalize()

    assert await safety_runtime.finalize() == result
    with pytest.raises(RuntimeError, match="finalized"):
        await safety_runtime.append_message({"role": "assistant", "content": "late"})
    with pytest.raises(RuntimeError, match="finalized"):
        await safety_runtime.simulate_tool_call(
            next(iter(safety_runtime.allowed_tool_names)),
            {},
            tool_call_id="late-call",
            turn_idx=0,
            call_idx=0,
        )


async def test_finalize_rejects_transcript_that_desynchronizes_tool_evidence(
    safety_runtime: ProbeEpisodeRuntime,
) -> None:
    tool_name = next(iter(safety_runtime.allowed_tool_names))
    payload = await safety_runtime.simulate_tool_call(
        tool_name,
        {},
        tool_call_id="call-1",
        turn_idx=2,
        call_idx=3,
    )

    with pytest.raises(ValueError, match="evidence"):
        await safety_runtime.finalize([])
    with pytest.raises(ValueError, match="evidence"):
        await safety_runtime.finalize([{"role": "tool", "tool_call_id": "call-1", "content": payload + "changed"}])


async def test_transcript_accepts_semantically_identical_json_tool_payload(
    safety_runtime: ProbeEpisodeRuntime,
) -> None:
    tool_name = next(iter(safety_runtime.allowed_tool_names))
    payload = await safety_runtime.simulate_tool_call(
        tool_name,
        {},
        tool_call_id="call-1",
        turn_idx=0,
        call_idx=0,
    )
    transcript = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": json.dumps(json.loads(payload), separators=(",", ":")),
        },
    ]

    await safety_runtime.synchronize_transcript(transcript)
