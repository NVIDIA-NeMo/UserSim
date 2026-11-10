# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""safety_agentic run by a host: its tools, its own tool loop, and a lifecycle a host can pause, resume and close."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest
from engine.core.test_episode_runtime import _AssistantTextModel, _drive, _text, _tool_call, _tool_calling_runtime

from usersim.engine.config import ConversationSimulatorConfig
from usersim.engine.core.episode_runtime import (
    ActivationRequest,
    ActivationResult,
    EpisodeContractError,
    EpisodeLifecycleComplete,
    ProbeEpisodeRuntime,
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
        config=ConversationSimulatorConfig(name="episode_runtime_test", random_seed=42, max_turns=3),
        data={"user_interaction_style": "direct"},
        profile={"patience": 0.75},
    )


def test_runtime_exposes_only_resolved_episode_tools(safety_runtime: ProbeEpisodeRuntime) -> None:
    schemas = safety_runtime.assistant_tools
    assert schemas
    schemas.clear()
    assert safety_runtime.assistant_tools


async def test_descriptor_is_deterministic_and_json_serializable(
    safety_runtime: ProbeEpisodeRuntime,
) -> None:
    first = (await safety_runtime.descriptor()).to_dict()
    second = (await safety_runtime.descriptor()).to_dict()
    tool_descriptor = (await _tool_calling_runtime().descriptor()).to_dict()

    assert first == second
    # Every per-call decision rides on the activation request, so the
    # descriptor stays this small.
    assert set(first) == {"probe_type", "assistant_tools"}
    assert first["probe_type"] == "safety_agentic"
    assert tool_descriptor["probe_type"] == "tool_calling"
    json.dumps(first)
    json.dumps(tool_descriptor)


async def test_tool_payloads_reach_the_host_verbatim(safety_runtime: ProbeEpisodeRuntime) -> None:
    """Payloads pass through unparsed; several safety mock responses are not JSON."""
    tool_name = safety_runtime.assistant_tools[0]["function"]["name"]

    def respond(activation: ActivationRequest) -> dict[str, object]:
        if activation.role == "assistant" and activation.tools_enabled:
            return _tool_call(tool_name, {"value": "example"}, "call-plain")
        return _text(activation)

    event, _seen = await _drive(safety_runtime, respond)

    assert isinstance(event, EpisodeLifecycleComplete)
    executed = await safety_runtime.executed_tool_calls()
    assert executed
    transcript = json.loads(event.result["conversation_messages"])
    tool_messages = [message for message in transcript if message["role"] == "tool"]
    assert tool_messages
    assert tool_messages[0]["content"] == executed[0].payload
    assert await safety_runtime.tool_result(executed[0].tool_call_id) == executed[0].payload


async def test_recording_an_unoffered_tool_is_rejected(safety_runtime: ProbeEpisodeRuntime) -> None:
    activation = await safety_runtime.advance()
    assert isinstance(activation, ActivationRequest)
    before = await safety_runtime.evidence()

    with pytest.raises(EpisodeContractError, match="not offered"):
        await safety_runtime.advance(
            ActivationResult(
                activation_id=activation.activation_id,
                response=_tool_call("not_this_episode", {}, "call-1"),
            )
        )

    assert await safety_runtime.evidence() == before
    await safety_runtime.close()


async def test_recorded_tool_calls_need_unique_non_empty_ids(safety_runtime: ProbeEpisodeRuntime) -> None:
    activation = await safety_runtime.advance()
    assert isinstance(activation, ActivationRequest)
    tool_name = safety_runtime.assistant_tools[0]["function"]["name"]

    with pytest.raises(EpisodeContractError, match="non-empty id"):
        await safety_runtime.advance(
            ActivationResult(activation_id=activation.activation_id, response=_tool_call(tool_name, {}, ""))
        )
    duplicate = _tool_call(tool_name, {}, "dup")
    duplicate["tool_calls"] = [duplicate["tool_calls"][0], deepcopy(duplicate["tool_calls"][0])]
    with pytest.raises(EpisodeContractError, match="repeated"):
        await safety_runtime.advance(ActivationResult(activation_id=activation.activation_id, response=duplicate))
    await safety_runtime.close()


async def test_finalize_requires_a_completed_native_lifecycle(
    safety_runtime: ProbeEpisodeRuntime,
) -> None:
    """Only UserSim's own dispatch produces a result; the host cannot synthesize one."""
    with pytest.raises(EpisodeContractError, match="has not completed"):
        await safety_runtime.finalize()


async def test_native_lifecycle_pauses_uniformly_and_is_idempotent(
    safety_runtime: ProbeEpisodeRuntime,
) -> None:
    activation = await safety_runtime.advance()

    assert isinstance(activation, ActivationRequest)
    assert activation.role == "assistant"
    assert activation.model_alias == "assistant_model"
    assert activation.messages[-1]["role"] == "user"
    assert activation.continues_turn is False
    json.dumps(activation.to_dict())

    result = ActivationResult(
        activation_id=activation.activation_id,
        response={"role": "assistant", "content": "I cannot perform that action."},
    )
    completed = await safety_runtime.advance(result)

    assert isinstance(completed, EpisodeLifecycleComplete)
    assert completed.result["conversation_status"] is True
    assert json.loads(completed.result["conversation_messages"])[-1]["content"] == "I cannot perform that action."
    assert await safety_runtime.advance(result) == completed
    with pytest.raises(EpisodeContractError, match="different result"):
        await safety_runtime.advance(
            ActivationResult(
                activation_id=activation.activation_id,
                response={"role": "assistant", "content": "changed"},
            )
        )


async def test_close_cancels_a_running_episode(safety_runtime: ProbeEpisodeRuntime) -> None:
    activation = await safety_runtime.advance()
    assert isinstance(activation, ActivationRequest)

    await safety_runtime.close()

    assert safety_runtime._lifecycle_task is not None
    assert safety_runtime._lifecycle_task.done()
    with pytest.raises(EpisodeContractError, match="closed"):
        await safety_runtime.advance(
            ActivationResult(activation_id=activation.activation_id, response={"role": "assistant", "content": "x"})
        )


async def test_activation_lifecycle_matches_native_safety_dispatch() -> None:
    external = ProbeEpisodeRuntime(
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
        config=ConversationSimulatorConfig(name="episode_runtime_test", random_seed=42, max_turns=3),
        data={"user_interaction_style": "direct"},
        profile={"patience": 0.75},
    )
    native = ProbeEpisodeRuntime(
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
        models={"assistant_model": _AssistantTextModel()},
        config=ConversationSimulatorConfig(name="episode_runtime_test", random_seed=42, max_turns=3),
        data={"user_interaction_style": "direct"},
        profile={"patience": 0.75},
    )

    activation = await external.advance()
    assert isinstance(activation, ActivationRequest)
    external_complete = await external.advance(
        ActivationResult(
            activation_id=activation.activation_id,
            response={"role": "assistant", "content": "I cannot perform that action."},
        )
    )
    native_result = await native.probe.run_dispatch(
        models=native.models,
        data=native.probe._data,
        cfg=native.config,
    )

    assert isinstance(external_complete, EpisodeLifecycleComplete)
    for key in ("conversation_messages", "conversation_metadata", "conversation_status", "num_turns", "num_tool_calls"):
        assert external_complete.result[key] == native_result[key]
