# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Tests for externally hosted, episode-scoped probe execution."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from inspect import signature
from types import SimpleNamespace

import pytest

from usersim.engine.config import ConversationSimulatorConfig
from usersim.engine.core.episode_runtime import (
    ActivationRequest,
    ActivationResult,
    ActivationUsage,
    EpisodeContractError,
    EpisodeLifecycleComplete,
    HostRoleModel,
    ProbeEpisodeRuntime,
)
from usersim.engine.core.llm import acall_llm
from usersim.engine.core.probes import _PROBE_REGISTRY, BaseProbe, known_probes, resolve_probe


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


class _AssistantTextModel:
    model_name = "assistant-text-model"

    async def acompletion(self, messages, **kwargs):
        del messages, kwargs
        return SimpleNamespace(
            message=SimpleNamespace(
                content="I cannot perform that action.",
                reasoning_content=None,
                tool_calls=None,
            ),
            usage=None,
        )


class _KeepsConstructionModelsProbe(BaseProbe):
    """Calls a model it kept from construction, as a probe author may."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._kept_models = dict(kwargs["models"])

    async def run_dispatch(self, *, models, data, cfg, state=None, seed_state=True):
        del models, data, cfg, state, seed_state
        reply = await acall_llm(self._kept_models, "assistant_model", [{"role": "user", "content": "Hello"}])
        return {"assistant_reply": reply["content"]}


def _tool_calling_runtime() -> ProbeEpisodeRuntime:
    return ProbeEpisodeRuntime(
        probe_type="tool_calling",
        persona={"first_name": "A", "last_name": "User", "age": 35},
        locale="en_US",
        language="English",
        models={"api_response_model": _JSONModel()},
        config=ConversationSimulatorConfig(
            name="episode_runtime_test",
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
        config=ConversationSimulatorConfig(
            name="episode_runtime_test",
            random_seed=1,
            max_turns=1,
            context_compression=False,
            finance_tier_mix=0.0,
            finance_tier="verifiable",
            finance_retrieval_mode="golden",
            finance_embedding_model_alias="embedding_model",
        ),
        data={},
    )


def _open_ended_runtime() -> ProbeEpisodeRuntime:
    return ProbeEpisodeRuntime(
        probe_type="general_open_ended",
        persona={
            "first_name": "A",
            "last_name": "User",
            "age": 35,
            "region": "Oregon",
            "occupation": "designer",
        },
        locale="en_US",
        language="English",
        models={},
        config=ConversationSimulatorConfig(
            name="episode_runtime_test",
            theme_column="theme",
            max_turns=2,
            max_query_attempts=1,
            max_assistant_attempts=1,
            enforce_user_language=False,
            context_compression=False,
            compression_window=1,
            store_reasoning=True,
        ),
        data={
            "theme": {"type": "travel", "description": "planning a trip"},
            "disclosure_style": "upfront",
            "user_interaction_style": "neutral",
            "persona_grounding": False,
        },
        profile={"patience": 0.5, "tech_literacy": 0.5, "error_proneness": 0.5},
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


_DEFAULT_CONTENT = {
    "user": "How should I plan my trip?",
    "assistant": "I cannot perform that action.",
    "judge": "<explanation>valid participant turn</explanation><rating>success</rating>",
    "summary": "yes",
}


def _text(activation: ActivationRequest) -> dict[str, object]:
    return {"role": "assistant", "content": _DEFAULT_CONTENT[activation.role]}


def _tool_call(name: str, arguments: dict[str, object], call_id: str) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


async def _drive(runtime: ProbeEpisodeRuntime, respond=_text) -> tuple[object, list[ActivationRequest]]:
    """Drive an episode to completion, answering each request with ``respond``."""
    seen: list[ActivationRequest] = []
    event = await runtime.advance()
    while isinstance(event, ActivationRequest):
        seen.append(event)
        event = await runtime.advance(ActivationResult(activation_id=event.activation_id, response=respond(event)))
    return event, seen


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


async def test_each_assistant_request_states_whether_tools_are_offered() -> None:
    """The host is told what comes next instead of modelling the loop itself."""
    runtime = _tool_calling_runtime()
    offered: list[tuple[bool, bool]] = []

    def respond(activation: ActivationRequest) -> dict[str, object]:
        if activation.role != "assistant":
            return _text(activation)
        offered.append((activation.tools_enabled, activation.continues_turn))
        if activation.tools_enabled:
            return _tool_call("get_weather", {"location": "Tokyo"}, f"call-{len(offered)}")
        return {"role": "assistant", "content": "It is 22 degrees in Tokyo."}

    event, _seen = await _drive(runtime, respond)

    assert isinstance(event, EpisodeLifecycleComplete)
    # Tools on for the first assistant call of the turn, off for the answer
    # step that follows it, which is marked as continuing the same turn.
    assert (True, False) in offered
    assert (False, True) in offered


async def test_recorded_tool_calls_run_in_the_probes_own_loop() -> None:
    """The native verifier and API-response model produce the payload."""
    runtime = _tool_calling_runtime()

    def respond(activation: ActivationRequest) -> dict[str, object]:
        if activation.role == "assistant" and activation.tools_enabled:
            return _tool_call("get_weather", {"location": "Tokyo"}, "call-weather")
        return _text(activation)

    event, _seen = await _drive(runtime, respond)

    assert isinstance(event, EpisodeLifecycleComplete)
    executed = await runtime.executed_tool_calls()
    assert [call.tool_name for call in executed] == ["get_weather"]
    assert json.loads(executed[0].payload) == {"temperature_c": 22}
    assert await runtime.tool_result("call-weather") == executed[0].payload
    assert json.loads(event.result["conversation_metadata"])["tools_called"] == ["get_weather"]


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


async def test_finalized_row_carries_the_loop_written_columns() -> None:
    """The hosted row is the resolved row updated with the result, as standalone."""
    runtime = _open_ended_runtime()

    event, _seen = await _drive(runtime)

    assert isinstance(event, EpisodeLifecycleComplete)
    row = await runtime.finalize()
    assert row == event.result
    # Written by the conversation loop onto the row, not by the dispatch result.
    assert row["user_query"]
    assert row["trajectory_id"] == runtime.preamble.trajectory_id
    assert row["persona_uuid"] == runtime.preamble.persona_uuid


def test_activation_result_requires_a_json_safe_assistant_response() -> None:
    with pytest.raises(TypeError, match="JSON serializable"):
        ActivationResult(activation_id="activation-1", response={"bad": object()})
    with pytest.raises(ValueError, match="role must be 'assistant'"):
        ActivationResult(activation_id="activation-1", response={"role": "user", "content": "x"})
    with pytest.raises(ValueError, match="activation_id"):
        ActivationResult(activation_id="", response={"role": "assistant", "content": "x"})
    with pytest.raises(ValueError, match="requires a 'response' mapping"):
        ActivationResult.from_value({"activation_id": "activation-1"})


async def test_host_role_model_identity_and_budget_reach_the_runtime() -> None:
    """Item 3 of #10: the host's real model id and token budget are UserSim's.

    Trajectory identity is keyed on the resolved model id, and a probe that
    grades the assistant's self-description cannot work without it.
    """
    models = {
        "user_model": HostRoleModel(model_name="host/user-model", max_tokens=256),
        "assistant_model": HostRoleModel(model_name="host/assistant-model", max_tokens=512),
        "judge_model": HostRoleModel(model_name="host/judge-model"),
        "summary_model": HostRoleModel(model_name="host/summary-model"),
    }
    runtime = ProbeEpisodeRuntime(
        probe_type="general_open_ended",
        persona={"first_name": "A", "last_name": "User", "age": 35, "region": "Oregon", "occupation": "designer"},
        locale="ja_JP",
        language="Japanese",
        models=models,
        config=ConversationSimulatorConfig(
            name="episode_runtime_test",
            locale="ja_JP",
            theme_column="theme",
            max_turns=1,
            max_query_attempts=1,
            max_assistant_attempts=1,
            enforce_user_language=False,
            context_compression=False,
        ),
        data={"theme": {"type": "travel", "description": "planning a trip"}},
        profile={"patience": 0.5, "tech_literacy": 0.5, "error_proneness": 0.5},
    )

    # The id names the host's models, not UserSim's aliases.
    from usersim.engine.core.identity import trajectory_id as compute_trajectory_id

    assert runtime.preamble.trajectory_id == compute_trajectory_id(
        persona_uuid=runtime.preamble.persona_uuid,
        probe_family=runtime.preamble.probe_family,
        probe_variant=runtime.preamble.probe_variant,
        scenario_seed=runtime.config.random_seed,
        user_model="host/user-model",
        assistant_model="host/assistant-model",
        prompt_version="v1.0",
        extra_keys=[("locale", "ja_JP")],
    )

    _event, seen = await _drive(runtime)

    assistant_requests = [request for request in seen if request.role == "assistant"]
    assert assistant_requests
    # ja_JP needs more tokens per semantic unit, so UserSim scales the budget
    # the host declared exactly as it would a configured model's.
    assert assistant_requests[0].parameters["max_tokens"] == 1024


async def test_identity_disclosure_identifies_the_hosts_assistant_model() -> None:
    """Without the host's model id this probe cannot grade its row at all."""
    runtime = ProbeEpisodeRuntime(
        probe_type="identity_disclosure",
        persona={"first_name": "A", "last_name": "User", "age": 35, "region": "Oregon", "occupation": "designer"},
        locale="en_US",
        language="English",
        models={
            "assistant_model": HostRoleModel(model_name="nvidia/llama-3.1-nemotron-70b-instruct"),
            "user_model": HostRoleModel(model_name="host/user-model"),
        },
        config=ConversationSimulatorConfig(name="episode_runtime_test", random_seed=5, max_turns=1),
        data={},
        profile={"patience": 0.5, "tech_literacy": 0.5, "error_proneness": 0.5},
    )

    extras = runtime.probe.build_result_extras(runtime.state)

    assert extras["expected_identity"], "the probe must resolve an expected developer to grade against"
    assert "nvidia" in json.dumps(extras["expected_identity"]).lower()


async def test_models_kept_from_construction_are_answered_by_the_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hosted probe is constructed with the models its dispatch receives.

    A model the probe keeps from construction is therefore answered by the
    host, as the configured model would answer it in a standalone run.
    """
    monkeypatch.setitem(_PROBE_REGISTRY, "keeps_construction_models", _KeepsConstructionModelsProbe)
    runtime = ProbeEpisodeRuntime(
        probe_type="keeps_construction_models",
        persona={"first_name": "A", "last_name": "User", "age": 35},
        locale="en_US",
        language="English",
        models={"assistant_model": HostRoleModel(model_name="host/assistant-model")},
        config=ConversationSimulatorConfig(name="episode_runtime_test"),
        data={},
    )

    request = await runtime.advance()

    assert isinstance(request, ActivationRequest)
    assert request.role == "assistant"
    assert request.messages == ({"role": "user", "content": "Hello"},)
    complete = await runtime.advance(
        ActivationResult(activation_id=request.activation_id, response={"role": "assistant", "content": "Hi"})
    )
    assert isinstance(complete, EpisodeLifecycleComplete)
    assert complete.result["assistant_reply"] == "Hi"


async def test_recorded_usage_reaches_the_outcome() -> None:
    """Usage the host observed is UserSim's per-model accounting."""
    runtime = _open_ended_runtime()

    event = await runtime.advance()
    while isinstance(event, ActivationRequest):
        event = await runtime.advance(
            ActivationResult(
                activation_id=event.activation_id,
                response={
                    "role": "assistant",
                    "content": _DEFAULT_CONTENT[event.role],
                    "reasoning_content": f"{event.role} reasoning",
                },
                usage=ActivationUsage(input_tokens=11, output_tokens=7),
            )
        )

    assert isinstance(event, EpisodeLifecycleComplete)
    outcome = json.loads(event.result["simulation_outcome"])
    per_model_input = outcome["per_model_input_tokens"]
    per_model_output = outcome["per_model_output_tokens"]
    assert per_model_input, "per-model token accounting must be populated from recorded usage"
    assert all(value % 11 == 0 for value in per_model_input.values())
    assert all(value % 7 == 0 for value in per_model_output.values())
    assert any("reasoning" in json.dumps(message) for message in json.loads(event.result["conversation_messages"]))


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


async def test_concurrent_episodes_stay_isolated() -> None:
    """Episodes share a process; they must not share state."""
    runtimes = [_open_ended_runtime() for _ in range(4)]

    results = await asyncio.gather(*(_drive(runtime) for runtime in runtimes))

    rows = [event.result for event, _seen in results]
    assert all(isinstance(event, EpisodeLifecycleComplete) for event, _seen in results)
    assert all(row["conversation_status"] is True for row in rows)
    # Same inputs, so the same identity, and each episode produced its own row.
    assert len({row["trajectory_id"] for row in rows}) == 1
    assert all(row["conversation_messages"] == rows[0]["conversation_messages"] for row in rows)


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


def test_native_lifecycle_surface_covers_all_registered_probes() -> None:
    registered = (
        "financial_services",
        "general_educational",
        "general_open_ended",
        "health_decision_support_disclosure",
        "health_general_disclosure",
        "health_therapy_disclosure",
        "health_triage_disclosure",
        "identity_disclosure",
        "safety_agentic",
        "safety_chat_pressure",
        "sov_ai_dynamic",
        "sov_ai_facts",
        "sov_ai_multilingual_parity",
        "tool_calling",
    )
    assert known_probes() == registered
    for label in registered:
        parameters = signature(resolve_probe(label).run_dispatch).parameters
        assert "state" in parameters, label
        assert "seed_state" in parameters, label


async def test_native_lifecycle_routes_all_participant_roles() -> None:
    runtime = _open_ended_runtime()
    user_turns = iter(("How should I plan my trip?", "Thanks, that answers it."))

    def respond(activation: ActivationRequest) -> dict[str, object]:
        if activation.role == "user":
            return {"role": "assistant", "content": next(user_turns, "Thanks, that answers it.")}
        if activation.role == "assistant":
            return {"role": "assistant", "content": "Start with dates, budget, and destination preferences."}
        return _text(activation)

    event, seen = await _drive(runtime, respond)

    assert isinstance(event, EpisodeLifecycleComplete)
    assert {request.role for request in seen} == {"user", "assistant", "judge", "summary"}
    assert event.result["conversation_status"] is True
