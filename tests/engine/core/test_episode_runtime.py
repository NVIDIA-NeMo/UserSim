# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for externally hosted, episode-scoped probe execution."""

from __future__ import annotations

import asyncio
import json
from inspect import signature
from types import SimpleNamespace

import pytest

from usersim.engine.config import ConversationSimulatorConfig
from usersim.engine.core.episode_runtime import (
    ActivationRequest,
    ActivationResult,
    ActivationUsage,
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
            probe_options={"tool_calling": {"max_tools": 1}},
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
    issued = False

    def respond(activation: ActivationRequest) -> dict[str, object]:
        nonlocal issued
        if activation.role == "assistant" and activation.tools_enabled and not issued:
            issued = True
            return _tool_call("get_weather", {"location": "Tokyo"}, "call-weather")
        return _text(activation)

    event, _seen = await _drive(runtime, respond)

    assert isinstance(event, EpisodeLifecycleComplete)
    executed = await runtime.executed_tool_calls()
    assert [call.tool_name for call in executed] == ["get_weather"]
    assert json.loads(executed[0].payload) == {"temperature_c": 22}
    assert await runtime.tool_result("call-weather") == executed[0].payload
    assert json.loads(event.result["conversation_metadata"])["tools_called"] == ["get_weather"]


async def test_a_rejected_tool_turn_is_requested_again_and_its_calls_stay_listed() -> None:
    """The host sees the rejected attempt's call run, so executed_tool_calls keeps it; the row keeps only the accepted one."""
    runtime = _tool_calling_runtime()
    runtime.config.max_assistant_attempts = 2
    runtime.config.max_turns = 1
    calls = 0
    rejected = False

    def respond(activation: ActivationRequest) -> dict[str, object]:
        nonlocal calls, rejected
        last = str(activation.messages[-1].get("content", "")) if activation.messages else ""
        sees_a_result = any(message.get("role") == "tool" for message in activation.messages)
        if activation.role == "assistant" and activation.tools_enabled and not sees_a_result:
            calls += 1
            return _tool_call("get_weather", {"location": "Tokyo"}, f"call-{calls}")
        if activation.role == "assistant":
            return {"role": "assistant", "content": "It is 22 degrees in Tokyo."}
        if activation.role == "judge" and "Judge the assistant's response" in last and not rejected:
            rejected = True
            return {
                "role": "assistant",
                "content": "<explanation>[UNGROUNDED] No tool returned that.</explanation><rating>failure</rating>",
            }
        return _text(activation)

    event, _seen = await _drive(runtime, respond)

    assert isinstance(event, EpisodeLifecycleComplete)
    executed = await runtime.executed_tool_calls()
    assert [call.tool_call_id for call in executed] == ["call-1", "call-2"]
    transcript = json.loads(event.result["conversation_messages"])
    assert [m["tool_call_id"] for m in transcript if m["role"] == "tool"] == ["call-2"]
    assert json.loads(event.result["conversation_metadata"])["tools_called"] == ["get_weather"]
    assert json.loads(event.result["simulation_outcome"])["n_assistant_retries"] == 1


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
    """The host's declared model id and token budget reach the runtime.

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


def test_native_lifecycle_surface_covers_all_registered_probes() -> None:
    """A host resumes an episode through ``state`` and ``seed_state``, so every
    registered probe's ``run_dispatch`` takes both, an override included. A probe
    defined here overrides it, so an override is checked whichever probes ship."""
    classes = [resolve_probe(label) for label in known_probes()] + [_KeepsConstructionModelsProbe]
    for cls in classes:
        parameters = signature(cls.run_dispatch).parameters
        assert "state" in parameters, cls.label
        assert "seed_state" in parameters, cls.label


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
