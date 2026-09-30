# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Behavioral parity between direct and externally resumed probe execution."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

from usersim.engine.config import ConversationSimulatorConfig
from usersim.engine.core.episode_runtime import (
    ActivationRequest,
    ActivationResult,
    EpisodeLifecycleComplete,
    ProbeEpisodeRuntime,
)
from usersim.engine.core.llm import get_current_outcome_builder, set_current_outcome_builder
from usersim.engine.core.probes import known_probes

_PERSONA = {
    "first_name": "Sarah",
    "last_name": "Johnson",
    "age": 35,
    "city": "Seattle",
    "state": "WA",
    "region": "Oregon",
    "occupation": "Software Engineer",
    "education_level": "Bachelor",
    "uuid": "parity-persona",
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


@dataclass(frozen=True)
class _Case:
    probe_type: str
    seed: int
    probe_data: Mapping[str, Any] | None = None
    max_turns: int = 1


_CASES = (
    _Case("general_open_ended", 1001),
    _Case("general_educational", 1002),
    _Case("tool_calling", 1003, {"tools": [_TOOL]}),
    _Case("sov_ai_facts", 1004),
    _Case("sov_ai_dynamic", 1005),
    _Case("sov_ai_multilingual_parity", 1006),
    _Case("safety_chat_pressure", 1007),
    _Case("safety_agentic", 42, {"user_interaction_style": "direct"}, 3),
    _Case("financial_services", 1009),
    _Case("health_general_disclosure", 1010, {"probe_variant": "default"}),
    _Case("health_therapy_disclosure", 1011, {"probe_variant": "default"}),
    _Case("health_triage_disclosure", 1012, {"probe_variant": "default"}),
    _Case("health_decision_support_disclosure", 1013, {"probe_variant": "default"}),
    _Case("identity_disclosure", 1014),
)


def _message_value(message: Any, name: str, default: Any = None) -> Any:
    if isinstance(message, Mapping):
        return message.get(name, default)
    return getattr(message, name, default)


def _arguments_for_schema(schema: Mapping[str, Any]) -> Any:
    if enum := schema.get("enum"):
        return enum[0]
    schema_type = schema.get("type")
    if schema_type == "object":
        required = set(schema.get("required", []))
        return {
            name: _arguments_for_schema(value)
            for name, value in schema.get("properties", {}).items()
            if name in required
        }
    return {
        "array": [],
        "boolean": True,
        "integer": 1,
        "number": 1,
        "string": "example",
    }.get(schema_type, "example")


class _ReplayableModel:
    """Return the same response for the same role, messages, and options."""

    model_name = "nvidia/nemotron-3-super-120b-a12b"

    def __init__(self, role: str, invocations: list[str]) -> None:
        self.role = role
        self.invocations = invocations

    async def acompletion(self, messages: Sequence[Any], **kwargs: Any) -> SimpleNamespace:
        self.invocations.append(self.role)
        tool_calls = None
        has_tool_result = any(str(_message_value(message, "role")) == "tool" for message in messages)
        if self.role == "assistant" and kwargs.get("tools") and not has_tool_result:
            function = kwargs["tools"][0]["function"]
            arguments = _arguments_for_schema(function.get("parameters", {}))
            tool_calls = [
                {
                    "id": "call-parity",
                    "type": "function",
                    "function": {
                        "name": function["name"],
                        "arguments": json.dumps(arguments, sort_keys=True),
                    },
                }
            ]
            content = ""
        else:
            content = {
                "api": '{"ok":true}',
                "assistant": "Scripted assistant response.",
                "judge": "<explanation>valid scripted turn</explanation><rating>success</rating>",
                "summary": "yes",
                "user": "Could you explain that a little more?",
            }[self.role]
        return SimpleNamespace(
            message=SimpleNamespace(
                content=content,
                reasoning_content=f"{self.role} reasoning",
                tool_calls=tool_calls,
            ),
            usage=None,
        )


def _runtime(case: _Case) -> tuple[ProbeEpisodeRuntime, list[str]]:
    data = {
        **dict(case.probe_data or {}),
        "persona": _PERSONA,
        "probe_type": case.probe_type,
        "theme": {
            "type": case.probe_type.replace("_", " "),
            "description": f"Exercise the {case.probe_type} probe.",
        },
    }
    config = ConversationSimulatorConfig(
        name="episode_runtime_parity",
        locale="en_US",
        max_turns=case.max_turns,
        random_seed=case.seed,
        tools_column="tools" if case.probe_type == "tool_calling" else None,
        finance_tier_mix=0.0,
        finance_tier="verifiable",
        finance_retrieval_mode="hybrid",
    )
    invocations: list[str] = []
    models = {
        "api_response_model": _ReplayableModel("api", invocations),
        "assistant_model": _ReplayableModel("assistant", invocations),
        "judge_model": _ReplayableModel("judge", invocations),
        "summary_model": _ReplayableModel("summary", invocations),
        "user_model": _ReplayableModel("user", invocations),
    }
    return (
        ProbeEpisodeRuntime(
            probe_type=case.probe_type,
            persona=_PERSONA,
            locale="en_US",
            language="English",
            models=models,
            config=config,
            data=data,
            profile=_PROFILE,
        ),
        invocations,
    )


def _response_dict(response: SimpleNamespace) -> dict[str, Any]:
    value = {
        "role": "assistant",
        "content": response.message.content or "",
        "reasoning_content": response.message.reasoning_content,
        "tool_calls": deepcopy(response.message.tool_calls),
    }
    return {key: item for key, item in value.items() if item is not None}


async def _execute_assistant_activation(
    runtime: ProbeEpisodeRuntime,
    activation: ActivationRequest,
    invocations: list[str],
) -> ActivationResult:
    model = _ReplayableModel("assistant", invocations)
    messages = list(activation.messages)
    response = _response_dict(await model.acompletion(messages, **activation.parameters))
    if not response.get("tool_calls"):
        return ActivationResult(activation_id=activation.activation_id, response=response)

    transcript = [response]
    # ToolCallingProbe's verifier records human turns one-based; the action
    # probes preserve their native zero-based assistant turn index.
    turn_index = (
        sum(1 for message in messages if str(_message_value(message, "role")) == "user")
        if runtime.probe_type == "tool_calling"
        else 0
    )
    for call_index, call in enumerate(response["tool_calls"]):
        function = call["function"]
        arguments = json.loads(function["arguments"])
        payload = await runtime.simulate_tool_call(
            function["name"],
            arguments,
            tool_call_id=call["id"],
            turn_idx=turn_index,
            call_idx=call_index,
        )
        transcript.append({"role": "tool", "content": payload, "tool_call_id": call["id"]})
    return ActivationResult(
        activation_id=activation.activation_id,
        transcript_delta=tuple(transcript),
    )


async def _run_external(
    runtime: ProbeEpisodeRuntime,
    invocations: list[str],
) -> tuple[dict[str, Any], list[str]]:
    event = await runtime.advance()
    roles: list[str] = []
    while isinstance(event, ActivationRequest):
        roles.append(event.role)
        if event.role == "assistant":
            result = await _execute_assistant_activation(runtime, event, invocations)
        else:
            response = await _ReplayableModel(event.role, invocations).acompletion(event.messages, **event.parameters)
            result = ActivationResult(
                activation_id=event.activation_id,
                response=_response_dict(response),
            )
        event = await runtime.advance(result)
    assert isinstance(event, EpisodeLifecycleComplete)
    return event.result, roles


async def _run_direct(runtime: ProbeEpisodeRuntime) -> dict[str, Any]:
    previous_builder = get_current_outcome_builder()
    set_current_outcome_builder(runtime.outcome)
    try:
        return await runtime.probe.run_dispatch(
            models=runtime.models,
            data=runtime.probe._data,
            cfg=runtime.config,
        )
    finally:
        set_current_outcome_builder(previous_builder)


def _normalized_result(result: Mapping[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(dict(result))
    for name in ("conversation_messages", "conversation_metadata", "simulation_outcome", "simulation_traces"):
        if isinstance(normalized.get(name), str):
            normalized[name] = json.loads(normalized[name])
    outcome = normalized["simulation_outcome"]
    outcome.pop("wall_clock_s", None)
    outcome.pop("wall_clock_s_by_alias", None)
    return normalized


_CORE_RESULT_FIELDS = {
    "conversation_messages",
    "conversation_metadata",
    "conversation_status",
    "simulation_outcome",
    "simulation_traces",
    "num_turns",
    "num_tool_calls",
}


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.probe_type)
async def test_direct_and_resumable_execution_have_deterministic_parity(case: _Case) -> None:
    """Replay one resolved episode through both execution paths and compare evidence."""
    direct_runtime, direct_invocations = _runtime(case)
    external_runtime, external_invocations = _runtime(case)

    direct = await _run_direct(direct_runtime)
    external, roles = await _run_external(external_runtime, external_invocations)
    direct_normalized = _normalized_result(direct)
    external_normalized = _normalized_result(external)

    assert direct_normalized == external_normalized
    assert direct_normalized["conversation_messages"] == external_normalized["conversation_messages"]
    assert direct_normalized["conversation_metadata"] == external_normalized["conversation_metadata"]
    assert direct_normalized["simulation_outcome"] == external_normalized["simulation_outcome"]
    assert direct_normalized["simulation_traces"] == external_normalized["simulation_traces"]
    assert {key: value for key, value in direct_normalized.items() if key not in _CORE_RESULT_FIELDS} == {
        key: value for key, value in external_normalized.items() if key not in _CORE_RESULT_FIELDS
    }
    assert direct_normalized["conversation_status"] is True
    assert direct_normalized["num_turns"] == external_normalized["num_turns"]
    assert direct_normalized["num_tool_calls"] == external_normalized["num_tool_calls"]
    assert direct_invocations == external_invocations
    assert roles == [role for role in external_invocations if role != "api"]
    assert {"user", "assistant"} <= {message["role"] for message in direct_normalized["conversation_messages"]}
    assert any(
        message.get("reasoning_content")
        for message in direct_normalized["conversation_messages"]
        if message["role"] == "assistant"
    )
    assert roles
    if direct_runtime.assistant_tools:
        tool_messages = [message for message in direct_normalized["conversation_messages"] if message["role"] == "tool"]
        assert tool_messages, f"{case.probe_type} did not exercise its registered tool evidence"


def test_parity_cases_cover_the_complete_registry() -> None:
    assert {case.probe_type for case in _CASES} == set(known_probes())
