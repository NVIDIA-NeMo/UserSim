# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Behavioral parity between direct and externally resumed probe execution."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

from usersim.engine.config import ConversationSimulatorConfig
from usersim.engine.core.episode_runtime import (
    ActivationRequest,
    ActivationResult,
    EpisodeContractError,
    EpisodeLifecycleComplete,
    ProbeEpisodeRuntime,
)
from usersim.engine.core.probes import known_probes, resolve_probe
from usersim.engine.generator import ConversationSimulatorGenerator

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


def _plain_message(message: Any) -> dict[str, Any]:
    """Normalize a chat message to a comparable dict, however it was supplied."""
    if isinstance(message, Mapping):
        return deepcopy(dict(message))
    value: dict[str, Any] = {
        "role": str(getattr(getattr(message, "role", "user"), "value", getattr(message, "role", "user"))),
        "content": getattr(message, "content", "") or "",
    }
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        value["tool_calls"] = [
            call.model_dump() if hasattr(call, "model_dump") else deepcopy(call) for call in tool_calls
        ]
    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id:
        value["tool_call_id"] = str(tool_call_id)
    return value


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

    def __init__(self, role: str, invocations: list[str], prompts: list[dict[str, Any]] | None = None) -> None:
        self.role = role
        self.invocations = invocations
        self.prompts = prompts if prompts is not None else []

    async def acompletion(self, messages: Sequence[Any], **kwargs: Any) -> SimpleNamespace:
        self.invocations.append(self.role)
        self.prompts.append(
            {
                "role": self.role,
                "messages": [_plain_message(message) for message in messages],
                "tools": deepcopy(kwargs.get("tools")),
            }
        )
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


def _activation_kwargs(activation: ActivationRequest) -> dict[str, Any]:
    """Reassemble the call options UserSim would have passed to the model."""
    kwargs = dict(activation.parameters)
    if activation.tools:
        kwargs["tools"] = [deepcopy(tool) for tool in activation.tools]
    return kwargs


async def _run_external(
    runtime: ProbeEpisodeRuntime,
    invocations: list[str],
    prompts: list[dict[str, Any]] | None = None,
    model_factory: Any = None,
) -> tuple[dict[str, Any], list[str]]:
    """Drive the episode the way a host does: record each response, nothing else."""
    build = model_factory or (lambda role: _ReplayableModel(role, invocations, prompts))
    event = await runtime.advance()
    roles: list[str] = []
    recorded_call_ids: set[str] = set()
    while isinstance(event, ActivationRequest):
        roles.append(event.role)
        response = await build(event.role).acompletion(list(event.messages), **_activation_kwargs(event))
        recorded = _response_dict(response)
        recorded_call_ids.update(call["id"] for call in recorded.get("tool_calls") or [])
        event = await runtime.advance(ActivationResult(activation_id=event.activation_id, response=recorded))
    assert isinstance(event, EpisodeLifecycleComplete)
    # UserSim ran the recorded calls its own loop accepted -- a probe may cap
    # how many it executes per turn -- and every one of those is retrievable.
    executed = await runtime.executed_tool_calls()
    assert {call.tool_call_id for call in executed} <= recorded_call_ids
    for call in executed:
        assert await runtime.tool_result(call.tool_call_id) == call.payload
    return event.result, roles


async def _run_direct(runtime: ProbeEpisodeRuntime) -> dict[str, Any]:
    registry = SimpleNamespace(get_model=lambda *, model_alias: runtime.models[model_alias])
    provider = SimpleNamespace(model_registry=registry)
    generator = ConversationSimulatorGenerator(runtime.config, provider)
    return await generator.agenerate(runtime.preamble.to_row())


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


def _seeded_probe_variant(probe_type: str) -> str:
    """The variant the dispatcher seeds before the probe runs (main's rule)."""
    module = sys.modules[resolve_probe(probe_type).__module__]
    variants = getattr(module, "PROBE_VARIANTS", None)
    if variants and isinstance(variants, (list, tuple)):
        return str(variants[0])
    return "default"


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.probe_type)
async def test_direct_and_resumable_execution_have_deterministic_parity(case: _Case) -> None:
    """Replay one resolved episode through both execution paths and compare evidence."""
    direct_runtime, direct_invocations = _runtime(case)
    external_runtime, external_invocations = _runtime(case)

    direct_row = await _run_direct(direct_runtime)
    external, roles = await _run_external(external_runtime, external_invocations)
    for field in (
        "persona_uuid",
        "locale",
        "behavioral_profile",
        "disclosure_style",
        "user_interaction_style",
        "persona_grounding",
        "probe_family",
        "trajectory_id",
        "usersim_provenance",
    ):
        assert direct_row[field] == direct_runtime.preamble.to_row()[field]
    # A probe that discovers its own variant reports it on the finished row, but
    # identity stays keyed on the dispatcher-seeded variant so resume and
    # deduplication keep working. Changing this changes every stored trajectory_id.
    assert direct_runtime.preamble.to_row()["probe_variant"] == _seeded_probe_variant(case.probe_type)
    direct = {key: direct_row[key] for key in external}
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


class _AlwaysToolModel(_ReplayableModel):
    """Issue tool calls on every activation that offers tools.

    The single-round scripted model cannot see cross-turn divergence: tool
    indices and the tool simulator's context only drift once a second round
    happens on a later turn.
    """

    async def acompletion(self, messages: Sequence[Any], **kwargs: Any) -> SimpleNamespace:
        self.invocations.append(self.role)
        self.prompts.append(
            {
                "role": self.role,
                "messages": [_plain_message(message) for message in messages],
                "tools": deepcopy(kwargs.get("tools")),
            }
        )
        tools = kwargs.get("tools")
        if self.role == "assistant" and tools:
            function = tools[0]["function"]
            call_index = sum(1 for item in self.invocations if item == "assistant")
            return SimpleNamespace(
                message=SimpleNamespace(
                    content="",
                    reasoning_content="assistant reasoning",
                    tool_calls=[
                        {
                            "id": f"call-round-{call_index}",
                            "type": "function",
                            "function": {
                                "name": function["name"],
                                "arguments": json.dumps(
                                    _arguments_for_schema(function.get("parameters", {})), sort_keys=True
                                ),
                            },
                        }
                    ],
                ),
                usage=None,
            )
        content = {
            "api": '{"ok":true}',
            "assistant": "Scripted assistant response.",
            "judge": "<explanation>valid scripted turn</explanation><rating>success</rating>",
            "summary": "yes",
            "user": "Could you explain that a little more?",
        }[self.role]
        return SimpleNamespace(
            message=SimpleNamespace(content=content, reasoning_content=f"{self.role} reasoning", tool_calls=None),
            usage=None,
        )


_MULTI_ROUND_CASES = (
    _Case("tool_calling", 2003, {"tools": [_TOOL]}, 3),
    _Case("financial_services", 2009, None, 3),
    _Case("safety_agentic", 2042, {"user_interaction_style": "direct"}, 3),
)


@pytest.mark.parametrize("case", _MULTI_ROUND_CASES, ids=lambda case: case.probe_type)
async def test_multi_turn_multi_round_tool_use_matches_standalone(case: _Case) -> None:
    """Tools on every turn: hosted and standalone agree on results and on prompts.

    This is the case single-round parity cannot see. Executing tool calls
    outside UserSim's loop made the tool simulator's context and the recorded
    ``turn_idx`` diverge from a standalone run from the second round onward.
    """
    direct_runtime, direct_invocations = _runtime(case)
    external_runtime, external_invocations = _runtime(case)
    direct_prompts: list[dict[str, Any]] = []
    external_prompts: list[dict[str, Any]] = []

    direct_runtime.models = {
        alias: _AlwaysToolModel(model.role, direct_invocations, direct_prompts)
        for alias, model in direct_runtime.models.items()
    }
    direct_row = await _run_direct(direct_runtime)

    # Models UserSim calls directly in the hosted run too (the tool-response
    # simulator) must record into the hosted prompt log, or the comparison is
    # only checking the halves the harness happens to drive.
    external_runtime.models = {
        alias: _AlwaysToolModel(model.role, external_invocations, external_prompts)
        for alias, model in external_runtime.models.items()
    }
    external, _roles = await _run_external(
        external_runtime,
        external_invocations,
        external_prompts,
        model_factory=lambda role: _AlwaysToolModel(role, external_invocations, external_prompts),
    )

    assert direct_invocations == external_invocations
    # The prompt each model saw, in order, including the assistant's own
    # tool-call line in the tool simulator's context.
    assert direct_prompts == external_prompts
    assert _normalized_result({key: direct_row[key] for key in external}) == _normalized_result(external)

    executed = await external_runtime.executed_tool_calls()
    assert executed, "the multi-round case must execute tool calls"
    # The indices the host observes are the loop's own, not a host-side
    # counter. Calibrate against the standalone run rather than a fixed
    # number: whatever rounds it spanned, the hosted run spans the same.
    direct_tool_turns = [
        trace["turn_idx"]
        for trace in json.loads(direct_row["simulation_traces"])
        if trace.get("kind") == "tool_call_verifier" and trace.get("turn_idx") is not None
    ]
    if direct_tool_turns:
        assert [call.turn_idx for call in executed] == direct_tool_turns
        assert len({call.turn_idx for call in executed}) == len(set(direct_tool_turns))


async def test_recording_tool_calls_when_tools_are_off_is_rejected() -> None:
    """A record that breaks the probe's loop rules fails at the host-facing call.

    ``tool_calling`` disables tools for its answer step. Keeping them on is a
    host bug: it must not consume a model retry or be attributed to the model
    under test.
    """
    runtime, invocations = _runtime(_Case("tool_calling", 3003, {"tools": [_TOOL]}, 1))

    event = await runtime.advance()
    answer_step: ActivationRequest | None = None
    while isinstance(event, ActivationRequest):
        if event.role == "assistant" and not event.tools_enabled:
            answer_step = event
            break
        response = _response_dict(
            await _ReplayableModel(event.role, invocations).acompletion(
                list(event.messages), **_activation_kwargs(event)
            )
        )
        event = await runtime.advance(ActivationResult(activation_id=event.activation_id, response=response))

    assert answer_step is not None, "tool_calling must reach a tools-disabled answer step"
    assert answer_step.continues_turn is True

    before = len([role for role in invocations if role == "assistant"])
    with pytest.raises(EpisodeContractError, match="offers no tools"):
        await runtime.advance(
            ActivationResult(
                activation_id=answer_step.activation_id,
                response={
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-illegal",
                            "type": "function",
                            "function": {"name": _TOOL["function"]["name"], "arguments": "{}"},
                        }
                    ],
                },
            )
        )
    # No retry was issued, and the episode is still waiting on the same step.
    assert len([role for role in invocations if role == "assistant"]) == before
    await runtime.close()


async def test_the_multi_round_suite_actually_spans_more_than_one_round() -> None:
    """Pin the coverage the single-round parity suite was missing.

    Divergence between hosted and standalone tool indices only appears from the
    second round onward, so a suite that silently degraded to one round per
    episode would stop testing the thing it exists to test.
    """
    runtime, invocations = _runtime(_Case("financial_services", 2009, None, 3))
    runtime.models = {alias: _AlwaysToolModel(model.role, invocations, []) for alias, model in runtime.models.items()}
    await _run_external(
        runtime,
        invocations,
        [],
        model_factory=lambda role: _AlwaysToolModel(role, invocations, []),
    )

    executed = await runtime.executed_tool_calls()
    assert len({call.turn_idx for call in executed}) > 1, (
        f"expected tool calls across more than one round, saw turn indices "
        f"{sorted({call.turn_idx for call in executed})}"
    )
