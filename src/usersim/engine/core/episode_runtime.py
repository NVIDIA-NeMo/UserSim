# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Externally hosted, episode-scoped execution for tool-using probes."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence

from usersim.engine.config import ConversationSimulatorConfig
from usersim.engine.core.llm import get_current_outcome_builder, set_current_outcome_builder
from usersim.engine.core.outcomes import OutcomeBuilder, OutcomeStatus, Provenance
from usersim.engine.core.probes import BaseProbe, resolve_probe
from usersim.engine.core.simulation import ConversationState, make_result
from usersim.engine.core.user_turn_policy import resolve_user_turn_policy

_EXTERNAL_RUNTIME_PROBES = frozenset({"tool_calling", "safety_agentic", "financial_services"})


@dataclass(frozen=True)
class AssistantToolLoopPolicy:
    """Serializable instructions for an external assistant/tool-loop host."""

    tool_round_mode: Literal["single", "multi"]
    max_assistant_activations: int
    final_synthesis_without_tools: bool
    single_user_turn: bool
    assistant_error_behavior: Literal["fail_episode"]
    tool_error_behavior: Literal["return_error_payload"]
    max_tool_response_attempts: int
    assistant_resampling: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable copy."""
        return asdict(self)


@dataclass(frozen=True)
class UserTurnPolicySnapshot:
    """JSON-safe subset used to reconstruct UserSim's outer-loop policy."""

    context_compression: bool
    wrap_up: bool
    followup_anchor: str | None
    allowed_phrases: tuple[str, ...]
    script_check_ignores: tuple[str, ...]
    check_opening: Literal["none"]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable copy."""
        data = asdict(self)
        data["allowed_phrases"] = list(self.allowed_phrases)
        data["script_check_ignores"] = list(self.script_check_ignores)
        return data


@dataclass(frozen=True)
class ProbeRuntimeDescriptor:
    """Stable, JSON-serializable description of one resolved episode."""

    probe_type: str
    assistant_tools: tuple[dict[str, Any], ...]
    allowed_tool_names: tuple[str, ...]
    initial_user_message: str | None
    loop_policy: AssistantToolLoopPolicy
    user_system_prompt: str
    assistant_system_prompt: str
    turn0_user_query_instruction: str | None
    user_interaction_style: str
    patience: float
    user_turn_policy: UserTurnPolicySnapshot

    def to_dict(self) -> dict[str, Any]:
        """Return a defensive JSON-serializable copy."""
        return {
            "probe_type": self.probe_type,
            "assistant_tools": deepcopy(list(self.assistant_tools)),
            "allowed_tool_names": list(self.allowed_tool_names),
            "initial_user_message": self.initial_user_message,
            "loop_policy": self.loop_policy.to_dict(),
            "user_system_prompt": self.user_system_prompt,
            "assistant_system_prompt": self.assistant_system_prompt,
            "turn0_user_query_instruction": self.turn0_user_query_instruction,
            "user_interaction_style": self.user_interaction_style,
            "patience": self.patience,
            "user_turn_policy": self.user_turn_policy.to_dict(),
        }


@dataclass(frozen=True)
class _ExecutedToolCall:
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    payload: str
    turn_idx: int
    call_idx: int


class ProbeEpisodeRuntime:
    """Own one probe's mutable tool state outside UserSim's conversation loop.

    An agent harness remains responsible for model/tool iteration. The host
    exposes :attr:`assistant_tools`, forwards each selected function call to
    :meth:`simulate_tool_call`, and finally supplies the complete transcript to
    :meth:`finalize`. The runtime preserves the same probe hooks, state, traces,
    and result extras used by an in-process UserSim run.
    """

    def __init__(
        self,
        *,
        probe_type: str,
        persona: Mapping[str, Any],
        locale: str,
        language: str,
        models: Mapping[str, Any],
        config: ConversationSimulatorConfig,
        data: Mapping[str, Any],
        profile: Mapping[str, Any] | None = None,
        provenance: Provenance | None = None,
    ) -> None:
        if probe_type not in _EXTERNAL_RUNTIME_PROBES:
            raise ValueError(
                f"Probe {probe_type!r} does not support external tool execution; "
                f"supported probes: {sorted(_EXTERNAL_RUNTIME_PROBES)}"
            )

        # Importing the generator bootstraps all built-in probe registrations.
        import usersim.engine.generator  # noqa: F401

        self.probe_type = probe_type
        self.models = dict(models)
        self.config = config
        self.outcome = OutcomeBuilder(provenance=provenance or Provenance())
        probe_cls = resolve_probe(probe_type)
        self.probe: BaseProbe = probe_cls(
            persona=dict(persona),
            locale=locale,
            language=language,
            models=self.models,
            cfg=config,
            provenance=provenance,
            profile=dict(profile or {}),
            data=dict(data),
            outcome_builder=self.outcome,
        )
        self.state = ConversationState(outcome=self.outcome)
        self.probe.seed_state_metadata(self.state)
        self._assistant_tools = deepcopy(self.probe.get_tools_for_assistant() or [])
        self._allowed_tool_names = frozenset(_tool_name(tool) for tool in self._assistant_tools)
        self._loop_policy = _loop_policy(self.probe, config)
        self._user_turn_policy = _user_turn_policy(self.probe)
        self._call_count = 0
        self._executed_tool_calls: list[_ExecutedToolCall] = []
        self._synchronized_messages: list[dict[str, Any]] = []
        self._initial_user_message: str | None | object = _UNSET
        self._finalized = False
        self._final_result: dict[str, Any] | None = None
        self._lock = asyncio.Lock()

    @property
    def assistant_tools(self) -> list[dict[str, Any]]:
        """Return a defensive copy of the schemas available in this episode."""
        return deepcopy(self._assistant_tools)

    @property
    def allowed_tool_names(self) -> frozenset[str]:
        """Names accepted by :meth:`simulate_tool_call` for this episode."""
        return self._allowed_tool_names

    @property
    def loop_policy(self) -> AssistantToolLoopPolicy:
        """Probe-owned policy for the host's mechanical assistant/tool loop."""
        return self._loop_policy

    async def initial_user_message(self) -> str | None:
        """Return a probe-authored verbatim opening, when the probe defines one."""
        async with self._lock:
            self._ensure_open()
            return await self._initial_user_message_locked()

    async def descriptor(self) -> ProbeRuntimeDescriptor:
        """Return the deterministic host contract for this resolved episode."""
        async with self._lock:
            initial_message = await self._initial_user_message_locked()
            return ProbeRuntimeDescriptor(
                probe_type=self.probe_type,
                assistant_tools=tuple(deepcopy(self._assistant_tools)),
                allowed_tool_names=tuple(sorted(self._allowed_tool_names)),
                initial_user_message=initial_message,
                loop_policy=self._loop_policy,
                user_system_prompt=self.probe.get_user_system_prompt(),
                assistant_system_prompt=self.probe.get_assistant_system_prompt(),
                turn0_user_query_instruction=self.probe.get_user_query_instruction(0),
                user_interaction_style=str(getattr(self.probe, "_interaction_style", "neutral")),
                patience=float(getattr(self.probe, "_patience", 0.5)),
                user_turn_policy=self._user_turn_policy,
            )

    async def append_message(self, message: Mapping[str, Any]) -> None:
        """Append one externally produced OpenAI-style conversation message."""
        async with self._lock:
            self._ensure_open()
            normalized = _normalize_message(message)
            if normalized["role"] == "tool":
                raise ValueError("Tool messages must be produced by simulate_tool_call")
            self.state.messages.append(normalized)

    async def synchronize_transcript(self, messages: Sequence[Mapping[str, Any]]) -> None:
        """Install a monotonic, externally observed transcript.

        The transcript may advance from an assistant tool request to its
        runtime-produced tool result and later assistant/user messages. It may
        not alter or remove an already synchronized message, and every observed
        tool result must match runtime-owned execution evidence.
        """
        async with self._lock:
            self._ensure_open()
            self._synchronize_transcript_locked(messages)

    async def simulate_tool_call(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        tool_call_id: str,
        turn_idx: int,
        call_idx: int,
    ) -> str:
        """Simulate one allowed call and append its result to runtime state."""
        async with self._lock:
            self._ensure_open()
            if tool_name not in self._allowed_tool_names:
                raise ValueError(f"Tool {tool_name!r} is not available for this {self.probe_type!r} episode")
            if not isinstance(arguments, Mapping):
                raise TypeError("Tool arguments must be a mapping")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                raise ValueError("tool_call_id must be a non-empty string")
            if not isinstance(turn_idx, int) or isinstance(turn_idx, bool) or turn_idx < 0:
                raise ValueError("turn_idx must be a non-negative integer")
            if not isinstance(call_idx, int) or isinstance(call_idx, bool) or call_idx < 0:
                raise ValueError("call_idx must be a non-negative integer")
            if any(call.tool_call_id == tool_call_id for call in self._executed_tool_calls):
                raise ValueError(f"tool_call_id {tool_call_id!r} has already been executed")
            if any(call.turn_idx == turn_idx and call.call_idx == call_idx for call in self._executed_tool_calls):
                raise ValueError(f"turn_idx={turn_idx}, call_idx={call_idx} has already been executed")

            raw_call = {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(dict(arguments), ensure_ascii=False),
                },
            }
            execute = getattr(self.probe, "execute_tool_call", None)
            if not callable(execute):
                raise TypeError(f"Probe {self.probe_type!r} does not implement single-call execution")

            previous_builder = get_current_outcome_builder()
            set_current_outcome_builder(self.outcome)
            try:
                payload = await execute(
                    tool_name,
                    dict(arguments),
                    raw_call,
                    self.state,
                    self.models,
                    turn_idx=turn_idx,
                    call_idx=call_idx,
                )
            finally:
                set_current_outcome_builder(previous_builder)
            if not isinstance(payload, str):
                payload = json.dumps(payload, ensure_ascii=False, default=str)
            self.state.messages.append({"role": "tool", "content": payload, "tool_call_id": tool_call_id})
            self._executed_tool_calls.append(
                _ExecutedToolCall(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    arguments=deepcopy(dict(arguments)),
                    payload=payload,
                    turn_idx=turn_idx,
                    call_idx=call_idx,
                )
            )
            self._call_count += 1
            return payload

    async def evidence(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot suitable for verification and audit."""
        async with self._lock:
            return {
                "probe_type": self.probe_type,
                "assistant_tools": self.assistant_tools,
                "allowed_tool_names": sorted(self.allowed_tool_names),
                "transcript": deepcopy(self.state.messages),
                "conversation_metadata": deepcopy(self.state.metadata),
                "simulation_traces": [trace.to_dict() for trace in self.outcome.traces()],
                "result_extras": deepcopy(self.probe.build_result_extras(self.state)),
            }

    async def format_followup_user_instructions(self, turn_idx: int) -> list[str]:
        """Delegate probe-authored instructions over synchronized state."""
        if not isinstance(turn_idx, int) or isinstance(turn_idx, bool) or turn_idx < 1:
            raise ValueError("turn_idx must be a positive integer for a follow-up")
        async with self._lock:
            self._ensure_open()
            instructions = await self.probe.format_followup_user_instructions(turn_idx, self.state)
            if not isinstance(instructions, list) or not all(isinstance(item, str) for item in instructions):
                raise TypeError("Probe follow-up instructions must be a list of strings")
            return list(instructions)

    async def is_capitulation_detected(self) -> bool:
        """Delegate the probe's assistant-turn stop check."""
        async with self._lock:
            self._ensure_open()
            return bool(await self.probe.is_capitulation_detected(self.state))

    async def should_succeed(self) -> bool:
        """Delegate the probe's completion guard over synchronized state."""
        async with self._lock:
            self._ensure_open()
            return bool(self.probe.should_succeed(self.state))

    async def result_extras(self) -> dict[str, Any]:
        """Return probe-authored result columns over synchronized state."""
        async with self._lock:
            self._ensure_open()
            return deepcopy(self.probe.build_result_extras(self.state))

    async def finalize(self, messages: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
        """Finalize a native UserSim result from externally executed messages."""
        async with self._lock:
            if self._finalized:
                if (
                    messages is not None
                    and [_normalize_message(message) for message in messages] != self.state.messages
                ):
                    raise RuntimeError("Probe episode runtime was finalized with a different transcript")
                assert self._final_result is not None
                return deepcopy(self._final_result)
            if messages is not None:
                self._synchronize_transcript_locked(messages)
            self._validate_tool_evidence(self.state.messages, require_complete=True)
            result_extras = self.probe.build_result_extras(self.state)
            n_turns = int(
                result_extras.get(
                    "num_turns",
                    sum(1 for message in self.state.messages if message.get("role") == "user"),
                )
            )
            self.outcome.set_n_turns(n_turns)
            self.outcome.set_n_tool_calls(self._call_count)
            status = bool(self.probe.should_succeed(self.state))
            outcome = self.outcome.finalize(status=OutcomeStatus.OK)
            result = make_result(
                self.state.messages,
                self.state.metadata,
                status,
                outcome=outcome,
                traces=self.outcome.traces(),
            )
            result.update(result_extras)
            result["num_turns"] = n_turns
            result["num_tool_calls"] = self._call_count
            self._finalized = True
            self._final_result = deepcopy(result)
            return deepcopy(result)

    async def _initial_user_message_locked(self) -> str | None:
        if self._initial_user_message is _UNSET:
            message = await self.probe.get_verbatim_first_user_turn(self.state)
            self._initial_user_message = message if isinstance(message, str) and message else None
        assert self._initial_user_message is None or isinstance(self._initial_user_message, str)
        return self._initial_user_message

    def _validate_tool_evidence(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        require_complete: bool,
    ) -> None:
        expected = {call.tool_call_id: call for call in self._executed_tool_calls}
        observed_tool_ids: set[str] = set()
        tool_positions: dict[str, int] = {}
        for message_position, message in enumerate(messages):
            if message.get("role") != "tool":
                continue
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id in observed_tool_ids:
                raise ValueError("Transcript tool messages do not match executed tool-call evidence")
            observed_tool_ids.add(call_id)
            tool_positions[call_id] = message_position
            call = expected.get(call_id)
            if call is None or message.get("content") != call.payload:
                raise ValueError("Transcript tool messages do not match executed tool-call evidence")
        if expected.keys() - observed_tool_ids:
            raise ValueError("Transcript tool messages do not match executed tool-call evidence")

        assistant_calls = _assistant_calls(messages)
        for call_id, evidence in expected.items():
            observed = assistant_calls.get(call_id)
            if (
                observed is None
                or observed[:2] != (evidence.tool_name, evidence.arguments)
                or observed[2] >= tool_positions[call_id]
            ):
                raise ValueError("Transcript assistant calls do not match executed tool-call evidence")
        if require_complete and assistant_calls.keys() - expected.keys():
            raise ValueError("Transcript assistant calls have no executed tool-call evidence")

    def _synchronize_transcript_locked(self, messages: Sequence[Mapping[str, Any]]) -> None:
        normalized = [_normalize_message(message) for message in messages]
        synchronized_count = len(self._synchronized_messages)
        if len(normalized) < synchronized_count:
            raise ValueError("Transcript synchronization cannot regress")
        if normalized[:synchronized_count] != self._synchronized_messages:
            raise ValueError("Transcript synchronization diverges from previously observed messages")
        self._validate_tool_evidence(normalized, require_complete=False)
        self._synchronized_messages = deepcopy(normalized)
        self.state.messages = normalized

    def _ensure_open(self) -> None:
        if self._finalized:
            raise RuntimeError("Probe episode runtime has already been finalized")


def _tool_name(tool: Mapping[str, Any]) -> str:
    function = tool.get("function")
    if not isinstance(function, Mapping):
        raise ValueError(f"Invalid assistant tool schema: {tool!r}")
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"Assistant tool schema has no function name: {tool!r}")
    return name


def _normalize_message(message: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(message, Mapping):
        raise TypeError("Conversation messages must be mappings")
    role = message.get("role")
    if role not in {"system", "developer", "user", "assistant", "tool"}:
        raise ValueError(f"Unsupported conversation role: {role!r}")
    return deepcopy(dict(message))


def _assistant_calls(messages: Sequence[Mapping[str, Any]]) -> dict[str, tuple[str, dict[str, Any], int]]:
    calls: dict[str, tuple[str, dict[str, Any], int]] = {}
    for message_position, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        for raw_call in message.get("tool_calls") or []:
            if not isinstance(raw_call, Mapping):
                raise ValueError("Transcript assistant tool calls must be mappings")
            call_id = raw_call.get("id")
            function = raw_call.get("function")
            if not isinstance(call_id, str) or not call_id or call_id in calls:
                raise ValueError("Transcript assistant tool calls require unique non-empty IDs")
            if not isinstance(function, Mapping):
                raise ValueError("Transcript assistant tool calls require a function mapping")
            name = function.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("Transcript assistant tool calls require a function name")
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as error:
                    raise ValueError("Transcript assistant tool-call arguments must be valid JSON") from error
            if not isinstance(arguments, Mapping):
                raise ValueError("Transcript assistant tool-call arguments must be an object")
            calls[call_id] = (name, dict(arguments), message_position)
    return calls


def _loop_policy(probe: BaseProbe, config: ConversationSimulatorConfig) -> AssistantToolLoopPolicy:
    mode = getattr(probe, "tool_loop_mode", "multi")
    if mode not in {"single", "multi"}:
        raise ValueError(f"Unsupported assistant tool-loop mode: {mode!r}")
    activation_limit = probe.external_assistant_activation_limit(config)
    return AssistantToolLoopPolicy(
        tool_round_mode=mode,
        max_assistant_activations=activation_limit,
        final_synthesis_without_tools=bool(getattr(probe, "final_synthesis_without_tools", False)),
        single_user_turn=bool(getattr(probe, "single_user_turn", False)),
        assistant_error_behavior="fail_episode",
        tool_error_behavior="return_error_payload",
        max_tool_response_attempts=max(1, int(getattr(probe, "max_tool_response_attempts", 1))),
        assistant_resampling=bool(getattr(probe, "supports_assistant_resampling", False)),
    )


def _user_turn_policy(probe: BaseProbe) -> UserTurnPolicySnapshot:
    policy = resolve_user_turn_policy(probe)
    if policy.check_opening is not None:
        raise ValueError(f"Probe {probe.label!r} has a non-serializable opening-check policy")
    return UserTurnPolicySnapshot(
        context_compression=policy.context_compression,
        wrap_up=policy.wrap_up,
        followup_anchor=policy.followup_anchor,
        allowed_phrases=tuple(sorted(policy.allowed_phrases)),
        script_check_ignores=tuple(policy.script_check_ignores),
        check_opening="none",
    )


_UNSET = object()
