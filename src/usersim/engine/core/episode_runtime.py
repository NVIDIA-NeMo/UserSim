# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Externally hosted, episode-scoped execution of UserSim's own probe loop.

The host does not reimplement any part of an episode. UserSim runs the
unmodified probe dispatch as a background task and pauses at every model
boundary; each pause becomes an :class:`ActivationRequest` the host answers
with the model response it obtained. Per-episode settings, user-turn gates,
judges, early stop, tool execution and finalize therefore stay UserSim's.

The host records the assistant's response — tool calls included — *before*
issuing those tool calls. UserSim's own loop then executes them, with its own
turn and call indices and its own context, and the recorded payloads are
available to the host through :meth:`ProbeEpisodeRuntime.tool_result`. The
next request tells the host what comes next: whether it continues the current
turn, whether tools are offered, and the exact input UserSim would send.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from copy import deepcopy
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

from usersim.engine.core.episode_input import EpisodeConstructionError, EpisodePreamble, construct_probe_episode
from usersim.engine.core.identity import resolve_model_name
from usersim.engine.core.llm import get_current_outcome_builder, set_current_outcome_builder
from usersim.engine.core.outcomes import Provenance
from usersim.engine.core.probes import BaseProbe
from usersim.engine.core.simulation import ConversationState

if TYPE_CHECKING:
    from usersim.engine.config import ConversationSimulatorConfig

ActivationRole = Literal["user", "assistant", "judge", "summary"]

_MODEL_ROLE: dict[str, ActivationRole] = {
    "user_model": "user",
    "assistant_model": "assistant",
    "judge_model": "judge",
    "summary_model": "summary",
}


@dataclass(frozen=True)
class HostRoleModel:
    """The host's real model identity and token budget for one role.

    Trajectory ids are keyed on the resolved model identity, and
    ``identity_disclosure`` cannot grade an assistant it cannot name, so a
    host that proxies model calls must still declare which model is behind
    each role. ``max_tokens`` is the role's budget; UserSim scales it for
    non-Latin-script locales exactly as it does for a configured model.
    """

    model_name: str
    max_tokens: int | None = None

    @property
    def _model_config(self) -> SimpleNamespace:
        """Mirror the attribute path ``core/llm.scaled_max_tokens`` reads."""
        return SimpleNamespace(inference_parameters=SimpleNamespace(max_tokens=self.max_tokens))


@dataclass(frozen=True)
class ExecutedToolCall:
    """One tool call executed by UserSim's own loop."""

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    payload: str
    turn_idx: int
    call_idx: int

    def to_dict(self) -> dict[str, Any]:
        """Return a defensive JSON-serializable copy."""
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "arguments": deepcopy(self.arguments),
            "payload": self.payload,
            "turn_idx": self.turn_idx,
            "call_idx": self.call_idx,
        }


@dataclass(frozen=True)
class ActivationRequest:
    """One model activation paused inside UserSim's own episode.

    ``messages`` is the exact input UserSim would send, so a host never has to
    assemble its own. ``tools`` is empty when UserSim is not offering tools for
    this call — recording tool calls against it is a contract error, not a
    model failure. ``continues_turn`` marks a request that continues the turn
    already in progress rather than opening a new one.
    """

    activation_id: str
    role: ActivationRole
    model_alias: str
    messages: tuple[dict[str, Any], ...]
    parameters: dict[str, Any]
    tools: tuple[dict[str, Any], ...] = ()
    continues_turn: bool = False

    def __post_init__(self) -> None:
        if not self.activation_id:
            raise ValueError("activation_id must be a non-empty string")
        if self.role not in _MODEL_ROLE.values():
            raise ValueError(f"Unsupported activation role: {self.role!r}")
        object.__setattr__(self, "messages", tuple(_normalize_message(message) for message in self.messages))
        object.__setattr__(self, "parameters", _json_safe_copy(self.parameters))
        object.__setattr__(self, "tools", tuple(_json_safe_copy(list(self.tools))))

    @property
    def tools_enabled(self) -> bool:
        """Whether UserSim is offering tools for this activation."""
        return bool(self.tools)

    @property
    def tool_names(self) -> frozenset[str]:
        """Names the host may record tool calls against for this activation."""
        return frozenset(_tool_name(tool) for tool in self.tools)

    def to_dict(self) -> dict[str, Any]:
        """Return a defensive JSON-serializable copy."""
        return {
            "activation_id": self.activation_id,
            "role": self.role,
            "model_alias": self.model_alias,
            "messages": deepcopy(list(self.messages)),
            "parameters": deepcopy(self.parameters),
            "tools": deepcopy(list(self.tools)),
            "tools_enabled": self.tools_enabled,
            "continues_turn": self.continues_turn,
        }


@dataclass(frozen=True)
class ActivationUsage:
    """Token usage the host observed for one activation."""

    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable copy."""
        return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens}


@dataclass(frozen=True)
class ActivationResult:
    """The model response the host recorded for one activation.

    ``response`` is an OpenAI-shaped assistant message. Reasoning belongs in
    ``reasoning_content`` and is retained exactly as a configured model's would
    be; ``usage`` feeds UserSim's per-model token accounting.
    """

    activation_id: str
    response: dict[str, Any]
    usage: ActivationUsage | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.activation_id, str) or not self.activation_id:
            raise ValueError("activation_id must be a non-empty string")
        if not isinstance(self.response, Mapping):
            raise TypeError("activation response must be a mapping")
        object.__setattr__(self, "response", _normalize_assistant_response(self.response))

    @classmethod
    def from_value(cls, value: ActivationResult | Mapping[str, Any]) -> ActivationResult:
        """Normalize a typed or wire-format activation result."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("activation result must be ActivationResult or a mapping")
        response = value.get("response")
        if not isinstance(response, Mapping):
            raise ValueError("activation result requires a 'response' mapping")
        raw_usage = value.get("usage")
        usage = None
        if isinstance(raw_usage, ActivationUsage):
            usage = raw_usage
        elif isinstance(raw_usage, Mapping):
            usage = ActivationUsage(
                input_tokens=int(raw_usage.get("input_tokens") or 0),
                output_tokens=int(raw_usage.get("output_tokens") or 0),
            )
        return cls(
            activation_id=str(value.get("activation_id") or ""),
            response=deepcopy(dict(response)),
            usage=usage,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a defensive JSON-serializable copy."""
        return {
            "activation_id": self.activation_id,
            "response": deepcopy(self.response),
            "usage": self.usage.to_dict() if self.usage else None,
        }


@dataclass(frozen=True)
class EpisodeLifecycleComplete:
    """Terminal event returned after UserSim's own dispatch finalizes."""

    result: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a defensive JSON-serializable copy."""
        return {"complete": True, "result": deepcopy(self.result)}


@dataclass(frozen=True)
class _LifecycleFailure:
    error: BaseException


class EpisodeContractError(ValueError):
    """The host broke the episode contract.

    Raised to the host at the call that broke it, never routed through the
    model under test: these are host bugs, not model failures, so they must
    not consume a model retry or be attributed to the assistant.
    """


@dataclass(frozen=True)
class ProbeRuntimeDescriptor:
    """Stable, JSON-serializable description of one resolved episode.

    Deliberately small: every per-call decision — whether tools are offered,
    whether the turn continues, what to send next — is carried by the
    activation request itself, so a host never has to model UserSim's loop.
    """

    probe_type: str
    assistant_tools: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a defensive JSON-serializable copy."""
        return {
            "probe_type": self.probe_type,
            "assistant_tools": deepcopy(list(self.assistant_tools)),
        }


@dataclass
class _PendingActivation:
    request: ActivationRequest
    waiter: asyncio.Future[ActivationResult]
    result: ActivationResult | None = field(default=None)


class ProbeEpisodeRuntime:
    """Run one UserSim episode with the model calls answered by a host.

    Construct from a materialized row with :meth:`from_resolved_row`, then
    drive :meth:`advance` until it returns :class:`EpisodeLifecycleComplete`.
    :meth:`finalize` returns the same row a standalone ``usersim simulate``
    would have written for the same inputs.
    """

    @classmethod
    def from_resolved_row(
        cls,
        row: Mapping[str, Any],
        *,
        models: Mapping[str, Any],
    ) -> ProbeEpisodeRuntime:
        """Create a runtime directly from a materialized UserSim episode row."""
        from usersim.engine.config import ConversationSimulatorConfig

        raw_config = row.get("usersim_config")
        if not isinstance(raw_config, Mapping):
            raise ValueError("Resolved episode row requires a usersim_config mapping")
        config = ConversationSimulatorConfig.model_validate(dict(raw_config))
        persona = row.get(config.persona_column)
        if not isinstance(persona, Mapping):
            raise ValueError(f"Resolved episode row requires a {config.persona_column!r} mapping")
        probe_type = row.get(config.probe_type_column)
        if not isinstance(probe_type, str) or not probe_type:
            raise ValueError(f"Resolved episode row requires a non-empty {config.probe_type_column!r}")
        language = row.get("conversation_language")
        if not isinstance(language, str) or not language:
            raise ValueError("Resolved episode row requires a non-empty conversation_language")
        return cls(
            probe_type=probe_type,
            persona=persona,
            locale=config.locale,
            language=language,
            models=models,
            config=config,
            data=row,
        )

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
        self.models = dict(models)
        self.config = config
        # Probes may keep the models they are constructed with, so construction
        # and dispatch share the activation models.
        self._activation_models: dict[str, Any] = {
            **self.models,
            **{alias: _ActivationModel(self, alias, self.models.get(alias)) for alias in _MODEL_ROLE},
        }
        try:
            constructed = construct_probe_episode(
                {
                    **dict(data),
                    config.persona_column: dict(persona),
                    config.probe_type_column: probe_type,
                    **({"behavioral_profile": dict(profile)} if profile is not None else {}),
                },
                config=config,
                models=self._activation_models,
                provenance=provenance,
            )
        except EpisodeConstructionError as error:
            raise error.cause from error
        preamble = constructed.preamble
        self.preamble: EpisodePreamble = preamble
        self.probe_type = preamble.probe_type
        self.outcome = constructed.outcome
        self.probe: BaseProbe = constructed.probe
        self.state = ConversationState(outcome=self.outcome)
        self.probe.seed_state_metadata(self.state)
        self._assistant_tools = deepcopy(self.probe.get_tools_for_assistant() or [])

        self._executed_tool_calls: list[ExecutedToolCall] = []
        self._finalized = False
        self._closed = False
        self._final_result: dict[str, Any] | None = None
        self._lock = asyncio.Lock()
        self._lifecycle_started = False
        self._lifecycle_task: asyncio.Task[None] | None = None
        self._activation_events: asyncio.Queue[ActivationRequest | EpisodeLifecycleComplete | _LifecycleFailure] = (
            asyncio.Queue()
        )
        self._pending: dict[str, _PendingActivation] = {}
        self._transitions: dict[str, tuple[str, ActivationRequest | EpisodeLifecycleComplete]] = {}
        self._active_activation_id: str | None = None
        self._activation_sequence = 0
        # A new assistant activation continues the current turn when no user
        # message has been appended since the previous one.
        self._assistant_user_message_count: int | None = None

    @property
    def assistant_tools(self) -> list[dict[str, Any]]:
        """Return a defensive copy of the schemas available in this episode."""
        return deepcopy(self._assistant_tools)

    async def descriptor(self) -> ProbeRuntimeDescriptor:
        """Return the deterministic host contract for this resolved episode."""
        async with self._lock:
            return ProbeRuntimeDescriptor(
                probe_type=self.probe_type,
                assistant_tools=tuple(deepcopy(self._assistant_tools)),
            )

    async def advance(
        self,
        result: ActivationResult | Mapping[str, Any] | None = None,
    ) -> ActivationRequest | EpisodeLifecycleComplete:
        """Start the episode, or record one model response and resume it.

        The first call takes no result. Every later call records the response
        for the pending activation; UserSim then runs to the next model
        boundary, executing any tool calls the response asked for, and returns
        the next request or the completed episode.

        Re-recording the same activation with an identical payload is
        idempotent and replays the same transition. Reusing an activation id
        with a different payload is rejected.
        """
        async with self._lock:
            if self._closed:
                raise EpisodeContractError("Probe episode runtime has been closed")
            if self._lifecycle_started and result is not None:
                normalized = ActivationResult.from_value(result)
                fingerprint = _json_fingerprint(normalized.to_dict())
                prior = self._transitions.get(normalized.activation_id)
                if prior is not None:
                    prior_fingerprint, event = prior
                    if prior_fingerprint != fingerprint:
                        raise EpisodeContractError(
                            f"activation_id {normalized.activation_id!r} was reused with a different result"
                        )
                    return deepcopy(event)
            self._ensure_open()
            if not self._lifecycle_started:
                if result is not None:
                    raise EpisodeContractError("The first lifecycle advance cannot include a result")
                self._lifecycle_started = True
                self._lifecycle_task = asyncio.create_task(self._drive_native_lifecycle())
                return await self._next_activation_event()

            if result is None:
                raise EpisodeContractError("A started lifecycle requires an activation result")
            if normalized.activation_id != self._active_activation_id:
                raise EpisodeContractError(
                    f"activation_id {normalized.activation_id!r} is not the pending activation "
                    f"{self._active_activation_id!r}"
                )
            pending = self._pending[normalized.activation_id]
            # Validated here, at the host-facing call, so a contract error is
            # raised to its author rather than surfacing inside UserSim's model
            # call where it would consume a retry and be blamed on the model.
            self._validate_recorded_response(pending.request, normalized)
            pending.result = normalized
            pending.waiter.set_result(normalized)
            event = await self._next_activation_event()
            self._transitions[normalized.activation_id] = (fingerprint, deepcopy(event))
            return event

    async def tool_result(self, tool_call_id: str) -> str:
        """Return the payload UserSim's loop produced for one recorded call.

        Available once the response carrying the call has been recorded with
        :meth:`advance`; UserSim executes the calls as part of that step.

        Raises if the call did not execute. A probe caps how many calls it
        executes per turn, so a recorded call beyond that cap never runs —
        :meth:`executed_tool_calls` is the authoritative list.
        """
        async with self._lock:
            for executed in self._executed_tool_calls:
                if executed.tool_call_id == tool_call_id:
                    return executed.payload
        raise EpisodeContractError(
            f"Tool call {tool_call_id!r} has not been executed. Either its assistant response has not "
            f"been recorded with advance() yet, the probe's per-turn cap was reached and User Sim "
            f"declined to run it, or it is a simulated user's call, which the probe consumes rather "
            f"than runs; see executed_tool_calls()."
        )

    async def executed_tool_calls(self) -> list[ExecutedToolCall]:
        """Return every tool call UserSim's loop has executed so far, in order."""
        async with self._lock:
            return list(self._executed_tool_calls)

    async def evidence(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot suitable for verification and audit."""
        async with self._lock:
            return {
                "probe_type": self.probe_type,
                "assistant_tools": self.assistant_tools,
                "transcript": deepcopy(self.state.messages),
                "conversation_metadata": deepcopy(self.state.metadata),
                "simulation_traces": [trace.to_dict() for trace in self.outcome.traces()],
                "executed_tool_calls": [call.to_dict() for call in self._executed_tool_calls],
                "result_extras": deepcopy(self.probe.build_result_extras(self.state)),
            }

    async def finalize(self) -> dict[str, Any]:
        """Return the row UserSim's own dispatch produced for this episode.

        This is the resolved input row updated with the dispatch result —
        byte-for-byte what ``ConversationSimulatorGenerator`` writes for the
        same inputs, including loop-written columns such as ``user_query``.
        """
        async with self._lock:
            if not self._finalized:
                raise EpisodeContractError(
                    "Probe episode lifecycle has not completed; drive it to completion with advance()"
                )
            assert self._final_result is not None
            return deepcopy(self._final_result)

    async def close(self) -> None:
        """Cancel the running episode and release its task."""
        task = self._lifecycle_task
        self._closed = True
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        for pending in self._pending.values():
            if not pending.waiter.done():
                pending.waiter.cancel()

    # ── internals ────────────────────────────────────────────────────

    def _ensure_open(self) -> None:
        if self._closed:
            raise EpisodeContractError("Probe episode runtime has been closed")
        if self._finalized:
            raise EpisodeContractError("Probe episode runtime has already been finalized")

    def _validate_recorded_response(self, request: ActivationRequest, result: ActivationResult) -> None:
        tool_calls = result.response.get("tool_calls") or []
        if not tool_calls:
            return
        # Either side of the conversation may call tools a probe offers it (a
        # probe can give the simulated user tools as well as the assistant), so a
        # hosted episode records the same calls a local run would.
        if request.role not in ("assistant", "user"):
            raise EpisodeContractError(f"A {request.role} activation cannot record tool calls")
        if not request.tools_enabled:
            raise EpisodeContractError(
                f"Activation {request.activation_id!r} offers no tools, so the recorded response "
                f"must not contain tool calls. User Sim offers tools only where the probe's loop "
                f"expects a call, and disables them when it asks the assistant for its final answer."
            )
        allowed = request.tool_names
        seen: set[str] = set()
        for raw_call in tool_calls:
            if not isinstance(raw_call, Mapping):
                raise EpisodeContractError("Recorded tool calls must be mappings")
            call_id = raw_call.get("id")
            if not isinstance(call_id, str) or not call_id:
                raise EpisodeContractError("Recorded tool calls require a non-empty id")
            if call_id in seen:
                raise EpisodeContractError(f"Recorded tool call id {call_id!r} is repeated in one response")
            seen.add(call_id)
            function = raw_call.get("function")
            name = function.get("name") if isinstance(function, Mapping) else None
            if not isinstance(name, str) or name not in allowed:
                raise EpisodeContractError(
                    f"Tool {name!r} is not offered for activation {request.activation_id!r}; "
                    f"available: {sorted(allowed)}"
                )

    def _record_executed_tool_call(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        payload: str,
        turn_idx: int,
        call_idx: int,
    ) -> None:
        """Observe one tool call UserSim's own loop just executed."""
        self._executed_tool_calls.append(
            ExecutedToolCall(
                tool_call_id=str(tool_call_id),
                tool_name=str(tool_name),
                arguments=deepcopy(dict(arguments)),
                payload=payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, default=str),
                turn_idx=int(turn_idx),
                call_idx=int(call_idx),
            )
        )

    async def _next_activation_event(self) -> ActivationRequest | EpisodeLifecycleComplete:
        event = await self._activation_events.get()
        if isinstance(event, _LifecycleFailure):
            raise event.error
        self._active_activation_id = event.activation_id if isinstance(event, ActivationRequest) else None
        return event

    async def _drive_native_lifecycle(self) -> None:
        previous_builder = get_current_outcome_builder()
        set_current_outcome_builder(self.outcome)
        self.probe.set_tool_call_observer(self._record_executed_tool_call)
        try:
            result = await self.probe.run_dispatch(
                models=self._activation_models,
                data=self.probe._data,
                cfg=self.config,
                state=self.state,
                seed_state=False,
            )
            # The generator's output row is the resolved input row updated with
            # the dispatch result; loop-written columns such as ``user_query``
            # live on the former. Hosted rows must match it exactly.
            final_row = {**deepcopy(self.probe._data), **deepcopy(result)}
            self._finalized = True
            self._final_result = final_row
            await self._activation_events.put(EpisodeLifecycleComplete(deepcopy(final_row)))
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self._activation_events.put(_LifecycleFailure(error))
        finally:
            set_current_outcome_builder(previous_builder)
            self.probe.set_tool_call_observer(None)

    async def _request_activation(
        self,
        alias: str,
        messages: list[dict[str, Any]],
        parameters: dict[str, Any],
    ) -> ActivationResult:
        self._activation_sequence += 1
        activation_id = f"activation-{self._activation_sequence:06d}"
        call_parameters = dict(parameters)
        tools = call_parameters.pop("tools", None) or ()
        continues_turn = False
        if alias == "assistant_model":
            user_messages = sum(1 for message in self.state.messages if message.get("role") == "user")
            continues_turn = self._assistant_user_message_count == user_messages
            self._assistant_user_message_count = user_messages
        request = ActivationRequest(
            activation_id=activation_id,
            role=_MODEL_ROLE[alias],
            model_alias=alias,
            messages=tuple(deepcopy(messages)),
            parameters=_json_safe_copy(call_parameters),
            tools=tuple(deepcopy(list(tools))),
            continues_turn=continues_turn,
        )
        waiter: asyncio.Future[ActivationResult] = asyncio.get_running_loop().create_future()
        self._pending[activation_id] = _PendingActivation(request=request, waiter=waiter)
        await self._activation_events.put(request)
        return await waiter


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


class _ActivationModel:
    """ModelFacade-shaped rendezvous used by ``acall_llm``.

    Mirrors the host's declared identity and token budget so UserSim resolves
    the real model id for trajectory identity and probe grading, and scales
    ``max_tokens`` for non-Latin-script locales as it would for a configured
    model.
    """

    def __init__(self, runtime: ProbeEpisodeRuntime, alias: str, host_model: Any) -> None:
        self._runtime = runtime
        self._alias = alias
        self.model_name = resolve_model_name(host_model, alias)
        self._model_config = getattr(host_model, "_model_config", None)

    async def acompletion(self, messages: Sequence[Any], **kwargs: Any) -> SimpleNamespace:
        result = await self._runtime._request_activation(
            self._alias,
            [_chat_message_to_dict(message) for message in messages],
            dict(kwargs),
        )
        response = result.response
        usage = result.usage
        return SimpleNamespace(
            message=SimpleNamespace(
                content=response.get("content", "") or "",
                reasoning_content=response.get("reasoning_content"),
                tool_calls=deepcopy(response.get("tool_calls")),
            ),
            usage=(
                SimpleNamespace(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens) if usage else None
            ),
        )


def _chat_message_to_dict(message: Any) -> dict[str, Any]:
    role = getattr(message, "role", "user")
    role_value = getattr(role, "value", role)
    normalized: dict[str, Any] = {
        "role": str(role_value),
        "content": getattr(message, "content", "") or "",
    }
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        normalized["tool_calls"] = [
            call.model_dump() if hasattr(call, "model_dump") else deepcopy(call) for call in tool_calls
        ]
    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id:
        normalized["tool_call_id"] = str(tool_call_id)
    return _json_safe_copy(normalized)


def _normalize_assistant_response(response: Mapping[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(dict(response))
    role = normalized.setdefault("role", "assistant")
    if role != "assistant":
        raise ValueError(f"Activation response role must be 'assistant', got {role!r}")
    normalized.setdefault("content", "")
    return _json_safe_copy(normalized)


def _json_safe_copy(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError) as error:
        raise TypeError("Activation payload must be JSON serializable") from error


def _json_fingerprint(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
