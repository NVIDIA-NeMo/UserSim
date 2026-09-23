# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Externally hosted, episode-scoped execution for tool-using probes."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from typing import Any, Mapping, Sequence

from usersim.engine.config import ConversationSimulatorConfig
from usersim.engine.core.llm import get_current_outcome_builder, set_current_outcome_builder
from usersim.engine.core.outcomes import OutcomeBuilder, OutcomeStatus, Provenance
from usersim.engine.core.probes import BaseProbe, resolve_probe
from usersim.engine.core.simulation import ConversationState, make_result

_EXTERNAL_RUNTIME_PROBES = frozenset({"tool_calling", "safety_agentic", "financial_services"})


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
        self._call_count = 0
        self._finalized = False
        self._lock = asyncio.Lock()

    @property
    def assistant_tools(self) -> list[dict[str, Any]]:
        """Return a defensive copy of the schemas available in this episode."""
        return deepcopy(self._assistant_tools)

    @property
    def allowed_tool_names(self) -> frozenset[str]:
        """Names accepted by :meth:`simulate_tool_call` for this episode."""
        return self._allowed_tool_names

    async def initial_user_message(self) -> str | None:
        """Return a probe-authored verbatim opening, when the probe defines one."""
        async with self._lock:
            self._ensure_open()
            message = await self.probe.get_verbatim_first_user_turn(self.state)
            return message if isinstance(message, str) and message else None

    def append_message(self, message: Mapping[str, Any]) -> None:
        """Append one externally produced OpenAI-style conversation message."""
        self._ensure_open()
        role = message.get("role")
        if role not in {"system", "developer", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported conversation role: {role!r}")
        self.state.messages.append(deepcopy(dict(message)))

    async def simulate_tool_call(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        tool_call_id: str | None = None,
        turn_idx: int = 0,
        call_idx: int | None = None,
    ) -> str:
        """Simulate one allowed call and append its result to runtime state."""
        async with self._lock:
            self._ensure_open()
            if tool_name not in self._allowed_tool_names:
                raise ValueError(f"Tool {tool_name!r} is not available for this {self.probe_type!r} episode")
            if not isinstance(arguments, Mapping):
                raise TypeError("Tool arguments must be a mapping")

            resolved_call_idx = self._call_count if call_idx is None else call_idx
            call_id = tool_call_id or f"call_{turn_idx}_{resolved_call_idx}"
            raw_call = {
                "id": call_id,
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
                    call_idx=resolved_call_idx,
                )
            finally:
                set_current_outcome_builder(previous_builder)
            if not isinstance(payload, str):
                payload = json.dumps(payload, ensure_ascii=False, default=str)
            self.state.messages.append({"role": "tool", "content": payload, "tool_call_id": call_id})
            self._call_count += 1
            return payload

    def replace_transcript(self, messages: Sequence[Mapping[str, Any]]) -> None:
        """Install the Agent-produced transcript without replaying tool effects."""
        self._ensure_open()
        normalized: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            if role not in {"system", "developer", "user", "assistant", "tool"}:
                raise ValueError(f"Unsupported conversation role: {role!r}")
            normalized.append(deepcopy(dict(message)))
        self.state.messages = normalized

    async def evidence(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot suitable for verification and audit."""
        async with self._lock:
            return {
                "probe_type": self.probe_type,
                "assistant_tools": self.assistant_tools,
                "allowed_tool_names": sorted(self.allowed_tool_names),
                "conversation_metadata": deepcopy(self.state.metadata),
                "simulation_traces": [trace.to_dict() for trace in self.outcome.traces()],
                "result_extras": deepcopy(self.probe.build_result_extras(self.state)),
            }

    async def finalize(self, messages: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
        """Finalize a native UserSim result from externally executed messages."""
        async with self._lock:
            self._ensure_open()
            if messages is not None:
                self.replace_transcript(messages)
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
            return result

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
