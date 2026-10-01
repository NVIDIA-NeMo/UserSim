# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Externally hosted, episode-scoped execution of UserSim's native probe loop.

``ConversationRuntime`` owns outer control flow and delegates each Assistant
turn as one completed-transcript boundary. ``ProbeToolSession`` independently
owns Agent-called native tool effects. ``ProbeEpisodeRuntime`` adapts both to
the backward-compatible per-model local facade.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from copy import deepcopy
from dataclasses import dataclass, field
from types import MethodType, SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence, cast

from usersim.engine.core.episode_input import EpisodeConstructionError, EpisodePreamble, construct_probe_episode
from usersim.engine.core.identity import resolve_model_name
from usersim.engine.core.llm import get_current_outcome_builder, set_current_outcome_builder
from usersim.engine.core.outcomes import Provenance
from usersim.engine.core.probes import BaseProbe
from usersim.engine.core.simulation import ConversationState

if TYPE_CHECKING:
    from usersim.engine.config import ConversationSimulatorConfig

ActivationRole = Literal["user", "assistant", "judge", "summary"]
type JSONValue = None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]

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
class AssistantLoopPolicy:
    """Declarative constraints for an Agent-owned assistant tool loop."""

    mode: Literal["none", "single", "multi"]
    max_model_calls: int
    max_tool_calls: int | None
    final_synthesis: bool
    replay_reasoning: bool = False
    project_document_tool_results: bool = False

    @classmethod
    def from_value(cls, value: AssistantLoopPolicy | Mapping[str, Any]) -> AssistantLoopPolicy:
        if isinstance(value, cls):
            return value
        return cls(
            mode=cast(Literal["none", "single", "multi"], value.get("mode")),
            max_model_calls=int(value.get("max_model_calls") or 0),
            max_tool_calls=(int(value["max_tool_calls"]) if value.get("max_tool_calls") is not None else None),
            final_synthesis=bool(value.get("final_synthesis")),
            replay_reasoning=bool(value.get("replay_reasoning")),
            project_document_tool_results=bool(value.get("project_document_tool_results")),
        )

    def should_continue(self, *, model_calls: int, tool_calls: int) -> bool:
        """Whether the Agent may issue another model call in this turn."""
        if model_calls >= self.max_model_calls:
            return False
        if self.mode == "none":
            return False
        return True

    def tools_enabled(self, *, model_calls: int, tool_calls: int, limit_reached: bool = False) -> bool:
        """Whether tools remain available on the next model call."""
        if limit_reached or self.mode == "none" or (self.mode == "single" and model_calls > 0):
            return False
        if self.max_tool_calls is None:
            return True
        return model_calls <= self.max_tool_calls

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "mode": self.mode,
            "max_model_calls": self.max_model_calls,
            "max_tool_calls": self.max_tool_calls,
            "final_synthesis": self.final_synthesis,
            "replay_reasoning": self.replay_reasoning,
            "project_document_tool_results": self.project_document_tool_results,
        }


@dataclass(frozen=True)
class ToolTurnContext:
    """Opaque Resources initialization for one Agent-owned assistant turn."""

    turn_id: str
    first_tool_turn_idx: int
    max_tool_calls: int | None
    state_snapshot: dict[str, JSONValue]

    def __post_init__(self) -> None:
        if not self.turn_id:
            raise ValueError("turn_id must be a non-empty string")
        object.__setattr__(self, "state_snapshot", _json_mapping(self.state_snapshot, "state_snapshot"))

    @classmethod
    def from_value(cls, value: ToolTurnContext | Mapping[str, Any]) -> ToolTurnContext:
        if isinstance(value, cls):
            return value
        return cls(
            turn_id=str(value.get("turn_id") or ""),
            first_tool_turn_idx=int(value.get("first_tool_turn_idx") or 0),
            max_tool_calls=(int(value["max_tool_calls"]) if value.get("max_tool_calls") is not None else None),
            state_snapshot=_json_mapping(value.get("state_snapshot"), "state_snapshot"),
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return _json_mapping(
            {
                "turn_id": self.turn_id,
                "first_tool_turn_idx": self.first_tool_turn_idx,
                "max_tool_calls": self.max_tool_calls,
                "state_snapshot": self.state_snapshot,
            },
            "tool_turn_context",
        )


@dataclass(frozen=True)
class AssistantTurnRequest:
    """One complete assistant turn delegated from Environment to Agent."""

    turn_id: str
    model_alias: str
    messages: tuple[dict[str, Any], ...]
    parameters: dict[str, Any]
    tools: tuple[dict[str, Any], ...]
    loop_policy: AssistantLoopPolicy
    tool_context: ToolTurnContext

    role: Literal["assistant"] = "assistant"

    def __post_init__(self) -> None:
        if not self.turn_id:
            raise ValueError("turn_id must be a non-empty string")
        object.__setattr__(self, "messages", tuple(_normalize_message(message) for message in self.messages))
        object.__setattr__(self, "parameters", _json_safe_copy(self.parameters))
        object.__setattr__(self, "tools", tuple(_json_safe_copy(list(self.tools))))

    @classmethod
    def from_value(cls, value: AssistantTurnRequest | Mapping[str, Any]) -> AssistantTurnRequest:
        if isinstance(value, cls):
            return value
        return cls(
            turn_id=str(value.get("turn_id") or ""),
            model_alias=str(value.get("model_alias") or "assistant_model"),
            messages=tuple(cast(Sequence[Mapping[str, Any]], value.get("messages") or ())),
            parameters=_json_mapping(value.get("parameters") or {}, "parameters"),
            tools=tuple(cast(Sequence[Mapping[str, Any]], value.get("tools") or ())),
            loop_policy=AssistantLoopPolicy.from_value(cast(Mapping[str, Any], value.get("loop_policy") or {})),
            tool_context=ToolTurnContext.from_value(cast(Mapping[str, Any], value.get("tool_context") or {})),
        )

    @property
    def activation_id(self) -> str:
        """Compatibility spelling shared with single-call activations."""
        return self.turn_id

    @property
    def tools_enabled(self) -> bool:
        return bool(self.tools)

    @property
    def tool_names(self) -> frozenset[str]:
        return frozenset(_tool_name(tool) for tool in self.tools)

    @property
    def continues_turn(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "role": self.role,
            "model_alias": self.model_alias,
            "messages": deepcopy(list(self.messages)),
            "parameters": deepcopy(self.parameters),
            "tools": deepcopy(list(self.tools)),
            "loop_policy": self.loop_policy.to_dict(),
            "tool_context": self.tool_context.to_dict(),
        }

    def model_messages(self, transcript: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
        """Build a later model input using the declared generic replay policy."""
        messages = [*deepcopy(list(self.messages)), *deepcopy(list(transcript))]
        if not self.loop_policy.replay_reasoning:
            for message in messages:
                message.pop("reasoning_content", None)
        if self.loop_policy.project_document_tool_results:
            messages = _project_document_tool_results(messages)
        return tuple(messages)


@dataclass(frozen=True)
class AssistantModelCall:
    """One model response and its usage inside a completed assistant turn."""

    response: dict[str, Any]
    usage: ActivationUsage | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "response", _normalize_assistant_response(self.response))

    @classmethod
    def from_value(cls, value: AssistantModelCall | Mapping[str, Any]) -> AssistantModelCall:
        if isinstance(value, cls):
            return value
        raw_usage = value.get("usage")
        usage = (
            ActivationUsage(
                input_tokens=int(raw_usage.get("input_tokens") or 0),
                output_tokens=int(raw_usage.get("output_tokens") or 0),
            )
            if isinstance(raw_usage, Mapping)
            else None
        )
        return cls(response=_json_mapping(value.get("response"), "response"), usage=usage)

    def to_dict(self) -> dict[str, Any]:
        return {
            "response": deepcopy(self.response),
            "usage": self.usage.to_dict() if self.usage else None,
        }


@dataclass(frozen=True)
class AssistantToolRound:
    """One ordered assistant tool-call batch submitted to Resources."""

    turn_id: str
    assistant_response: dict[str, Any]
    round_id: str = ""

    def __post_init__(self) -> None:
        if not self.turn_id:
            raise ValueError("turn_id must be a non-empty string")
        object.__setattr__(self, "assistant_response", _normalize_assistant_response(self.assistant_response))
        if not self.round_id:
            object.__setattr__(self, "round_id", _json_fingerprint(self.assistant_response))

    @classmethod
    def from_value(cls, value: AssistantToolRound | Mapping[str, Any]) -> AssistantToolRound:
        if isinstance(value, cls):
            return value
        return cls(
            turn_id=str(value.get("turn_id") or ""),
            round_id=str(value.get("round_id") or ""),
            assistant_response=_json_mapping(value.get("assistant_response"), "assistant_response"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "round_id": self.round_id,
            "assistant_response": deepcopy(self.assistant_response),
        }


@dataclass(frozen=True)
class ToolCallReceipt:
    """Opaque payload plus Resources-owned semantic indices and effect state."""

    tool_call_id: str
    tool_name: str
    arguments: dict[str, JSONValue]
    raw_tool_call: dict[str, JSONValue]
    payload: str
    turn_idx: int
    call_idx: int
    effect_state: dict[str, JSONValue]

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", _json_mapping(self.arguments, "arguments"))
        object.__setattr__(self, "raw_tool_call", _json_mapping(self.raw_tool_call, "raw_tool_call"))
        object.__setattr__(self, "effect_state", _json_mapping(self.effect_state, "effect_state"))

    @classmethod
    def from_value(cls, value: ToolCallReceipt | Mapping[str, Any]) -> ToolCallReceipt:
        if isinstance(value, cls):
            return value
        payload = value.get("payload")
        if not isinstance(payload, str):
            raise TypeError("tool receipt payload must be a string")
        return cls(
            tool_call_id=str(value.get("tool_call_id") or ""),
            tool_name=str(value.get("tool_name") or ""),
            arguments=_json_mapping(value.get("arguments"), "arguments"),
            raw_tool_call=_json_mapping(value.get("raw_tool_call"), "raw_tool_call"),
            payload=payload,
            turn_idx=int(value.get("turn_idx") or 0),
            call_idx=int(value.get("call_idx") or 0),
            effect_state=_json_mapping(value.get("effect_state"), "effect_state"),
        )

    @property
    def tool_message(self) -> dict[str, Any]:
        return {"role": "tool", "content": self.payload, "tool_call_id": self.tool_call_id}

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "arguments": deepcopy(self.arguments),
            "raw_tool_call": deepcopy(self.raw_tool_call),
            "payload": self.payload,
            "turn_idx": self.turn_idx,
            "call_idx": self.call_idx,
            "effect_state": deepcopy(self.effect_state),
        }


@dataclass(frozen=True)
class ToolRoundReceipt:
    """Resources result for one ordered tool batch."""

    turn_id: str
    round_id: str
    receipts: tuple[ToolCallReceipt, ...]
    limit_reached: bool = False

    @classmethod
    def from_value(cls, value: ToolRoundReceipt | Mapping[str, Any]) -> ToolRoundReceipt:
        if isinstance(value, cls):
            return value
        return cls(
            turn_id=str(value.get("turn_id") or ""),
            round_id=str(value.get("round_id") or ""),
            receipts=tuple(
                ToolCallReceipt.from_value(receipt)
                for receipt in cast(Sequence[Mapping[str, Any]], value.get("receipts") or ())
            ),
            limit_reached=bool(value.get("limit_reached")),
        )

    @property
    def tool_messages(self) -> tuple[dict[str, Any], ...]:
        return tuple(receipt.tool_message for receipt in self.receipts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "round_id": self.round_id,
            "receipts": [receipt.to_dict() for receipt in self.receipts],
            "limit_reached": self.limit_reached,
        }


@dataclass(frozen=True)
class CompletedTurnEvidence:
    """Typed Resources evidence attached to an Agent's completed turn."""

    turn_id: str
    rounds: tuple[ToolRoundReceipt, ...]
    final_effect_state: dict[str, JSONValue]

    def __post_init__(self) -> None:
        object.__setattr__(self, "final_effect_state", _json_mapping(self.final_effect_state, "final_effect_state"))

    @classmethod
    def from_value(cls, value: CompletedTurnEvidence | Mapping[str, Any]) -> CompletedTurnEvidence:
        if isinstance(value, cls):
            return value
        return cls(
            turn_id=str(value.get("turn_id") or ""),
            rounds=tuple(
                ToolRoundReceipt.from_value(round_receipt)
                for round_receipt in cast(Sequence[Mapping[str, Any]], value.get("rounds") or ())
            ),
            final_effect_state=_json_mapping(value.get("final_effect_state"), "final_effect_state"),
        )

    @property
    def receipts(self) -> tuple[ToolCallReceipt, ...]:
        return tuple(receipt for round_receipt in self.rounds for receipt in round_receipt.receipts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "rounds": [round_receipt.to_dict() for round_receipt in self.rounds],
            "final_effect_state": deepcopy(self.final_effect_state),
        }

    @classmethod
    def from_unexecuted_turn(cls, context: ToolTurnContext | Mapping[str, Any]) -> CompletedTurnEvidence:
        """Build evidence for a completed Assistant turn that called no tools."""
        normalized = ToolTurnContext.from_value(context)
        return cls(
            turn_id=normalized.turn_id,
            rounds=(),
            final_effect_state=_effect_state_from_snapshot(normalized.state_snapshot),
        )


@dataclass(frozen=True)
class ProbeToolCallRequest:
    """One normal Resources tool call plus hidden UserSim turn context.

    ``tool_name`` and ``arguments`` are the ordinary Resources route path and
    body. The remaining fields are transport metadata that let UserSim validate
    the original Assistant response and assign native semantic indices.
    """

    context: ToolTurnContext
    assistant_response: dict[str, Any]
    tool_call_id: str
    tool_name: str
    arguments: dict[str, JSONValue]
    round_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "context", ToolTurnContext.from_value(self.context))
        object.__setattr__(self, "assistant_response", _normalize_assistant_response(self.assistant_response))
        object.__setattr__(self, "arguments", _json_mapping(self.arguments, "arguments"))
        if not self.tool_call_id or not self.tool_name:
            raise ValueError("tool_call_id and tool_name must be non-empty strings")
        if not self.round_id:
            object.__setattr__(self, "round_id", _json_fingerprint(self.assistant_response))

    @classmethod
    def from_value(cls, value: ProbeToolCallRequest | Mapping[str, Any]) -> ProbeToolCallRequest:
        if isinstance(value, cls):
            return value
        return cls(
            context=ToolTurnContext.from_value(cast(Mapping[str, Any], value.get("context") or {})),
            assistant_response=_json_mapping(value.get("assistant_response"), "assistant_response"),
            tool_call_id=str(value.get("tool_call_id") or ""),
            tool_name=str(value.get("tool_name") or ""),
            arguments=_json_mapping(value.get("arguments"), "arguments"),
            round_id=str(value.get("round_id") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "context": self.context.to_dict(),
            "assistant_response": deepcopy(self.assistant_response),
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "arguments": deepcopy(self.arguments),
            "round_id": self.round_id,
        }


@dataclass(frozen=True)
class ProbeToolCallResult:
    """Model-visible payload plus hidden Resources evidence for one call."""

    receipt: ToolCallReceipt | None
    evidence: CompletedTurnEvidence
    limit_reached: bool = False

    @property
    def payload(self) -> str | None:
        """Return the opaque model-visible payload when the call executed."""
        return self.receipt.payload if self.receipt is not None else None

    @classmethod
    def from_value(cls, value: ProbeToolCallResult | Mapping[str, Any]) -> ProbeToolCallResult:
        if isinstance(value, cls):
            return value
        raw_receipt = value.get("receipt")
        return cls(
            receipt=ToolCallReceipt.from_value(raw_receipt) if isinstance(raw_receipt, Mapping) else None,
            evidence=CompletedTurnEvidence.from_value(cast(Mapping[str, Any], value.get("evidence") or {})),
            limit_reached=bool(value.get("limit_reached")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt": self.receipt.to_dict() if self.receipt is not None else None,
            "evidence": self.evidence.to_dict(),
            "limit_reached": self.limit_reached,
        }


@dataclass(frozen=True)
class CompletedAssistantTurn:
    """The complete Agent-owned assistant transcript returned to Environment."""

    turn_id: str
    transcript: tuple[dict[str, Any], ...]
    model_calls: tuple[AssistantModelCall, ...]
    evidence: CompletedTurnEvidence

    def __post_init__(self) -> None:
        if not self.turn_id:
            raise ValueError("turn_id must be a non-empty string")
        object.__setattr__(self, "transcript", tuple(_normalize_message(message) for message in self.transcript))

    @classmethod
    def from_value(cls, value: CompletedAssistantTurn | Mapping[str, Any]) -> CompletedAssistantTurn:
        if isinstance(value, cls):
            return value
        return cls(
            turn_id=str(value.get("turn_id") or ""),
            transcript=tuple(cast(Sequence[Mapping[str, Any]], value.get("transcript") or ())),
            model_calls=tuple(
                AssistantModelCall.from_value(model_call)
                for model_call in cast(Sequence[Mapping[str, Any]], value.get("model_calls") or ())
            ),
            evidence=CompletedTurnEvidence.from_value(cast(Mapping[str, Any], value.get("evidence") or {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "transcript": deepcopy(list(self.transcript)),
            "model_calls": [call.to_dict() for call in self.model_calls],
            "evidence": self.evidence.to_dict(),
        }


@dataclass(frozen=True)
class ToolExecutionRequest:
    """One native probe tool effect paused for a resources host."""

    request_id: str
    tool_call_id: str
    tool_name: str
    arguments: dict[str, JSONValue]
    raw_tool_call: dict[str, JSONValue]
    turn_idx: int
    call_idx: int
    state_snapshot: dict[str, JSONValue]

    def __post_init__(self) -> None:
        if not self.request_id or not self.tool_call_id:
            raise ValueError("Tool execution request ids must be non-empty strings")
        object.__setattr__(self, "arguments", _json_mapping(self.arguments, "arguments"))
        object.__setattr__(self, "raw_tool_call", _json_mapping(self.raw_tool_call, "raw_tool_call"))
        object.__setattr__(self, "state_snapshot", _json_mapping(self.state_snapshot, "state_snapshot"))

    @classmethod
    def from_value(cls, value: ToolExecutionRequest | Mapping[str, object]) -> ToolExecutionRequest:
        """Normalize a typed or wire-format tool request."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("tool execution request must be ToolExecutionRequest or a mapping")
        return cls(
            request_id=str(value.get("request_id") or ""),
            tool_call_id=str(value.get("tool_call_id") or ""),
            tool_name=str(value.get("tool_name") or ""),
            arguments=_json_mapping(value.get("arguments"), "arguments"),
            raw_tool_call=_json_mapping(value.get("raw_tool_call"), "raw_tool_call"),
            turn_idx=int(value.get("turn_idx") or 0),
            call_idx=int(value.get("call_idx") or 0),
            state_snapshot=_json_mapping(value.get("state_snapshot"), "state_snapshot"),
        )

    def to_dict(self) -> dict[str, JSONValue]:
        """Return a defensive JSON-serializable copy."""
        return _json_mapping(
            {
                "request_id": self.request_id,
                "tool_call_id": self.tool_call_id,
                "tool_name": self.tool_name,
                "arguments": self.arguments,
                "raw_tool_call": self.raw_tool_call,
                "turn_idx": self.turn_idx,
                "call_idx": self.call_idx,
                "state_snapshot": self.state_snapshot,
            },
            "tool_execution_request",
        )


@dataclass(frozen=True)
class ToolExecutionResult:
    """Verbatim tool payload and resulting JSON-safe effect state."""

    request_id: str
    payload: str
    state_snapshot: dict[str, JSONValue]

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be a non-empty string")
        if not isinstance(self.payload, str):
            raise TypeError("tool payload must be a string")
        object.__setattr__(self, "state_snapshot", _json_mapping(self.state_snapshot, "state_snapshot"))

    @classmethod
    def from_value(cls, value: ToolExecutionResult | Mapping[str, object]) -> ToolExecutionResult:
        """Normalize a typed or wire-format tool result."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("tool execution result must be ToolExecutionResult or a mapping")
        payload = value.get("payload")
        if not isinstance(payload, str):
            raise TypeError("tool execution result requires a string 'payload'")
        return cls(
            request_id=str(value.get("request_id") or ""),
            payload=payload,
            state_snapshot=_json_mapping(value.get("state_snapshot"), "state_snapshot"),
        )

    def to_dict(self) -> dict[str, JSONValue]:
        """Return a defensive JSON-serializable copy."""
        return _json_mapping(
            {
                "request_id": self.request_id,
                "payload": self.payload,
                "state_snapshot": self.state_snapshot,
            },
            "tool_execution_result",
        )


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

    Deliberately small: Assistant loop constraints and exact inputs travel on
    each turn request, so Environment never models the Agent's inner loop.
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


@dataclass
class _PendingAssistantTurn:
    request: AssistantTurnRequest
    waiter: asyncio.Future[CompletedAssistantTurn]
    result: CompletedAssistantTurn | None = field(default=None)


@dataclass
class _AssistantReplay:
    completed: CompletedAssistantTurn
    model_index: int = 0
    receipt_index: int = 0


@dataclass
class _ToolRoundState:
    batch: AssistantToolRound
    fingerprint: str
    turn_idx: int
    receipts: list[ToolCallReceipt] = field(default_factory=list)
    call_fingerprints: dict[str, str] = field(default_factory=dict)
    limit_reached: bool = False

    def snapshot(self) -> ToolRoundReceipt:
        return ToolRoundReceipt(
            turn_id=self.batch.turn_id,
            round_id=self.batch.round_id,
            receipts=tuple(deepcopy(self.receipts)),
            limit_reached=self.limit_reached,
        )


@dataclass
class _ToolTurnState:
    context: ToolTurnContext
    rounds: list[_ToolRoundState] = field(default_factory=list)
    rounds_by_id: dict[str, _ToolRoundState] = field(default_factory=dict)


@dataclass
class _PendingToolExecution:
    request: ToolExecutionRequest
    waiter: asyncio.Future[ToolExecutionResult]
    result: ToolExecutionResult | None = field(default=None)


class ConversationRuntime:
    """Run outer conversation control while Agent and Resources own each turn.

    Construct from a materialized row with :meth:`from_resolved_row`, then
    drive :meth:`advance` across participant and completed-Assistant events.
    :meth:`finalize` returns the same row a standalone ``usersim simulate``
    would have written for the same inputs.
    """

    @classmethod
    def from_resolved_row(
        cls,
        row: Mapping[str, Any],
        *,
        models: Mapping[str, Any],
    ) -> ConversationRuntime:
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
        self._lifecycle_events: asyncio.Queue[
            ActivationRequest | AssistantTurnRequest | EpisodeLifecycleComplete | _LifecycleFailure
        ] = asyncio.Queue()
        self._pending: dict[str, _PendingActivation] = {}
        self._pending_assistant_turns: dict[str, _PendingAssistantTurn] = {}
        self._pending_tools: dict[str, _PendingToolExecution] = {}
        self._transitions: dict[
            str, tuple[str, ActivationRequest | AssistantTurnRequest | EpisodeLifecycleComplete]
        ] = {}
        self._active_activation_id: str | None = None
        self._active_assistant_turn_id: str | None = None
        self._assistant_replay: _AssistantReplay | None = None
        self._activation_sequence = 0
        self._assistant_turn_sequence = 0
        self._tool_sequence = 0
        # A new assistant activation continues the current turn when no user
        # message has been appended since the previous one.
        self._assistant_user_message_count: int | None = None
        self.probe._conversation_runtime = self
        self.probe.execute_tool_call = MethodType(_proxy_execute_tool_call, self.probe)

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
        result: ActivationResult | CompletedAssistantTurn | Mapping[str, Any] | None = None,
    ) -> ActivationRequest | AssistantTurnRequest | EpisodeLifecycleComplete:
        """Start the episode, or record one participant/completed-turn result.

        The first call takes no result. Every later call records the response
        for the pending lifecycle event and resumes to the next boundary.

        Re-recording the same activation with an identical payload is
        idempotent and replays the same transition. Reusing an activation id
        with a different payload is rejected.
        """
        async with self._lock:
            if self._closed:
                raise EpisodeContractError("Probe episode runtime has been closed")
            if self._lifecycle_started and result is not None:
                is_completed_turn = isinstance(result, CompletedAssistantTurn) or (
                    isinstance(result, Mapping) and "model_calls" in result and "evidence" in result
                )
                if is_completed_turn:
                    normalized: ActivationResult | CompletedAssistantTurn = CompletedAssistantTurn.from_value(result)  # type: ignore[arg-type]
                    result_id = normalized.turn_id
                else:
                    normalized = ActivationResult.from_value(result)
                    result_id = normalized.activation_id
                fingerprint = _json_fingerprint(normalized.to_dict())
                prior = self._transitions.get(result_id)
                if prior is not None:
                    prior_fingerprint, event = prior
                    if prior_fingerprint != fingerprint:
                        raise EpisodeContractError(f"lifecycle id {result_id!r} was reused with a different result")
                    return deepcopy(event)
            self._ensure_open()
            if not self._lifecycle_started:
                if result is not None:
                    raise EpisodeContractError("The first lifecycle advance cannot include a result")
                self._lifecycle_started = True
                self._lifecycle_task = asyncio.create_task(self._drive_native_lifecycle())
                return await self._next_activation_event()

            if result is None:
                raise EpisodeContractError("A started lifecycle requires a matching result")
            if isinstance(normalized, CompletedAssistantTurn):
                if normalized.turn_id != self._active_assistant_turn_id:
                    raise EpisodeContractError(
                        f"turn_id {normalized.turn_id!r} is not the pending assistant turn "
                        f"{self._active_assistant_turn_id!r}"
                    )
                pending_turn = self._pending_assistant_turns[normalized.turn_id]
                self._validate_completed_turn(pending_turn.request, normalized)
                pending_turn.result = normalized
                pending_turn.waiter.set_result(normalized)
                event = await self._next_activation_event()
                self._transitions[normalized.turn_id] = (fingerprint, deepcopy(event))
                return event
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
            f"been recorded with advance() yet, or the probe's per-turn cap was reached and UserSim "
            f"declined to run it; see executed_tool_calls()."
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
        for pending in self._pending_assistant_turns.values():
            if not pending.waiter.done():
                pending.waiter.cancel()
        for pending in self._pending_tools.values():
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
        if request.role != "assistant":
            raise EpisodeContractError(f"A {request.role} activation cannot record tool calls")
        if not request.tools_enabled:
            raise EpisodeContractError(
                f"Activation {request.activation_id!r} offers no tools, so the recorded response "
                f"must not contain tool calls. UserSim disables tools when the probe's loop asks "
                f"the assistant for its final answer."
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

    def _validate_completed_turn(self, request: AssistantTurnRequest, completed: CompletedAssistantTurn) -> None:
        if completed.evidence.turn_id != request.turn_id:
            raise EpisodeContractError("Completed assistant turn evidence belongs to a different turn")
        expected: list[dict[str, Any]] = []
        tool_count = 0
        round_index = 0
        limit_reached = False
        policy_error: str | None = None
        for model_index, model_call in enumerate(completed.model_calls):
            tools = (
                request.tools
                if request.loop_policy.tools_enabled(
                    model_calls=model_index,
                    tool_calls=tool_count,
                    limit_reached=limit_reached,
                )
                else ()
            )
            self._validate_recorded_response(
                ActivationRequest(
                    activation_id=request.turn_id,
                    role="assistant",
                    model_alias=request.model_alias,
                    messages=request.messages,
                    parameters=request.parameters,
                    tools=tools,
                ),
                ActivationResult(activation_id=request.turn_id, response=model_call.response, usage=model_call.usage),
            )
            expected.append(model_call.response)
            raw_calls = model_call.response.get("tool_calls") or []
            if not raw_calls:
                if model_index != len(completed.model_calls) - 1:
                    policy_error = "Completed assistant turn continues after a final text response"
                continue
            if round_index >= len(completed.evidence.rounds):
                raise EpisodeContractError("Completed assistant transcript has a tool batch without Resources evidence")
            round_receipt = completed.evidence.rounds[round_index]
            round_index += 1
            for raw_call, receipt in zip(raw_calls, round_receipt.receipts, strict=False):
                if receipt.raw_tool_call != raw_call:
                    raise EpisodeContractError("Completed assistant transcript and Resources evidence disagree")
                expected.append(receipt.tool_message)
            executable = (
                len(raw_calls)
                if request.loop_policy.max_tool_calls is None
                else min(len(raw_calls), max(0, request.loop_policy.max_tool_calls - tool_count))
            )
            if len(round_receipt.receipts) != executable:
                raise EpisodeContractError("Completed assistant transcript and Resources receipt count disagree")
            tool_count += len(round_receipt.receipts)
            limit_reached = round_receipt.limit_reached
            should_continue = request.loop_policy.should_continue(
                model_calls=model_index + 1,
                tool_calls=tool_count,
            )
            is_last = model_index == len(completed.model_calls) - 1
            if should_continue == is_last:
                state = "ends before" if is_last else "continues after"
                policy_error = f"Completed assistant turn {state} its declared loop policy"
        if round_index != len(completed.evidence.rounds):
            raise EpisodeContractError("Completed assistant evidence contains a round absent from the transcript")
        if list(completed.transcript) != expected:
            raise EpisodeContractError("Completed assistant transcript and model/evidence records disagree")
        if policy_error is not None:
            raise EpisodeContractError(policy_error)
        if not completed.model_calls:
            raise EpisodeContractError("Completed assistant turn requires at least one model call")
        if len(completed.model_calls) > request.loop_policy.max_model_calls:
            raise EpisodeContractError("Completed assistant turn exceeds its model-call limit")
        if request.loop_policy.max_tool_calls is not None and tool_count > request.loop_policy.max_tool_calls:
            raise EpisodeContractError("Completed assistant turn exceeds its tool-call limit")

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

    async def _next_activation_event(
        self,
    ) -> ActivationRequest | AssistantTurnRequest | EpisodeLifecycleComplete:
        event = await self._lifecycle_events.get()
        if isinstance(event, _LifecycleFailure):
            raise event.error
        self._active_activation_id = event.activation_id if isinstance(event, ActivationRequest) else None
        self._active_assistant_turn_id = event.turn_id if isinstance(event, AssistantTurnRequest) else None
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
            self._ensure_assistant_replay_consumed()
            # The generator's output row is the resolved input row updated with
            # the dispatch result; loop-written columns such as ``user_query``
            # live on the former. Hosted rows must match it exactly.
            final_row = {**deepcopy(self.probe._data), **deepcopy(result)}
            self._finalized = True
            self._final_result = final_row
            await self._lifecycle_events.put(EpisodeLifecycleComplete(deepcopy(final_row)))
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self._lifecycle_events.put(_LifecycleFailure(error))
        finally:
            set_current_outcome_builder(previous_builder)
            self.probe.set_tool_call_observer(None)

    async def _request_activation(
        self,
        alias: str,
        messages: list[dict[str, Any]],
        parameters: dict[str, Any],
    ) -> ActivationResult:
        if alias == "assistant_model":
            return await self._request_assistant_turn(messages, parameters)
        self._ensure_assistant_replay_consumed()
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
        await self._lifecycle_events.put(request)
        return await waiter

    async def _request_assistant_turn(
        self,
        messages: list[dict[str, Any]],
        parameters: dict[str, Any],
    ) -> ActivationResult:
        replay = self._assistant_replay
        if replay is None:
            self._assistant_turn_sequence += 1
            turn_id = f"assistant-turn-{self._assistant_turn_sequence:06d}"
            call_parameters = dict(parameters)
            tools = tuple(deepcopy(list(call_parameters.pop("tools", None) or ())))
            policy = self._assistant_loop_policy(bool(tools))
            first_tool_turn_idx = sum(1 for message in self.state.messages if message.get("role") == "assistant")
            if hasattr(self.probe, "tool_loop_mode"):
                first_tool_turn_idx += 1
            request = AssistantTurnRequest(
                turn_id=turn_id,
                model_alias="assistant_model",
                messages=tuple(deepcopy(messages)),
                parameters=_json_safe_copy(call_parameters),
                tools=tools,
                loop_policy=policy,
                tool_context=ToolTurnContext(
                    turn_id=turn_id,
                    first_tool_turn_idx=first_tool_turn_idx,
                    max_tool_calls=policy.max_tool_calls,
                    state_snapshot=self._state_snapshot(),
                ),
            )
            waiter: asyncio.Future[CompletedAssistantTurn] = asyncio.get_running_loop().create_future()
            self._pending_assistant_turns[turn_id] = _PendingAssistantTurn(request=request, waiter=waiter)
            await self._lifecycle_events.put(request)
            completed = await waiter
            self._assistant_replay = _AssistantReplay(completed=completed)
            replay = self._assistant_replay

        if replay.model_index >= len(replay.completed.model_calls):
            raise EpisodeContractError("Completed assistant turn ended before UserSim's native loop")
        model_call = replay.completed.model_calls[replay.model_index]
        replay.model_index += 1
        return ActivationResult(
            activation_id=replay.completed.turn_id,
            response=model_call.response,
            usage=model_call.usage,
        )

    def _assistant_loop_policy(self, has_tools: bool) -> AssistantLoopPolicy:
        if not has_tools:
            return AssistantLoopPolicy(mode="none", max_model_calls=1, max_tool_calls=0, final_synthesis=False)
        if hasattr(self.probe, "tool_loop_mode"):
            mode = cast(Literal["single", "multi"], getattr(self.probe, "tool_loop_mode", "single"))
            max_calls = int(getattr(self.probe, "tool_max_calls_per_turn", 8))
            return AssistantLoopPolicy(
                mode=mode,
                max_model_calls=(2 if mode == "single" else max_calls + 2),
                max_tool_calls=max_calls,
                final_synthesis=True,
                project_document_tool_results=self.probe_type == "financial_services",
            )
        max_model_calls = max(1, int(getattr(self.config, "max_turns", 5)))
        return AssistantLoopPolicy(
            mode="multi",
            max_model_calls=max_model_calls,
            max_tool_calls=None,
            final_synthesis=False,
        )

    def _ensure_assistant_replay_consumed(self) -> None:
        replay = self._assistant_replay
        if replay is None:
            return
        if replay.model_index != len(replay.completed.model_calls):
            raise EpisodeContractError(
                "Completed assistant turn contains unused model calls "
                f"({replay.model_index} consumed of {len(replay.completed.model_calls)})"
            )
        if replay.receipt_index != len(replay.completed.evidence.receipts):
            raise EpisodeContractError("Completed assistant turn contains unused tool receipts")
        self._assistant_replay = None

    def _state_snapshot(self) -> dict[str, JSONValue]:
        return _json_mapping(
            {
                "messages": self.state.messages,
                "metadata": self.state.metadata,
                "outcome": self.outcome.snapshot(),
            },
            "state_snapshot",
        )

    def _apply_state_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        messages = snapshot.get("messages")
        metadata = snapshot.get("metadata")
        outcome = snapshot.get("outcome")
        if not isinstance(messages, list) or not isinstance(metadata, Mapping) or not isinstance(outcome, Mapping):
            raise EpisodeContractError("Tool result requires messages, metadata, and outcome snapshots")
        self.state.messages = _json_safe_copy(messages)
        self.state.metadata = _json_safe_copy(dict(metadata))
        self.outcome.apply_snapshot(outcome)

    def _apply_effect_state(self, snapshot: Mapping[str, Any]) -> None:
        metadata = snapshot.get("metadata")
        outcome = snapshot.get("outcome")
        if not isinstance(metadata, Mapping) or not isinstance(outcome, Mapping):
            raise EpisodeContractError("Tool receipt requires metadata and outcome effect state")
        current = self.outcome.snapshot()
        current_outcome = cast(dict[str, Any], current["outcome"])
        self.state.metadata = _json_safe_copy(dict(metadata))
        self.outcome.apply_snapshot(outcome)
        merged = self.outcome.snapshot()
        merged_outcome = cast(dict[str, Any], merged["outcome"])
        for stat_field in (
            "per_model_calls",
            "per_model_input_tokens",
            "per_model_output_tokens",
            "wall_clock_s_by_alias",
        ):
            current_values = cast(dict[str, Any], current_outcome[stat_field])
            merged_values = cast(dict[str, Any], merged_outcome[stat_field])
            if "assistant_model" in current_values:
                merged_values["assistant_model"] = current_values["assistant_model"]
        self.outcome.apply_snapshot(merged)

    async def _request_tool_execution(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        raw_tool_call: Any,
        turn_idx: int,
        call_idx: int,
    ) -> str:
        self._tool_sequence += 1
        request_id = f"tool-{self._tool_sequence:06d}"
        raw_id = raw_tool_call.get("id") if isinstance(raw_tool_call, Mapping) else None
        request = ToolExecutionRequest(
            request_id=request_id,
            tool_call_id=str(raw_id or f"call_{turn_idx}_{call_idx}"),
            tool_name=tool_name,
            arguments=_json_mapping(arguments, "arguments"),
            raw_tool_call=_json_mapping(raw_tool_call, "raw_tool_call"),
            turn_idx=turn_idx,
            call_idx=call_idx,
            state_snapshot=self._state_snapshot(),
        )
        waiter: asyncio.Future[ToolExecutionResult] = asyncio.get_running_loop().create_future()
        self._pending_tools[request_id] = _PendingToolExecution(request=request, waiter=waiter)
        await self._lifecycle_events.put(request)
        return (await waiter).payload


async def _proxy_execute_tool_call(
    probe: BaseProbe,
    name: str,
    args: dict[str, Any],
    tc: Any,
    state: ConversationState,
    models: dict[str, Any],
    *,
    turn_idx: int,
    call_idx: int,
) -> str:
    """Replay a Resources-owned effect without executing it again."""
    del state, models
    runtime = getattr(probe, "_conversation_runtime")
    replay = runtime._assistant_replay
    if replay is None or replay.receipt_index >= len(replay.completed.evidence.receipts):
        raise EpisodeContractError("Native UserSim replay requested a tool receipt the Agent did not provide")
    receipt = replay.completed.evidence.receipts[replay.receipt_index]
    if (
        receipt.tool_name != name
        or receipt.arguments != args
        or receipt.raw_tool_call != tc
        or receipt.turn_idx != turn_idx
        or receipt.call_idx != call_idx
    ):
        raise EpisodeContractError("Resources receipt does not match UserSim's native tool replay")
    replay.receipt_index += 1
    runtime._apply_effect_state(receipt.effect_state)
    return receipt.payload


class ProbeToolSession:
    """Independently constructed owner of one episode's native tool effects."""

    @classmethod
    def from_resolved_row(
        cls,
        row: Mapping[str, Any],
        *,
        models: Mapping[str, Any],
    ) -> ProbeToolSession:
        """Create a tool session from the same immutable row as a conversation runtime."""
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
        try:
            constructed = construct_probe_episode(
                {
                    **dict(data),
                    config.persona_column: dict(persona),
                    config.probe_type_column: probe_type,
                    **({"behavioral_profile": dict(profile)} if profile is not None else {}),
                },
                config=config,
                models=self.models,
                provenance=deepcopy(provenance),
            )
        except EpisodeConstructionError as error:
            raise error.cause from error
        self.preamble = constructed.preamble
        self.probe_type = self.preamble.probe_type
        self.outcome = constructed.outcome
        self.probe = constructed.probe
        self.state = ConversationState(outcome=self.outcome)
        self.probe.seed_state_metadata(self.state)
        self._assistant_tools = deepcopy(self.probe.get_tools_for_assistant() or [])
        self._receipts: dict[str, tuple[str, ToolExecutionResult]] = {}
        self._turns: dict[str, _ToolTurnState] = {}
        self._closed = False
        self._lock = asyncio.Lock()

    @property
    def assistant_tools(self) -> list[dict[str, Any]]:
        """Return a defensive tool catalog copy."""
        return deepcopy(self._assistant_tools)

    async def descriptor(self) -> ProbeRuntimeDescriptor:
        """Return the independently resolved tool catalog."""
        return ProbeRuntimeDescriptor(probe_type=self.probe_type, assistant_tools=tuple(self.assistant_tools))

    async def begin_turn(self, context: ToolTurnContext | Mapping[str, Any]) -> None:
        """Initialize Resources state before an Agent owns the model/tool loop."""
        context = ToolTurnContext.from_value(context)
        async with self._lock:
            self._begin_turn_locked(context)

    async def execute_round(self, batch: AssistantToolRound | Mapping[str, Any]) -> ToolRoundReceipt:
        """Execute one ordered batch; Resources assigns every semantic index."""
        batch = AssistantToolRound.from_value(batch)
        async with self._lock:
            if self._closed:
                raise EpisodeContractError("Probe tool session has been closed")
            turn = self._turns.get(batch.turn_id)
            if turn is None:
                raise EpisodeContractError(f"Tool turn {batch.turn_id!r} has not been initialized")
            round_state = self._round_locked(turn, batch)
            for raw_call in batch.assistant_response.get("tool_calls") or []:
                if not isinstance(raw_call, Mapping):
                    raise EpisodeContractError("Tool calls must be mappings")
                raw = _json_mapping(raw_call, "raw_tool_call")
                name, arguments = self._parse_and_validate_call(raw)
                await self._execute_call_locked(
                    turn,
                    round_state,
                    raw=raw,
                    name=name,
                    arguments=arguments,
                )
            return round_state.snapshot()

    async def execute_call(
        self,
        request: ProbeToolCallRequest | Mapping[str, Any],
    ) -> ProbeToolCallResult:
        """Execute one normal Resources tool call and return hidden native evidence.

        The first call lazily initializes its turn from ``request.context``.
        Calls in one Assistant response must arrive in model order. The
        returned ``payload`` is the normal model-visible tool result; the
        receipt and evidence are host metadata and must not be shown to the
        model.
        """
        normalized = ProbeToolCallRequest.from_value(request)
        async with self._lock:
            turn = self._begin_turn_locked(normalized.context)
            batch = AssistantToolRound(
                turn_id=normalized.context.turn_id,
                round_id=normalized.round_id,
                assistant_response=normalized.assistant_response,
            )
            raw_calls = batch.assistant_response.get("tool_calls") or []
            matches = [
                raw_call
                for raw_call in raw_calls
                if isinstance(raw_call, Mapping) and raw_call.get("id") == normalized.tool_call_id
            ]
            if len(matches) != 1:
                raise EpisodeContractError(
                    f"tool_call_id {normalized.tool_call_id!r} must identify exactly one call in the Assistant response"
                )
            raw = _json_mapping(matches[0], "raw_tool_call")
            name, arguments = self._parse_and_validate_call(raw)
            if name != normalized.tool_name or arguments != normalized.arguments:
                raise EpisodeContractError(
                    "Resources route name or arguments disagree with the raw Assistant tool call"
                )
            round_state = self._round_locked(turn, batch)
            receipt = await self._execute_call_locked(
                turn,
                round_state,
                raw=raw,
                name=name,
                arguments=arguments,
            )
            return ProbeToolCallResult(
                receipt=deepcopy(receipt),
                evidence=self._completed_evidence_locked(turn),
                limit_reached=round_state.limit_reached,
            )

    async def rounds(self, turn_id: str) -> tuple[ToolRoundReceipt, ...]:
        """Return Resources receipts for a turn in execution order."""
        async with self._lock:
            turn = self._turns.get(turn_id)
            if turn is None:
                raise EpisodeContractError(f"Tool turn {turn_id!r} has not been initialized")
            return tuple(round_state.snapshot() for round_state in turn.rounds)

    async def complete_turn(self, turn_id: str) -> CompletedTurnEvidence:
        """Snapshot typed Resources evidence for the Agent result."""
        async with self._lock:
            turn = self._turns.get(turn_id)
            if turn is None:
                raise EpisodeContractError(f"Tool turn {turn_id!r} has not been initialized")
            return self._completed_evidence_locked(turn)

    def _begin_turn_locked(self, context: ToolTurnContext) -> _ToolTurnState:
        if self._closed:
            raise EpisodeContractError("Probe tool session has been closed")
        fingerprint = _json_fingerprint(context.to_dict())
        prior = self._turns.get(context.turn_id)
        if prior is not None:
            if _json_fingerprint(prior.context.to_dict()) != fingerprint:
                raise EpisodeContractError(f"turn_id {context.turn_id!r} was reused with different context")
            return prior
        self._apply_request_snapshot(context.state_snapshot)
        turn = _ToolTurnState(context=context)
        self._turns[context.turn_id] = turn
        return turn

    def _round_locked(self, turn: _ToolTurnState, batch: AssistantToolRound) -> _ToolRoundState:
        fingerprint = _json_fingerprint(batch.assistant_response)
        prior = turn.rounds_by_id.get(batch.round_id)
        if prior is not None:
            if prior.fingerprint != fingerprint:
                raise EpisodeContractError(f"round_id {batch.round_id!r} was reused with a different batch")
            return prior
        raw_calls = batch.assistant_response.get("tool_calls") or []
        if not raw_calls:
            raise EpisodeContractError("A tool round requires at least one tool call")
        call_ids = [raw_call.get("id") for raw_call in raw_calls if isinstance(raw_call, Mapping)]
        if len(call_ids) != len(raw_calls) or any(not isinstance(call_id, str) or not call_id for call_id in call_ids):
            raise EpisodeContractError("Every tool call requires a non-empty string id")
        if len(set(call_ids)) != len(call_ids):
            raise EpisodeContractError("Tool call ids must be unique within one Assistant response")
        self.state.messages.append(deepcopy(batch.assistant_response))
        round_state = _ToolRoundState(
            batch=batch,
            fingerprint=fingerprint,
            turn_idx=turn.context.first_tool_turn_idx + len(turn.rounds),
        )
        turn.rounds.append(round_state)
        turn.rounds_by_id[batch.round_id] = round_state
        return round_state

    def _parse_and_validate_call(self, raw: dict[str, JSONValue]) -> tuple[str, dict[str, JSONValue]]:
        if hasattr(self.probe, "parse_tool_call"):
            name, arguments = self.probe.parse_tool_call(raw)
        else:
            name, arguments = _parse_tool_call(raw)
        allowed = frozenset(_tool_name(tool) for tool in self._assistant_tools)
        if name not in allowed:
            raise EpisodeContractError(f"Tool {name!r} is not offered by this Resources session")
        return name, _json_mapping(arguments, "arguments")

    async def _execute_call_locked(
        self,
        turn: _ToolTurnState,
        round_state: _ToolRoundState,
        *,
        raw: dict[str, JSONValue],
        name: str,
        arguments: dict[str, JSONValue],
    ) -> ToolCallReceipt | None:
        raw_calls = round_state.batch.assistant_response.get("tool_calls") or []
        call_id = str(raw["id"])
        call_idx = next(
            index
            for index, raw_call in enumerate(raw_calls)
            if isinstance(raw_call, Mapping) and raw_call.get("id") == call_id
        )
        fingerprint = _json_fingerprint(
            {
                "raw_tool_call": raw,
                "tool_name": name,
                "arguments": arguments,
            }
        )
        prior_fingerprint = round_state.call_fingerprints.get(call_id)
        if prior_fingerprint is not None:
            if prior_fingerprint != fingerprint:
                raise EpisodeContractError(f"tool_call_id {call_id!r} was reused with different arguments")
            return next(
                (receipt for receipt in round_state.receipts if receipt.tool_call_id == call_id),
                None,
            )
        if call_idx != len(round_state.call_fingerprints):
            raise EpisodeContractError("Resources tool calls must arrive in Assistant response order")
        round_state.call_fingerprints[call_id] = fingerprint
        calls_made = sum(len(prior.receipts) for prior in turn.rounds[:-1]) + len(round_state.receipts)
        if turn.context.max_tool_calls is not None and calls_made >= turn.context.max_tool_calls:
            round_state.limit_reached = True
            return None

        previous_builder = get_current_outcome_builder()
        set_current_outcome_builder(self.outcome)
        try:
            payload = await self.probe.execute_tool_call(
                name,
                deepcopy(arguments),
                deepcopy(raw),
                self.state,
                self.models,
                turn_idx=round_state.turn_idx,
                call_idx=call_idx,
            )
        finally:
            set_current_outcome_builder(previous_builder)
        if not isinstance(payload, str):
            payload = json.dumps(payload, ensure_ascii=False, default=str)
        self.state.messages.append({"role": "tool", "content": payload, "tool_call_id": call_id})
        receipt = ToolCallReceipt(
            tool_call_id=call_id,
            tool_name=name,
            arguments=arguments,
            raw_tool_call=raw,
            payload=payload,
            turn_idx=round_state.turn_idx,
            call_idx=call_idx,
            effect_state=self._effect_state(),
        )
        round_state.receipts.append(receipt)
        return receipt

    def _completed_evidence_locked(self, turn: _ToolTurnState) -> CompletedTurnEvidence:
        return CompletedTurnEvidence(
            turn_id=turn.context.turn_id,
            rounds=tuple(round_state.snapshot() for round_state in turn.rounds),
            final_effect_state=self._effect_state(),
        )

    def _effect_state(self) -> dict[str, JSONValue]:
        return _json_mapping(
            {"metadata": self.state.metadata, "outcome": self.outcome.snapshot()},
            "effect_state",
        )

    async def execute(
        self,
        request: ToolExecutionRequest | Mapping[str, object],
    ) -> ToolExecutionResult:
        """Execute one request idempotently through the native probe hook."""
        normalized = ToolExecutionRequest.from_value(request)
        fingerprint = _json_fingerprint(normalized.to_dict())
        async with self._lock:
            if self._closed:
                raise EpisodeContractError("Probe tool session has been closed")
            prior = self._receipts.get(normalized.request_id)
            if prior is not None:
                prior_fingerprint, result = prior
                if prior_fingerprint != fingerprint:
                    raise EpisodeContractError(
                        f"request_id {normalized.request_id!r} was reused with a different request"
                    )
                return deepcopy(result)
            self._apply_request_snapshot(normalized.state_snapshot)
            previous_builder = get_current_outcome_builder()
            set_current_outcome_builder(self.outcome)
            try:
                payload = await self.probe.execute_tool_call(
                    normalized.tool_name,
                    deepcopy(normalized.arguments),
                    deepcopy(normalized.raw_tool_call),
                    self.state,
                    self.models,
                    turn_idx=normalized.turn_idx,
                    call_idx=normalized.call_idx,
                )
            finally:
                set_current_outcome_builder(previous_builder)
            if not isinstance(payload, str):
                payload = json.dumps(payload, ensure_ascii=False, default=str)
            result = ToolExecutionResult(
                request_id=normalized.request_id,
                payload=payload,
                state_snapshot=self.snapshot(),
            )
            self._receipts[normalized.request_id] = (fingerprint, result)
            return deepcopy(result)

    def _apply_request_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        messages = snapshot.get("messages")
        metadata = snapshot.get("metadata")
        outcome = snapshot.get("outcome")
        if not isinstance(messages, list) or not isinstance(metadata, Mapping) or not isinstance(outcome, Mapping):
            raise EpisodeContractError("Tool request requires messages, metadata, and outcome snapshots")
        self.state.messages = _json_safe_copy(messages)
        self.state.metadata = _json_safe_copy(dict(metadata))
        self.outcome.apply_snapshot(outcome)

    def snapshot(self) -> dict[str, JSONValue]:
        """Return JSON-safe mutable tool evidence for the conversation host."""
        return _json_mapping(
            {
                "messages": self.state.messages,
                "metadata": self.state.metadata,
                "outcome": self.outcome.snapshot(),
            },
            "state_snapshot",
        )

    async def evidence(self) -> dict[str, Any]:
        """Return the tool session's current auditable state."""
        async with self._lock:
            completed_receipts = [
                receipt.to_dict()
                for turn in self._turns.values()
                for round_receipt in turn.rounds
                for receipt in round_receipt.receipts
            ]
            return {
                "probe_type": self.probe_type,
                "assistant_tools": self.assistant_tools,
                "conversation_metadata": deepcopy(self.state.metadata),
                "simulation_traces": [trace.to_dict() for trace in self.outcome.traces()],
                "receipts": [
                    *[result.to_dict() for _fingerprint, result in self._receipts.values()],
                    *completed_receipts,
                ],
            }

    async def close(self) -> None:
        """Reject future effects and release receipt state."""
        self._closed = True


class ProbeEpisodeRuntime:
    """Compatibility facade composing conversation control and local tools."""

    @classmethod
    def from_resolved_row(
        cls,
        row: Mapping[str, Any],
        *,
        models: Mapping[str, Any],
    ) -> ProbeEpisodeRuntime:
        """Create the compatibility runtime from one resolved row."""
        conversation = ConversationRuntime.from_resolved_row(row, models=models)
        tools = ProbeToolSession.from_resolved_row(row, models=models)
        return cls._from_parts(conversation, tools)

    @classmethod
    def _from_parts(
        cls,
        conversation: ConversationRuntime,
        tools: ProbeToolSession,
    ) -> ProbeEpisodeRuntime:
        instance = cls.__new__(cls)
        instance.conversation = conversation
        instance.tool_session = tools
        instance._compat_turn = None
        instance._compat_request_id = None
        instance._compat_active_activation = None
        instance._compat_transitions = {}
        return instance

    def __init__(self, **kwargs: Any) -> None:
        self.conversation = ConversationRuntime(**kwargs)
        self.tool_session = ProbeToolSession(**kwargs)
        self._compat_turn: dict[str, Any] | None = None
        self._compat_request_id: str | None = None
        self._compat_active_activation: ActivationRequest | None = None
        self._compat_transitions: dict[str, tuple[str, ActivationRequest | EpisodeLifecycleComplete]] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.conversation, name)

    @property
    def assistant_tools(self) -> list[dict[str, Any]]:
        return self.conversation.assistant_tools

    async def descriptor(self) -> ProbeRuntimeDescriptor:
        return await self.conversation.descriptor()

    async def advance(
        self,
        result: ActivationResult | Mapping[str, Any] | None = None,
    ) -> ActivationRequest | EpisodeLifecycleComplete:
        if result is not None:
            candidate = ActivationResult.from_value(result)
            prior = self._compat_transitions.get(candidate.activation_id)
            if prior is not None:
                fingerprint = _json_fingerprint(candidate.to_dict())
                if fingerprint != prior[0]:
                    raise EpisodeContractError(
                        f"lifecycle id {candidate.activation_id!r} was reused with a different result"
                    )
                return deepcopy(prior[1])
        if self._compat_turn is not None:
            if result is None:
                raise EpisodeContractError("A started assistant turn requires a matching activation result")
            normalized = ActivationResult.from_value(result)
            if normalized.activation_id != self._compat_request_id:
                raise EpisodeContractError(
                    f"activation_id {normalized.activation_id!r} is not the pending activation "
                    f"{self._compat_request_id!r}"
                )
            assert self._compat_active_activation is not None
            self.conversation._validate_recorded_response(self._compat_active_activation, normalized)
            turn = self._compat_turn
            request = cast(AssistantTurnRequest, turn["request"])
            model_calls = cast(list[AssistantModelCall], turn["model_calls"])
            transcript = cast(list[dict[str, Any]], turn["transcript"])
            model_calls.append(AssistantModelCall(response=normalized.response, usage=normalized.usage))
            transcript.append(normalized.response)
            tool_calls = normalized.response.get("tool_calls") or []
            if tool_calls:
                receipt = await self.tool_session.execute_round(
                    AssistantToolRound(
                        turn_id=request.turn_id,
                        round_id=f"round-{len(model_calls):06d}",
                        assistant_response=normalized.response,
                    )
                )
                transcript.extend(receipt.tool_messages)
                turn["tool_limit_reached"] = receipt.limit_reached
            tool_count = sum(
                len(round_receipt.receipts) for round_receipt in await self.tool_session.rounds(request.turn_id)
            )
            if tool_calls and request.loop_policy.should_continue(
                model_calls=len(model_calls),
                tool_calls=tool_count,
            ):
                event = self._compat_activation(request, model_calls=len(model_calls), tool_calls=tool_count)
                self._compat_transitions[normalized.activation_id] = (
                    _json_fingerprint(normalized.to_dict()),
                    deepcopy(event),
                )
                return event
            completed = CompletedAssistantTurn(
                turn_id=request.turn_id,
                transcript=tuple(transcript),
                model_calls=tuple(model_calls),
                evidence=await self.tool_session.complete_turn(request.turn_id),
            )
            self._compat_turn = None
            self._compat_request_id = None
            self._compat_active_activation = None
            event = await self.conversation.advance(completed)
        else:
            event = await self.conversation.advance(result)
        adapted = await self._adapt_assistant_event(event)
        if result is not None:
            normalized = ActivationResult.from_value(result)
            self._compat_transitions[normalized.activation_id] = (
                _json_fingerprint(normalized.to_dict()),
                deepcopy(adapted),
            )
        return adapted

    async def _adapt_assistant_event(
        self,
        event: ActivationRequest | AssistantTurnRequest | EpisodeLifecycleComplete,
    ) -> ActivationRequest | EpisodeLifecycleComplete:
        if not isinstance(event, AssistantTurnRequest):
            return event
        await self.tool_session.begin_turn(event.tool_context)
        self._compat_turn = {
            "request": event,
            "model_calls": [],
            "transcript": [],
            "tool_limit_reached": False,
        }
        return self._compat_activation(event, model_calls=0, tool_calls=0)

    def _compat_activation(
        self,
        request: AssistantTurnRequest,
        *,
        model_calls: int,
        tool_calls: int,
    ) -> ActivationRequest:
        self._compat_request_id = f"{request.turn_id}-call-{model_calls + 1:06d}"
        parameters = deepcopy(request.parameters)
        tools = (
            request.tools
            if request.loop_policy.tools_enabled(
                model_calls=model_calls,
                tool_calls=tool_calls,
                limit_reached=bool(self._compat_turn["tool_limit_reached"]),
            )
            else ()
        )
        transcript = cast(list[dict[str, Any]], self._compat_turn["transcript"])
        activation = ActivationRequest(
            activation_id=self._compat_request_id,
            role="assistant",
            model_alias=request.model_alias,
            messages=request.model_messages(transcript),
            parameters=parameters,
            tools=tools,
            continues_turn=model_calls > 0,
        )
        self._compat_active_activation = activation
        return activation

    async def tool_result(self, tool_call_id: str) -> str:
        return await self.conversation.tool_result(tool_call_id)

    async def executed_tool_calls(self) -> list[ExecutedToolCall]:
        return await self.conversation.executed_tool_calls()

    async def evidence(self) -> dict[str, Any]:
        return await self.conversation.evidence()

    async def finalize(self) -> dict[str, Any]:
        return await self.conversation.finalize()

    async def close(self) -> None:
        await self.conversation.close()
        await self.tool_session.close()


def _tool_name(tool: Mapping[str, Any]) -> str:
    function = tool.get("function")
    if not isinstance(function, Mapping):
        raise ValueError(f"Invalid assistant tool schema: {tool!r}")
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"Assistant tool schema has no function name: {tool!r}")
    return name


def _parse_tool_call(raw_call: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    function = raw_call.get("function")
    if not isinstance(function, Mapping):
        return str(raw_call.get("name") or "<unknown>"), {}
    name = str(function.get("name") or "<unknown>")
    arguments = function.get("arguments")
    if isinstance(arguments, Mapping):
        return name, deepcopy(dict(arguments))
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except (TypeError, ValueError):
            return name, {}
        return name, parsed if isinstance(parsed, dict) else {}
    return name, {}


def _effect_state_from_snapshot(snapshot: Mapping[str, Any]) -> dict[str, JSONValue]:
    metadata = snapshot.get("metadata")
    outcome = snapshot.get("outcome")
    if not isinstance(metadata, Mapping) or not isinstance(outcome, Mapping):
        raise EpisodeContractError("Tool turn snapshot requires metadata and outcome mappings")
    return _json_mapping(
        {
            "metadata": metadata,
            "outcome": outcome,
        },
        "effect_state",
    )


def _project_document_tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project repeated JSON document results to one current full-body copy."""
    result_names: dict[int, str] = {}
    pending: list[tuple[str | None, str]] = []
    current_user_turn = 0
    retrievals: list[tuple[int, int, dict[str, Any]]] = []
    for index, message in enumerate(messages):
        if message.get("role") == "user":
            current_user_turn += 1
            pending = []
        elif message.get("role") == "assistant":
            pending = []
            for raw_call in message.get("tool_calls") or []:
                if isinstance(raw_call, Mapping):
                    function = raw_call.get("function") or {}
                    pending.append(
                        (
                            str(raw_call.get("id")) if raw_call.get("id") else None,
                            str(function.get("name") or "") if isinstance(function, Mapping) else "",
                        )
                    )
        elif message.get("role") == "tool":
            call_id = str(message.get("tool_call_id") or "")
            match = next(
                (position for position, (pending_id, _name) in enumerate(pending) if pending_id == call_id),
                0 if pending else None,
            )
            if match is not None:
                _pending_id, name = pending.pop(match)
                result_names[index] = name
                if name == "kb_search":
                    try:
                        payload = json.loads(str(message.get("content") or ""))
                    except (TypeError, ValueError):
                        continue
                    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
                        retrievals.append((index, current_user_turn, payload))
    newest: dict[str, tuple[int, int]] = {}
    for message_index, _turn, payload in retrievals:
        for document_index, document in enumerate(payload["results"]):
            if isinstance(document, Mapping) and document.get("id"):
                newest[str(document["id"])] = (message_index, document_index)
    projected = deepcopy(messages)
    for message_index, document_turn, payload in retrievals:
        documents: list[Any] = []
        changed = False
        for document_index, document in enumerate(payload["results"]):
            if not isinstance(document, Mapping) or not document.get("id"):
                documents.append(document)
                continue
            keep_body = (
                newest[str(document["id"])] == (message_index, document_index) and current_user_turn - document_turn < 8
            )
            if keep_body:
                documents.append(document)
            else:
                changed = True
                documents.append({key: value for key, value in document.items() if key != "body"})
        if changed:
            projected[message_index] = {
                **projected[message_index],
                "content": json.dumps({**payload, "results": documents}, ensure_ascii=False),
            }
    return projected


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

    def __init__(self, runtime: ConversationRuntime, alias: str, host_model: Any) -> None:
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


def _json_mapping(value: object, label: str) -> dict[str, JSONValue]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a JSON mapping")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{label} keys must be strings")
    normalized = _json_safe_copy(dict(value))
    if not isinstance(normalized, dict):
        raise TypeError(f"{label} must be a JSON mapping")
    return cast(dict[str, JSONValue], normalized)


def _json_safe_copy(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError) as error:
        raise TypeError("Activation payload must be JSON serializable") from error


def _json_fingerprint(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
