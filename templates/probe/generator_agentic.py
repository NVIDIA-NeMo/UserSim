# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Probe template — agentic single-user-turn + tool-interception loop.

Use this shape when the probe has ONE user turn (verbatim from a
bank) followed by a multi-assistant-turn loop where every tool call
the assistant emits is intercepted and answered with a canned mock
response. The ``safety_agentic`` probe is the only example today.

**This shape OVERRIDES ``run_dispatch`` and bypasses ``ConversationLoop.run``.**
The unified loop's machinery (fourth-wall prefilter, user-judge gate,
follow-up generation, role-violation telemetry) all assume back-and-
forth conversation with a user-LLM. This probe has NO user-LLM at
all (single user turn from the bank, zero follow-ups), so threading
the agentic shape through the unified loop would either skip every
hook or call user-LLM machinery uselessly. The substrate
(``BaseProbe`` + ``BankBackedProbe`` + ``AgenticMixin``) still
provides bank load + placeholder warning + version pinning + the
``should_succeed`` / ``build_result_extras`` hooks; this probe owns
the simulate loop directly.

What ``AgenticMixin`` does:
- Inherits from ``BankVerbatimMixin``.
- Sets ``should_continue_after_turn = False`` once the assistant
  emits a turn with no tool calls.
- Sets ``should_inline_judge_user_turn = False`` for turn 0 (the
  user's verbatim action request is auditable as-is — no judge call).

What you provide:
- ``derive_task(persona, bank, *, cfg)`` — picks the action_request.
- ``__init__`` — pre-formats the OpenAI tool specs and the declared
  tool names set.
- ``get_tools_for_assistant`` — returns the OpenAI tool specs (the
  assistant SHOULD know what tools exist; ``tools=`` parameter is
  not "system prompt" — it's API surface).
- ``run_dispatch(*, models, data, cfg)`` — your custom simulate
  loop. Inject verbatim user turn-1, loop assistant calls, intercept
  tool calls + append mock responses.
- ``build_result_extras`` — overrides ``num_turns`` to assistant-
  message count (NOT user-turn count — single user turn here).

Total LOC: ~250. ``safety_agentic`` is a real example at ~370 LOC
(close to this template since they share the same custom-loop
pattern; the extra LOC is bank-specific class-construction handling).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from usersim.engine.core.outcomes import (
    OutcomeBuilder,
    OutcomeStatus,
    WarningKind,
)
from usersim.engine.core.probes import (
    AgenticMixin,
    BankBackedProbe,
    register_probe,
)
from usersim.engine.core.simulation import ConversationState, make_result


@dataclass
class _DemoActionRequest:
    id: str
    placeholder: bool
    sub_protocol: str
    initial_user_prompt: str
    simulated_tools: list  # list of tool specs
    mock_tool_responses: dict  # tool_name -> canned response


def _demo_agentic_loader():
    raise NotImplementedError("replace with your real agentic-bank loader")


@register_probe(
    family="demo_agentic",
    prompt_version="v1.0",
    variants=("default",),
)
class DemoAgenticProbe(AgenticMixin, BankBackedProbe):
    """Single-user-turn + multi-assistant-turn tool-interception probe."""

    label = "demo_agentic"
    bank_loader = staticmethod(_demo_agentic_loader)
    placeholder_warning_kind = WarningKind.USED_PLACEHOLDER_AGENTIC_ACTION
    bank_version_key = "agentic"  # non-locale

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self._task is None:
            raise ValueError("no resolvable action_request")
        # Pre-format OpenAI tool specs once.
        self._api_tools = []  # _format_tools_for_api(self._task.simulated_tools)
        self._declared_tool_names: set[str] = set()  # {t.name for t in ...}

    def derive_task(
        self,
        persona: dict,
        bank: Any,
        *,
        cfg: Any,
    ) -> _DemoActionRequest | None:
        return _DemoActionRequest(
            id="AR-DEMO-001",
            placeholder=True,
            sub_protocol="unsanctioned_action",
            initial_user_prompt="Please do X with my data.",
            simulated_tools=[],
            mock_tool_responses={},
        )

    def get_user_system_prompt(self) -> str:
        return ""  # Unused — no user-LLM call.

    def get_assistant_system_prompt(self) -> str:
        return ""  # Pure-capability-test policy.

    def get_tools_for_assistant(self) -> list | None:
        return self._api_tools

    def after_assistant_turn(
        self,
        models: dict,
        state: Any,
        response: Any,
        cfg: Any,
    ) -> str:
        return ""  # Unused — run_dispatch owns the loop.

    def build_result_extras(self, state: Any) -> dict:
        extras = super().build_result_extras(state)
        if self._task is not None:
            extras["probe_variant"] = f"{self._task.sub_protocol}::demo"
            extras["action_request_id"] = self._task.id
            extras["sub_protocol"] = self._task.sub_protocol
            # AgenticMixin convention: num_turns = assistant-message count.
            extras["num_turns"] = sum(1 for m in state.messages if m.get("role") == "assistant")
            extras["num_tool_calls"] = len(state.metadata.get("attempted_actions") or [])
        return extras

    def should_succeed(self, state: Any) -> bool:
        return bool(state.metadata.get("action_request_id"))

    def run_dispatch(
        self,
        *,
        models: dict[str, Any],
        data: dict[str, Any],
        cfg: Any,
    ) -> dict:
        """Custom agentic simulate loop — see safety_agentic for the real one."""
        builder = self._outcome_builder or OutcomeBuilder(
            provenance=self._provenance,
        )
        state = ConversationState(outcome=builder)
        # Seed metadata BEFORE turn-1 — survives even if turn-1 fails.
        state.metadata["action_request_id"] = self._task.id
        state.metadata["sub_protocol"] = self._task.sub_protocol
        state.metadata["attempted_actions"] = []
        # Inject verbatim user turn-1.
        state.messages.append(
            {
                "role": "user",
                "content": self._task.initial_user_prompt,
            }
        )
        # Multi-assistant-turn tool-interception loop.
        # See safety_agentic.SafetyAgenticProbe.run_dispatch for the
        # real implementation (call_llm wrapping, tool-call extraction,
        # mock-response injection, infrastructure-failure handling).
        outcome = builder.finalize(status=OutcomeStatus.OK)
        result = make_result(
            state.messages,
            state.metadata,
            self.should_succeed(state),
            outcome=outcome,
            traces=builder.traces(),
        )
        result.update(self.build_result_extras(state))
        return result
