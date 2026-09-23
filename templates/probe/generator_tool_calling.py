# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Probe template — tool-calling capability test.

Use this shape when the probe tests the assistant's ability to use
tools the user provides. Different from agentic-safety: the goal is
NOT mock-environment safety probing but rather quality scoring of
how well the assistant chooses + parameterizes + sequences tools.
The ``tool_calling`` probe is the only example today.

What ``ToolCallingMixin`` does:
- Sets ``allow_early_stop_at_turn = False`` until at least one tool
  call has been recorded in ``state.metadata["tools_called"]``.
  Rationale: a tool_calling trajectory must engage tools before
  "user satisfied" is meaningful — early-stop on a never-tooled
  trajectory would silently mask the model's failure to use them.
- (Does NOT bypass ``ConversationLoop`` — this shape goes through
  the unified loop. The tool-execution glue lives in
  ``after_assistant_turn``.)

What you provide:
- ``__init__(...)`` — sample tools from ``self._data[cfg.tools_column]``
  (typically a CLI-provided tool registry), validate, store on
  ``self``.
- ``get_user_system_prompt`` — typical persona-grounded user prompt.
- ``get_tools_for_assistant`` — returns the sampled OpenAI tool specs.
- ``after_assistant_turn`` — intercepts tool calls, executes them
  (or simulates execution), appends tool-response messages, records
  in ``state.metadata["tools_called"]``.
- ``should_succeed`` — `len(tools_called) > 0` (no tool calls = sim
  failure).
- ``build_result_extras`` — surface ``num_tool_calls`` and the tool
  subset used.

Total LOC: ~200. ``tool_calling`` is a real example at ~410 LOC
(extra LOC: tool-name validation, tool-result simulation via API-
response model, tool-call schema verifier — substantive logic the
template can't fully express).
"""

from __future__ import annotations

import json
from typing import Any

from usersim.engine.core.probes import (
    BaseProbe,
    ToolCallingMixin,
    register_probe,
)


@register_probe(
    family="demo_tool_calling",
    prompt_version="v1.0",
    variants=("default",),
)
class DemoToolCallingProbe(ToolCallingMixin, BaseProbe):
    """Tool-calling capability probe with mid-loop tool execution."""

    label = "demo_tool_calling"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        cfg = self._cfg
        if not getattr(cfg, "tools_column", None):
            raise ValueError("demo_tool_calling: cfg.tools_column not configured")
        # Real probe: parse self._data[cfg.tools_column], validate,
        # sample subset. See tool_calling/generator.py for the
        # canonical tool-sampling logic.
        self._tools: list[dict] = []  # populated from row

    def get_user_system_prompt(self) -> str:
        return (
            f"You are a user named "
            f"{self._persona.get('first_name', 'there')} "
            f"who needs to accomplish a task that requires using tools. "
            f"Make a concrete request the assistant can fulfill via "
            f"the available tools."
        )

    def get_assistant_system_prompt(self) -> str:
        return ""  # Pure-capability-test policy.

    def get_tools_for_assistant(self) -> list | None:
        return self._tools

    async def after_assistant_turn(
        self,
        models: dict,
        state: Any,
        response: Any,
        cfg: Any,
    ) -> str:
        # Real probe: extract tool_calls, validate against the tool
        # schema, simulate execution via an API-response model
        # (or mock), append tool-response messages, record in
        # state.metadata["tools_called"].
        content = response.get("content", "") if isinstance(response, dict) else ""
        tool_calls = response.get("tool_calls") if isinstance(response, dict) else None
        state.messages.append(
            {
                "role": "assistant",
                "content": content or "",
                "tool_calls": tool_calls or None,
            }
        )
        if tool_calls:
            for idx, tc in enumerate(tool_calls):
                tool_name = tc.get("function", {}).get("name", "<unknown>")
                state.metadata.setdefault("tools_called", []).append(tool_name)
                # Simulated tool response.
                state.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id", f"call_{idx}"),
                        "content": json.dumps({"ok": True}),
                    }
                )
        return content or ""

    def should_succeed(self, state: Any) -> bool:
        # Sim-side guardrail: zero tool calls = the probe didn't
        # really happen. Mirrors the tool_calling probe's policy.
        return len(state.metadata.get("tools_called", [])) > 0

    def build_result_extras(self, state: Any) -> dict:
        return {
            "num_turns": sum(1 for m in state.messages if m.get("role") == "user"),
            "num_tool_calls": len(state.metadata.get("tools_called", [])),
            "tool_subset": json.dumps(state.metadata.get("tools_called", [])),
        }
