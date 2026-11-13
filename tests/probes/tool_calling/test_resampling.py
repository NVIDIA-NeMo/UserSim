# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A tool_calling turn the assistant judge rejects is undone, tool calls included, and regenerated."""

from __future__ import annotations

import json

from usersim.engine.probes.tool_calling.probe import ToolCallingProbe
from usersim.extension import ConversationSimulatorConfig
from usersim.testing import ScriptedModel, run_episode

_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
_ASSISTANT_JUDGE = "Judge the assistant's response in this conversation."


class _RejectsTheFirstReply(ScriptedModel):
    """A judge that rejects the first assistant reply it is shown and passes everything else."""

    rejected = 0

    async def acompletion(self, messages, **kwargs):
        reply = await super().acompletion(messages, **kwargs)
        last = messages[-1]
        text = str(last.get("content", "") if isinstance(last, dict) else getattr(last, "content", ""))
        if self.role == "judge" and _ASSISTANT_JUDGE in text and not type(self).rejected:
            type(self).rejected += 1
            reply.message.content = "<explanation>Wrong city.</explanation><rating>failure</rating>"
        return reply


async def test_a_rejected_tool_turn_leaves_only_its_verdict() -> None:
    _RejectsTheFirstReply.rejected = 0
    run = await run_episode(
        "tool_calling",
        config=ConversationSimulatorConfig(
            name="conversation_messages", locale="en_US", max_turns=2, max_assistant_attempts=2
        ),
        persona={"first_name": "Ana", "last_name": "Souza", "age": 35, "city": "Austin"},
        data={"theme": {"type": "weather", "description": "Check the weather."}, "tools": [_TOOL]},
        model_factory=_RejectsTheFirstReply,
    )

    messages = json.loads(run.row["conversation_messages"])
    metadata = json.loads(run.row["conversation_metadata"])
    outcome = json.loads(run.row["simulation_outcome"])
    traces = json.loads(run.row["simulation_traces"])
    tool_messages = [m for m in messages if m["role"] == "tool"]
    assistant_calls = [p for p in run.prompts if p["role"] == "assistant" and p["tools"]]

    assert _RejectsTheFirstReply.rejected == 1
    assert outcome["n_assistant_retries"] == 1
    assert len(tool_messages) == len(metadata["tools_called"]) == run.row["num_tool_calls"]
    assert len(assistant_calls) > len(tool_messages)
    assert len([t for t in traces if t["kind"] == "api_response_judge"]) == len(tool_messages)
    assert [r["attempt"] for r in metadata["assistant_judge_ratings"] if r["turn_idx"] == 0] == [0, 1]


async def test_the_assistant_judge_reads_the_turns_tool_results() -> None:
    _RejectsTheFirstReply.rejected = 1
    run = await run_episode(
        "tool_calling",
        config=ConversationSimulatorConfig(
            name="conversation_messages", locale="en_US", max_turns=1, assistant_judge=True
        ),
        persona={"first_name": "Ana", "last_name": "Souza", "age": 35, "city": "Austin"},
        data={"theme": {"type": "weather", "description": "Check the weather."}, "tools": [_TOOL]},
        model_factory=_RejectsTheFirstReply,
    )
    judged = [
        p["messages"][-1]["content"]
        for p in run.prompts
        if p["role"] == "judge" and _ASSISTANT_JUDGE in p["messages"][-1]["content"]
    ]
    assert judged
    assert all("Assistant [tool calls]: get_weather" in prompt and "Tool response: " in prompt for prompt in judged)


def test_the_turn_state_hooks_restore_the_held_credentials() -> None:
    probe = ToolCallingProbe.__new__(ToolCallingProbe)
    probe.held_credentials = {"Account password": "old-password"}
    snapshot = probe.snapshot_turn_state()
    probe.held_credentials["Account password"] = "new-password"

    probe.restore_turn_state(snapshot)

    assert probe.held_credentials == {"Account password": "old-password"}
