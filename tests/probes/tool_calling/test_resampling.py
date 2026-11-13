# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A tool_calling turn the assistant judge rejects is undone, tool calls included, and regenerated."""

from __future__ import annotations

import json
from typing import Any

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
_UNGROUNDED = "[UNGROUNDED] No tool returned that temperature."


def _last_text(messages: Any) -> str:
    last = messages[-1]
    return str(last.get("content", "") if isinstance(last, dict) else getattr(last, "content", ""))


def _judge_failing(
    replies: int, explanation: str, *, user_done: bool = True
) -> tuple[type[ScriptedModel], dict[str, int]]:
    """A model factory whose judge fails the first ``replies`` assistant replies it is shown, and a tally.

    ``user_done=False`` has the summary model say the user is not done, so the
    conversation runs every turn.
    """
    tally = {"shown": 0, "failed": 0}

    class _Judge(ScriptedModel):
        CONTENT = {**ScriptedModel.CONTENT, **({} if user_done else {"summary": "no"})}

        async def acompletion(self, messages, **kwargs):
            reply = await super().acompletion(messages, **kwargs)
            if self.role == "judge" and _ASSISTANT_JUDGE in _last_text(messages):
                tally["shown"] += 1
                if tally["failed"] < replies:
                    tally["failed"] += 1
                    reply.message.content = f"<explanation>{explanation}</explanation><rating>failure</rating>"
            return reply

    return _Judge, tally


async def _episode(model_factory: type[ScriptedModel], **config: Any):
    return await run_episode(
        "tool_calling",
        config=ConversationSimulatorConfig(name="conversation_messages", locale="en_US", **config),
        persona={"first_name": "Ana", "last_name": "Souza", "age": 35, "city": "Austin"},
        data={"theme": {"type": "weather", "description": "Check the weather."}, "tools": [_TOOL]},
        model_factory=model_factory,
    )


async def test_a_rejected_tool_turn_leaves_only_its_verdict() -> None:
    judge, tally = _judge_failing(1, _UNGROUNDED)
    run = await _episode(judge, max_turns=2, max_assistant_attempts=2)

    messages = json.loads(run.row["conversation_messages"])
    metadata = json.loads(run.row["conversation_metadata"])
    outcome = json.loads(run.row["simulation_outcome"])
    traces = json.loads(run.row["simulation_traces"])
    tool_messages = [m for m in messages if m["role"] == "tool"]
    assistant_calls = [p for p in run.prompts if p["role"] == "assistant" and p["tools"]]

    assert tally["failed"] == 1
    assert outcome["n_assistant_retries"] == outcome["n_assistant_ungrounded"] == 1
    assert len(tool_messages) == len(metadata["tools_called"]) == run.row["num_tool_calls"]
    assert len(assistant_calls) > len(tool_messages)
    assert len([t for t in traces if t["kind"] == "api_response_judge"]) == len(tool_messages)
    assert [r["attempt"] for r in metadata["assistant_judge_ratings"] if r["turn_idx"] == 0] == [0, 1]


async def test_the_assistant_judge_reads_the_turns_tool_results() -> None:
    run = await _episode(ScriptedModel, max_turns=1, assistant_judge=True)
    judged = [
        p["messages"][-1]["content"]
        for p in run.prompts
        if p["role"] == "judge" and _ASSISTANT_JUDGE in p["messages"][-1]["content"]
    ]
    assert judged
    assert all("Assistant [tool calls]: get_weather" in prompt and "Tool response: " in prompt for prompt in judged)


async def test_an_ungrounded_reply_is_flagged_without_frustrating_the_user() -> None:
    """The simulated user never sees tool results, so a reply failed only for them leaves its patience alone."""
    ungrounded, _ = _judge_failing(99, _UNGROUNDED, user_done=False)
    off_topic, _ = _judge_failing(99, "The reply ignores the question.", user_done=False)

    flagged = await _episode(ungrounded, max_turns=3, assistant_judge=True)
    unhelpful = await _episode(off_topic, max_turns=3, assistant_judge=True)

    def failures_seen(run) -> list[int]:
        return [e["assistant_failures"] for e in json.loads(run.row["conversation_metadata"])["frustration_events"]]

    flagged_outcome = json.loads(flagged.row["simulation_outcome"])
    assert flagged_outcome["n_assistant_inline_failures"] == flagged_outcome["n_assistant_ungrounded"] > 0
    assert all(seen == 0 for seen in failures_seen(flagged))
    assert max(failures_seen(unhelpful), default=0) >= 2


def test_the_turn_state_hooks_restore_the_held_credentials() -> None:
    probe = ToolCallingProbe.__new__(ToolCallingProbe)
    probe.held_credentials = {"Account password": "old-password"}
    snapshot = probe.snapshot_turn_state()
    probe.held_credentials["Account password"] = "new-password"

    probe.restore_turn_state(snapshot)

    assert probe.held_credentials == {"Account password": "old-password"}
