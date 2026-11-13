# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The judge checks each simulated tool response before the assistant sees it, and a rejected one is regenerated."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

from usersim.engine.probes.tool_calling.probe import ToolCallingProbe, _simulate_tool_response
from usersim.extension import ConversationSimulatorConfig
from usersim.testing import ScriptedModel, run_episode

_SPEC = {"name": "get_weather", "description": "Current weather for a city.", "parameters": {"type": "object"}}
_CALL = {"function": {"name": "get_weather", "arguments": '{"city":"Tokyo"}'}}
_PROBE = "usersim.engine.probes.tool_calling.probe"


async def _simulate(api_replies: list[str], verdicts: list[Any], **kwargs: Any):
    """Run the simulator with scripted API replies and judge verdicts; return the result and the calls."""
    api = iter(api_replies)
    judged = iter(verdicts)
    api_prompts: list[str] = []
    judge_prompts: list[str] = []

    async def fake_call_llm(models, alias, messages, **_):
        api_prompts.append(messages[-1]["content"])
        return {"content": next(api)}

    async def fake_judge(models, alias, prompt):
        judge_prompts.append(prompt)
        verdict = next(judged)
        if isinstance(verdict, Exception):
            raise verdict
        passed, explanation = verdict
        return explanation, "success" if passed else "failure", passed

    with (
        patch(f"{_PROBE}.acall_llm", side_effect=fake_call_llm),
        patch(f"{_PROBE}.run_inline_judge", side_effect=fake_judge),
    ):
        result = await _simulate_tool_response({}, _SPEC, _CALL, [], today="Friday, 2026-11-06", **kwargs)
    return result, api_prompts, judge_prompts


async def test_a_rejected_response_is_regenerated_with_the_judges_reason() -> None:
    result, api_prompts, judge_prompts = await _simulate(
        ['{"temp_f": 70}', '{"temp_c": 21}'],
        [(False, "Tokyo reports Celsius."), (True, "Fine.")],
        judge=True,
    )
    assert result.content == '{"temp_c": 21}'
    assert result.verdicts == [(False, "Tokyo reports Celsius."), (True, "Fine.")]
    assert result.unverified is None
    assert "Tokyo reports Celsius." in api_prompts[1] and '{"temp_f": 70}' in api_prompts[1]
    assert "Tokyo reports Celsius." not in api_prompts[0]
    assert '{"temp_c": 21}' in judge_prompts[1] and "Tokyo" in judge_prompts[1]


async def test_when_every_attempt_is_rejected_the_last_stands_unverified() -> None:
    result, _, _ = await _simulate(
        ['{"temp": 1}', '{"temp": 2}'], [(False, "Wrong city."), (False, "Still wrong.")], judge=True
    )
    assert result.content == '{"temp": 2}'
    assert result.unverified == "the judge rejected all 2 attempts it saw"


async def test_a_reply_that_is_not_json_uses_an_attempt_before_the_judge_sees_any() -> None:
    result, _, judge_prompts = await _simulate(['{"temp": 2', '{"temp": 22}'], [(True, "Fine.")], judge=True)
    assert (result.content, result.rerolls, len(judge_prompts)) == ('{"temp": 22}', 1, 1)


async def test_a_judge_outage_leaves_the_response_standing_unverified() -> None:
    result, _, _ = await _simulate(['{"temp": 22}'], [RuntimeError("rate limited")], judge=True)
    assert result.content == '{"temp": 22}'
    assert result.unverified == "the judge gave no verdict (RuntimeError)"


async def test_without_the_judge_the_first_json_reply_stands() -> None:
    result, _, judge_prompts = await _simulate(['{"temp": 22}'], [], judge=False)
    assert (result.content, result.verdicts, judge_prompts) == ('{"temp": 22}', [], [])


async def test_the_judge_knows_the_credentials_were_checked() -> None:
    _, _, judge_prompts = await _simulate(
        ['{"temp": 22}'], [(True, "Fine.")], judge=True, credential_parameters=("api_key",)
    )
    assert "credentials (`api_key`) are valid" in judge_prompts[0]


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


def _last_content(messages: Any) -> str:
    last = messages[-1]
    return str(last.get("content", "") if isinstance(last, dict) else getattr(last, "content", ""))


class _RejectsEveryResponse(ScriptedModel):
    """A judge that rejects every simulated tool response and passes everything else."""

    async def acompletion(self, messages, **kwargs):
        reply = await super().acompletion(messages, **kwargs)
        if self.role == "judge" and "simulated API response" in _last_content(messages):
            reply.message.content = "<explanation>The city is wrong.</explanation><rating>failure</rating>"
        return reply


async def _episode(*, model_factory=ScriptedModel, probe_options: dict | None = None):
    return await run_episode(
        "tool_calling",
        config=ConversationSimulatorConfig(
            name="conversation_messages", locale="en_US", max_turns=2, probe_options=probe_options or {}
        ),
        persona={"first_name": "Ana", "last_name": "Souza", "age": 35, "city": "Austin"},
        data={"theme": {"type": "weather", "description": "Check the weather."}, "tools": [_TOOL]},
        model_factory=model_factory,
    )


def _response_checks(run) -> list[str]:
    return [
        _last_content(p["messages"])
        for p in run.prompts
        if p["role"] == "judge" and "simulated API response" in _last_content(p["messages"])
    ]


async def test_every_simulated_response_is_judged_and_traced() -> None:
    run = await _episode()
    outcome = json.loads(run.row["simulation_outcome"])
    traces = [t for t in json.loads(run.row["simulation_traces"]) if t["kind"] == "api_response_judge"]
    assert _response_checks(run)
    assert len(traces) == len(_response_checks(run))
    assert all(t["rating"] == "success" for t in traces)
    assert outcome["n_api_response_rejections"] == 0
    assert not [w for w in outcome["warnings"] if w["kind"] == ToolCallingProbe.unverified_response_warning_kind]


async def test_rejections_are_counted_and_an_unverified_response_is_flagged() -> None:
    run = await _episode(model_factory=_RejectsEveryResponse)
    outcome = json.loads(run.row["simulation_outcome"])
    flagged = [w for w in outcome["warnings"] if w["kind"] == ToolCallingProbe.unverified_response_warning_kind]
    assert outcome["n_api_response_rejections"] == 2 * len(flagged) > 0
    assert outcome["status"] == "completed_with_warnings"
    assert all("get_weather: the judge rejected all 2 attempts" in w["detail"] for w in flagged)


async def test_the_judge_is_off_when_the_probe_option_says_so() -> None:
    run = await _episode(probe_options={"tool_calling": {"judge_responses": False}})
    assert _response_checks(run) == []


async def test_the_probe_option_sets_the_attempt_budget() -> None:
    run = await _episode(
        model_factory=_RejectsEveryResponse, probe_options={"tool_calling": {"max_response_attempts": 3}}
    )
    outcome = json.loads(run.row["simulation_outcome"])
    flagged = [w for w in outcome["warnings"] if w["kind"] == ToolCallingProbe.unverified_response_warning_kind]
    assert outcome["n_api_response_rejections"] == 3 * len(flagged) > 0
