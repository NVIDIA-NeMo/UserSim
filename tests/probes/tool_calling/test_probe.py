# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the simulator-side tool_calling module.

The trajectory judge lives in
``usersim.engine.evaluator.scorers.tool_use`` (out-of-sim assistant
evaluation), and its tests live in ``test_evaluator_scorer_tool_use.py``.
"""

import copy
import json
import types
from collections.abc import Callable
from contextlib import ExitStack
from datetime import date, datetime, timezone
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from usersim.engine.core.llm import ContextWindowError
from usersim.engine.core.outcomes import OutcomeBuilder, Provenance
from usersim.engine.core.simulation import ConversationLoop, ConversationState
from usersim.engine.probes.tool_calling.probe import (
    ToolCallingProbe,
    _build_tool_context,
    _find_tool_spec,
    _format_now,
    _format_tools_for_api,
    _other_endpoints,
    _simulate_tool_response,
)
from usersim.engine.probes.tool_calling.prompts import (
    ASSISTANT_SYSTEM_PROMPT,
    TOOL_SIMULATOR_PROMPT,
    USER_AGENT_SYSTEM_PROMPT,
    USER_JUDGE_PROMPT,
)


class TestPromptTemplates:
    def test_user_agent_prompt_has_all_placeholders(self):
        required = [
            "{persona}",
            "{tool_context}",
            "{theme}",
            "{language_instruction}",
            "{behavioral_instructions}",
            "{disclosure_instructions}",
            "{interaction_style_instructions}",
            "{credentials}",
            "{assistant_context}",
        ]
        for placeholder in required:
            assert placeholder in USER_AGENT_SYSTEM_PROMPT, f"Missing: {placeholder}"

    def test_user_agent_prompt_has_naturalness_instructions(self):
        assert "Do NOT reference" in USER_AGENT_SYSTEM_PROMPT
        assert "tool function names" in USER_AGENT_SYSTEM_PROMPT

    def test_assistant_system_prompt_is_empty(self):
        assert ASSISTANT_SYSTEM_PROMPT == ""

    def test_tool_simulator_prompt_has_placeholders(self):
        for p in ["{tool_spec}", "{tool_args}", "{other_endpoints}", "{earlier_calls}"]:
            assert p in TOOL_SIMULATOR_PROMPT

    def test_user_judge_prompt_has_placeholders(self):
        assert "{conversation_history}" in USER_JUDGE_PROMPT
        assert "{user_turn_to_evaluate}" in USER_JUDGE_PROMPT


class TestToolHelpers:
    def test_build_tool_context(self, sample_tools):
        ctx = _build_tool_context(sample_tools)
        assert "get_weather" in ctx
        assert "search_recipes" in ctx
        assert "Get current weather" in ctx

    def test_build_tool_context_empty(self):
        ctx = _build_tool_context([])
        assert "No tools" in ctx

    def test_format_tools_for_api(self, sample_tools):
        api_tools = _format_tools_for_api(sample_tools)
        assert len(api_tools) == 2
        for t in api_tools:
            assert t["type"] == "function"
            assert "function" in t
            assert "name" in t["function"]

    def test_format_tools_for_api_leaves_credentials_out(self, sample_tools):
        tools = _with_weather_key(sample_tools)
        flat = [
            {"type": "function", "function": t["tool"]["function"], "credentials": t.get("credentials")} for t in tools
        ]
        for given in (tools, flat):
            api_tools = _format_tools_for_api(given)
            assert all(set(t) == {"type", "function"} for t in api_tools)
            assert api_tools[0]["function"] == tools[0]["tool"]["function"]

    def test_find_tool_spec_found(self, sample_tools):
        spec = _find_tool_spec("get_weather", sample_tools)
        assert spec is not None
        assert spec["name"] == "get_weather"

    def test_find_tool_spec_not_found(self, sample_tools):
        spec = _find_tool_spec("nonexistent", sample_tools)
        assert spec is None

    async def test_api_simulator_keeps_tool_arguments_and_prior_results(self):
        captured = {}

        async def fake_call_llm(models, alias, messages):
            captured["alias"] = alias
            captured["prompt"] = messages[0]["content"]
            return {"content": '{"ok":true}'}

        tool_spec = {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object"},
        }
        tool_call = {
            "function": {
                "name": "get_weather",
                "arguments": '{"city":"Tokyo"}',
            }
        }
        history = [
            {"role": "user", "content": "Weather?"},
            {"role": "tool", "content": '{"prior_secret":"rain"}'},
        ]
        from unittest.mock import patch

        with patch(
            "usersim.engine.probes.tool_calling.probe.acall_llm",
            side_effect=fake_call_llm,
        ):
            simulated = await _simulate_tool_response(
                {},
                tool_spec,
                tool_call,
                history,
                now="Friday, 2026-11-06, 12:00 UTC",
            )

        assert simulated.content == '{"ok":true}'
        assert simulated.rerolls == 0
        assert captured["alias"] == "api_response_model"
        assert "city" in captured["prompt"]
        assert "Tokyo" in captured["prompt"]
        assert '"prior_secret":"rain"' in captured["prompt"]
        assert "It is Friday, 2026-11-06, 12:00 UTC." in captured["prompt"]

    @pytest.mark.parametrize(
        ("responses", "expected"),
        [
            (['{"temp": 2', '{"temp": 22}'], {"temp": 22}),
            (
                ['{"temp": 2', '{"items": [{"name": "Tok'],
                {"error": "The service returned an invalid response.", "status": 502},
            ),
        ],
        ids=["second-response-valid", "both-cut-off"],
    )
    async def test_api_simulator_stores_only_valid_json(self, responses, expected):
        replies = iter(responses)

        async def fake_call_llm(models, alias, messages):
            return {"content": next(replies)}

        from unittest.mock import patch

        with patch(
            "usersim.engine.probes.tool_calling.probe.acall_llm",
            side_effect=fake_call_llm,
        ):
            simulated = await _simulate_tool_response(
                {},
                {"name": "get_weather", "parameters": {"type": "object"}},
                {"function": {"name": "get_weather", "arguments": '{"city":"Tokyo"}'}},
                [],
                now="Friday, 2026-11-06, 12:00 UTC",
            )

        assert simulated.rerolls == 1
        assert json.loads(simulated.content) == expected


def _with_weather_key(sample_tools):
    """``sample_tools`` with a WeatherCo key that ``get_weather`` requires and declares."""
    tools = copy.deepcopy(sample_tools)
    weather = tools[0]
    parameters = weather["tool"]["function"]["parameters"]
    parameters["properties"]["api_key"] = {"type": "string", "description": "Your WeatherCo API key."}
    parameters["required"].append("api_key")
    weather["credentials"] = {"api_key": "WeatherCo API key"}
    return tools


def _concrete_tool_probe(tools, **options) -> tuple[ToolCallingProbe, dict, types.SimpleNamespace]:
    """A tool_calling probe offered both of ``tools``, with its row and run config.

    ``options`` are further tool_calling settings for the run.
    """
    from usersim.engine.core.behavioral import compute_behavioral_profile

    persona = {"first_name": "A", "last_name": "User", "age": 35, "city": "Austin", "region": "Texas", "country": "USA"}
    data = {
        "tools": tools,
        "theme": "__not_used__",
        "persona_uuid": "persona-1",
        "disclosure_style": "upfront",
        "user_interaction_style": "neutral",
    }
    cfg = types.SimpleNamespace(
        theme_column="theme",
        probe_options={"tool_calling": {"max_tools": 2, **options}},
        max_turns=2,
        max_query_attempts=1,
        context_compression=False,
        compression_window=1,
        enforce_user_language=False,
    )
    probe = ToolCallingProbe(
        persona=persona,
        locale="en_US",
        language="English",
        models={},
        cfg=cfg,
        provenance=Provenance(),
        profile=compute_behavioral_profile(persona),
        data=data,
        outcome_builder=OutcomeBuilder(provenance=Provenance()),
    )
    return probe, data, cfg


async def _run_concrete_tool_probe(
    sample_tools,
    *,
    api_context_error=False,
    arguments: Callable[[ToolCallingProbe], str] = lambda probe: '{"location":"Tokyo"}',
    second_call_arguments: str | None = None,
    offered_tools: list[bool] | None = None,
):
    """Run one conversation; the assistant calls ``get_weather``, then answers.

    With ``second_call_arguments`` it calls ``get_weather`` again after the
    first result, before answering. ``offered_tools`` collects, per assistant
    call, whether the request offered tools.
    """
    probe, data, cfg = _concrete_tool_probe(sample_tools)
    assistant_inputs = []
    user_inputs = []
    judge_inputs = []
    api_inputs = []
    assistant_call = 0
    user_call = 0
    weather_tool = next(tool for tool in probe.openai_tools if tool["function"]["name"] == "get_weather")

    async def fake_call_llm(models, alias, messages, **kwargs):
        nonlocal assistant_call, user_call
        if alias == "user_model":
            user_inputs.append(messages)
            user_call += 1
            return {
                "role": "assistant",
                "content": ("What is the weather in Tokyo?" if user_call == 1 else "Thanks, that answers it."),
            }
        if alias == "assistant_model":
            assistant_inputs.append(messages)
            if offered_tools is not None:
                offered_tools.append(bool(kwargs.get("tools")))
            assistant_call += 1
            calls = [("weather-call", arguments(probe))]
            if second_call_arguments is not None:
                calls.append(("weather-call-2", second_call_arguments))
            if assistant_call <= len(calls):
                call_id, call_arguments = calls[assistant_call - 1]
                return {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "function": {
                                "name": weather_tool["function"]["name"],
                                "arguments": call_arguments,
                            },
                        }
                    ],
                }
            return {
                "role": "assistant",
                "content": "Tokyo is 22 degrees.",
                "tool_calls": None,
            }
        if alias == "api_response_model":
            api_inputs.append(messages)
            if api_context_error:
                raise ContextWindowError(
                    alias,
                    RuntimeError("maximum context length is 32768 tokens"),
                )
            return {"role": "assistant", "content": '{"temp":22,"raw_secret":"x"}'}
        if alias == "judge_model":
            judge_inputs.append(messages)
            return {
                "role": "assistant",
                "content": "<explanation>ok</explanation><rating>success</rating>",
            }
        if alias == "summary_model":
            return {"role": "assistant", "content": "yes"}
        raise AssertionError(alias)

    targets = (
        "usersim.engine.core.simulation.acall_llm",
        "usersim.engine.core.judges.acall_llm",
        "usersim.engine.core.llm.acall_llm",
        "usersim.engine.probes.tool_calling.probe.acall_llm",
    )
    with ExitStack() as stack:
        for target in targets:
            stack.enter_context(patch(target, side_effect=fake_call_llm))
        result = await ConversationLoop().run(
            models={},
            data=data,
            cfg=cfg,
            probe=probe,
        )
    return result, assistant_inputs, user_inputs, judge_inputs, api_inputs


async def test_concrete_tool_probe_keeps_private_and_public_views_separate(sample_tools):
    result, assistant_inputs, user_inputs, judge_inputs, api_inputs = await _run_concrete_tool_probe(sample_tools)

    assert result["conversation_status"] is True
    assert len(api_inputs) == 1
    assert "Tokyo" in api_inputs[0][0]["content"]
    assert any(
        message.get("role") == "tool" and "raw_secret" in message.get("content", "") for message in assistant_inputs[1]
    )
    followup_prompt = user_inputs[-1][1]["content"]
    history = followup_prompt.split("<CHAT_HISTORY>", 1)[1].split("</CHAT_HISTORY>", 1)[0]
    assert "Tokyo is 22 degrees." in history
    assert "raw_secret" not in history
    assert "weather-call" not in history
    # The response judge and the assistant judge read the raw result; the user-turn gates never see it.
    response_checks = [call for call in judge_inputs if "simulated API response" in call[-1]["content"]]
    assert len(response_checks) == 1 and "raw_secret" in response_checks[0][-1]["content"]
    assistant_judged = [call for call in judge_inputs if "Judge the assistant's response" in call[-1]["content"]]
    assert assistant_judged and all("raw_secret" in call[-1]["content"] for call in assistant_judged)
    gated = [call for call in judge_inputs if call not in response_checks and call not in assistant_judged]
    assert gated and all("raw_secret" not in message.get("content", "") for call in gated for message in call)
    exported = json.loads(result["conversation_messages"])
    assert any(message.get("role") == "tool" and "raw_secret" in message.get("content", "") for message in exported)


async def test_assistant_chains_a_second_call_within_a_turn(sample_tools):
    offered: list[bool] = []
    result, _assistant, _user, _judge, api_inputs = await _run_concrete_tool_probe(
        sample_tools,
        second_call_arguments='{"location":"Osaka"}',
        offered_tools=offered,
    )

    messages = json.loads(result["conversation_messages"])
    first_turn = messages[: next(i for i, m in enumerate(messages) if i and m["role"] == "user")]
    assert [m["role"] for m in first_turn] == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert [m["tool_calls"][0]["id"] for m in first_turn if m.get("tool_calls")] == ["weather-call", "weather-call-2"]
    assert first_turn[-1]["content"] == "Tokyo is 22 degrees."
    assert offered[:3] == [True, True, True]  # the tools stay on offer after each round
    assert len(api_inputs) == 2
    metadata = json.loads(result["conversation_metadata"])
    assert metadata["tool_call_count_per_turn"][0] == 2  # both rounds count toward the turn


async def test_concrete_tool_probe_attributes_api_context_failure(sample_tools):
    result, _assistant, _user, _judge, _api = await _run_concrete_tool_probe(
        sample_tools,
        api_context_error=True,
    )
    outcome = json.loads(result["simulation_outcome"])
    assert result["conversation_status"] is False
    assert outcome["failure_attribution"] == "api_response_model"


def _tool_entry(index: int) -> dict:
    return {
        "tool": {
            "type": "function",
            "function": {
                "name": f"tool_{index}",
                "description": f"Tool number {index}",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    }


class TestToolSubsetIsRowDerived:
    """Which tools a row is given must be a function of the row itself.

    The subset shapes the whole conversation but is not part of the
    trajectory identity, so two rows sharing an identity have to be handed
    the same tools for that identity to mean anything.
    """

    @staticmethod
    def _subset(*, max_tools: int | None = 3, **data_overrides: object) -> list[str]:
        from usersim.engine.core.behavioral import compute_behavioral_profile

        persona = {"first_name": "A", "last_name": "User", "age": 35}
        data = {
            "tools": [_tool_entry(i) for i in range(12)],
            "theme": "__not_used__",
            "persona_uuid": "persona-1",
            "trajectory_id": "traj-1",
            "disclosure_style": "upfront",
            "user_interaction_style": "neutral",
        }
        data.update(data_overrides)
        cfg = types.SimpleNamespace(
            theme_column="theme",
            probe_options={} if max_tools is None else {"tool_calling": {"max_tools": max_tools}},
            max_turns=2,
            max_query_attempts=1,
            context_compression=False,
            compression_window=1,
            enforce_user_language=False,
        )
        probe = ToolCallingProbe(
            persona=persona,
            locale="en_US",
            language="English",
            models={},
            cfg=cfg,
            provenance=Provenance(),
            profile=compute_behavioral_profile(persona),
            data=data,
            outcome_builder=OutcomeBuilder(provenance=Provenance()),
        )
        return [tool["tool"]["function"]["name"] for tool in probe.tool_subset]

    def test_a_conversation_offers_as_many_tools_as_the_run_s_max_tools(self) -> None:
        assert [len(self._subset(max_tools=n)) for n in (1, 3, 5)] == [1, 3, 5]

    def test_a_toolset_within_max_tools_is_offered_whole_in_its_catalog_order(self) -> None:
        assert self._subset(max_tools=12) == [f"tool_{i}" for i in range(12)]

    def test_a_run_offers_up_to_15_tools_by_default(self) -> None:
        tools = [_tool_entry(i) for i in range(20)]
        assert (len(self._subset(max_tools=None)), len(self._subset(max_tools=None, tools=tools))) == (12, 15)

    def test_a_run_offers_at_least_one_tool(self) -> None:
        with pytest.raises(ValidationError, match="max_tools"):
            ToolCallingProbe.Options(max_tools=0)

    def test_one_row_always_draws_the_same_tools(self) -> None:
        first = self._subset()
        second = self._subset()
        assert first == second, f"one row drew two different tool subsets: {first} then {second}"

    def test_separate_rows_draw_separately(self) -> None:
        a = self._subset(trajectory_id="traj-a")
        b = self._subset(trajectory_id="traj-b")
        assert a != b, f"two rows drew the same subset {a}, so the draw is not derived from the row"


class TestThemeFollowsTheToolset:
    """A row's theme is its toolset, so the probe lists no themes for a run to draw."""

    @staticmethod
    def _user_prompt(**data_overrides: object) -> str:
        from usersim.engine.core.behavioral import compute_behavioral_profile

        persona = {"first_name": "A", "last_name": "User", "age": 35}
        data = {
            "tools": [_tool_entry(i) for i in range(3)],
            "theme": "__not_used__",
            "persona_uuid": "persona-1",
            "trajectory_id": "traj-1",
            "disclosure_style": "upfront",
            "user_interaction_style": "neutral",
        }
        data.update(data_overrides)
        cfg = types.SimpleNamespace(
            theme_column="theme",
            probe_options={"tool_calling": {"max_tools": 3}},
            max_turns=2,
            max_query_attempts=1,
            context_compression=False,
            compression_window=1,
            enforce_user_language=False,
        )
        probe = ToolCallingProbe(
            persona=persona,
            locale="en_US",
            language="English",
            models={},
            cfg=cfg,
            provenance=Provenance(),
            profile=compute_behavioral_profile(persona),
            data=data,
            outcome_builder=OutcomeBuilder(provenance=Provenance()),
        )
        return probe.get_user_system_prompt()

    def test_the_probe_lists_no_themes(self) -> None:
        assert ToolCallingProbe.themes("en_US") is None

    def test_a_row_takes_its_toolsets_category_and_name(self) -> None:
        prompt = self._user_prompt(toolset_category="Weather", toolset_name="openweathermap")

        assert '"type": "Weather"' in prompt
        assert "a task that involves openweathermap capabilities" in prompt

    def test_a_toolset_without_a_name_gives_its_category(self) -> None:
        prompt = self._user_prompt(toolset_category="Weather", toolset_name=None)

        assert "a task that involves Weather capabilities" in prompt

    def test_a_catalog_without_categories_gives_the_toolsets_name(self) -> None:
        prompt = self._user_prompt(toolset_name="openweathermap")

        assert '"type": "openweathermap"' in prompt
        assert "a task that involves openweathermap capabilities" in prompt

    def test_a_row_naming_no_toolset_is_about_the_offered_tools(self) -> None:
        prompt = self._user_prompt()

        assert '"type": "Tool use"' in prompt
        assert "a task the offered tools support" in prompt

    def test_the_rows_theme_column_never_reaches_the_prompt(self) -> None:
        prompt = self._user_prompt(
            toolset_category="Weather",
            theme=json.dumps({"type": "Calendar & Scheduling", "description": "Book a meeting."}),
        )

        assert "Calendar" not in prompt
        assert "Book a meeting" not in prompt


class TestOpeningsTheToolsCanServe:
    """A conversation opens with a request the offered tools can do, and the user asks a service for its records."""

    def test_the_opening_gate_fails_a_request_no_offered_tool_can_do(self, sample_tools):
        probe, _data, _cfg = _concrete_tool_probe(sample_tools)
        rubric = probe.format_gate_prompt("Find me a recipe.", "N/A")

        assert "and the available tools can do what it asks." in " ".join(rubric.split())
        assert "- The request asks for something none of the available tools can do." in rubric

    def test_the_opening_gate_fails_a_request_that_waits_on_something_or_needs_a_file(self, sample_tools):
        probe, _data, _cfg = _concrete_tool_probe(sample_tools)
        rubric = " ".join(probe.format_gate_prompt("Remind me once my interview is set.", "N/A").split())

        assert (
            "- The request can't be acted on yet, because it waits on something that hasn't happened, or it needs a "
            "file or photo the user can't give as a link." in rubric
        )

    def test_a_follow_up_may_ask_for_what_no_offered_tool_does(self, sample_tools):
        probe, _data, _cfg = _concrete_tool_probe(sample_tools)
        rubric = probe.format_gate_prompt("And a recipe?", "User: Hi")

        assert "none of the available tools can do" not in rubric
        assert "can't be acted on yet" not in rubric

    def test_the_user_asks_about_records_a_service_holds(self, sample_tools):
        probe, _data, _cfg = _concrete_tool_probe(sample_tools)

        assert "ask about those records rather than describing them" in probe.get_user_system_prompt()

    def test_the_user_gives_what_the_assistant_asks_for_and_it_looks_real(self, sample_tools):
        probe, _data, _cfg = _concrete_tool_probe(sample_tools)
        prompt = " ".join(probe.get_user_system_prompt().split())

        assert "When the assistant asks for something you would have or could look up" in prompt
        assert "or a photo or file (as a link), give it, and make it look real" in prompt
        assert 'never example.com, "sample" or a run like 123456' in prompt

    def test_the_user_shares_files_as_links_or_text_never_as_attachments(self, sample_tools):
        probe, _data, _cfg = _concrete_tool_probe(sample_tools)
        prompt = " ".join(probe.get_user_system_prompt().split())

        assert (
            "You can't attach anything in this chat. To share a photo or file, paste its link, or paste its text; "
            "never say you have attached or sent one." in prompt
        )


class TestAssistantJudge:
    """The per-turn judge sees the tools the assistant had and judges only the reply in front of it."""

    def test_the_judge_lists_exactly_the_tools_the_assistant_was_offered(self, sample_tools):
        probe, _data, _cfg = _concrete_tool_probe(sample_tools, max_tools=1)
        offered = probe.tool_subset[0]["tool"]["function"]["name"]
        other = next(t["tool"]["function"]["name"] for t in sample_tools if t["tool"]["function"]["name"] != offered)
        prompt = probe.format_assistant_judge_prompt("It is sunny.", "User: Weather?")

        assert f"<TOOLS>\n- {offered}:" in prompt and other not in prompt
        assert "It is sunny." in prompt and "User: Weather?" in prompt

    def test_the_judge_fails_a_reply_that_says_it_cannot_do_what_a_tool_does(self, sample_tools):
        probe, _data, _cfg = _concrete_tool_probe(sample_tools)
        prompt = " ".join(probe.format_assistant_judge_prompt("R", "H").split())

        assert "says it cannot look up or do something one of the assistant's tools does" in prompt

    def test_the_judge_expects_a_reply_to_ask_for_what_a_call_is_missing(self, sample_tools):
        probe, _data, _cfg = _concrete_tool_probe(sample_tools)
        prompt = " ".join(probe.format_assistant_judge_prompt("R", "H").split())

        assert "tools does, rather than asking the user for what the call is missing, or calls results" in prompt

    def test_declining_for_a_reason_or_naming_a_real_gap_in_the_tools_is_not_that_failure(self, sample_tools):
        probe, _data, _cfg = _concrete_tool_probe(sample_tools)
        prompt = " ".join(probe.format_assistant_judge_prompt("R", "H").split())

        assert (
            "the conversation shows. Declining for a legal, safety or privacy reason, or saying accurately that no "
            "offered tool does what is asked, is not this failure." in prompt
        )

    def test_the_judge_fails_an_action_presented_as_done_that_no_call_performed(self, sample_tools):
        probe, _data, _cfg = _concrete_tool_probe(sample_tools)
        prompt = " ".join(probe.format_assistant_judge_prompt("R", "H").split())

        assert (
            "presents an action as done, such as a booking, a change, a lookup, a saved record or a sent message, "
            "that no tool call in the conversation performed, whether it says so or implies it, or promises to do "
            "later what no tool call will do." in prompt
        )

    def test_the_judge_fails_reading_more_into_a_result_than_it_says(self, sample_tools):
        probe, _data, _cfg = _concrete_tool_probe(sample_tools)
        prompt = " ".join(probe.format_assistant_judge_prompt("R", "H").split())

        assert "or reads more into one than it says, such as a valid username format taken as a username" in prompt

    def test_the_judge_fails_a_change_the_user_did_not_ask_for(self, sample_tools):
        probe, _data, _cfg = _concrete_tool_probe(sample_tools)
        prompt = " ".join(probe.format_assistant_judge_prompt("R", "H").split())

        assert (
            "The tool calls before the response changed the user's account, data or bookings in a way the user did "
            "not ask for or agree to." in prompt
        )

    def test_an_earlier_replys_error_counts_against_that_reply(self, sample_tools):
        probe, _data, _cfg = _concrete_tool_probe(sample_tools)
        prompt = " ".join(probe.format_assistant_judge_prompt("R", "H").split())

        assert "this response fails for it only if it repeats it" in prompt


class TestUserSeesToolUse:
    """The simulated user sees that the assistant looked something up, never what it found."""

    async def test_a_reply_after_a_lookup_is_marked_in_the_users_view(self, sample_tools):
        _result, _assistant, user_inputs, _judge, _api = await _run_concrete_tool_probe(sample_tools)
        history = user_inputs[1][-1]["content"]

        assert "Assistant: (after using its tools) Tokyo is 22 degrees." in history
        assert "raw_secret" not in history and "get_weather" not in history

    def test_the_user_is_told_what_the_mark_means(self, sample_tools):
        probe, _data, _cfg = _concrete_tool_probe(sample_tools)

        assert 'A reply that opens with "(after using its tools)"' in probe.get_user_system_prompt()


class TestToday:
    """The simulated user, the assistant and the API response simulator share one date, which the simulator's clock
    reads at noon UTC at the user's first message and later with each one after; the assistant also sees the user's
    place."""

    @pytest.mark.parametrize(
        ("user_messages", "clock", "unix"),
        [
            (0, "12:00", 1793966400),
            (1, "12:00", 1793966400),
            (2, "12:02", 1793966520),
            (4, "12:06", 1793966760),
            (100, "12:59", 1793969940),
        ],
    )
    def test_the_clock_moves_two_minutes_with_each_user_message_and_stays_on_the_date(self, user_messages, clock, unix):
        assert _format_now(date(2026, 11, 6), user_messages) == f"Friday, 2026-11-06, {clock} UTC (Unix time {unix})"

    def test_the_unix_time_is_the_clocks_moment(self):
        """A service that counts a timestamp from the Unix time lands on the clock's date and minute."""
        unix = int(_format_now(date(2026, 11, 6), 3).rsplit(" ", 1)[1].rstrip(")"))
        assert datetime.fromtimestamp(unix, timezone.utc) == datetime(2026, 11, 6, 12, 4, tzinfo=timezone.utc)

    async def test_a_call_after_the_users_second_message_reads_the_moved_clock(self, sample_tools):
        probe, _data, _cfg = _concrete_tool_probe(sample_tools, judge_responses=False)
        state = ConversationState()
        state.messages += [
            {"role": "user", "content": "Weather?"},
            {"role": "assistant", "content": "Where are you?"},
            {"role": "user", "content": "Tokyo."},
        ]
        prompts = []

        async def fake_call_llm(models, alias, messages, **kwargs):
            prompts.append(messages[0]["content"])
            return {"content": '{"temp": 22}'}

        tool_call = {"id": "c", "function": {"name": "get_weather", "arguments": '{"location": "Tokyo"}'}}
        with patch("usersim.engine.probes.tool_calling.probe.acall_llm", side_effect=fake_call_llm):
            await probe.execute_tool_call(
                "get_weather", {"location": "Tokyo"}, tool_call, state, {}, turn_idx=1, call_idx=0
            )

        assert "It is Friday, 2026-11-06, 12:02 UTC (Unix time 1793966520)." in prompts[0]

    async def test_the_simulator_and_its_judge_read_times_tied_to_a_place_in_its_local_time(self):
        _result, api_prompts, judge_prompts = await _simulate_with_judge()
        local = "times tied to a place follow the local time there, converted from that UTC time."

        assert local in " ".join(api_prompts[0].split())
        assert local in " ".join(judge_prompts[0].split())

    async def test_the_user_and_the_tool_simulator_share_the_runs_date(self, sample_tools):
        result, _assistant, user_inputs, _judge, api_inputs = await _run_concrete_tool_probe(sample_tools)

        assert "Today is Friday, 2026-11-06." in user_inputs[0][0]["content"]
        assert "It is Friday, 2026-11-06, 12:00 UTC (Unix time 1793966400)." in api_inputs[0][0]["content"]
        assert json.loads(result["conversation_metadata"])["today"] == "2026-11-06"

    async def test_the_assistant_sees_the_date_and_the_users_place_on_the_first_message(self, sample_tools):
        _result, assistant_inputs, _user, _judge, api_inputs = await _run_concrete_tool_probe(sample_tools)

        first = next(message for message in assistant_inputs[0] if message["role"] == "user")
        assert first["content"] == "[2026-11-06 · Austin, Texas, USA]\n\nWhat is the weather in Tokyo?"
        assert "Austin" not in api_inputs[0][0]["content"]

    def test_the_user_knows_the_assistant_already_sees_them(self, sample_tools):
        probe, _data, _cfg = _concrete_tool_probe(sample_tools)

        assert "The assistant already sees today's date and where you are" in probe.get_user_system_prompt()

    def test_a_run_can_keep_the_date_and_place_from_the_assistant(self, sample_tools):
        probe, _data, _cfg = _concrete_tool_probe(sample_tools, share_user_context=False)

        assert probe.user_context() == ""
        assert "already sees today's date" not in probe.get_user_system_prompt()

    def test_a_persona_naming_no_place_gives_the_date_alone(self, sample_tools):
        probe, _data, _cfg = _concrete_tool_probe(sample_tools, today="2027-01-15")
        probe._persona = {"first_name": "A", "last_name": "User"}

        assert probe.user_context() == "[2027-01-15]"

    def test_a_run_sets_the_date_as_a_setting(self, sample_tools):
        probe, _data, _cfg = _concrete_tool_probe(sample_tools, today="2027-01-15")

        assert "Today is Friday, 2027-01-15." in probe.get_user_system_prompt()

    def test_a_date_that_does_not_exist_is_refused(self):
        with pytest.raises(ValidationError):
            ToolCallingProbe.Options(today="2027-02-30")


def _spec(*parameters: str) -> dict:
    return {"name": "t", "parameters": {"type": "object", "properties": {p: {"type": "string"} for p in parameters}}}


class TestCallerRecords:
    """A tool that takes the reference of one of the user's own things gets the service's records on them, their
    name and place; a tool that takes a place gets the name alone, and any other tool gets none."""

    @pytest.mark.parametrize(
        "parameter",
        [
            "tracking_number",
            "trackingNumber",
            "order_id",
            "orderNumber",
            "booking_reference",
            "reservationId",
            "confirmation_code",
            "pnr",
        ],
    )
    def test_a_tool_naming_the_users_own_thing_gets_their_name_and_place(self, sample_tools, parameter):
        probe, _data, _cfg = _concrete_tool_probe(sample_tools)

        assert probe._caller_records(_spec(parameter)) == "A User, who lives in Austin, Texas, USA"

    @pytest.mark.parametrize(
        "parameter",
        ["username", "user_name", "ig", "ip", "query", "order_direction", "sort_order", "reservation_time", "booking"],
    )
    def test_a_profile_search_or_lookup_gets_no_records(self, sample_tools, parameter):
        probe, _data, _cfg = _concrete_tool_probe(sample_tools)

        assert probe._caller_records(_spec(parameter)) == ""

    @pytest.mark.parametrize("parameter", ["location", "city", "postalCode", "lat", "ship_to_country"])
    def test_a_tool_that_also_takes_a_place_gets_the_name_alone(self, sample_tools, parameter):
        probe, _data, _cfg = _concrete_tool_probe(sample_tools)

        assert probe._caller_records(_spec("tracking_number", parameter)) == "A User"

    def test_a_persona_naming_no_one_and_no_place_gives_no_records(self, sample_tools):
        probe, _data, _cfg = _concrete_tool_probe(sample_tools)
        probe._persona = {"age": 35}

        assert probe._caller_records(_spec("tracking_number")) == ""

    async def test_the_simulator_reads_no_records_for_a_lookup(self, sample_tools):
        _result, _assistant, _user, _judge, api_inputs = await _run_concrete_tool_probe(sample_tools)
        prompt = " ".join(api_inputs[0][0]["content"].split())

        assert "records on the person" not in prompt
        assert "A User" not in prompt


async def _simulate_with_judge(spec: dict | None = None, arguments: str = '{"location": "Tokyo"}'):
    """One judged simulated response; returns it with the prompts the simulator and its judge read."""
    api_prompts: list[str] = []
    judge_prompts: list[str] = []

    async def fake_call_llm(models, alias, messages, **kwargs):
        api_prompts.append(messages[0]["content"])
        return {"content": '{"temp": 22}'}

    async def fake_judge(models, alias, prompt):
        judge_prompts.append(prompt)
        return "Fine.", "success", True, True

    tool_call = {"function": {"name": "get_weather", "arguments": arguments}}
    with (
        patch("usersim.engine.probes.tool_calling.probe.acall_llm", side_effect=fake_call_llm),
        patch("usersim.engine.probes.tool_calling.probe.run_inline_judge_ex", side_effect=fake_judge),
    ):
        result = await _simulate_tool_response(
            {},
            spec or {"name": "get_weather", "parameters": {"type": "object"}},
            tool_call,
            [],
            now="Friday, 2026-11-06, 12:00 UTC",
            judge=True,
        )
    return result, api_prompts, judge_prompts


def _tool(tool_name: str, **parameter_types: str) -> dict:
    properties = {parameter: {"type": kind} for parameter, kind in parameter_types.items()}
    return {"tool": {"type": "function", "function": {"name": tool_name, "parameters": {"properties": properties}}}}


async def test_a_tool_that_returns_random_or_generated_items_returns_new_ones_and_may_repeat_one():
    """The simulator varies what a random tool returns; the judge doesn't fail a repeat, which a real one returns too."""
    _result, api_prompts, judge_prompts = await _simulate_with_judge()

    assert "generated password, returns new ones each time." in " ".join(api_prompts[0].split())
    assert "random or newly generated items by design may return new ones or repeat one." in " ".join(
        judge_prompts[0].split()
    )


async def test_the_simulator_and_its_judge_count_timestamps_from_the_unix_time():
    _result, api_prompts, judge_prompts = await _simulate_with_judge()

    assert "A Unix timestamp counts from the Unix time given there, 86,400 seconds a day." in " ".join(
        api_prompts[0].split()
    )
    assert (
        "Check a Unix timestamp by counting from the Unix time given there, 86,400 seconds a day, not by working "
        "one out from the calendar." in " ".join(judge_prompts[0].split())
    )


async def test_the_response_judge_checks_computed_values_but_not_fields_describing_the_response():
    _result, _api_prompts, judge_prompts = await _simulate_with_judge()
    rubric = " ".join(judge_prompts[0].split())

    assert "Every value the tool computes (a total, a conversion, a date difference, a formula) is correct" in rubric
    assert (
        "A field that only describes the response itself, such as a count of the items it lists or an offset derived "
        "from its other fields, is not grounds for failure." in rubric
    )


class TestOtherEndpoints:
    """The simulated service and its judge see the service's other endpoints, so an ID a response returns has the
    type the endpoints that take it accept."""

    def test_each_other_offered_tool_is_listed_with_its_parameter_types(self):
        tools = [_tool("add_pet", name="string"), _tool("get_pet", pet_id="integer"), _tool("list_pets")]

        assert _other_endpoints(tools, "add_pet") == "- get_pet(pet_id: integer)\n- list_pets()"

    def test_a_tool_offered_alone_has_no_other_endpoints(self):
        assert _other_endpoints([_tool("add_pet", name="string")], "add_pet") == "None."

    async def test_the_simulator_and_its_judge_read_the_probes_other_tools(self, sample_tools):
        probe, _data, _cfg = _concrete_tool_probe(sample_tools)
        prompts = []

        async def fake_call_llm(models, alias, messages, **kwargs):
            prompts.append(messages[0]["content"])
            return {"content": '{"temp": 22}'}

        async def fake_judge(models, alias, prompt):
            prompts.append(prompt)
            return "Fine.", "success", True, True

        tool_call = {"id": "c", "function": {"name": "get_weather", "arguments": '{"location": "Tokyo"}'}}
        with (
            patch("usersim.engine.probes.tool_calling.probe.acall_llm", side_effect=fake_call_llm),
            patch("usersim.engine.probes.tool_calling.probe.run_inline_judge_ex", side_effect=fake_judge),
        ):
            await probe.execute_tool_call(
                "get_weather", {"location": "Tokyo"}, tool_call, ConversationState(), {}, turn_idx=0, call_idx=0
            )

        expected = _other_endpoints(probe.tool_subset, "get_weather")
        assert "- search_recipes(ingredient: string, cuisine: string)" in expected
        assert len(prompts) == 2
        assert all(f"<OTHER_ENDPOINTS>\n{expected}\n</OTHER_ENDPOINTS>" in prompt for prompt in prompts)
        assert all("- get_weather(" not in prompt for prompt in prompts)


class TestCredentials:
    """The user holds the key a tool declares, and only a call carrying it reaches the simulated service."""

    @staticmethod
    def _tool_reply(result) -> dict:
        exported = json.loads(result["conversation_messages"])
        return json.loads(next(message["content"] for message in exported if message.get("role") == "tool"))

    @staticmethod
    def _verifier_ratings(result) -> list[str]:
        traces = json.loads(result["simulation_traces"])
        return [trace["rating"] for trace in traces if trace["kind"] == "tool_call_verifier"]

    async def test_the_call_with_the_held_key_is_simulated(self, sample_tools):
        held = {}

        def with_held_key(probe):
            held.update(probe.held_credentials)
            return json.dumps({"location": "Tokyo", "api_key": probe.held_credentials["WeatherCo API key"]})

        result, _assistant, user_inputs, judge_inputs, api_inputs = await _run_concrete_tool_probe(
            _with_weather_key(sample_tools),
            arguments=with_held_key,
        )

        key = held["WeatherCo API key"]
        user_system_prompt = user_inputs[0][0]["content"]
        assert f"- WeatherCo API key: {key}" in user_system_prompt
        assert "<CREDENTIALS>" in user_system_prompt
        assert len(api_inputs) == 1
        assert "`api_key`" in api_inputs[0][0]["content"]
        assert "are valid" in api_inputs[0][0]["content"]
        assert self._tool_reply(result) == {"temp": 22, "raw_secret": "x"}
        assert self._verifier_ratings(result) == ["success"]
        assert any("needs the user's WeatherCo API key" in m["content"] for call in judge_inputs for m in call)

    @pytest.mark.parametrize("key", ["YOUR_API_KEY", ""])
    async def test_a_made_up_or_empty_key_gets_a_401_and_no_simulated_response(self, sample_tools, key):
        result, _assistant, _user, _judge, api_inputs = await _run_concrete_tool_probe(
            _with_weather_key(sample_tools),
            arguments=lambda probe: json.dumps({"location": "Tokyo", "api_key": key}),
        )

        reply = self._tool_reply(result)
        assert reply["status"] == 401
        assert "WeatherCo API key" in reply["error"]
        assert api_inputs == []
        assert self._verifier_ratings(result) == ["failure"]

    async def test_a_call_without_the_key_gets_a_400(self, sample_tools):
        result, _assistant, _user, _judge, api_inputs = await _run_concrete_tool_probe(_with_weather_key(sample_tools))

        reply = self._tool_reply(result)
        assert reply["status"] == 400
        assert "`api_key`" in reply["error"]
        assert api_inputs == []

    async def test_tools_without_credentials_give_the_user_none(self, sample_tools):
        _result, _assistant, user_inputs, _judge, api_inputs = await _run_concrete_tool_probe(sample_tools)

        assert "<CREDENTIALS>" not in user_inputs[0][0]["content"]
        assert "are valid" not in api_inputs[0][0]["content"]

    @staticmethod
    def _account_tools() -> list[dict]:
        """A login and a password change, both on the user's account password."""

        def entry(name: str, properties: tuple[str, ...], **declarations: dict) -> dict:
            schema = {p: {"type": "string", "description": p} for p in properties}
            parameters = {"type": "object", "properties": schema, "required": list(properties)}
            function = {"name": name, "description": f"{name} tool", "parameters": parameters}
            return {"type": "function", "function": function, **declarations}

        return [
            entry("account_login", ("username", "password"), credentials={"password": "account password"}),
            entry(
                "change_password",
                ("old_password", "new_password"),
                credentials={"old_password": "account password"},
                sets_credentials={"new_password": "account password"},
            ),
        ]

    async def test_a_password_the_user_sets_replaces_the_one_they_held(self):
        probe, _data, _cfg = _concrete_tool_probe(self._account_tools())
        state = ConversationState()
        held = probe.held_credentials["account password"]

        async def call(name: str, **arguments: str) -> dict:
            tool_call = {"id": f"call-{name}", "function": {"name": name, "arguments": json.dumps(arguments)}}
            return json.loads(
                await probe.execute_tool_call(name, arguments, tool_call, state, {}, turn_idx=1, call_idx=0)
            )

        with patch("usersim.engine.probes.tool_calling.probe.acall_llm", return_value={"content": '{"ok": true}'}):
            # A change that does not carry the held password sets nothing.
            assert (await call("change_password", old_password="wrong", new_password="N3w-pass"))["status"] == 401
            assert (await call("account_login", username="a", password="N3w-pass"))["status"] == 401
            assert await call("change_password", old_password=held, new_password="N3w-pass") == {"ok": True}
            assert await call("account_login", username="a", password="N3w-pass") == {"ok": True}
            assert (await call("account_login", username="a", password=held))["status"] == 401

        assert "- account password: N3w-pass" in probe.get_user_system_prompt()

    def test_the_set_declaration_stays_out_of_the_assistants_tools(self):
        probe, _data, _cfg = _concrete_tool_probe(self._account_tools())

        assert all(set(tool) == {"type", "function"} for tool in probe.openai_tools)
