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
    _format_tools_for_api,
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
        ]
        for placeholder in required:
            assert placeholder in USER_AGENT_SYSTEM_PROMPT, f"Missing: {placeholder}"

    def test_user_agent_prompt_has_naturalness_instructions(self):
        assert "Do NOT reference" in USER_AGENT_SYSTEM_PROMPT
        assert "tool function names" in USER_AGENT_SYSTEM_PROMPT

    def test_assistant_system_prompt_is_empty(self):
        assert ASSISTANT_SYSTEM_PROMPT == ""

    def test_tool_simulator_prompt_has_placeholders(self):
        for p in ["{tool_spec}", "{tool_args}", "{conversation_context}"]:
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
                today="Friday, 2026-11-06",
            )

        assert simulated.content == '{"ok":true}'
        assert simulated.rerolls == 0
        assert captured["alias"] == "api_response_model"
        assert "city" in captured["prompt"]
        assert "Tokyo" in captured["prompt"]
        assert '"prior_secret":"rain"' in captured["prompt"]
        assert "Today is Friday, 2026-11-06." in captured["prompt"]

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
                today="Friday, 2026-11-06",
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

    persona = {"first_name": "A", "last_name": "User", "age": 35}
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

    def test_a_follow_up_may_ask_for_what_no_offered_tool_does(self, sample_tools):
        probe, _data, _cfg = _concrete_tool_probe(sample_tools)

        assert "none of the available tools can do" not in probe.format_gate_prompt("And a recipe?", "User: Hi")

    def test_the_user_asks_about_records_a_service_holds(self, sample_tools):
        probe, _data, _cfg = _concrete_tool_probe(sample_tools)

        assert "ask about those records rather than describing them" in probe.get_user_system_prompt()


class TestToday:
    """The simulated user and the API response simulator share one date; the assistant is not told it."""

    async def test_the_user_and_the_tool_simulator_share_the_runs_date(self, sample_tools):
        result, assistant_inputs, user_inputs, _judge, api_inputs = await _run_concrete_tool_probe(sample_tools)

        assert "Today is Friday, 2026-11-06." in user_inputs[0][0]["content"]
        assert "Today is Friday, 2026-11-06." in api_inputs[0][0]["content"]
        assert not any("2026-11-06" in (message.get("content") or "") for message in assistant_inputs[0])
        assert json.loads(result["conversation_metadata"])["today"] == "2026-11-06"

    def test_a_run_sets_the_date_as_a_setting(self, sample_tools):
        probe, _data, _cfg = _concrete_tool_probe(sample_tools, today="2027-01-15")

        assert "Today is Friday, 2027-01-15." in probe.get_user_system_prompt()

    def test_a_date_that_does_not_exist_is_refused(self):
        with pytest.raises(ValidationError):
            ToolCallingProbe.Options(today="2027-02-30")


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
