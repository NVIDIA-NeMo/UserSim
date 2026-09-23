# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Tests for the simulator-side tool_calling module.

The 8-axis trajectory judge lives in
``usersim.engine.evaluator.scorers.tool_use`` (out-of-sim assistant
evaluation), and its tests live in ``test_evaluator_scorer_tool_use.py``.
"""

import json
import types
from contextlib import ExitStack
from unittest.mock import patch

from usersim.engine.core.llm import ContextWindowError
from usersim.engine.core.outcomes import OutcomeBuilder, Provenance
from usersim.engine.core.simulation import ConversationLoop
from usersim.engine.probes.tool_calling.generator import (
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

    def test_find_tool_spec_found(self, sample_tools):
        spec = _find_tool_spec("get_weather", sample_tools)
        assert spec is not None
        assert spec["name"] == "get_weather"

    def test_find_tool_spec_not_found(self, sample_tools):
        spec = _find_tool_spec("nonexistent", sample_tools)
        assert spec is None

    def test_api_simulator_keeps_tool_arguments_and_prior_results(self):
        captured = {}

        def fake_call_llm(models, alias, messages):
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
            "usersim.engine.probes.tool_calling.generator.call_llm",
            side_effect=fake_call_llm,
        ):
            content, rerolled = _simulate_tool_response(
                {},
                tool_spec,
                tool_call,
                history,
            )

        assert content == '{"ok":true}'
        assert rerolled is False
        assert captured["alias"] == "api_response_model"
        assert "city" in captured["prompt"]
        assert "Tokyo" in captured["prompt"]
        assert '"prior_secret":"rain"' in captured["prompt"]


def _run_concrete_tool_probe(sample_tools, *, api_context_error=False):
    from usersim.engine.core.behavioral import compute_behavioral_profile

    persona = {"first_name": "A", "last_name": "User", "age": 35}
    data = {
        "tools": sample_tools,
        "theme": json.dumps(
            {
                "type": "Weather & Location Lookup",
                "description": "Look up weather.",
                "tool_expected": True,
            }
        ),
        "persona_uuid": "persona-1",
        "disclosure_style": "upfront",
        "user_interaction_style": "neutral",
    }
    cfg = types.SimpleNamespace(
        tools_column="tools",
        theme_column="theme",
        max_tools=2,
        max_turns=2,
        max_query_attempts=1,
        context_compression=False,
        compression_window=1,
        enforce_user_language=False,
    )
    outcome = OutcomeBuilder(provenance=Provenance())
    probe = ToolCallingProbe(
        persona=persona,
        locale="en_US",
        language="English",
        models={},
        cfg=cfg,
        provenance=Provenance(),
        profile=compute_behavioral_profile(persona),
        data=data,
        outcome_builder=outcome,
    )
    assistant_inputs = []
    user_inputs = []
    judge_inputs = []
    api_inputs = []
    assistant_call = 0
    user_call = 0
    weather_tool = next(tool for tool in probe.openai_tools if tool["function"]["name"] == "get_weather")

    def fake_call_llm(models, alias, messages, **kwargs):
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
            assistant_call += 1
            if assistant_call == 1:
                return {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "weather-call",
                            "function": {
                                "name": weather_tool["function"]["name"],
                                "arguments": '{"location":"Tokyo"}',
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
        "usersim.engine.core.simulation.call_llm",
        "usersim.engine.core.judges.call_llm",
        "usersim.engine.core.llm.call_llm",
        "usersim.engine.probes.tool_calling.generator.call_llm",
    )
    with ExitStack() as stack:
        for target in targets:
            stack.enter_context(patch(target, side_effect=fake_call_llm))
        result = ConversationLoop().run(
            models={},
            data=data,
            cfg=cfg,
            probe=probe,
        )
    return result, assistant_inputs, user_inputs, judge_inputs, api_inputs


def test_concrete_tool_probe_keeps_private_and_public_views_separate(sample_tools):
    result, assistant_inputs, user_inputs, judge_inputs, api_inputs = _run_concrete_tool_probe(sample_tools)

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
    assert all("raw_secret" not in message.get("content", "") for call in judge_inputs for message in call)
    exported = json.loads(result["conversation_messages"])
    assert any(message.get("role") == "tool" and "raw_secret" in message.get("content", "") for message in exported)


def test_concrete_tool_probe_attributes_api_context_failure(sample_tools):
    result, _assistant, _user, _judge, _api = _run_concrete_tool_probe(
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
    def _subset(**data_overrides: object) -> list[str]:
        from usersim.engine.core.behavioral import compute_behavioral_profile

        persona = {"first_name": "A", "last_name": "User", "age": 35}
        data = {
            "tools": [_tool_entry(i) for i in range(12)],
            "theme": json.dumps({"type": "T", "description": "d", "tool_expected": True}),
            "persona_uuid": "persona-1",
            "trajectory_id": "traj-1",
            "disclosure_style": "upfront",
            "user_interaction_style": "neutral",
        }
        data.update(data_overrides)
        cfg = types.SimpleNamespace(
            tools_column="tools",
            theme_column="theme",
            max_tools=3,
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

    def test_one_row_always_draws_the_same_tools(self) -> None:
        first = self._subset()
        second = self._subset()
        assert first == second, f"one row drew two different tool subsets: {first} then {second}"

    def test_separate_rows_draw_separately(self) -> None:
        a = self._subset(trajectory_id="traj-a")
        b = self._subset(trajectory_id="traj-b")
        assert a != b, f"two rows drew the same subset {a}, so the draw is not derived from the row"
