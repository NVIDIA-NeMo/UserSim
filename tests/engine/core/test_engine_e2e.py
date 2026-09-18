# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the unified simulation engine with stubbed models.

These tests run full probe loops (general_open_ended, general_educational, tool_calling)
using deterministic stubbed LLM responses to verify transcript shape,
message ordering, turn count, and status propagation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict
from unittest.mock import MagicMock, patch


from usersim.engine.core.probes import BaseProbe
from usersim.engine.core.simulation import (
    ConversationLoop,
    _check_conversation_complete,
    _is_poor_assistant_response,
    make_failed,
    make_result,
)


class _StubProbe(BaseProbe):
    """Minimal BaseProbe subclass for engine-level e2e tests.

    Direct instantiation here exercises the unified loop without going through
    the dispatcher / probe registry.
    """

    label = "_test_stub"

    def __init__(
        self,
        *,
        user_system_prompt: str = "You are a user.",
        gate_prompt_template: str = ("{conversation_history}\n{user_turn_to_evaluate}"),
    ) -> None:
        super().__init__(
            persona={"first_name": "Test", "last_name": "User"},
            locale="en_US",
            language="English",
            models={},
        )
        self._user_system_prompt = user_system_prompt
        self._gate_prompt_template = gate_prompt_template

    def get_user_system_prompt(self) -> str:
        return self._user_system_prompt

    def get_assistant_system_prompt(self) -> str:
        return ""

    def format_gate_prompt(
        self,
        user_query: str,
        conversation_history: str,
    ) -> str:
        return self._gate_prompt_template.format(
            conversation_history=conversation_history,
            user_turn_to_evaluate=user_query,
        )


# ---------------------------------------------------------------------------
# Stub model that returns canned responses
# ---------------------------------------------------------------------------


class StubFacade:
    """Minimal ModelFacade stub that returns canned responses in sequence."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._idx = 0

    def completion(self, messages, **kwargs):
        text = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        msg = MagicMock()
        msg.content = text
        msg.reasoning_content = None
        msg.tool_calls = None
        resp = MagicMock()
        resp.message = msg
        resp.usage = None
        return resp


@dataclass
class StubConfig:
    max_turns: int = 2
    max_query_attempts: int = 2
    context_compression: bool = False
    compression_window: int = 1
    persona_column: str = "persona"
    probe_type_column: str = "probe_type"
    theme_column: str = "theme"
    tools_column: str | None = None
    max_tools: int = 5
    max_steps: int = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_models(
    user_responses: list[str],
    assistant_responses: list[str],
    judge_responses: list[str],
) -> dict:
    return {
        "user_model": StubFacade(user_responses),
        "assistant_model": StubFacade(assistant_responses),
        "judge_model": StubFacade(judge_responses),
        "summary_model": StubFacade(["Summary."]),
        "api_response_model": StubFacade(['{"result": "ok"}']),
    }


def _judge_success():
    return "<explanation>Good turn.</explanation>\n<rating>success</rating>"


def _judge_failure():
    return "<explanation>Bad turn.</explanation>\n<rating>failure</rating>"


def _judge_warn():
    return "<explanation>Minor issue.</explanation>\n<rating>warn</rating>"


# ---------------------------------------------------------------------------
# Tests: early stopping detection
# ---------------------------------------------------------------------------


class TestEarlyStopping:
    def test_llm_based_satisfaction_detected(self):
        models = {"summary_model": StubFacade(["yes"])}
        assert _check_conversation_complete(models, "Thanks, that's all I needed!")

    def test_llm_based_not_stopped(self):
        models = {"summary_model": StubFacade(["no"])}
        assert not _check_conversation_complete(models, "Can you tell me more?")

    def test_llm_based_handles_whitespace(self):
        models = {"summary_model": StubFacade(["  Yes  \n"])}
        assert _check_conversation_complete(models, "Perfect, thanks!")


# ---------------------------------------------------------------------------
# Tests: assistant quality check
# ---------------------------------------------------------------------------


class TestAssistantQualityCheck:
    def test_empty_response_flagged(self):
        assert _is_poor_assistant_response("")
        assert _is_poor_assistant_response("   ")

    def test_very_long_response_flagged(self):
        assert _is_poor_assistant_response("word " * 801)

    def test_over_formatted_response_flagged(self):
        content = "\n# Section 1\ntext\n# Section 2\ntext\n# Section 3\ntext\n# Section 4\ntext"
        assert _is_poor_assistant_response(content)

    def test_normal_response_passes(self):
        assert not _is_poor_assistant_response("Here is your weather forecast for today.")


# ---------------------------------------------------------------------------
# Tests: unified loop with simple adapter
# ---------------------------------------------------------------------------


class TestConversationLoopSimple:
    def test_successful_conversation(self):
        models = _build_models(
            user_responses=["What is Python?", "Tell me more about types."],
            assistant_responses=["Python is a language.", "Python has dynamic types."],
            judge_responses=[_judge_success(), _judge_success(), _judge_success()],
        )
        data: Dict[str, Any] = {}
        cfg = StubConfig(max_turns=2)

        adapter = _StubProbe(
            gate_prompt_template=("Judge: {conversation_history}\n{user_turn_to_evaluate}"),
        )

        loop = ConversationLoop()
        result = loop.run(models, data, cfg, adapter)

        assert result["conversation_status"] is True
        messages = json.loads(result["conversation_messages"])
        assert messages[0]["role"] == "user"
        roles = [m["role"] for m in messages]
        assert "assistant" in roles

    def test_gate_failure_returns_failed(self):
        models = _build_models(
            user_responses=["Bad query", "Still bad"],
            assistant_responses=[],
            judge_responses=[_judge_failure(), _judge_failure()],
        )
        data: Dict[str, Any] = {}
        cfg = StubConfig(max_turns=2, max_query_attempts=2)

        adapter = _StubProbe(
            gate_prompt_template=("Judge: {conversation_history}\n{user_turn_to_evaluate}"),
        )

        loop = ConversationLoop()
        result = loop.run(models, data, cfg, adapter)

        assert result["conversation_status"] is False
        assert result["num_turns"] == 0

    def test_warn_rating_continues_conversation(self):
        models = _build_models(
            user_responses=["What is Python?", "Tell me more."],
            assistant_responses=["Python is a language.", "It has types."],
            judge_responses=[_judge_success(), _judge_warn(), _judge_success()],
        )
        data: Dict[str, Any] = {}
        cfg = StubConfig(max_turns=2)

        adapter = _StubProbe(
            gate_prompt_template=("Judge: {conversation_history}\n{user_turn_to_evaluate}"),
        )

        loop = ConversationLoop()
        result = loop.run(models, data, cfg, adapter)

        assert result["conversation_status"] is True


# ---------------------------------------------------------------------------
# Tests: message ordering invariants
# ---------------------------------------------------------------------------


class TestMessageInvariants:
    def test_user_assistant_alternation(self):
        models = _build_models(
            user_responses=["Hello", "Thanks"],
            assistant_responses=["Hi there!", "You're welcome."],
            judge_responses=[_judge_success(), _judge_success(), _judge_success()],
        )
        data: Dict[str, Any] = {}
        cfg = StubConfig(max_turns=2)

        adapter = _StubProbe(user_system_prompt="System.")

        loop = ConversationLoop()
        result = loop.run(models, data, cfg, adapter)
        messages = json.loads(result["conversation_messages"])

        non_system = [m for m in messages if m["role"] != "system"]
        for i in range(len(non_system) - 1):
            assert non_system[i]["role"] != non_system[i + 1]["role"], (
                f"Adjacent messages have same role at positions {i} and {i + 1}: {non_system[i]['role']}"
            )

    def test_num_turns_matches_user_messages(self):
        models = _build_models(
            user_responses=["Q1", "Q2"],
            assistant_responses=["A1", "A2"],
            judge_responses=[_judge_success(), _judge_success(), _judge_success()],
        )
        data: Dict[str, Any] = {}
        cfg = StubConfig(max_turns=2)

        adapter = _StubProbe(user_system_prompt="System.")

        loop = ConversationLoop()
        result = loop.run(models, data, cfg, adapter)
        messages = json.loads(result["conversation_messages"])

        user_count = sum(1 for m in messages if m["role"] == "user")
        assert result["num_turns"] == user_count


class _ProtocolHistoryProbe(_StubProbe):
    """Inject private protocol messages around one visible assistant reply."""

    def __init__(self) -> None:
        super().__init__()
        self.gate_histories: list[str] = []

    def get_verbatim_first_user_turn(self, state):
        return "Please check the private result."

    def format_gate_prompt(self, user_query, conversation_history):
        self.gate_histories.append(conversation_history)
        return f"{conversation_history}\nCANDIDATE:{user_query}"

    def after_assistant_turn(self, models, state, response, cfg):
        state.messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "Let me check.",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {
                                "name": "internal_lookup",
                                "arguments": '{"secret_arg":"x"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"raw_secret_result":"42"}',
                    "tool_call_id": "c1",
                },
                {"role": "assistant", "content": "The result is 42."},
            ]
        )
        return "The result is 42."


def test_user_and_inline_judges_see_only_public_dialogue():
    probe = _ProtocolHistoryProbe()
    user_prompts: list[str] = []
    judge_prompts: list[str] = []

    def fake_call_llm(models, alias, messages, **kwargs):
        if alias == "assistant_model":
            return {"role": "assistant", "content": ""}
        if alias == "user_model":
            user_prompts.append(messages[1]["content"])
            return {"role": "assistant", "content": "Thanks, one more check."}
        if alias == "summary_model":
            return {"role": "assistant", "content": "no"}
        raise AssertionError(alias)

    def fake_judge(models, alias, prompt):
        judge_prompts.append(prompt)
        return ("ok", "success", True)

    def fake_judge_ex(models, alias, prompt):
        expl, rating, ok = fake_judge(models, alias, prompt)
        return (expl, rating, ok, True)

    with (
        patch(
            "usersim.engine.core.simulation.call_llm",
            side_effect=fake_call_llm,
        ),
        patch(
            "usersim.engine.core.simulation.run_inline_judge",
            side_effect=fake_judge,
        ),
        patch(
            "usersim.engine.core.simulation.run_inline_judge_ex",
            side_effect=fake_judge_ex,
        ),
    ):
        ConversationLoop().run(
            models={},
            data={},
            cfg=StubConfig(max_turns=2),
            probe=probe,
        )

    assert len(user_prompts) == 1
    history = user_prompts[0].split("<CHAT_HISTORY>", 1)[1].split("</CHAT_HISTORY>", 1)[0]
    assert "Let me check." in history
    assert "The result is 42." in history
    assert "internal_lookup" not in history
    assert "secret_arg" not in history
    assert "raw_secret_result" not in history
    assert probe.gate_histories == [history.strip()]
    assert all("raw_secret_result" not in prompt for prompt in judge_prompts)
    assert all("secret_arg" not in prompt for prompt in judge_prompts)


def test_tool_hook_context_failure_preserves_model_attribution():
    from usersim.engine.core.llm import ContextWindowError

    class _ApiContextFailureProbe(_StubProbe):
        def get_verbatim_first_user_turn(self, state):
            return "Use the tool."

        def after_assistant_turn(self, models, state, response, cfg):
            state.messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "c1", "function": {"name": "lookup"}}],
                }
            )
            raise ContextWindowError(
                "api_response_model",
                RuntimeError("maximum context length is 32768 tokens"),
            )

    with patch(
        "usersim.engine.core.simulation.call_llm",
        return_value={"role": "assistant", "content": ""},
    ):
        result = ConversationLoop().run(
            models={},
            data={},
            cfg=StubConfig(max_turns=1),
            probe=_ApiContextFailureProbe(),
        )

    outcome = json.loads(result["simulation_outcome"])
    assert outcome["status"] == "failed"
    assert outcome["failure_attribution"] == "api_response_model"
    assert json.loads(result["conversation_messages"])[-1]["tool_calls"]


# ---------------------------------------------------------------------------
# Tests: make_result / make_failed
# ---------------------------------------------------------------------------


class TestResultHelpers:
    def test_make_result_structure(self):
        result = make_result(
            [{"role": "user", "content": "hi"}],
            {"ratings": []},
            True,
        )
        assert result["conversation_status"] is True
        assert json.loads(result["conversation_messages"]) == [{"role": "user", "content": "hi"}]

    def test_make_failed_structure(self):
        result = make_failed("test reason")
        assert result["conversation_status"] is False
        assert result["num_turns"] == 0
        assert result["num_tool_calls"] == 0
        meta = json.loads(result["conversation_metadata"])
        assert meta["reason"] == "test reason"
