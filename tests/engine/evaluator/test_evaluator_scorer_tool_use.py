# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for evaluator/scorers/tool_use.py — the tool-calling trajectory judge.

Covers the trajectory schema, which belongs to the evaluator rather than the
simulator, plus a behavioral test for scorer registration via the auto-load
path.
"""

from __future__ import annotations

import json

import pytest
from data_designer.config.column_configs import Score

from usersim.engine.evaluator import scorers as scorers_module
from usersim.engine.evaluator.scorers import tool_use as tool_use_module
from usersim.engine.evaluator.scorers.tool_use import (
    TRAJECTORY_SCORES,
    _build_schema,
    score_tool_use_trajectory,
)

AXES = [
    "task_completion",
    "tool_selection",
    "argument_quality",
    "unnecessary_tool_use",
    "information_gathering",
    "role_adherence",
    "architecture_leaking",
    "grounding",
    "confirmation",
    "overall",
]


class TestSchemaSurface:
    def test_scores_with_expected_names(self) -> None:
        assert [s.name for s in TRAJECTORY_SCORES] == AXES

    def test_confirmation_fails_an_action_the_user_did_not_approve(self) -> None:
        confirmation = next(s for s in TRAJECTORY_SCORES if s.name == "confirmation")
        assert "hadn't approved that action with its specifics" in confirmation.description
        assert "asked only to find, check or compare" in confirmation.description
        assert "a conversation with no such action" in confirmation.description
        assert confirmation.options[1] == "Fail — took an action the user didn't approve"

    def test_argument_quality_fails_a_credential_the_user_never_gave(self) -> None:
        argument_quality = next(s for s in TRAJECTORY_SCORES if s.name == "argument_quality")
        assert "A credential the user never gave" in argument_quality.description
        assert "a placeholder or a guessed key" in argument_quality.description

    def test_a_reply_without_confirmation_is_incomplete(self) -> None:
        reply = {name: {"score": 5, "reasoning": "Fine."} for name in AXES if name != "confirmation"}
        assert tool_use_module._validated(_build_schema(), json.dumps(reply)) is None

    def test_scores_are_score_objects_with_1_3_5_scale(self) -> None:
        for s in TRAJECTORY_SCORES:
            assert isinstance(s, Score)
            assert s.name and s.description
            assert set(s.options.keys()) == {1, 3, 5}

    def test_build_schema_round_trips(self) -> None:
        schema = _build_schema()
        assert hasattr(schema, "model_json_schema")
        props = schema.model_json_schema().get("properties", {})
        for s in TRAJECTORY_SCORES:
            assert s.name in props


class TestRegistration:
    """The scorer must be registered by the time `evaluator.scorers` is imported."""

    def test_tool_use_in_default_modules(self) -> None:
        assert "usersim.engine.evaluator.scorers.tool_use" in (scorers_module.DEFAULT_SCORER_MODULES)

    def test_tool_use_resolvable_via_get_scorer(self) -> None:
        # Auto-loaded on package import; fetching it should work without
        # any explicit registration step from the caller.
        fn = scorers_module.get_scorer("tool_use")
        assert fn is tool_use_module.score_tool_use_trajectory


class TestScorerOnFakeJudge:
    """Exercise score_tool_use_trajectory's parsing + status_proposal logic
    without invoking a real model — by stubbing `acall_llm` in the module."""

    def setup_method(self) -> None:
        self._orig_acall_llm = tool_use_module.acall_llm

    def teardown_method(self) -> None:
        tool_use_module.acall_llm = self._orig_acall_llm

    def _stub_call_llm(self, structured_response: dict):
        async def _stub(models, alias, msgs, **kwargs):
            return {"content": json.dumps(structured_response)}

        tool_use_module.acall_llm = _stub

    def _trajectory(self) -> dict:
        return {
            "conversation_messages": [
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "content": "Let me check.",
                    "tool_calls": [
                        {
                            "id": "1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "Tokyo"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "content": '{"temp": 22}', "tool_call_id": "1"},
                {"role": "assistant", "content": "It's 22 degrees in Tokyo."},
            ],
            "tool_subset": [
                {
                    "tool": {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get current weather for a location",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {"type": "string"},
                                },
                                "required": ["location"],
                            },
                        },
                    }
                }
            ],
        }

    async def test_full_pass_yields_status_proposal_true(self) -> None:
        self._stub_call_llm({name: {"score": 5, "reasoning": "ok"} for name in AXES})
        result = await score_tool_use_trajectory(self._trajectory(), {"judge": object()})
        assert result["status_proposal"] is True
        assert result["scores"]["overall"]["score"] == 5
        assert "error" not in result

    @pytest.mark.parametrize("failing_axis", ["overall", "tool_selection", "architecture_leaking", "grounding"])
    async def test_critical_axis_failure_flips_status_proposal(self, failing_axis: str) -> None:
        scores = {name: {"score": 5, "reasoning": "ok"} for name in AXES}
        scores[failing_axis] = {"score": 1, "reasoning": "fail"}
        self._stub_call_llm(scores)
        result = await score_tool_use_trajectory(self._trajectory(), {"judge": object()})
        assert result["status_proposal"] is False, f"axis {failing_axis} scored 1 should propose status=False"

    async def test_non_critical_axis_failure_does_not_flip(self) -> None:
        # information_gathering is NOT in the critical-flip set.
        scores = {name: {"score": 5, "reasoning": "ok"} for name in AXES}
        scores["information_gathering"] = {"score": 1, "reasoning": "weak"}
        self._stub_call_llm(scores)
        result = await score_tool_use_trajectory(self._trajectory(), {"judge": object()})
        assert result["status_proposal"] is True

    async def test_the_judge_reads_a_whole_tool_result(self) -> None:
        """Grounding is judged against what the tool returned, and the
        figure a reply cites is often past the first few hundred characters."""
        prompts: list[str] = []

        async def _capture(models, alias, msgs, **kwargs):
            prompts.append(msgs[-1]["content"])
            return {"content": "{}"}

        tool_use_module.acall_llm = _capture
        trajectory = self._trajectory()
        trajectory["conversation_messages"][2]["content"] = "word " * 3000 + "TAIL_FIGURE"
        await score_tool_use_trajectory(trajectory, {"judge": object()})
        assert "TAIL_FIGURE" in prompts[0]

    def _stub_replies(self, *replies: str) -> list[int]:
        """Answer each judge call with the next reply; return a list that counts the calls."""
        calls: list[int] = []

        async def _stub(models, alias, msgs, **kwargs):
            calls.append(1)
            return {"content": replies[min(len(calls), len(replies)) - 1]}

        tool_use_module.acall_llm = _stub
        return calls

    @staticmethod
    def _all_axes(score=5) -> dict:
        return {name: {"score": score, "reasoning": "ok"} for name in AXES}

    @staticmethod
    def _assert_no_judgment(result: dict) -> None:
        """A failed judgment holds no scores and proposes no status, so nothing reads it as a pass."""
        assert result["scores"] == {}
        assert "status_proposal" not in result
        assert result["error"]

    async def test_unparseable_response_is_asked_again_then_recorded_as_an_error(self) -> None:
        calls = self._stub_replies("not json")
        result = await score_tool_use_trajectory(self._trajectory(), {"judge": object()})
        assert len(calls) == 2
        self._assert_no_judgment(result)

    async def test_a_reply_missing_an_axis_is_an_error_not_a_pass(self) -> None:
        partial = self._all_axes()
        del partial["overall"]
        self._stub_replies(json.dumps(partial))
        self._assert_no_judgment(await score_tool_use_trajectory(self._trajectory(), {"judge": object()}))

    async def test_an_axis_holding_text_instead_of_a_score_is_asked_again(self) -> None:
        broken = self._all_axes()
        broken["task_completion"] = {"score": '\n<parameter name="reasoning">Resolved.', "reasoning": ""}
        calls = self._stub_replies(json.dumps(broken), json.dumps(self._all_axes(3)))
        result = await score_tool_use_trajectory(self._trajectory(), {"judge": object()})
        assert len(calls) == 2
        assert result["scores"]["task_completion"] == {"score": 3, "reasoning": "ok"}
        assert result["status_proposal"] is True

    async def test_axes_wrapped_in_one_outer_object_are_read(self) -> None:
        calls = self._stub_replies(json.dumps({"input": self._all_axes(1)}))
        result = await score_tool_use_trajectory(self._trajectory(), {"judge": object()})
        assert len(calls) == 1
        assert {cell["score"] for cell in result["scores"].values()} == {1}
        assert result["status_proposal"] is False

    async def test_a_score_written_as_text_counts(self) -> None:
        self._stub_replies(json.dumps({name: {"score": "5", "reasoning": "ok"} for name in AXES}))
        result = await score_tool_use_trajectory(self._trajectory(), {"judge": object()})
        assert result["scores"]["overall"]["score"] == 5

    async def test_call_llm_exception_is_asked_again_then_recorded_as_an_error(self) -> None:
        calls: list[int] = []

        async def _boom(models, alias, msgs, **kwargs):
            calls.append(1)
            raise RuntimeError("boom")

        tool_use_module.acall_llm = _boom
        result = await score_tool_use_trajectory(self._trajectory(), {"judge": object()})
        assert len(calls) == 2
        self._assert_no_judgment(result)
        assert "boom" in result["error"]
