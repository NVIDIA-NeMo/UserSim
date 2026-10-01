# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for evaluator/scorers/tool_use.py — the 8-axis tool-calling judge.

Covers the 8-axis schema, which belongs to the evaluator rather than the
simulator, plus a behavioral test for scorer registration via the auto-load
path.
"""

from __future__ import annotations

import pytest
from data_designer.config.column_configs import Score

from usersim.engine.evaluator import scorers as scorers_module
from usersim.engine.evaluator.scorers import tool_use as tool_use_module
from usersim.engine.evaluator.scorers.tool_use import (
    TRAJECTORY_SCORES,
    _build_schema,
    score_tool_use_trajectory,
)


class TestSchemaSurface:
    def test_8_scores_with_expected_names(self) -> None:
        names = {s.name for s in TRAJECTORY_SCORES}
        assert names == {
            "task_completion",
            "tool_selection",
            "argument_quality",
            "unnecessary_tool_use",
            "information_gathering",
            "role_adherence",
            "architecture_leaking",
            "overall",
        }

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
        scorers_module.load_default_scorers()  # idempotent re-trigger
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
        import json as _json

        async def _stub(models, alias, msgs, **kwargs):
            return {"content": _json.dumps(structured_response)}

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
        self._stub_call_llm(
            {
                name: {"score": 5, "reasoning": "ok"}
                for name in [
                    "task_completion",
                    "tool_selection",
                    "argument_quality",
                    "unnecessary_tool_use",
                    "information_gathering",
                    "role_adherence",
                    "architecture_leaking",
                    "overall",
                ]
            }
        )
        result = await score_tool_use_trajectory(self._trajectory(), {"judge": object()})
        assert result["status_proposal"] is True
        assert result["scores"]["overall"]["score"] == 5
        assert "error" not in result

    @pytest.mark.parametrize("failing_axis", ["overall", "tool_selection", "architecture_leaking"])
    async def test_critical_axis_failure_flips_status_proposal(self, failing_axis: str) -> None:
        scores = {
            name: {"score": 5, "reasoning": "ok"}
            for name in [
                "task_completion",
                "tool_selection",
                "argument_quality",
                "unnecessary_tool_use",
                "information_gathering",
                "role_adherence",
                "architecture_leaking",
                "overall",
            ]
        }
        scores[failing_axis] = {"score": 1, "reasoning": "fail"}
        self._stub_call_llm(scores)
        result = await score_tool_use_trajectory(self._trajectory(), {"judge": object()})
        assert result["status_proposal"] is False, f"axis {failing_axis} scored 1 should propose status=False"

    async def test_non_critical_axis_failure_does_not_flip(self) -> None:
        # information_gathering is NOT in the critical-flip set.
        scores = {
            name: {"score": 5, "reasoning": "ok"}
            for name in [
                "task_completion",
                "tool_selection",
                "argument_quality",
                "unnecessary_tool_use",
                "information_gathering",
                "role_adherence",
                "architecture_leaking",
                "overall",
            ]
        }
        scores["information_gathering"] = {"score": 1, "reasoning": "weak"}
        self._stub_call_llm(scores)
        result = await score_tool_use_trajectory(self._trajectory(), {"judge": object()})
        assert result["status_proposal"] is True

    async def test_unparseable_response_returns_parse_error(self) -> None:
        async def _stub(models, alias, msgs, **kwargs):
            return {"content": "not json"}

        tool_use_module.acall_llm = _stub
        result = await score_tool_use_trajectory(self._trajectory(), {"judge": object()})
        assert result["error"] == "parse_failure"
        # All scores marked None.
        for s in TRAJECTORY_SCORES:
            assert result["scores"][s.name]["score"] is None

    async def test_call_llm_exception_returns_error_field(self) -> None:
        async def _boom(models, alias, msgs, **kwargs):
            raise RuntimeError("boom")

        tool_use_module.acall_llm = _boom
        result = await score_tool_use_trajectory(self._trajectory(), {"judge": object()})
        assert "error" in result
        assert "boom" in result["error"]
        # status_proposal defaults to True when scoring fails (we can't
        # claim sim-side failure on the assistant's behalf without data).
        assert result["status_proposal"] is True
