# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the unified dispatcher in generator.py."""

import json

from usersim.engine.generator import _make_failed_result


class TestMakeFailedResult:
    """_make_failed_result produces simulation-level columns.

    Dispatcher-level columns (behavioral_profile, locale, conversation_language)
    are set by the generator before calling _make_failed_result.
    """

    SIMULATION_COLUMNS = [
        "user_query", "conversation_messages", "conversation_metadata",
        "conversation_status", "simulation_outcome", "simulation_traces",
        "num_turns", "num_tool_calls", "tool_subset",
        "disclosure_style", "user_interaction_style",
    ]

    def test_returns_all_simulation_keys(self):
        result = _make_failed_result("test failure")
        for col in self.SIMULATION_COLUMNS:
            assert col in result, f"Missing column: {col}"

    def test_status_is_false(self):
        result = _make_failed_result("test failure")
        assert result["conversation_status"] is False

    def test_simulation_outcome_carries_failure(self):
        result = _make_failed_result("test failure")
        outcome = json.loads(result["simulation_outcome"])
        assert outcome["status"] == "failed"
        assert outcome["failure_detail"] == "test failure"

    def test_tool_subset_none(self):
        result = _make_failed_result("test failure")
        assert result["tool_subset"] is None

    def test_metadata_contains_reason(self):
        result = _make_failed_result("test reason")
        meta = json.loads(result["conversation_metadata"])
        assert meta["reason"] == "test reason"
        assert meta["skipped"] is True
