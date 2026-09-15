# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tool_calling/verifiers.py."""

from usersim.engine.probes.tool_calling.verifiers import ToolCallVerifier


class TestToolCallVerifier:
    def test_valid_tool_call(self, sample_tools):
        verifier = ToolCallVerifier()
        tool_calls = [{
            "id": "call_1",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "Tokyo"}',
            },
        }]
        all_ok, errors, _, _, correct = verifier.verify(tool_calls, sample_tools)
        assert all_ok
        assert len(correct) == 1

    def test_missing_required_param(self, sample_tools):
        verifier = ToolCallVerifier()
        tool_calls = [{
            "id": "call_1",
            "function": {
                "name": "get_weather",
                "arguments": '{}',
            },
        }]
        all_ok, errors, _, _, _ = verifier.verify(tool_calls, sample_tools)
        assert not all_ok

    def test_unknown_tool(self, sample_tools):
        verifier = ToolCallVerifier()
        tool_calls = [{
            "id": "call_1",
            "function": {
                "name": "nonexistent_tool",
                "arguments": '{}',
            },
        }]
        all_ok, _, _, _, _ = verifier.verify(tool_calls, sample_tools)
        assert not all_ok

    def test_empty_tool_calls(self, sample_tools):
        verifier = ToolCallVerifier()
        all_ok, _, _, _, _ = verifier.verify([], sample_tools)
        assert all_ok
