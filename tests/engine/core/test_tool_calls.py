# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared OpenAI-compatible tool-call parser (``core.tool_calls``).

Covers the provider-variance shapes that consumers (safety_agentic tool
execution, health_disclosure move-space) rely on being parsed uniformly.
"""

from __future__ import annotations

from usersim.engine.core.tool_calls import (
    json_object_from_content,
    parse_arguments,
    parse_tool_call,
    recover_tool_call,
)


class TestParseArguments:
    def test_dict_passthrough(self):
        assert parse_arguments({"a": 1}) == {"a": 1}

    def test_json_string(self):
        assert parse_arguments('{"a": 1}') == {"a": 1}

    def test_bad_json_string(self):
        assert parse_arguments("{not json") == {}

    def test_non_object_json(self):
        assert parse_arguments("[1, 2, 3]") == {}

    def test_none_and_other_types(self):
        assert parse_arguments(None) == {}
        assert parse_arguments(42) == {}


class TestParseToolCall:
    def test_standard_openai_shape(self):
        tc = {"function": {"name": "f", "arguments": '{"x": 1}'}}
        assert parse_tool_call(tc) == ("f", {"x": 1})

    def test_args_already_dict(self):
        tc = {"function": {"name": "f", "arguments": {"x": 1}}}
        assert parse_tool_call(tc) == ("f", {"x": 1})

    def test_no_function_uses_top_level_name(self):
        assert parse_tool_call({"name": "loose"}) == ("loose", {})

    def test_non_dict_is_malformed(self):
        assert parse_tool_call("garbage") == ("<malformed>", {})

    def test_unparseable_args_yield_empty(self):
        tc = {"function": {"name": "f", "arguments": "{bad"}}
        assert parse_tool_call(tc) == ("f", {})


class TestJsonObjectFromContent:
    def test_plain_json(self):
        assert json_object_from_content('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        assert json_object_from_content('```json\n{"a": 1}\n```') == {"a": 1}

    def test_prose_wrapped(self):
        assert json_object_from_content('Sure! {"a": 1} hope that helps') == {"a": 1}

    def test_no_json(self):
        assert json_object_from_content("no json here") == {}

    def test_non_string(self):
        assert json_object_from_content(None) == {}


class TestRecoverToolCall:
    def test_structured_tool_call_by_name(self):
        result = {
            "tool_calls": [
                {"function": {"name": "other", "arguments": "{}"}},
                {"function": {"name": "commit_move", "arguments": '{"move": "withhold"}'}},
            ]
        }
        args, call = recover_tool_call(result, name="commit_move")
        assert args == {"move": "withhold"}
        assert call is not None and call["function"]["name"] == "commit_move"

    def test_first_call_when_no_name(self):
        result = {"tool_calls": [{"function": {"name": "f", "arguments": '{"x": 1}'}}]}
        args, call = recover_tool_call(result)
        assert args == {"x": 1} and call is not None

    def test_args_as_dict(self):
        result = {"tool_calls": [{"function": {"name": "f", "arguments": {"x": 1}}}]}
        args, _ = recover_tool_call(result, name="f")
        assert args == {"x": 1}

    def test_json_in_content_fallback_when_no_tool_call(self):
        args, call = recover_tool_call({"content": '{"move": "deflect"}'}, name="commit_move")
        assert args == {"move": "deflect"} and call is None

    def test_empty_stub_falls_back_to_content_but_keeps_raw_call(self):
        # A tool-call stub with empty args, plus usable JSON in content.
        result = {
            "tool_calls": [{"id": "c1", "function": {"name": "commit_move", "arguments": ""}}],
            "content": '{"move": "reveal_risk"}',
        }
        args, call = recover_tool_call(result, name="commit_move")
        assert args == {"move": "reveal_risk"}
        assert call is not None and call["id"] == "c1"  # raw call still surfaced for audit

    def test_no_tool_calls_no_content(self):
        args, call = recover_tool_call({}, name="commit_move")
        assert args == {} and call is None

    def test_non_list_tool_calls_guarded(self):
        args, call = recover_tool_call({"tool_calls": "nope", "content": "{}"}, name="x")
        assert args == {} and call is None
