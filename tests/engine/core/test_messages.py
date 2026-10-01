# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for core/messages.py formatting helpers."""

from usersim.engine.core.messages import (
    _parse_theme,
    format_conversation_history_for_prompt,
    format_themes_for_prompt,
    format_tools_for_prompt,
    project_public_dialogue,
)


class TestFormatToolsForPrompt:
    def test_formats_nested_tools(self, sample_tools):
        result = format_tools_for_prompt(sample_tools)
        assert "get_weather" in result
        assert "search_recipes" in result

    def test_empty_tools(self):
        assert format_tools_for_prompt([]) == ""


class TestFormatThemes:
    def test_formats_themes(self):
        themes = [{"type": "health", "description": "Health topics"}]
        result = format_themes_for_prompt(themes)
        assert "health" in result


class TestParseTheme:
    def test_dict_passthrough(self):
        d = {"type": "test", "description": "desc"}
        assert _parse_theme(d) == d

    def test_json_string(self):
        result = _parse_theme('{"type": "test", "description": "desc"}')
        assert result["type"] == "test"

    def test_plain_string(self):
        result = _parse_theme("health")
        assert result["type"] == "health"

    def test_list_returns_first(self):
        result = _parse_theme([{"type": "a"}, {"type": "b"}])
        assert result["type"] == "a"


class TestFormatConversationHistory:
    def test_formats_user_assistant(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = format_conversation_history_for_prompt(msgs)
        assert "User: Hello" in result
        assert "Assistant: Hi there" in result

    def test_empty_returns_na(self):
        assert format_conversation_history_for_prompt([]) == "N/A"

    def test_reasoning_content_is_never_rendered(self):
        """Judge and user-agent prompts are built from this. A trace
        reaching one would leak the assistant's private thinking into a
        prompt that is supposed to see only the visible dialogue."""
        result = format_conversation_history_for_prompt(
            [
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "content": "Hi there",
                    "reasoning_content": "SECRET_TRACE the user seems friendly",
                },
            ]
        )
        assert "SECRET_TRACE" not in result


class TestProjectPublicDialogue:
    def test_removes_protocol_but_keeps_visible_assistant_text(self):
        messages = [
            {"role": "system", "content": "private system"},
            {"role": "user", "content": "What is the weather?"},
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"Tokyo"}',
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "content": '{"secret_payload":"sunny"}',
                "tool_call_id": "c1",
            },
            {"role": "assistant", "content": "It is sunny."},
        ]
        assert project_public_dialogue(messages) == [
            {"role": "user", "content": "What is the weather?"},
            {"role": "assistant", "content": "Let me check."},
            {"role": "assistant", "content": "It is sunny."},
        ]

    def test_reasoning_content_is_dropped(self):
        """A thinking trace is not text a real user could observe, so it
        must not survive the projection — this is what keeps traces out
        of the follow-up and judge prompts built downstream."""
        assert project_public_dialogue(
            [
                {
                    "role": "assistant",
                    "content": "It is sunny.",
                    "reasoning_content": "SECRET_TRACE checked the tool result",
                },
            ]
        ) == [{"role": "assistant", "content": "It is sunny."}]

    def test_projection_does_not_mutate_source(self):
        messages = [
            {
                "role": "assistant",
                "content": "Checking.",
                "tool_calls": [{"function": {"name": "lookup", "arguments": "{}"}}],
            }
        ]
        before = repr(messages)
        project_public_dialogue(messages)
        assert repr(messages) == before
