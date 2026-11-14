# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for core/messages.py formatting helpers."""

import pytest

from usersim.engine.core.messages import (
    _parse_theme,
    format_conversation_history_for_prompt,
    format_judge_dialogue,
    format_tools_for_prompt,
    project_public_dialogue,
    tool_result_weight,
)


class TestFormatToolsForPrompt:
    def test_formats_nested_tools(self, sample_tools):
        result = format_tools_for_prompt(sample_tools)
        assert "get_weather" in result
        assert "search_recipes" in result

    def test_empty_tools(self):
        assert format_tools_for_prompt([]) == ""

    def test_names_the_credentials_a_tool_declares(self, sample_tools):
        sample_tools[0]["credentials"] = {"api_key": "WeatherCo API key", "api_secret": "WeatherCo API secret"}
        assert format_tools_for_prompt(sample_tools).splitlines() == [
            "- get_weather: Get current weather for a location "
            "(needs the user's WeatherCo API key and WeatherCo API secret)",
            "- search_recipes: Search for recipes by ingredient",
        ]


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


class TestFormatJudgeDialogue:
    """The assistant judge reads every tool result as whole as the evaluator does."""

    def _messages(self):
        return [
            {"role": "user", "content": "Weather in Tokyo?"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "get_weather"}}]},
            {"role": "tool", "content": "e" * 1000, "tool_call_id": "1"},
            {"role": "assistant", "content": "It is sunny.", "reasoning_content": "SECRET_TRACE"},
            {"role": "user", "content": "And Osaka?"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "get_weather"}}]},
            {"role": "tool", "content": "t" * 30000, "tool_call_id": "2"},
        ]

    def test_results_are_read_up_to_the_evaluators_cap(self):
        text = format_judge_dialogue(self._messages())
        assert "Tool response: " + "e" * 1000 + "\n" in text
        assert "t" * 24000 + " [cut: 24,000 of 30,000 characters shown]" in text

    def test_it_shows_the_tool_calls_and_never_a_reasoning_trace(self):
        text = format_judge_dialogue(self._messages())
        assert text.count("Assistant [tool calls]: get_weather") == 2
        assert "SECRET_TRACE" not in text

    def test_an_empty_conversation_reads_as_na(self):
        assert format_judge_dialogue([]) == "N/A"


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

    def test_tool_results_are_cut_at_the_given_length(self):
        msgs = [{"role": "tool", "content": "x" * 600 + "TAIL"}]
        assert "TAIL" not in format_conversation_history_for_prompt(msgs)
        assert "TAIL" in format_conversation_history_for_prompt(msgs, tool_result_chars=1000)

    def test_a_cut_tool_result_says_how_much_is_shown(self):
        msgs = [{"role": "tool", "content": "x" * 1200}]
        assert format_conversation_history_for_prompt(msgs).endswith(" [cut: 500 of 1,200 characters shown]")
        assert "[cut:" not in format_conversation_history_for_prompt(msgs, tool_result_chars=1200)

    @pytest.mark.parametrize("dense", ["漢", "か", "カ", "한"])
    def test_han_kana_and_hangul_count_three_to_a_character(self, dense):
        """A cap holds comparable content in every language: one of these
        characters carries about as much as three Latin ones."""
        msgs = [{"role": "tool", "content": '{"note": "' + dense * 400 + '"}'}]
        result = format_conversation_history_for_prompt(msgs, tool_result_chars=600)
        assert result.endswith(" [cut: 206 of 412 characters shown]")
        assert result.count(dense) == 196

    def test_other_scripts_count_one_to_a_character(self):
        msgs = [{"role": "tool", "content": "नमस्ते" * 100}]
        assert "[cut:" not in format_conversation_history_for_prompt(msgs, tool_result_chars=600)

    def test_a_result_weighs_what_the_cut_counts(self):
        """``tool_result_weight`` measures a result as the cut does, so a result of exactly the cap is shown whole."""
        content = '{"city": "東京", "note": "नमस्ते"}'
        assert tool_result_weight(content) == len(content) + 2 * 2
        assert "[cut:" not in format_conversation_history_for_prompt(
            [{"role": "tool", "content": content}], tool_result_chars=tool_result_weight(content)
        )
        assert "[cut:" in format_conversation_history_for_prompt(
            [{"role": "tool", "content": content}], tool_result_chars=tool_result_weight(content) - 1
        )

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

    def test_a_reply_after_tool_results_can_be_marked_without_showing_them(self):
        """A chat app shows that the assistant looked something up, never what it found."""
        messages = [
            {"role": "user", "content": "What is the weather?"},
            {"role": "assistant", "content": "Let me check.", "tool_calls": [{"id": "c1", "function": {"name": "w"}}]},
            {"role": "tool", "content": '{"secret_payload":"sunny"}', "tool_call_id": "c1"},
            {"role": "assistant", "content": "It is sunny."},
            {"role": "user", "content": "And tomorrow?"},
            {"role": "assistant", "content": "I can't say yet."},
        ]
        assert project_public_dialogue(messages, mark_tool_use=True) == [
            {"role": "user", "content": "What is the weather?"},
            {"role": "assistant", "content": "Let me check."},
            {"role": "assistant", "content": "(after using its tools) It is sunny."},
            {"role": "user", "content": "And tomorrow?"},
            {"role": "assistant", "content": "I can't say yet."},
        ]
        assert "(after using its tools)" not in str(project_public_dialogue(messages))

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
