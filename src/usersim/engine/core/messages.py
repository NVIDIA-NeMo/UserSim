# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Message and data formatting helpers for conversation simulation."""

from __future__ import annotations

import json
from typing import Any


def format_tools_for_prompt(tools: list[dict[str, Any]]) -> str:
    """Format a list of tool definitions into a human-readable prompt block.

    A tool entry may declare ``credentials`` beside its function spec, mapping
    a parameter to the credential it takes, such as ``{"api_key": "SerpApi
    key"}``. Its line then names those credentials, so a judge reading the
    block knows that a call needs the user's credential and that asking the
    user for it is part of making the call.
    """
    lines = []
    for tool_data in tools:
        tool = tool_data.get("tool", {}) if "tool" in tool_data else tool_data
        name = tool.get("function", {}).get("name", "Unknown")
        description = tool.get("function", {}).get("description", "No description")
        line = f"- {name}: {description}"
        credentials = tool_data.get("credentials")
        if isinstance(credentials, dict) and credentials:
            line += f" (needs the user's {' and '.join(dict.fromkeys(credentials.values()))})"
        lines.append(line)
    return "\n".join(lines)


#: The most of each tool result the evaluator's judges read, in characters
#: weighed as ``_cut_tool_result`` does. They judge whether the assistant's
#: answers match what the tools returned, so they need the whole result; the
#: cap only keeps one runaway result from flooding a judge call. Tool results
#: run 2.6 to 3.8 characters a token, so this is about 8,000 tokens.
EVALUATOR_TOOL_RESULT_CHARS = 24_000

#: What one Han, Kana or Hangul character weighs against a cap. They count two
#: to a word (``llm._word_count``), and an English word with its space averages
#: about six characters, so a cap holds comparable content in every language.
_DENSE_CHARACTER_WEIGHT = 3


def _cut_tool_result(content: str, limit: int) -> str:
    """``content`` up to ``limit`` characters, each Han, Kana or Hangul one counting as ``_DENSE_CHARACTER_WEIGHT``."""
    from usersim.engine.core.llm import _CHARACTER_COUNTED  # llm imports this module through engine.config

    used = 0
    for i, ch in enumerate(content):
        used += _DENSE_CHARACTER_WEIGHT if _CHARACTER_COUNTED.match(ch) else 1
        if used > limit:
            return f"{content[:i]} [cut: {i:,} of {len(content):,} characters shown]"
    return content


def format_conversation_history_for_prompt(messages: list[dict[str, Any]], *, tool_result_chars: int = 500) -> str:
    """Format conversation messages into a text block for judge prompts.

    Each tool result is cut at ``tool_result_chars`` characters, a Han, Kana
    or Hangul character counting as three. A cut result says how much of it
    is shown, so a judge doesn't count what the cut removed as missing from
    the result.
    """
    lines = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls")

        if role == "system":
            continue
        elif role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            if tool_calls:
                tc_summary = []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    tc_summary.append(f"{fn.get('name', '?')}({fn.get('arguments', '{}')})")
                lines.append(f"Assistant [tool calls]: {'; '.join(tc_summary)}")
            if content:
                lines.append(f"Assistant: {content}")
        elif role == "tool":
            lines.append(f"Tool response: {_cut_tool_result(content, tool_result_chars)}")

    return "\n".join(lines) if lines else "N/A"


def project_public_dialogue(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return only text a real user could observe.

    Assistant text remains public even when its source message also contains a
    tool call. Protocol envelopes, arguments, and raw tool results do not.
    """
    result: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "user":
            result.append({"role": "user", "content": content})
        elif role == "assistant" and content:
            result.append({"role": "assistant", "content": content})
    return result


def _parse_theme(raw: Any) -> dict[str, Any]:
    """Normalise a theme value into a dict with 'type' and 'description' keys."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed[0] if parsed else {"type": "general", "description": "General conversation"}
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        return {"type": raw, "description": raw}
    if isinstance(raw, list):
        return raw[0] if raw else {"type": "general", "description": "General conversation"}
    return {"type": "general", "description": "General conversation"}
