# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Message and data formatting helpers for conversation simulation."""

from __future__ import annotations

import json
from typing import Any, Dict, List


def format_tools_for_prompt(tools: List[Dict[str, Any]]) -> str:
    """Format a list of tool definitions into a human-readable prompt block."""
    lines = []
    for tool_data in tools:
        tool = tool_data.get("tool", {}) if "tool" in tool_data else tool_data
        name = tool.get("function", {}).get("name", "Unknown")
        description = tool.get("function", {}).get("description", "No description")
        lines.append(f"- {name}: {description}")
    return "\n".join(lines)


def format_themes_for_prompt(themes: List[Dict[str, Any]]) -> str:
    """Format a list of theme dicts into a human-readable prompt block."""
    lines = []
    for theme in themes:
        lines.append(f"- {theme.get('type', 'Unknown')}: {theme.get('description', 'No description')}")
    return "\n".join(lines)


def format_conversation_history_for_prompt(messages: List[Dict[str, Any]]) -> str:
    """Format conversation messages into a text block for judge prompts."""
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
            lines.append(f"Tool response: {content[:500]}")

    return "\n".join(lines) if lines else "N/A"


def project_public_dialogue(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return only text a real user could observe.

    Assistant text remains public even when its source message also contains a
    tool call. Protocol envelopes, arguments, and raw tool results do not.
    """
    result: List[Dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "user":
            result.append({"role": "user", "content": content})
        elif role == "assistant" and content:
            result.append({"role": "assistant", "content": content})
    return result


def _parse_theme(raw: Any) -> Dict[str, Any]:
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
