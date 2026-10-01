# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared OpenAI-compatible tool-call parsing.

One hardened place to pull a tool call (name + arguments) out of a model
completion, tolerant of the shape variance seen across OpenAI-"compatible"
providers/hosts:

- a structured ``tool_calls`` array,
- ``arguments`` already parsed to a ``dict``,
- JSON embedded in ``content`` (fenced or prose-wrapped),
- a tool-call *stub* whose args are empty/unparseable (recover from ``content``).

Consumers: the ``safety_agentic`` probe (tool execution) and the
``health_disclosure`` move-space (the ``commit_move`` call). Hardening the
contract in one place means every consumer benefits — this is framed as
robustness across the OpenAI-compatible boundary, not model testing.
"""

from __future__ import annotations

import json
from typing import Any


def parse_arguments(raw: Any) -> dict[str, Any]:
    """Coerce a tool-call ``arguments`` payload into a ``dict``.

    Accepts an already-parsed dict or a JSON string; anything else (or a JSON
    value that isn't an object) yields ``{}`` rather than raising.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def parse_tool_call(tc: Any) -> tuple[str, dict[str, Any]]:
    """Pull ``(tool_name, args_dict)`` out of a single ``tool_call`` dict.

    Defensive: tool-call shapes vary slightly across providers. Falls back to a
    sentinel name and/or empty args on any malformation rather than crashing the
    trajectory.
    """
    if not isinstance(tc, dict):
        return ("<malformed>", {})
    fn = tc.get("function")
    if not isinstance(fn, dict):
        return (str(tc.get("name") or "<unknown>"), {})
    name = str(fn.get("name") or "<unknown>")
    return (name, parse_arguments(fn.get("arguments")))


def json_object_from_content(content: Any) -> dict[str, Any]:
    """Best-effort recovery of a single JSON object embedded in free text.

    Handles fenced or prose-wrapped JSON by slicing the outermost ``{ ... }``.
    Returns ``{}`` when nothing parseable is found.
    """
    if not isinstance(content, str):
        return {}
    t = content.strip()
    if "{" in t and "}" in t:
        t = t[t.find("{") : t.rfind("}") + 1]
    try:
        parsed = json.loads(t)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def recover_tool_call(
    result: dict[str, Any],
    name: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Recover ``(args_dict, raw_tool_call | None)`` from a completion result.

    Tolerant of provider variance: looks for a ``tool_calls`` entry (matching
    ``name`` when given, otherwise the first), parses its arguments, and — when
    the args are missing/empty/unparseable, or there is no tool call at all —
    falls back to JSON embedded in ``content``. The raw native tool call is
    returned (when present) for audit, even if the args were recovered from
    ``content``.
    """
    calls = result.get("tool_calls") or []
    call: dict[str, Any] | None = None
    if isinstance(calls, list) and calls:
        if name is None:
            call = calls[0] if isinstance(calls[0], dict) else None
        else:
            call = next(
                (c for c in calls if isinstance(c, dict) and (c.get("function") or {}).get("name") in (None, name)),
                None,
            )
    args: dict[str, Any] = {}
    if call is not None:
        args = parse_arguments((call.get("function") or {}).get("arguments"))
    if not args:  # provider-variance fallback: JSON in content
        args = json_object_from_content(result.get("content"))
    return args, call
