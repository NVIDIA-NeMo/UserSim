# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool surface for the financial_services probe.

The assistant is offered the PERMANENT tools up front (``kb_search``,
``verify_identity``, ...) plus a single ``call_tool(tool_name, arguments)`` gate
for DISCOVERABLE tools. A discoverable tool is only usable once its
documentation has been retrieved (its name appears in a retrieved document's
``mentions_tools``) — the tau-Banking discoverable-tool mechanic. This module is
pure helpers; the probe's ``after_assistant_turn`` tool loop owns state.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from usersim.engine.core.finance_bank import (
    Institution,
    framework_primitive_tool,
    verify_identity_schema,
)

CALL_TOOL_NAME = "call_tool"


def build_offered_tools(
    institution: Institution,
    *,
    identity_required_fields: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """OpenAI-style tool schemas offered to the assistant, SCOPED to ``institution``.

    Permanent (non-discoverable) tools are offered by name; discoverable tools
    are reached only through the ``call_tool`` gate. Framework primitives
    (``kb_search`` / ``verify_identity``) use the canonical schema from
    ``core.finance_bank`` so the assistant always sees their real parameters
    even if a bank asset ships them empty.

    ``identity_required_fields`` specializes ``verify_identity`` to the region's
    KYC contract (e.g. India also requires a PAN). Omit it for the universal
    name + DOB default.
    """
    offered: List[Dict[str, Any]] = []
    for t in institution.permanent_tools():
        canonical = framework_primitive_tool(t.name)
        if t.name == "verify_identity" and identity_required_fields:
            canonical = verify_identity_schema(identity_required_fields)
        if canonical is not None:
            offered.append(
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": canonical["description"],
                        "parameters": canonical["parameters"],
                    },
                }
            )
        else:
            offered.append(t.to_openai_tool())
    offered.append(
        {
            "type": "function",
            "function": {
                "name": CALL_TOOL_NAME,
                "description": (
                    "Invoke a tool documented in the knowledge base. You must first "
                    "retrieve the tool's documentation via kb_search before calling it."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tool_name": {"type": "string"},
                        "arguments": {"type": "object"},
                    },
                    "required": ["tool_name"],
                },
            },
        }
    )
    return offered


def extract_call(tc: Any) -> Tuple[str, Dict[str, Any]]:
    """Pull ``(function_name, arguments_dict)`` from one tool_call dict."""
    if not isinstance(tc, dict):
        return ("<malformed>", {})
    fn = tc.get("function")
    if not isinstance(fn, dict):
        return (str(tc.get("name") or "<unknown>"), {})
    name = str(fn.get("name") or "<unknown>")
    raw = fn.get("arguments")
    if isinstance(raw, dict):
        return (name, raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return (name, parsed if isinstance(parsed, dict) else {})
        except (json.JSONDecodeError, TypeError):
            return (name, {})
    return (name, {})


def resolve_invoked_tool(function_name: str, arguments: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Map an assistant tool call to the *domain* tool it targets.

    ``call_tool(tool_name=..., arguments=...)`` targets the named discoverable
    tool; any other function call targets itself (permanent tool or a
    directly-named tool).
    """
    if function_name == CALL_TOOL_NAME:
        target = str(arguments.get("tool_name") or "<unknown>")
        return (target, arguments.get("arguments") or {})
    return (function_name, arguments)
