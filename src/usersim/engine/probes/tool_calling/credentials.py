# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Credentials the simulated user holds for the services a row's tools call.

A tool entry may declare ``credentials`` beside its function spec: a mapping
from a parameter of the function to the credential that parameter takes, named
the way a user knows it::

    {"type": "function", "function": {...}, "credentials": {"api_key": "SerpApi key"}}

Each conversation draws one random value per credential its tools declare, so
tools that share a credential take the same value and no two conversations
share one. The simulated user holds those values and gives one when the
assistant asks for it. A call whose argument is not the value the user holds,
because it is missing, empty or made up, is rejected before any response is
simulated, as the service would reject it.

A tool that sets a new value for one of those credentials, such as a password
change, declares ``sets_credentials`` the same way::

    {"type": "function", "function": {...}, "sets_credentials": {"new_password": "account password"}}

Once such a call is accepted, the value it passed is the one the user holds,
so later calls must carry the new value and the old one is rejected.
"""

from __future__ import annotations

import hashlib
import random
import string
from collections.abc import Iterable, Mapping
from typing import Any

_ALPHABET = string.ascii_lowercase + string.digits
_LENGTH = 24


def declared_credentials(tool_entry: Mapping[str, Any]) -> dict[str, str]:
    """The ``{parameter: credential}`` mapping a tool entry declares.

    Raises ``ValueError`` when the declaration is not a mapping of parameter
    names to non-empty credential names, or names a parameter the function
    does not define, since no call could then pass the credential.
    """
    return _parameter_credentials(tool_entry, "credentials")


def credentials_set_by(tool_entry: Mapping[str, Any]) -> dict[str, str]:
    """The ``{parameter: credential}`` mapping whose new values the tool sets.

    Read from the entry's ``sets_credentials`` and checked like
    ``declared_credentials``.
    """
    return _parameter_credentials(tool_entry, "sets_credentials")


def _parameter_credentials(tool_entry: Mapping[str, Any], key: str) -> dict[str, str]:
    declared = tool_entry.get(key)
    if declared is None:
        return {}
    spec = tool_entry["tool"]["function"] if "tool" in tool_entry else tool_entry.get("function", tool_entry)
    name = spec.get("name", "unknown")
    if not isinstance(declared, Mapping) or not all(
        isinstance(parameter, str) and isinstance(credential, str) and credential.strip()
        for parameter, credential in declared.items()
    ):
        raise ValueError(f"tool_calling: `{name}` {key} must map parameter names to credential names")
    properties = (spec.get("parameters") or {}).get("properties") or {}
    undefined = sorted(set(declared) - set(properties))
    if undefined:
        raise ValueError(f"tool_calling: `{name}` declares {key} for parameters it does not define: {undefined}")
    return dict(declared)


def draw_credentials(tools: Iterable[Mapping[str, Any]], identity: str) -> dict[str, str]:
    """One random value for each credential the tools declare, keyed by credential name.

    The values depend only on ``identity`` and the credential's name, so one
    conversation always holds the same values and another holds different ones.
    """
    names = sorted({credential for tool in tools for credential in declared_credentials(tool).values()})
    held = {}
    for name in names:
        seed = int(hashlib.sha256(f"{identity}\x00{name}".encode()).hexdigest(), 16)
        rng = random.Random(seed)
        held[name] = "".join(rng.choice(_ALPHABET) for _ in range(_LENGTH))
    return held


def rejected_credential(
    declared: Mapping[str, str],
    arguments: Mapping[str, Any],
    held: Mapping[str, str],
) -> tuple[str, str] | None:
    """The first ``(parameter, credential)`` whose argument is not the value the user holds."""
    for parameter, credential in declared.items():
        if credential not in held or arguments.get(parameter) != held[credential]:
            return parameter, credential
    return None


def format_held_credentials(held: Mapping[str, str]) -> str:
    """One ``- <credential>: <value>`` line per credential, for the simulated user's prompt."""
    return "\n".join(f"- {name}: {value}" for name, value in held.items())
