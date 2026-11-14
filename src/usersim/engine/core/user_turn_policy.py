# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How the conversation loop treats a probe's simulated user turns and the model's history.

A probe returns a :class:`UserTurnPolicy` from ``user_turn_policy()``. The
defaults are the loop's own behaviour, so a probe that keeps them changes
nothing.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Any


@dataclass(frozen=True)
class UserTurnPolicy:
    """What a probe changes about the loop. The defaults change nothing."""

    #: Whether long assistant replies may be replaced by summaries in later turns, when
    #: the run's ``context_compression`` allows it. Off, the model under test and the
    #: simulated user see every earlier reply as written.
    context_compression: bool = True
    #: Whether the last follow-ups may tell the simulated user to thank the assistant and wrap up.
    wrap_up: bool = True
    #: Replaces the loop's follow-up role anchor, which casts the user as asking for help.
    followup_anchor: str | None = None
    #: Fourth-wall and refusal-echo phrases this probe's user turns may contain.
    allowed_phrases: frozenset[str] = frozenset()
    #: Names left out of a user turn's script check, such as brand names written in
    #: Latin script whatever the locale.
    script_check_ignores: tuple[str, ...] = ()
    #: Why a generated opening must be written again, or None to accept it. The loop
    #: runs it on each draft of turn 1 before the gate; a verbatim opening is not checked.
    check_opening: Callable[[str], str | None] | None = None
    #: Whether the simulated user sees that the assistant used its tools: a reply
    #: after tool results opens with ``(after using its tools)``, and no tool name,
    #: argument or result is shown.
    show_tool_use: bool = False

    def without_ignored_names(self, text: str) -> str:
        """``text`` with ``script_check_ignores`` removed, for the script check."""
        names = tuple(name for name in self.script_check_ignores if name)
        return _names_pattern(names).sub(" ", text) if names else text


def resolve_user_turn_policy(probe: Any) -> UserTurnPolicy:
    """The probe's ``user_turn_policy()``, or the defaults for an adapter without the hook."""
    hook = getattr(probe, "user_turn_policy", None)
    return hook() if callable(hook) else UserTurnPolicy()


@cache
def _names_pattern(names: tuple[str, ...]) -> re.Pattern[str]:
    longest_first = sorted(names, key=len, reverse=True)
    return re.compile("|".join(re.escape(name) for name in longest_first), re.IGNORECASE)
