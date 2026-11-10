# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The guarded health variants run by a host: a vetoed move is asked again within the same user turn."""

from __future__ import annotations

import json
from typing import Any

import pytest
from engine.core.test_episode_runtime_parity import (
    _GUARDED_CASES,
    _activation_kwargs,
    _ReplayableModel,
    _response_dict,
    _runtime,
)

from usersim.engine.core.episode_runtime import ActivationRequest, ActivationResult, EpisodeLifecycleComplete


async def test_a_vetoed_move_is_committed_again_within_the_same_user_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guide's veto path, hosted: a rejected call is answered with a tool message and asked again.

    The first proposal is vetoed. The second, made with the veto in view, is the
    move recorded; the proposal, the re-ask and the utterance are one user turn;
    and none of the exchange reaches the exported conversation.
    """
    from usersim.engine.probes.health_disclosure.moves import Guard

    check, checks = Guard.check, []

    def veto_the_first(self: Guard, move: Any, env: Any) -> tuple[bool, str]:
        checks.append(move.move)
        return (False, "forced veto") if len(checks) == 1 else check(self, move, env)

    monkeypatch.setattr(Guard, "check", veto_the_first)
    case = next(case for case in _GUARDED_CASES if case.probe_type == "health_therapy_disclosure")
    runtime, invocations = _runtime(case)
    user_activations: list[tuple[bool, bool, bool]] = []  # (offers tools, continues_turn, tool reply in view)
    proposals: list[dict[str, Any]] = []
    event = await runtime.advance()
    while isinstance(event, ActivationRequest):
        response = _response_dict(
            await _ReplayableModel(event.role, invocations).acompletion(
                list(event.messages), **_activation_kwargs(event)
            )
        )
        if event.role == "user":
            in_view = any(message.get("role") == "tool" for message in event.messages)
            user_activations.append((event.tools_enabled, event.continues_turn, in_view))
            if event.tools_enabled:
                call = response["tool_calls"][0]["function"]
                arguments = {**json.loads(call["arguments"]), "reasoning": f"proposal {len(proposals) + 1}"}
                call["arguments"] = json.dumps(arguments)
                proposals.append(arguments)
        event = await runtime.advance(ActivationResult(activation_id=event.activation_id, response=response))

    assert isinstance(event, EpisodeLifecycleComplete)
    first = next(i for i, (offers_tools, _, _) in enumerate(user_activations) if offers_tools)
    assert user_activations[first : first + 3] == [
        (True, False, False),  # the proposal opens the user's turn
        (True, True, True),  # the re-ask, with the veto in view, continues it
        (False, True, False),  # and so does the utterance, written without the private exchange
    ]
    # The vetoed check was the user's first proposal, not the probe's own risk pacing.
    assert checks[0] == proposals[0]["move"]
    assert event.result["guard_veto_count"] == 1
    moves = event.result["moves_detail"]
    moves = json.loads(moves) if isinstance(moves, str) else moves
    assert moves[0]["reasoning"] == "proposal 2"
    assert [veto["move"] for veto in moves[0]["vetoes"]] == [proposals[0]["move"]]
    messages = event.result["conversation_messages"]
    messages = json.loads(messages) if isinstance(messages, str) else messages
    assert not any(message.get("tool_calls") or message.get("role") == "tool" for message in messages)
