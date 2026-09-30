# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""identity_contradictions lists the turns the identity scorer labelled contradictions, with the question each answered."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from usersim.engine.core.identity_spec import load_identity_spec_default, reset_identity_spec_cache
from usersim.engine.evaluator.scorers.identity_disclosure import score_identity_disclosure_trajectory
from usersim.reporting import identity_contradictions
from usersim.reporting.identity import CONTRADICTION_COLUMNS

NEMOTRON = "nvidia/nemotron-3-super-120b-a12b"
PATCH = "usersim.engine.evaluator.scorers.identity_disclosure.acall_llm"


@pytest.fixture(scope="module")
def shipped():
    reset_identity_spec_cache()
    yield load_identity_spec_default()
    reset_identity_spec_cache()


def _row(spec, user: str, reply: str, tactic: str) -> dict[str, Any]:
    return {
        "conversation_messages": [{"role": "user", "content": user}, {"role": "assistant", "content": reply}],
        "identity_tactic_id": tactic,
        "expected_identity": json.dumps(asdict(spec.resolve(NEMOTRON))),
        "identity_spec": json.dumps(
            {"version": spec.version, "digest": spec.digest, "vocabulary": spec.vocabulary(), "rules": spec.rules()}
        ),
    }


async def _block(row: dict[str, Any], claims: list[dict[str, Any]], stance: str) -> dict[str, Any]:
    """The scorer's block for a one-turn ``row`` whose extraction has ``claims`` and ``stance``."""
    extraction = {"turns": [{"turn": 1, "premise_stance": stance, "claims": claims, "self_details": []}]}
    reply = {"role": "assistant", "content": json.dumps({**extraction, "reasoning": []})}
    with patch(PATCH, AsyncMock(return_value=reply)):
        return await score_identity_disclosure_trajectory(row, {"evaluator_model": object()})


def _frames(rows: list[tuple[dict[str, Any], dict[str, Any]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trajectories and evaluations as stored: joined on ``trajectory_id``, the evaluation cell JSON-encoded."""
    ids = [f"t{i:015d}" for i in range(len(rows))]
    trajectories = pd.DataFrame(
        [
            {
                "trajectory_id": tid,
                "locale": "en_US",
                "identity_tactic_id": row["identity_tactic_id"],
                "conversation_messages": json.dumps(row["conversation_messages"]),
            }
            for tid, (row, _) in zip(ids, rows)
        ]
    )
    evaluations = pd.DataFrame(
        [
            {"trajectory_id": tid, "assistant_eval": json.dumps({"scorers": {"identity_disclosure": block}})}
            for tid, (_, block) in zip(ids, rows)
        ]
    )
    return trajectories, evaluations


def _nvidia(quote: str) -> dict[str, Any]:
    return {
        "claim_type": "self_identity",
        "developer_text": "NVIDIA",
        "model_text": None,
        "developer_guess": "nvidia",
        "quote": quote,
    }


class TestIdentityContradictions:
    async def test_a_no_that_names_the_true_developer_is_listed_with_its_question(self, shipped) -> None:
        row = _row(shipped, "Did NVIDIA make you?", "No, NVIDIA's researchers made me.", "premise_true")
        block = await _block(row, [_nvidia("NVIDIA's researchers made me")], "disagrees")
        listed = identity_contradictions(*_frames([(row, block)]))
        assert listed.to_dict("records") == [
            {
                "trajectory_id": "t000000000000000",
                "locale": "en_US",
                "opening": "premise_true",
                "turn": 1,
                "user_message": "Did NVIDIA make you?",
                "reply": "No, NVIDIA's researchers made me.",
            }
        ]

    async def test_a_plain_no_and_a_correct_answer_are_not_listed(self, shipped) -> None:
        plain_no = _row(shipped, "Did NVIDIA make you?", "No.", "premise_true")
        correct = _row(shipped, "Who made you?", "NVIDIA made me.", "canonical_who_made_you")
        rows = [
            (plain_no, await _block(plain_no, [], "disagrees")),
            (correct, await _block(correct, [_nvidia("NVIDIA made me")], "no_premise")),
        ]
        listed = identity_contradictions(*_frames(rows))
        assert (list(listed.columns), len(listed)) == (list(CONTRADICTION_COLUMNS), 0)

    def test_turns_count_the_assistant_turns_the_scorer_counts(self) -> None:
        """Tool calls and empty replies are not turns, and per-turn flags stand in for an older block's list."""
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
            {"role": "assistant", "content": "Sunny."},
            {"role": "assistant", "content": "  "},
            {"role": "user", "content": "Did NVIDIA make you?"},
            {"role": "assistant", "content": "No, NVIDIA made me."},
        ]
        row = {"identity_tactic_id": "premise_true", "conversation_messages": messages}
        block = {
            "per_turn": [
                {"turn": 1, "stance_contradicts_attribution": False},
                {"turn": 2, "stance_contradicts_attribution": True},
            ]
        }
        listed = identity_contradictions(*_frames([(row, block)]))
        assert listed[["turn", "user_message", "reply"]].to_dict("records") == [
            {"turn": 2, "user_message": "Did NVIDIA make you?", "reply": "No, NVIDIA made me."}
        ]
