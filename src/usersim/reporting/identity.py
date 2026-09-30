# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Identity replies that contradict themselves, listed turn by turn.

A contradiction answers no to a question naming the model's own developer,
then names that developer as its maker: "No, X made me" in reply to
"Did X make you?" from a model X made. The capability dashboard counts it
against "Identity: confirms its developer"; this lists each one so it can be
read.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from usersim.reporting.capability_report import _scorer_block
from usersim.reporting.diagnostics import _decode_messages

CONTRADICTION_COLUMNS = ("trajectory_id", "locale", "opening", "turn", "user_message", "reply")


def identity_contradictions(
    trajectories: pd.DataFrame,
    evaluations: pd.DataFrame,
    *,
    eval_column: str = "assistant_eval",
) -> pd.DataFrame:
    """One row per assistant turn the identity scorer labelled a contradiction.

    ``turn`` counts the assistant turns as the scorer does, ``user_message``
    is the message that turn answered and ``opening`` the row's identity
    tactic. Evaluations written before the scorer listed
    ``contradiction_turns`` are read from its per-turn flags.
    """
    turns_by_row: dict[str, list[int]] = {}
    for _, row in evaluations.iterrows():
        block = _scorer_block(row, eval_column, "identity_disclosure")
        turns = block.get("contradiction_turns")
        if turns is None:
            turns = [t.get("turn") for t in block.get("per_turn") or [] if t.get("stance_contradicts_attribution")]
        if turns:
            turns_by_row[str(row["trajectory_id"])] = [int(turn) for turn in turns]

    records: list[dict[str, Any]] = []
    rows = trajectories[trajectories["trajectory_id"].astype(str).isin(turns_by_row)]
    for _, row in rows.iterrows():
        trajectory_id = str(row["trajectory_id"])
        exchanges = _exchanges(row.get("conversation_messages"))
        for turn in turns_by_row[trajectory_id]:
            if 1 <= turn <= len(exchanges):
                user_message, reply = exchanges[turn - 1]
                records.append(
                    {
                        "trajectory_id": trajectory_id,
                        "locale": row.get("locale"),
                        "opening": row.get("identity_tactic_id"),
                        "turn": turn,
                        "user_message": user_message,
                        "reply": reply,
                    }
                )
    frame = pd.DataFrame(records, columns=list(CONTRADICTION_COLUMNS))
    return frame.sort_values(["locale", "trajectory_id", "turn"], ignore_index=True)


def _exchanges(messages: Any) -> list[tuple[str, str]]:
    """Each assistant turn with text, numbered as the scorer numbers them, with the user message before it."""
    exchanges: list[tuple[str, str]] = []
    last_user = ""
    for message in _decode_messages(messages):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        text = content if isinstance(content, str) else ""
        if message.get("role") == "user":
            last_user = text
        elif message.get("role") == "assistant" and text.strip() and not message.get("tool_calls"):
            exchanges.append((last_user, text))
    return exchanges
