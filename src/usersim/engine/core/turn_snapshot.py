# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Undoing an assistant turn the judge rejected, so the loop can regenerate it."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from usersim.engine.core.outcomes import OutcomeCheckpoint


@dataclass(frozen=True)
class TurnSnapshot:
    """What an assistant turn can change: the transcript, the metadata, the outcome record and the probe's own state."""

    n_messages: int
    metadata: dict[str, Any]
    outcome: OutcomeCheckpoint
    probe_state: Any


def take_turn_snapshot(state: Any, probe: Any) -> TurnSnapshot:
    snapshot_probe = getattr(probe, "snapshot_turn_state", None)
    return TurnSnapshot(
        n_messages=len(state.messages),
        metadata=copy.deepcopy(state.metadata),
        outcome=state.outcome.checkpoint(),
        probe_state=snapshot_probe() if snapshot_probe else None,
    )


def restore_turn_snapshot(state: Any, probe: Any, snapshot: TurnSnapshot) -> None:
    """Put ``state`` and ``probe`` back as ``snapshot`` found them."""
    del state.messages[snapshot.n_messages :]
    state.metadata.clear()
    state.metadata.update(copy.deepcopy(snapshot.metadata))
    state.outcome.rollback(snapshot.outcome)
    restore_probe = getattr(probe, "restore_turn_state", None)
    if restore_probe:
        restore_probe(snapshot.probe_state)
