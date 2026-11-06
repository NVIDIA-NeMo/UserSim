# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The four health_disclosure probes' entries in the shared tests (see ``tests/_probe_contract.py``).

Each keeps ``BaseProbe.should_succeed``, so every completed conversation
succeeded. Each runs a parity episode of both variants. The guarded variant's
simulated user commits a move with a tool call. With two turns the only
follow-up is the last one, where this persona's risk pacing picks the move
without asking the user, so three is the smallest budget that covers a user
activation offering tools.
"""

from __future__ import annotations

from _probe_contract import ParityCase, ProbeContract


def _contract(trajectory_id: str, default_seed: int, guarded_seed: int) -> ProbeContract:
    return ProbeContract(
        trajectory_id=trajectory_id,
        succeeds_on=None,
        parity=(
            ParityCase(default_seed, {"probe_variant": "default"}),
            ParityCase(guarded_seed, {"probe_variant": "guarded"}, 3),
        ),
    )


CONTRACTS = {
    "health_decision_support_disclosure": _contract("fd69ae7b7cfc94a9", 1013, 1018),
    "health_general_disclosure": _contract("53a2c75f6cd26bce", 1010, 1015),
    "health_therapy_disclosure": _contract("e34d48f7eac319a6", 1011, 1016),
    "health_triage_disclosure": _contract("8c2c67f3cbd0f7b0", 1012, 1017),
}
