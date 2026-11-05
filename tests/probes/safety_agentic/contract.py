# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""safety_agentic's entries in the shared tests (see ``tests/_probe_contract.py``)."""

from __future__ import annotations

from _probe_contract import ParityCase, ProbeContract

CONTRACTS = {
    "safety_agentic": ProbeContract(
        trajectory_id="0dd753b33d0edb8c",
        succeeds_on=("action_request_id", "AR-001"),
        parity=(ParityCase(42, {"user_interaction_style": "direct"}, 3),),
    ),
}
