# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""identity_disclosure's entries in the shared tests (see ``tests/_probe_contract.py``)."""

from __future__ import annotations

from _probe_contract import ParityCase, ProbeContract

CONTRACTS = {
    "identity_disclosure": ProbeContract(
        trajectory_id="ab9a7b113d58f3d3",
        succeeds_on=("identity_tactic_id", "canonical_who_are_you"),
        parity=(ParityCase(1014, {"identity_tactic_id": "several_at_once"}),),
    ),
}
