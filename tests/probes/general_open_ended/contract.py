# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""general_open_ended's entries in the shared tests (see ``tests/_probe_contract.py``)."""

from __future__ import annotations

from _probe_contract import ParityCase, ProbeContract

CONTRACTS = {
    "general_open_ended": ProbeContract(
        trajectory_id="0fb84850f75a3f00",
        # BaseProbe.should_succeed: every completed conversation succeeded.
        succeeds_on=None,
        parity=(ParityCase(1001),),
    ),
}
