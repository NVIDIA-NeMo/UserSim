# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""safety_chat_pressure's entries in the shared tests (see ``tests/_probe_contract.py``)."""

from __future__ import annotations

from _probe_contract import ParityCase, ProbeContract

CONTRACTS = {
    "safety_chat_pressure": ProbeContract(
        trajectory_id="34650e5f725ae448",
        succeeds_on=("target_request_id", "T-001"),
        parity=(ParityCase(1007),),
    ),
}
