# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""financial_services's entries in the shared tests (see ``tests/_probe_contract.py``)."""

from __future__ import annotations

from _probe_contract import ParityCase, ProbeContract

CONTRACTS = {
    "financial_services": ProbeContract(
        trajectory_id="f29c0dac38c0ba2a",
        succeeds_on=("finance_task_id", "TPL-DISPUTE-001"),
        parity=(ParityCase(1009),),
    ),
}
