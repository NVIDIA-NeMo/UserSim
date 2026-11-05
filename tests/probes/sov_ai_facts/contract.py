# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""sov_ai_facts's entries in the shared tests (see ``tests/_probe_contract.py``)."""

from __future__ import annotations

from _probe_contract import ParityCase, ProbeContract

CONTRACTS = {
    "sov_ai_facts": ProbeContract(
        trajectory_id="dfc5df24a426aad0",
        succeeds_on=("facts_probed", ["F-001"]),
        parity=(ParityCase(1004),),
    ),
}
