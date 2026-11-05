# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""sov_ai_multilingual_parity's entries in the shared tests (see ``tests/_probe_contract.py``)."""

from __future__ import annotations

from _probe_contract import ParityCase, ProbeContract

CONTRACTS = {
    "sov_ai_multilingual_parity": ProbeContract(
        trajectory_id="6d1ca33ea06f0977",
        succeeds_on=("query_id", "Q-001"),
        parity=(ParityCase(1006),),
    ),
}
