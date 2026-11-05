# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""sov_ai_dynamic's entries in the shared tests (see ``tests/_probe_contract.py``)."""

from __future__ import annotations

from _probe_contract import ParityCase, ProbeContract

CONTRACTS = {
    "sov_ai_dynamic": ProbeContract(
        trajectory_id="f2fb3d9172800ca5",
        succeeds_on=("probing_categories_explored", ["category-1"]),
        parity=(ParityCase(1005),),
    ),
}
