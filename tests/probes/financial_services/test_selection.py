# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The selection exports' finance group holds exactly financial_services' trajectory columns."""

from __future__ import annotations


def test_finance_group_matches_probe_contract():
    # Schema-parity guard across the package boundary: the selection
    # `finance` group must match the probe's canonical column contract, so
    # a rename/add on either side can't silently drop export columns.
    from usersim.engine.probes.financial_services.probe import (
        FINANCE_TRAJECTORY_COLUMNS,
    )
    from usersim.selection.schemas import LOGICAL_GROUPS

    assert set(LOGICAL_GROUPS["finance"]) == set(FINANCE_TRAJECTORY_COLUMNS)
