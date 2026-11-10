# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""financial_services refuses settings outside their range, so a run stops before any model call."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from usersim.engine.core.probes import resolve_probe


@pytest.mark.parametrize(
    ("setting", "message"),
    [
        ({"tier_mix": 1.5}, "Input should be less than or equal to 1"),
        ({"tier": "both"}, "Input should be 'verifiable' or 'dynamic'"),
    ],
    ids=["tier_mix_out_of_range", "unknown_tier"],
)
def test_out_of_range_settings_are_refused(setting, message) -> None:
    with pytest.raises(ValidationError, match=message):
        resolve_probe("financial_services").Options(**setting)
