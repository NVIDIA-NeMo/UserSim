# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The pressure bank's default path holds a bank its loader accepts."""

from __future__ import annotations

import pytest

from usersim.engine.probes.safety_chat_pressure.bank import (
    default_pressure_bank_path,
    load_pressure_bank,
    reset_pressure_bank_cache,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    reset_pressure_bank_cache()
    yield


def test_pressure_bank_default_path_exists_and_loads() -> None:
    p = default_pressure_bank_path()
    assert p.exists(), f"pressure_bank default path missing: {p}"
    bank = load_pressure_bank(p)
    assert bank.bank_id, "pressure_bank loaded but bank_id is empty"
