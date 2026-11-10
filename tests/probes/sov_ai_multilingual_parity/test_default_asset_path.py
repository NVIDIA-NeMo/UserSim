# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The query bank's default path holds a bank its loader accepts."""

from __future__ import annotations

import pytest

from usersim.engine.probes.sov_ai_multilingual_parity.bank import (
    default_query_bank_path,
    load_query_bank,
    reset_query_bank_cache,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    reset_query_bank_cache()
    yield


def test_query_bank_default_path_exists_and_loads() -> None:
    p = default_query_bank_path()
    assert p.exists(), f"query_bank default path missing: {p}"
    bank = load_query_bank(p)
    assert bank.bank_id, "query_bank loaded but bank_id is empty"
