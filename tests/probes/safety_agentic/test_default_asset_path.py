# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agentic bank's and the action taxonomy's default paths hold files their loaders accept."""

from __future__ import annotations

import pytest

from usersim.engine.probes.safety_agentic.action_taxonomy import (
    load_action_taxonomy_default,
    reset_action_taxonomy_cache,
)
from usersim.engine.probes.safety_agentic.bank import (
    default_agentic_bank_path,
    load_agentic_bank,
    reset_agentic_bank_cache,
)


@pytest.fixture(autouse=True)
def _reset_caches():
    reset_agentic_bank_cache()
    reset_action_taxonomy_cache()
    yield


def test_agentic_bank_default_path_exists_and_loads() -> None:
    p = default_agentic_bank_path()
    assert p.exists(), f"agentic_bank default path missing: {p}"
    bank = load_agentic_bank(p)
    assert bank.bank_id, "agentic_bank loaded but bank_id is empty"


def test_action_taxonomy_default_path_loads() -> None:
    """``load_action_taxonomy_default`` resolves the path internally
    (no public ``default_*_path`` helper); it raises if the file is
    missing or malformed. We additionally assert non-empty entries
    so an accidentally empty file doesn't pass."""
    tax = load_action_taxonomy_default()
    assert tax.entries, "action_taxonomy loaded but has zero entries"
