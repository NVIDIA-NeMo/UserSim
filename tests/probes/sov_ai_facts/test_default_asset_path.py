# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The fact bank's default path, for every shipped locale, holds a bank its loader accepts."""

from __future__ import annotations

import pytest

from usersim.engine.core.locale import SHIPPED_LOCALES
from usersim.engine.probes.sov_ai_facts.bank import default_fact_bank_path, load_fact_bank, reset_fact_bank_cache


@pytest.fixture(autouse=True)
def _reset_cache():
    reset_fact_bank_cache()
    yield


@pytest.mark.parametrize("locale", SHIPPED_LOCALES)
def test_fact_bank_default_path_exists_and_loads(locale: str) -> None:
    p = default_fact_bank_path(locale)
    assert p.exists(), (
        f"fact_bank default path missing for {locale}: {p}\n"
        "Either ship the file or update default_fact_bank_path() to "
        "point at the right directory."
    )
    bank = load_fact_bank(p)
    assert bank.bank_version, f"fact_bank for {locale} loaded but bank_version is empty"
    assert bank.locale == locale, (
        f"fact_bank at {p} declares locale={bank.locale!r} but the "
        f"directory it lives in declares {locale!r} — drift between "
        "directory layout and file content."
    )
