# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The probing taxonomy's default path, for every shipped locale, holds a taxonomy its loader accepts."""

from __future__ import annotations

import pytest

from usersim.engine.core.locale import SHIPPED_LOCALES
from usersim.engine.core.probing_taxonomy import load_probing_taxonomy, reset_probing_taxonomy_cache
from usersim.engine.probes.sov_ai_dynamic.taxonomy import default_probing_taxonomy_path


@pytest.fixture(autouse=True)
def _reset_cache():
    reset_probing_taxonomy_cache()
    yield


@pytest.mark.parametrize("locale", SHIPPED_LOCALES)
def test_probing_taxonomy_default_path_exists_and_loads(locale: str) -> None:
    p = default_probing_taxonomy_path(locale)
    assert p.exists(), (
        f"probing_taxonomy default path missing for {locale}: {p}\n"
        "Either ship the file or update default_probing_taxonomy_path() to "
        "point at the right directory."
    )
    tax = load_probing_taxonomy(p)
    assert tax.taxonomy_version, f"probing_taxonomy for {locale} loaded but taxonomy_version is empty"
    assert tax.locale == locale, (
        f"probing_taxonomy at {p} declares locale={tax.locale!r} but the "
        f"directory it lives in declares {locale!r} — drift between "
        "directory layout and file content."
    )
