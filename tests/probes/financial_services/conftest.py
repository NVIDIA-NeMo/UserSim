# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for financial_services' tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from test_extension_seams import extension  # noqa: F401  (an extension-installing fixture, shared with that file)


@pytest.fixture
def finance_assets_without_fr_tasks(tmp_path: Path) -> Path:
    """An asset root whose financial_services bank runs in en_US but not in fr_FR.

    The fr_FR folder is a copy of the en_US bank without its tasks, the shape of a
    bank whose corpus has been generated but whose tasks are not written yet:
    present on disk, yet every row would abort.
    """
    import shutil

    from usersim.engine.core._assets import probe_assets_dir

    source = probe_assets_dir("financial_services") / "en_US"
    root = tmp_path / "assets"
    for locale in ("en_US", "fr_FR"):
        dest = root / "financial_services" / locale
        (dest / "bank").mkdir(parents=True)
        shutil.copy(source / "region_meta.yaml", dest / "region_meta.yaml")
        shutil.copytree(source / "bank" / "northwind_bank", dest / "bank" / "northwind_bank")
    (root / "financial_services" / "fr_FR" / "bank" / "northwind_bank" / "tasks.yaml").unlink()
    return root
