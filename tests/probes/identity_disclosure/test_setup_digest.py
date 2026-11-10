# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The identity spec reads every asset root below the first, so an edit to any of them is a new setup."""

from __future__ import annotations

import shutil

from usersim.cli import _inputs
from usersim.engine.core._assets import probe_assets_dir


def test_a_change_in_a_lower_priority_asset_root_is_a_new_setup(monkeypatch, tmp_path):
    """A spec that ``extends`` its own probe reads the copy below it, so every root counts."""
    top, below = tmp_path / "top", tmp_path / "below"
    shutil.copytree(probe_assets_dir("identity_disclosure").parent, top)
    shutil.copytree(probe_assets_dir("identity_disclosure").parent, below)
    monkeypatch.setenv("USERSIM_ASSET_PATH", str(below))
    before = _inputs._probe_assets_digest("identity_disclosure", top)
    shadowed = next(below.rglob("spec.yaml"))
    shadowed.write_text(shadowed.read_text() + "\n# edited\n")

    assert _inputs._probe_assets_digest("identity_disclosure", top) != before
