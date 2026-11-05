# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Each probe that checks its shipped assets reports the ones it cannot find.

``BaseProbe.asset_problems`` is what ``usersim smoke`` and
``assert_probe_conforms`` run for every registered probe. Every built-in ships
assets and checks them. The shipped assets pass every check
(``test_conformance_suite.py``); here each check is pointed at an empty asset
folder and has to say so.
"""

from __future__ import annotations

import inspect

import pytest
from _probe_contract import built_in_labels

from usersim.engine.core._assets import reset_runtime_assets_dir, set_runtime_assets_dir
from usersim.engine.core.probes import BaseProbe, known_probes, resolve_probe


def _checks_its_assets(label: str) -> bool:
    check = inspect.getattr_static(resolve_probe(label), "asset_problems")
    return check is not inspect.getattr_static(BaseProbe, "asset_problems")


@pytest.fixture
def empty_assets(tmp_path):
    """An asset root, searched first, holding an empty folder for every registered probe."""
    for label in known_probes():
        (tmp_path / label).mkdir()
    token = set_runtime_assets_dir(tmp_path)
    try:
        yield tmp_path
    finally:
        reset_runtime_assets_dir(token)


@pytest.mark.parametrize("label", built_in_labels())
def test_every_built_in_probe_checks_its_assets(label: str) -> None:
    assert _checks_its_assets(label), f"{label} ships assets but defines no asset_problems to check them"


@pytest.mark.parametrize("label", [label for label in known_probes() if _checks_its_assets(label)])
def test_a_probe_with_an_empty_asset_folder_is_reported(label: str, empty_assets) -> None:
    problems = resolve_probe(label).asset_problems()

    assert problems
    assert all(isinstance(problem, str) and problem for problem in problems)
