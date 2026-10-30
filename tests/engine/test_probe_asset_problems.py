# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Each probe that checks its shipped assets reports the ones it cannot find.

``BaseProbe.asset_problems`` is what ``usersim smoke`` and
``assert_probe_conforms`` run for every registered probe. The shipped assets
pass every check (``test_conformance_suite.py``); here each check is pointed
at an empty asset folder and has to say so.
"""

from __future__ import annotations

import inspect

import pytest

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


def test_these_probes_check_their_assets() -> None:
    probes = (
        "financial_services",
        "general_educational",
        "general_open_ended",
        "identity_disclosure",
        "safety_agentic",
        "safety_chat_pressure",
        "sov_ai_dynamic",
        "sov_ai_facts",
        "sov_ai_multilingual_parity",
        "tool_calling",
    )

    assert [label for label in probes if not _checks_its_assets(label)] == []


@pytest.mark.parametrize("label", [label for label in known_probes() if _checks_its_assets(label)])
def test_a_probe_with_an_empty_asset_folder_is_reported(label: str, empty_assets) -> None:
    problems = resolve_probe(label).asset_problems()

    assert problems
    assert all(isinstance(problem, str) and problem for problem in problems)
