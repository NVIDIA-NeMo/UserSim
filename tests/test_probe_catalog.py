# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The catalog in ``docs/probes.md`` has one row for each built-in probe, and no other."""

from __future__ import annotations

import re
from pathlib import Path

from usersim.engine.core.builtin_packages import builtin_modules
from usersim.engine.core.probes import known_probes, resolve_probe

_CATALOG = Path(__file__).resolve().parents[1] / "docs" / "probes.md"
_HEADER = "| Family | Probe | Catches |"


def _catalog_probes() -> list[str]:
    """The probe each row of the catalog table names, in its second column."""
    lines = _CATALOG.read_text(encoding="utf-8").splitlines()
    start = lines.index(_HEADER) + 2
    rows = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        cell = line.split("|")[2]
        match = re.fullmatch(r"\s*`([a-z0-9_]+)`\s*", cell)
        assert match, f"a catalog row names no probe in its Probe column: {line}"
        rows.append(match.group(1))
    return rows


def test_the_catalog_lists_each_built_in_probe_once():
    built_in = sorted(label for label in known_probes() if resolve_probe(label).__module__ in builtin_modules("probe"))
    assert sorted(_catalog_probes()) == built_in
