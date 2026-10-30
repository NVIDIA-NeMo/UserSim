# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The reference tables in ``architecture/probes.md`` list what the probe classes define.

A hook, declaration or helper added to ``BaseProbe``, ``BankBackedProbe`` or a
mixin without a row fails here, and so does a row left behind when one is
removed.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

from usersim.engine.core import probes

_PROBES_DOC = Path(__file__).resolve().parents[1] / "architecture" / "probes.md"

#: The classes a probe builds on, in the order a name is looked up.
_PROBE_CLASSES = (
    probes.BaseProbe,
    probes.BankBackedProbe,
    probes.BankVerbatimMixin,
    probes.BankReframingMixin,
    probes.CustomTurn1InstructionMixin,
    probes.ToolCallingMixin,
    probes.ToolExecutionMixin,
    probes.AgenticMixin,
)

_NO_VALUE = object()


def _public_members() -> dict[str, object]:
    """Each public name the probe classes define, with the value of its first definition.

    A declaration with a type and no value, such as ``bank_loader``, has
    ``_NO_VALUE``.
    """
    members: dict[str, object] = {}
    for cls in _PROBE_CLASSES:
        for name, value in vars(cls).items():
            if not name.startswith("_"):
                members.setdefault(name, value)
    for cls in _PROBE_CLASSES:
        for name in inspect.get_annotations(cls):
            if not name.startswith("_"):
                members.setdefault(name, _NO_VALUE)
    return members


def _table_rows(heading: str) -> list[list[str]]:
    """The cells of each row of the table whose first column is headed ``heading``."""
    lines = _PROBES_DOC.read_text(encoding="utf-8").splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(f"| {heading} |")), None)
    assert start is not None, f"architecture/probes.md has no table headed {heading!r}"
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split(" | ")])
    return rows


def _names(heading: str) -> set[str]:
    """The names in the first column of the table headed ``heading``."""
    return {name for row in _table_rows(heading) for name in re.findall(r"`(\w+)`", row[0])}


def test_every_public_name_on_the_probe_classes_has_a_row() -> None:
    documented = _names("Hook") | _names("Declaration") | _names("Helper")
    members = set(_public_members())
    assert sorted(members - documented) == [], "add a row for each to architecture/probes.md"
    assert sorted(documented - members) == [], "these rows name nothing the probe classes define"


def test_no_name_has_rows_in_two_tables() -> None:
    hooks, declarations, helpers = _names("Hook"), _names("Declaration"), _names("Helper")
    assert hooks & declarations == set()
    assert hooks & helpers == set()
    assert declarations & helpers == set()


def test_hooks_are_methods_and_declarations_are_not() -> None:
    members = _public_members()
    for name in _names("Hook"):
        assert inspect.isfunction(members[name]) or isinstance(members[name], staticmethod), name
    for name in _names("Declaration"):
        assert not inspect.isfunction(members[name]), f"{name} is a method: give it a row in the hook table"


def test_each_documented_default_value_matches_the_class() -> None:
    members = _public_members()
    checked = 0
    for cells in _table_rows("Declaration"):
        names = re.findall(r"`(\w+)`", cells[0])
        literal = re.match(r"`([^`]+)`", cells[1])
        if len(names) != 1 or literal is None or isinstance(members[names[0]], classmethod):
            continue
        assert members[names[0]] == ast.literal_eval(literal.group(1)), names[0]
        checked += 1
    assert checked >= 10, "the declaration table's Default column lost its values"
