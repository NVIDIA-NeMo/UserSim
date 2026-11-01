# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``usersim.extension`` and ``usersim.testing`` export what ``docs/extension_api.md`` lists.

A name added to either module's ``__all__`` without a row on the page fails
here, and so does a row naming something neither exports. ``usersim.extension``
imports each name on first use, so the tests also check that every name
resolves to the object its home module defines and that the imports a type
checker reads name the same objects.
"""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

import usersim.extension
import usersim.testing

_REPO = Path(__file__).resolve().parents[1]
_REFERENCE = _REPO / "docs" / "extension_api.md"


def _documented(module: str) -> list[str]:
    """The names in the first column of the tables under ``## `module```."""
    names, inside = [], False
    for line in _REFERENCE.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            inside = line == f"## `{module}`"
        elif inside and line.startswith("| `"):
            names.append(line.split("`")[1])
    return names


@pytest.mark.parametrize("module", [usersim.extension, usersim.testing], ids=lambda m: m.__name__)
def test_the_reference_lists_every_exported_name_once(module):
    documented = _documented(module.__name__)
    assert len(documented) == len(set(documented)), "a name has two rows"
    assert len(module.__all__) == len(set(module.__all__)), "a name is in __all__ twice"
    unlisted = sorted(set(module.__all__) - set(documented))
    unexported = sorted(set(documented) - set(module.__all__))
    assert not unlisted, f"add a row to docs/extension_api.md for: {unlisted}"
    assert not unexported, f"docs/extension_api.md has a row for a name {module.__name__} does not export: {unexported}"


def test_each_name_is_the_object_its_home_module_defines():
    for module, names in usersim.extension._EXPORTS.items():
        home = importlib.import_module(module)
        for name in names:
            assert getattr(usersim.extension, name) is getattr(home, name), name
    mapped, listed = set(usersim.extension._MODULE_OF), set(usersim.extension.__all__)
    assert mapped == listed, f"in __all__ only: {sorted(listed - mapped)}; in _EXPORTS only: {sorted(mapped - listed)}"


def test_an_unknown_name_raises_attribute_error():
    with pytest.raises(AttributeError, match="no_such_name"):
        usersim.extension.no_such_name


def test_the_type_checking_imports_match_the_exports():
    tree = ast.parse(Path(usersim.extension.__file__).read_text(encoding="utf-8"))
    (block,) = [
        node
        for node in tree.body
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING"
    ]
    imported = {(node.module, alias.name) for node in block.body for alias in node.names}
    exported = {(module, name) for module, names in usersim.extension._EXPORTS.items() for name in names}
    assert imported == exported, (
        f"in the TYPE_CHECKING imports only: {sorted(imported - exported)}; "
        f"in _EXPORTS only: {sorted(exported - imported)}"
    )


def test_an_asset_domain_or_backend_import_leaves_the_conversation_loop_unloaded():
    code = (
        "import sys\n"
        "from usersim.extension import AssetDomain, CapabilityDefinition, ExecutionBackend, JobSpec\n"
        "print(sorted(m for m in ('usersim.engine.core.simulation', 'data_designer') if m in sys.modules))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", f"loaded by the import: {result.stdout.strip()}"
