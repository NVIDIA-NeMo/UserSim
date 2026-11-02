# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``usersim.extension`` and ``usersim.testing`` export what ``docs/extension_api.md`` lists.

A name added to either module's ``__all__`` without a row on the page fails
here, and so does a row naming something neither exports. ``usersim.extension``
imports each name on first use, so the tests also check that every name
resolves to the object its home module defines and that the imports a type
checker reads name the same objects.

The code an extension author copies (the probe templates, the extension
fixture, and the Python examples in the extension docs) imports ``usersim``
only through these two modules, apart from the imports listed in
``_KEPT_INTERNAL_IMPORTS``.
"""

from __future__ import annotations

import ast
import importlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

import usersim.extension
import usersim.testing

_REPO = Path(__file__).resolve().parents[1]
_REFERENCE = _REPO / "docs" / "extension_api.md"

_PUBLIC = {module.__name__: set(module.__all__) for module in (usersim.extension, usersim.testing)}

#: Files an extension author copies code from.
_EXAMPLE_FILES = (
    *sorted((_REPO / "templates" / "probe").glob("*.py")),
    _REPO / "templates" / "probe" / "test_probe.py.template",
    *sorted((_REPO / "tests" / "fixtures" / "extension_package" / "usersim_extension_fixture").glob("*.py")),
)

#: Pages whose Python examples an extension author copies.
_EXAMPLE_DOCS = (
    "CONTRIBUTING.md",
    "architecture/plugins.md",
    "docs/engine/AUTHORING_A_PROBE.md",
    "docs/extension_api.md",
    "docs/extensions.md",
)

#: Imports from other ``usersim`` modules that the examples keep, as (module, name).
_KEPT_INTERNAL_IMPORTS = {
    # A new in-tree probe's own modules, under the names the examples give them.
    ("usersim.engine.probes.YOUR_PROBE", "probe"),
    ("usersim.engine.probes.bookings.bank", "LOCALES"),
    ("usersim.engine.probes.bookings.bank", "Booking"),
    ("usersim.engine.probes.bookings.bank", "draw_booking"),
    ("usersim.engine.probes.bookings.bank", "read_booking_bank"),
    ("usersim.engine.probes.bookings.prompts", "BOOKINGS_USER_PROMPTS"),
    ("usersim.engine.probes.my_probe.bank", "load_my_bank_default"),
    ("usersim.engine.templates.probe.prompts", "DEMO_GATE_PROMPT_TEMPLATE"),
    ("usersim.engine.templates.probe.prompts", "DEMO_USER_AGENT_PROMPTS"),
    # The worked example's in-tree shim, and the test template's standard
    # tests, which load the probe registry before patching the loop's model calls.
    ("usersim.engine.core.probes", "dispatch_episode"),
    ("usersim.engine.generator", None),
    # The fixture's probe that layers a spec over a shipped probe's.
    ("usersim.engine.probes.identity_disclosure.generator", "IdentityDisclosureProbe"),
}


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


def _examples() -> list[tuple[str, str]]:
    """Each example as (where, source): a whole file, or one Python block of a page."""
    examples = [(path.relative_to(_REPO).as_posix(), path.read_text(encoding="utf-8")) for path in _EXAMPLE_FILES]
    for doc in _EXAMPLE_DOCS:
        blocks = re.findall(r"```python\n(.*?)```", (_REPO / doc).read_text(encoding="utf-8"), re.S)
        examples += [(f"{doc}, Python block {index + 1}", block) for index, block in enumerate(blocks)]
    return examples


def _usersim_imports(source: str) -> list[tuple[str, str | None]]:
    """Every ``usersim`` import in ``source`` as (module, name); name is None for ``import module``."""
    imports: list[tuple[str, str | None]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "usersim":
            imports += [(node.module, alias.name) for alias in node.names]
        elif isinstance(node, ast.Import):
            imports += [(alias.name, None) for alias in node.names if alias.name.split(".")[0] == "usersim"]
    return imports


def test_the_examples_import_usersim_through_the_public_modules():
    problems, kept = [], set()
    for where, source in _examples():
        try:
            imports = _usersim_imports(source)
        except SyntaxError:
            # A pseudo-code block cannot be checked, so it may not import usersim at all.
            if re.search(r"^\s*(from|import) usersim", source, re.M):
                problems.append(f"{where}: does not parse, and imports usersim")
            continue
        for module, name in imports:
            if name in _PUBLIC.get(module, ()):
                continue
            if (module, name) in _KEPT_INTERNAL_IMPORTS:
                kept.add((module, name))
            else:
                problems.append(f"{where}: {module} {name or ''}".rstrip())
    assert not problems, "import these from usersim.extension or usersim.testing:\n" + "\n".join(problems)
    assert kept == _KEPT_INTERNAL_IMPORTS, f"no example makes these imports: {sorted(_KEPT_INTERNAL_IMPORTS - kept)}"
