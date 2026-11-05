# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Each layer imports only the layers below it.

The data flows from configuration through generation and storage to reporting
and selection, with the CLI on top. So the engine imports none of ``cli``,
``asset_gen``, ``reporting`` or ``selection``, and outside ``engine/probes``
it imports no probe package: the registry loads the built-ins by module path.
``taxonomy`` sits below ``reporting``, so it imports neither that nor what is
above it. The extension API exports nothing from the CLI. Probe packages reach
the rest of ``usersim`` only through ``usersim.extension`` and
``usersim.testing``, which ``test_probe_private_imports.py`` checks.

``engine/external.py`` is the one exception: the run pipeline it drives still
lives in ``cli``, so it may import exactly the modules ``_EXTERNAL_CLI_IMPORTS``
names, and the list fails when it is longer than needed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import usersim.extension

_SRC = Path(__file__).resolve().parents[1] / "src" / "usersim"

#: (the modules a rule covers, the import prefixes they may not use).
_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("taxonomy", ("usersim.reporting", "usersim.selection", "usersim.cli")),
    (
        "engine",
        ("usersim.cli", "usersim.asset_gen", "usersim.reporting", "usersim.selection", "usersim.engine.probes"),
    ),
)

#: The CLI modules ``engine/external.py`` imports, for the run pipeline that lives there.
_EXTERNAL_CLI_IMPORTS = frozenset({"usersim.cli._inputs", "usersim.cli._pipeline"})


def _imports(path: Path) -> set[str]:
    """Every module ``path`` imports, at its top or inside a function."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
    return found


def _covered(package: str) -> list[Path]:
    """The modules of ``package`` a rule covers: probe packages and ``external.py`` are checked elsewhere."""
    root = _SRC / package
    return sorted(
        path
        for path in root.rglob("*.py")
        if not path.is_relative_to(_SRC / "engine" / "probes") and path != _SRC / "engine" / "external.py"
    )


def test_each_layer_imports_none_of_the_layers_above_it() -> None:
    offenders = []
    for package, forbidden in _RULES:
        modules = _covered(package)
        assert modules, f"no module found under {package}"
        for path in modules:
            for module in sorted(_imports(path)):
                if module.startswith(forbidden):
                    offenders.append(f"{path.relative_to(_SRC.parent)} imports {module}")
    assert not offenders, "\n".join(offenders)


def test_external_imports_only_the_cli_modules_its_exception_names() -> None:
    cli = {module for module in _imports(_SRC / "engine" / "external.py") if module.startswith("usersim.cli")}

    assert cli <= _EXTERNAL_CLI_IMPORTS, (
        f"engine/external.py imports more of the CLI: {sorted(cli - _EXTERNAL_CLI_IMPORTS)}"
    )
    assert cli == _EXTERNAL_CLI_IMPORTS, f"drop from _EXTERNAL_CLI_IMPORTS: {sorted(_EXTERNAL_CLI_IMPORTS - cli)}"


def test_the_extension_api_exports_nothing_from_the_cli() -> None:
    from_cli = sorted(module for module in usersim.extension._EXPORTS if module.startswith("usersim.cli"))

    assert not from_cli, f"usersim.extension exports names from {from_cli}"
