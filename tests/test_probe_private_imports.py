# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-in probes import ``usersim`` only through its extension API.

An extension imports only ``usersim.extension`` and ``usersim.testing``, and so
does every package under ``usersim/engine/probes/``, apart from its own
modules, so moving a built-in out of this repository changes none of its
imports. A name a probe needs from elsewhere in ``usersim`` is exported from
``usersim.extension``, or kept in the package that uses it.
"""

from __future__ import annotations

import ast
from pathlib import Path

from usersim.engine.core.probes import BUILTIN_PROBE_MODULES

_PROBE_PACKAGES = Path(__file__).resolve().parents[1] / "src" / "usersim" / "engine" / "probes"
_PUBLIC = frozenset({"usersim.extension", "usersim.testing"})


def _packages() -> list[Path]:
    return sorted(path for path in _PROBE_PACKAGES.iterdir() if (path / "__init__.py").is_file())


def _internal_imports(package: Path) -> set[str]:
    """``module:name`` for each name ``package`` imports from an internal ``usersim`` module."""
    own = f"usersim.engine.probes.{package.name}"
    found = set()
    for path in package.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "usersim":
                pairs = [(node.module, alias.name) for alias in node.names]
            elif isinstance(node, ast.Import):
                pairs = [(alias.name, "") for alias in node.names if alias.name.split(".")[0] == "usersim"]
            else:
                continue
            for module, name in pairs:
                if module in _PUBLIC or module == own or module.startswith(f"{own}."):
                    continue
                found.add(f"{module}:{name}")
    return found


def test_the_scan_covers_every_built_in_probe():
    scanned = {package.name for package in _packages()}
    unscanned = sorted(module for module in BUILTIN_PROBE_MODULES if module.split(".")[3] not in scanned)
    assert not unscanned, f"built-in probe modules outside the scanned packages: {unscanned}"


def test_built_in_probes_import_usersim_only_through_the_extension_api():
    found = {package.name: sorted(names) for package in _packages() if (names := _internal_imports(package))}
    assert not found, (
        "A built-in probe package imports from an internal usersim module. Import the name from "
        f"usersim.extension, export it there, or keep it in the probe's own package: {found}"
    )
