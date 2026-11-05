# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-in probes take no more from internal ``usersim`` modules than ``_ALLOWED`` says.

An extension imports only ``usersim.extension`` and ``usersim.testing``. Each name a
built-in package imports from any other ``usersim`` module, outside its own package,
is an import to change if the probe leaves this repository. It is counted once per
package, however many of the package's files import it. A package over its allowance
fails, and so does an allowance above the count, so a name taken out cannot come
back unnoticed. A new built-in package adds its count in the same change.
"""

from __future__ import annotations

import ast
from pathlib import Path

_PROBE_PACKAGES = Path(__file__).resolve().parents[1] / "src" / "usersim" / "engine" / "probes"
_PUBLIC = frozenset({"usersim.extension", "usersim.testing"})

#: How many distinct ``(module, name)`` imports from internal ``usersim`` modules
#: each built-in package holds.
_ALLOWED: dict[str, int] = {
    "financial_services": 8,
    "general_educational": 0,
    "general_open_ended": 0,
    "health_disclosure": 0,
    "identity_disclosure": 1,
    "safety_agentic": 3,
    "safety_chat_pressure": 0,
    "sov_ai_dynamic": 0,
    "sov_ai_facts": 0,
    "sov_ai_multilingual_parity": 0,
    "tool_calling": 0,
}


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


def test_every_built_in_package_has_an_allowance():
    missing = {package.name: len(_internal_imports(package)) for package in _packages() if package.name not in _ALLOWED}
    assert not missing, (
        f"add each to _ALLOWED with its count, after importing what you can from usersim.extension: {missing}"
    )


def test_no_built_in_package_imports_more_internal_names_than_allowed():
    over = {
        package.name: sorted(found)
        for package in _packages()
        if len(found := _internal_imports(package)) > _ALLOWED.get(package.name, len(found))
    }
    assert not over, (
        "A built-in package imports more names from internal usersim modules than its allowance. "
        f"Import them from usersim.extension, or keep them in the probe's own package: {over}"
    )


def test_every_allowance_is_still_needed():
    counts = {package.name: len(_internal_imports(package)) for package in _packages()}
    stale = {name: (allowed, counts.get(name)) for name, allowed in _ALLOWED.items() if counts.get(name, 0) < allowed}
    assert not stale, (
        f"lower each allowance to its package's count, or drop a removed package, (allowed, counted): {stale}"
    )
