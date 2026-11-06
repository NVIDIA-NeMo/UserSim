# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The tracked notebooks call code that exists, with arguments it takes.

No CI job runs the notebooks, so a rename in ``usersim`` would show only when
someone opens one. This test parses every code cell (IPython magics and shell
lines left out) and checks that each name a notebook imports from ``usersim``
resolves, and that each keyword argument it passes to an imported callable, a
settings class included, is one the callable takes.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_NOTEBOOKS = sorted(
    path
    for path in (*(_REPO / "notebooks").glob("*.ipynb"), *(_REPO / "src").rglob("*.ipynb"))
    if ".ipynb_checkpoints" not in path.parts
)
_MISSING = object()


def _cells(path: Path) -> list[ast.Module]:
    """Each code cell of ``path`` as parsed Python, with magics and shell lines left out."""
    cells = []
    for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
        if cell["cell_type"] != "code":
            continue
        lines = "".join(cell["source"]).splitlines()
        cells.append(ast.parse("\n".join(line for line in lines if not line.lstrip().startswith(("%", "!")))))
    return cells


def _resolve(module: str, name: str) -> object:
    """``name`` from ``module``, as ``from module import name`` binds it."""
    imported = importlib.import_module(module)
    if hasattr(imported, name):
        return getattr(imported, name)
    try:
        return importlib.import_module(f"{module}.{name}")
    except ModuleNotFoundError:
        return _MISSING


def _problems(cells: list[ast.Module]) -> tuple[list[str], int]:
    """What the cells import or call that the code does not provide, and how many calls were checked."""
    names: dict[str, object] = {}
    problems: list[str] = []
    for cell in cells:
        for node in ast.walk(cell):
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "usersim":
                for alias in node.names:
                    value = _resolve(node.module, alias.name)
                    if value is _MISSING:
                        problems.append(f"cannot import {alias.name} from {node.module}")
                    else:
                        names[alias.asname or alias.name] = value
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == "usersim":
                        module = importlib.import_module(alias.name)
                        names[alias.asname or "usersim"] = (
                            module if alias.asname else importlib.import_module("usersim")
                        )

    checked = 0
    for cell in cells:
        for node in ast.walk(cell):
            if not isinstance(node, ast.Call):
                continue
            target = _target(node.func, names)
            if target is None:
                continue
            if target is _MISSING:
                problems.append(f"{ast.unparse(node.func)} does not exist")
                continue
            try:
                parameters = inspect.signature(target).parameters
            except (TypeError, ValueError):
                continue
            checked += 1
            if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
                continue
            for keyword in node.keywords:
                if keyword.arg is not None and keyword.arg not in parameters:
                    problems.append(f"{ast.unparse(node.func)} takes no argument {keyword.arg!r}")
    return problems, checked


def _target(func: ast.expr, names: dict[str, object]) -> object | None:
    """The imported object a call's function names, if it names one; attributes are followed on modules only."""
    if isinstance(func, ast.Name):
        return names.get(func.id)
    if isinstance(func, ast.Attribute):
        base = _target(func.value, names)
        if inspect.ismodule(base):
            return getattr(base, func.attr, _MISSING)
    return None


def test_the_notebooks_are_found() -> None:
    assert _NOTEBOOKS, "no notebook found under notebooks/ or src/"


@pytest.mark.parametrize("path", _NOTEBOOKS, ids=lambda path: path.relative_to(_REPO).as_posix())
def test_a_notebook_calls_code_that_exists(path: Path) -> None:
    problems, checked = _problems(_cells(path))

    assert checked, f"{path.name} calls nothing it imports from usersim, so nothing here checks it"
    assert not problems, f"{path.relative_to(_REPO)}:\n" + "\n".join(problems)


def test_a_renamed_argument_or_a_missing_name_is_reported() -> None:
    cell = ast.parse(
        "from usersim.extension import ConversationSimulatorConfig, no_such_name\n"
        "ConversationSimulatorConfig(name='conversation', max_turns=2, no_such_setting=1)\n"
    )

    problems, checked = _problems([cell])

    assert checked == 1
    assert problems == [
        "cannot import no_such_name from usersim.extension",
        "ConversationSimulatorConfig takes no argument 'no_such_setting'",
    ]
