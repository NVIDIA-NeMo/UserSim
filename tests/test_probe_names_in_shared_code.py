# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared code names a built-in probe no more often than ``_ALLOWED`` says.

A probe label written as a string in ``src/usersim/`` outside ``engine/probes/``
ties shared code to one probe, and a probe installed as an extension has no
such code written for it. What a probe needs from the core it declares on its
class instead (``architecture/probes.md``, "What a probe declares").

Labels are counted file by file, docstrings excluded. A file over its allowance
fails, and so does an allowance above the count, so a label taken out cannot
come back unnoticed. A new built-in probe's own scorer module and its rows in
``taxonomy/capabilities.py`` write its label, and raise their allowances in the
same change.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

from usersim.engine.core.probes import BUILTIN_PROBE_MODULES, known_probes, resolve_probe

_SRC = Path(__file__).resolve().parents[1] / "src" / "usersim"
_PROBE_PACKAGES = _SRC / "engine" / "probes"

#: How many built-in probe labels each file outside ``engine/probes/`` may
#: write as strings, by path under ``src/usersim/``.
_ALLOWED: dict[str, int] = {
    "asset_gen/registry.py": 3,
    "cli/_pipeline.py": 7,
    "cli/smoke.py": 2,
    "engine/core/action_taxonomy.py": 1,
    "engine/core/agentic_bank.py": 2,
    "engine/core/clinical_profile_bank.py": 4,
    "engine/core/finance_bank.py": 1,
    "engine/core/identity_spec.py": 1,
    "engine/core/preview.py": 3,
    "engine/evaluator/axes.py": 6,
    "engine/evaluator/scorers/financial_services.py": 4,
    "engine/evaluator/scorers/identity_disclosure.py": 1,
    "engine/evaluator/scorers/safety_agentic.py": 1,
    "reporting/capability_report.py": 1,
    "reporting/identity.py": 1,
    "taxonomy/capabilities.py": 41,
}


def _builtin_labels() -> set[str]:
    return {label for label in known_probes() if resolve_probe(label).__module__ in BUILTIN_PROBE_MODULES}


def _docstring_nodes(tree: ast.Module) -> set[int]:
    nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            first = node.body[0] if node.body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                nodes.add(id(first.value))
    return nodes


def _label_uses() -> list[tuple[str, int, str]]:
    """Every built-in label written as a string outside ``engine/probes/``: path, line, label."""
    labels = _builtin_labels()
    uses = []
    for path in sorted(_SRC.rglob("*.py")):
        if path.is_relative_to(_PROBE_PACKAGES):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = _docstring_nodes(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value in labels
                and id(node) not in docstrings
            ):
                uses.append((path.relative_to(_SRC).as_posix(), node.lineno, node.value))
    return sorted(uses)


def test_no_file_names_probes_more_often_than_allowed():
    uses = _label_uses()
    counts = Counter(path for path, _, _ in uses)
    over = {path for path, count in counts.items() if count > _ALLOWED.get(path, 0)}
    found = [f"{path}:{line}: {label!r}" for path, line, label in uses if path in over]
    assert not over, (
        "Shared code names a probe more often than allowed. Declare what the core needs on the "
        "probe class instead (architecture/probes.md, 'What a probe declares'). A new built-in "
        "probe's own scorer module and capability rows write its label: raise their allowances "
        "in _ALLOWED in the same change.\n" + "\n".join(found)
    )


def test_each_allowance_matches_the_count():
    counts = Counter(path for path, _, _ in _label_uses())
    stale = {path: counts[path] for path, allowed in _ALLOWED.items() if counts[path] < allowed}
    assert not stale, f"Fewer probe labels than allowed; lower these allowances in _ALLOWED to the counts: {stale}"
