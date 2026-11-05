# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared code names a built-in probe only where ``_ALLOWED`` says it may.

A probe label in ``src/usersim/`` outside ``engine/probes/`` ties shared code to
one probe, and a probe installed as an extension has no such code written for
it. What a probe needs from the core it declares on its class instead
(``architecture/probes.md``, "What a probe declares"), and its scorer,
capabilities and judge axes live in its own package, where its label is not
counted.

A label counts wherever it stands as a whole word: in a string, such as a help
text or a module path, in an identifier, or in a ``.md`` or ``.yaml`` file;
docstrings and comments do not count. Each allowance pins one (file, label)
pair. A pair not pinned fails, and so does a pinned pair the file no longer
writes, so a label taken out cannot come back unnoticed. The roster names every
built-in package by design, so it is not scanned.
"""

from __future__ import annotations

import ast
import re
import sys
from collections.abc import Iterator
from pathlib import Path

from usersim.engine.core.builtin_packages import builtin_modules
from usersim.engine.core.probes import known_probes, resolve_probe

_SRC = Path(__file__).resolve().parents[1] / "src" / "usersim"
_PROBE_PACKAGES = _SRC / "engine" / "probes"
_ROSTER = "engine/core/builtin_packages.py"
_TEXT_SUFFIXES = (".md", ".yaml", ".yml")

#: Each (file, label) pair shared code may write, by path under ``src/usersim/``, with why.
_ALLOWED: dict[tuple[str, str], str] = {
    ("cli/_pipeline.py", "general_educational"): "the default probe mix",
    ("cli/_pipeline.py", "general_open_ended"): "the default probe mix",
    ("cli/_pipeline.py", "tool_calling"): "the default probe mix",
    ("cli/_pipeline.py", "sov_ai_multilingual_parity"): "matched-pair sampling draws whole query pairs",
    ("cli/evaluate_assets.py", "financial_services"): "--domain help example",
    ("cli/gen_assets.py", "financial_services"): "--domain help example",
    ("cli/validate_assets.py", "financial_services"): "--domain help example",
    ("cli/simulate.py", "general_educational"): "the --probe-mix default",
    ("cli/simulate.py", "general_open_ended"): "the --probe-mix default",
    ("cli/simulate.py", "tool_calling"): "the --probe-mix default and the --toolsets help",
    ("cli/simulate.py", "financial_services"): "the help of its two run settings",
    ("cli/smoke.py", "general_open_ended"): "the synthetic rows the report smoke check renders",
    ("engine/core/preview.py", "safety_agentic"): "the attempted-actions table",
    ("engine/core/preview.py", "tool_calling"): "the tool-call panels",
    ("reporting/capability_report.py", "safety_chat_pressure"): "the pressure findings on its evidence cards",
    ("reporting/capability_report.py", "tool_calling"): "the api_response_model spend tooltip",
    ("reporting/dashboard.py", "tool_calling"): "the api_response_model spend tooltip",
    ("reporting/identity.py", "identity_disclosure"): "the identity report reads its scorer block",
}


def _label_pattern() -> re.Pattern[str]:
    """Each built-in label as a whole word: no letter or digit on either side."""
    labels = sorted(
        (label for label in known_probes() if resolve_probe(label).__module__ in builtin_modules("probe")),
        key=len,
        reverse=True,
    )
    return re.compile(r"(?<![A-Za-z0-9])(" + "|".join(map(re.escape, labels)) + r")(?![A-Za-z0-9])")


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


def _python_texts(path: Path) -> Iterator[tuple[int, str]]:
    """Each string, identifier and imported module path in ``path``, docstrings left out, with its line."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = _docstring_nodes(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            yield node.lineno, node.value
        elif isinstance(node, ast.Name):
            yield node.lineno, node.id
        elif isinstance(node, ast.Attribute):
            yield node.lineno, node.attr
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node.lineno, node.name
        elif isinstance(node, ast.arg):
            yield node.lineno, node.arg
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.lineno, node.module
        elif isinstance(node, ast.Import):
            yield from ((node.lineno, alias.name) for alias in node.names)


def _label_uses() -> list[tuple[str, int, str]]:
    """Every built-in label shared code writes outside ``engine/probes/``: path, line, label."""
    pattern = _label_pattern()
    uses = []
    for path in sorted(_SRC.rglob("*")):
        relative = path.relative_to(_SRC).as_posix()
        if not path.is_file() or path.is_relative_to(_PROBE_PACKAGES) or relative == _ROSTER:
            continue
        if path.suffix == ".py":
            texts = _python_texts(path)
        elif path.suffix in _TEXT_SUFFIXES:
            texts = enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        else:
            continue
        uses.extend((relative, line, match) for line, text in texts for match in pattern.findall(text))
    return sorted(uses)


def test_shared_code_names_a_probe_only_where_allowed():
    found = [f"{path}:{line}: {label!r}" for path, line, label in _label_uses() if (path, label) not in _ALLOWED]
    assert not found, (
        "Shared code names a probe. Declare what the core needs on the probe class instead "
        "(architecture/probes.md, 'What a probe declares'), and keep its scorer, capabilities "
        "and judge axes in its own package.\n" + "\n".join(found)
    )


def test_each_allowance_is_still_used():
    used = {(path, label) for path, _, label in _label_uses()}
    stale = sorted(pair for pair in _ALLOWED if pair not in used)
    assert not stale, f"These files no longer name these probes; drop the pairs from _ALLOWED: {stale}"


def test_a_label_is_found_as_a_whole_word_in_text_and_identifiers(tmp_path, monkeypatch):
    """Inside a help text or an identifier it counts; as part of a longer word it does not."""
    module = tmp_path / "shared.py"
    module.write_text(
        "HELP = \"run tool_calling=0.5 here\"\ndef run_tool_calling_rows():\n    return 'retool_callings'\n",
        encoding="utf-8",
    )
    (tmp_path / "notes.md").write_text("The tool_calling probe.\n", encoding="utf-8")
    this = sys.modules[__name__]
    monkeypatch.setattr(this, "_SRC", tmp_path)
    monkeypatch.setattr(this, "_PROBE_PACKAGES", tmp_path / "engine" / "probes")

    assert _label_uses() == [
        ("notes.md", 1, "tool_calling"),
        ("shared.py", 1, "tool_calling"),
        ("shared.py", 2, "tool_calling"),
    ]
