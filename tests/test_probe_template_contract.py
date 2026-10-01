# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Probe templates must stay on the current probe contract.

A template is copied verbatim into a new probe, so a template that is a
contract behind ships broken probes. These checks are AST-only; the templates
are illustrative files and are not importable on their own.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_TEMPLATES = sorted((Path(__file__).resolve().parent.parent / "templates" / "probe").glob("generator_*.py"))


def test_probe_templates_are_present() -> None:
    assert _TEMPLATES, "no probe templates found"


@pytest.mark.parametrize("template", _TEMPLATES, ids=lambda path: path.name)
def test_custom_run_dispatch_accepts_the_hosted_parameters(template: Path) -> None:
    """A template that overrides run_dispatch must be hostable.

    ``core/episode_runtime.py`` resumes an episode with
    ``run_dispatch(..., state=..., seed_state=False)``. A template missing
    either parameter produces probes that raise ``TypeError`` when hosted.
    """
    tree = ast.parse(template.read_text(encoding="utf-8"), filename=str(template))

    dispatches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run_dispatch"
    ]
    if not dispatches:
        pytest.skip(f"{template.name} does not override run_dispatch")

    for node in dispatches:
        names = {arg.arg for arg in node.args.kwonlyargs} | {arg.arg for arg in node.args.args}
        missing = {"state", "seed_state"} - names
        assert not missing, (
            f"{template.name}:{node.lineno} run_dispatch is missing {sorted(missing)}; "
            f"probes copied from it cannot be driven by the external episode runtime"
        )
