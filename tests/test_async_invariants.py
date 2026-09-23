# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""The engine runs on the loop it is handed, and never starts one of its own.

Data Designer owns the event loop: it schedules each cell as a task on a
process-wide loop, which is also what lets a synchronous call work inside a
notebook. Engine code that reaches for a loop itself either raises (calling
``asyncio.run`` from a running loop) or, worse, binds state to a second loop
and produces cross-loop errors far from the cause.

A CLI entry point is a different matter: it is the boundary where a
synchronous caller meets async code, and starting a loop there is the whole
job. So the ban is scoped to the engine.
"""

from __future__ import annotations

import ast
from pathlib import Path

#: Names that create or reach for a loop. ``asyncio.sleep`` and the rest of
#: the module are fine; these three are the ones that take ownership.
_LOOP_OWNING = ("run", "get_event_loop", "new_event_loop", "set_event_loop")


def _engine_files() -> list[Path]:
    root = Path(__file__).resolve().parent.parent / "src" / "usersim" / "engine"
    return sorted(root.rglob("*.py"))


def test_the_engine_never_takes_ownership_of_a_loop() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    offenders: list[str] = []
    for path in _engine_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute) or func.attr not in _LOOP_OWNING:
                continue
            if isinstance(func.value, ast.Name) and func.value.id == "asyncio":
                offenders.append(f"{path.relative_to(repo_root)}:{node.lineno} asyncio.{func.attr}")
    assert not offenders, (
        f"the engine runs on the loop Data Designer provides; starting or swapping one here "
        f"raises under a running loop and binds state across loops: {offenders}"
    )


def test_the_ban_would_catch_a_real_offender() -> None:
    """The scan above passes trivially if it matches nothing, so prove it bites."""
    tree = ast.parse("import asyncio\n\ndef f():\n    return asyncio.run(g())\n")
    hits = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _LOOP_OWNING
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "asyncio"
    ]
    assert len(hits) == 1


def test_asyncio_sleep_is_not_banned() -> None:
    """The retry backoff awaits it, so a blanket ban on ``asyncio`` would fail."""
    tree = ast.parse("import asyncio\n\nasync def f():\n    await asyncio.sleep(1)\n")
    hits = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _LOOP_OWNING
    ]
    assert hits == []
