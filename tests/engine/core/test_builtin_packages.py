# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The roster of built-in probe packages, and what a package on it registers.

``BUILTIN_PACKAGES`` names every package under ``usersim/engine/probes/``. A
package on it registers its probe, and its scorer, capability rows and asset
domain when it has them, by the names of its modules alone.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

import usersim.engine.probes
from usersim.engine.core import builtin_packages
from usersim.engine.core.builtin_packages import BUILTIN_PACKAGES, BUILTIN_ROOT, builtin_modules

_PROBES = Path(usersim.engine.probes.__file__).resolve().parent


def _packages_on_disk() -> set[str]:
    return {path.parent.name for path in _PROBES.glob("*/__init__.py")}


def _roster_problems(roster: tuple[str, ...]) -> list[str]:
    """What is wrong with ``roster`` as the list of the packages on disk."""
    problems = []
    if len(roster) != len(set(roster)):
        problems.append("a package is listed twice")
    on_disk = _packages_on_disk()
    if unlisted := sorted(on_disk - set(roster)):
        problems.append(f"not in BUILTIN_PACKAGES: {unlisted}")
    if missing := sorted(set(roster) - on_disk):
        problems.append(f"listed without a folder: {missing}")
    if no_probe := sorted(package for package in roster if not (_PROBES / package / "probe.py").is_file()):
        problems.append(f"listed without a probe.py: {no_probe}")
    return problems


def test_the_roster_lists_every_probe_package_on_disk() -> None:
    assert _packages_on_disk(), f"no probe package found under {_PROBES}"
    assert _roster_problems(BUILTIN_PACKAGES) == []


def test_a_package_left_off_the_roster_is_reported() -> None:
    assert _roster_problems(BUILTIN_PACKAGES[1:]) == [f"not in BUILTIN_PACKAGES: {[BUILTIN_PACKAGES[0]]}"]


def test_the_modules_of_a_kind_follow_the_roster_order() -> None:
    probes = builtin_modules("probe")
    assert probes == tuple(f"{BUILTIN_ROOT}.{package}.probe" for package in BUILTIN_PACKAGES)
    assert builtin_modules("no_such_module") == ()


_THROWAWAY = {
    "__init__.py": '"""A throwaway built-in package."""\n',
    "probe.py": """
        from usersim.extension import BaseProbe, register_probe


        @register_probe
        class ThrowawayProbe(BaseProbe):
            label = "throwaway_probe"

            def get_user_system_prompt(self) -> str:
                return "You are a user."

            def get_assistant_system_prompt(self) -> str:
                return ""
    """,
    "scorer.py": """
        from usersim.extension import register_scorer


        async def score(row, models):
            return {}


        register_scorer("throwaway_scorer", score)
    """,
    "capabilities.py": """
        from usersim.extension import CapabilityDefinition, EvidenceSource

        CAPABILITIES = (
            CapabilityDefinition(
                id="throwaway_capability",
                label="Throwaway",
                description="A capability a throwaway package declares.",
                sources=(EvidenceSource(probe="throwaway_probe", scorer="throwaway_scorer", axes=("throwaway.axis",)),),
                threshold=4.0,
                scale="score_1_5",
                aggregation_policy="mean",
            ),
        )
        AXIS_DESCRIPTIONS = {"throwaway.axis": "What the throwaway axis measures."}
    """,
}


@pytest.fixture
def throwaway_package(tmp_path, monkeypatch, restored_registries):
    """A package named ``throwaway_pkg`` under ``usersim.engine.probes``, on the roster."""
    from usersim.taxonomy import capabilities

    folder = tmp_path / "throwaway_pkg"
    folder.mkdir()
    for name, source in _THROWAWAY.items():
        (folder / name).write_text(textwrap.dedent(source), encoding="utf-8")
    monkeypatch.setattr(usersim.engine.probes, "__path__", [*usersim.engine.probes.__path__, str(tmp_path)])
    monkeypatch.setattr(builtin_packages, "BUILTIN_PACKAGES", (*BUILTIN_PACKAGES, "throwaway_pkg"))
    capabilities._builtin_capabilities.cache_clear()
    capabilities._axis_descriptions.cache_clear()
    yield
    for module in [name for name in sys.modules if name.startswith(f"{BUILTIN_ROOT}.throwaway_pkg")]:
        del sys.modules[module]
    capabilities._builtin_capabilities.cache_clear()
    capabilities._axis_descriptions.cache_clear()


def test_a_package_on_the_roster_registers_its_probe_scorer_and_capabilities(throwaway_package) -> None:
    from usersim.engine.core.probes import known_probes
    from usersim.engine.evaluator.scorers import list_scorers, load_default_scorers
    from usersim.taxonomy.capabilities import axis_description, capability_definitions

    load_default_scorers()

    assert "throwaway_probe" in known_probes()
    assert "throwaway_scorer" in list_scorers()
    assert "throwaway_capability" in {definition.id for definition in capability_definitions()}
    assert axis_description("throwaway.axis") == "What the throwaway axis measures."
