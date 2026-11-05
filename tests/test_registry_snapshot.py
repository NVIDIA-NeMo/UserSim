# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``registry_snapshot`` puts the probe and scorer registries back as they were on entry.

A test that registers its own probe or scorer, or clears a registry, runs
inside it, so the next test in the same process sees the built-ins and
nothing else.
"""

from __future__ import annotations

import subprocess
import sys

from usersim.engine.core import probes
from usersim.engine.core.probes import BaseProbe, clear_registry, known_probes, register_probe
from usersim.engine.evaluator import scorers
from usersim.engine.evaluator.scorers import deterministic_scorers, list_scorers, register_scorer
from usersim.testing import registry_snapshot


async def _score(row, models):
    return {}


def test_what_the_block_registers_is_gone_after_it():
    labels, names, deterministic = known_probes(), list_scorers(), deterministic_scorers()

    with registry_snapshot():
        register_probe(family="my_snapshot_probe", prompt_version="v1.0", variants=("default",))(
            type("MySnapshotProbe", (BaseProbe,), {"label": "my_snapshot_probe"})
        )
        register_scorer("my_snapshot_scorer", _score, deterministic=True)
        assert "my_snapshot_probe" in known_probes()
        assert "my_snapshot_scorer" in deterministic_scorers()

    assert (known_probes(), list_scorers(), deterministic_scorers()) == (labels, names, deterministic)


def test_a_cleared_registry_is_refilled_in_place():
    probe_registry, scorer_registry = probes._PROBE_REGISTRY, scorers._REGISTRY
    labels, names = known_probes(), list_scorers()

    with registry_snapshot():
        clear_registry()
        scorers.clear_registry()
        assert not probe_registry
        assert not scorer_registry

    assert probes._PROBE_REGISTRY is probe_registry
    assert scorers._REGISTRY is scorer_registry
    assert (known_probes(), list_scorers()) == (labels, names)


def test_the_built_ins_are_kept_when_nothing_has_looked_a_probe_up():
    # In a fresh process the built-ins register on the first lookup, which
    # imports their modules once; a snapshot taken before that lookup would
    # drop them for the rest of the process.
    code = (
        "from usersim.engine.core import probes\n"
        "from usersim.testing import registry_snapshot\n"
        "assert not probes._PROBE_REGISTRY, 'a probe was registered before the snapshot'\n"
        "with registry_snapshot():\n"
        "    probes.known_probes()\n"
        "print(len(probes.known_probes()))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert int(result.stdout) == len(known_probes())
