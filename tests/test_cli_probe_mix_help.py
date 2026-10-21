# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`usersim simulate --probe-mix` help names every built-in probe.

The help spells the probe list out by hand rather than reading the registry,
which would import every probe module on each `usersim` invocation. So a new
probe has to be added to the help as well, and this test says when it is not.
Probes from installed extensions are left out: the help lists built-ins only.
"""

from __future__ import annotations

import argparse
import re


def _probe_mix_help() -> str:
    from usersim.cli import simulate

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    simulate.register(sub)
    (action,) = [a for a in sub.choices["simulate"]._actions if a.dest == "probe_mix"]
    return action.help


def test_probe_mix_help_names_every_built_in_probe() -> None:
    from usersim.engine.core.probes import known_probes, resolve_probe

    built_in = [
        label for label in known_probes() if resolve_probe(label).__module__.startswith("usersim.engine.probes.")
    ]
    assert built_in, "no built-in probes registered"
    named = set(re.findall(r"[a-z_]+", _probe_mix_help()))
    missing = [label for label in built_in if label not in named]
    assert not missing, f"--probe-mix help does not name: {missing}"
