# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`usersim simulate --help` names every registered probe under `--probe-mix`.

The list is read from the registry when the help is asked for, so it names a
new probe without an edit. Building the parser, which every `usersim` command
does, imports no probe module.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

import pytest


def _usersim_parser() -> argparse.ArgumentParser:
    from usersim.cli import simulate

    parser = argparse.ArgumentParser(prog="usersim")
    simulate.register(parser.add_subparsers(dest="cmd"))
    return parser


def test_the_help_names_every_registered_probe(monkeypatch, capsys) -> None:
    from usersim.engine.core.probes import known_probes

    monkeypatch.setenv("COLUMNS", "10000")  # one line per option, so no name is broken
    with pytest.raises(SystemExit) as exit_:
        _usersim_parser().parse_args(["simulate", "--help"])
    assert exit_.value.code == 0
    listed = capsys.readouterr().out.split("Registered probes:", 1)[1].splitlines()[0]
    assert known_probes(), "no probes registered"
    missing = [label for label in known_probes() if label not in set(re.findall(r"[a-z_]+", listed))]
    assert not missing, f"--probe-mix help does not name: {missing}"


def test_only_asking_for_the_help_loads_the_probes() -> None:
    code = (
        "import argparse, contextlib, io, sys\n"
        "from usersim.cli import simulate\n"
        "parser = argparse.ArgumentParser(prog='usersim')\n"
        "simulate.register(parser.add_subparsers())\n"
        "loaded = lambda: any(m.startswith('usersim.engine.probes.') for m in sys.modules)\n"
        "before = loaded()\n"
        "with contextlib.redirect_stdout(io.StringIO()), contextlib.suppress(SystemExit):\n"
        "    parser.parse_args(['simulate', '--help'])\n"
        "print(before, loaded())\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["False", "True"]
