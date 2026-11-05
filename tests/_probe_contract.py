# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Each built-in probe's entries in the shared tests, kept in the probe's own test folder.

``tests/probes/<package>/contract.py`` defines ``CONTRACTS``, one
:class:`ProbeContract` for each label the package registers. The shared tests
read a probe's entry with :func:`probe_contract`, which fails, naming the file
to add it to, when there is none; so a new probe states its contract in its own
folder, and no shared test file lists the probes.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pytest

from usersim.engine.core.probes import BUILTIN_PROBE_MODULES, known_probes, resolve_probe

_TESTS = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ParityCase:
    """One episode that must give the same row run directly and through a host."""

    seed: int
    probe_data: Mapping[str, Any] | None = None
    max_turns: int = 1


@dataclass(frozen=True)
class ProbeContract:
    """What the shared tests hold one built-in probe to."""

    #: The ``trajectory_id`` of the pinned episode in ``test_episode_input.py``.
    #: It pins the probe's identity scheme: a change to it changes the id of
    #: every trajectory the probe has stored.
    trajectory_id: str
    #: The metadata key and value that make ``should_succeed`` true, or None for
    #: a probe that keeps ``BaseProbe.should_succeed``, which is always true.
    succeeds_on: tuple[str, Any] | None
    #: The episodes ``test_episode_runtime_parity.py`` runs directly and through a host.
    parity: tuple[ParityCase, ...]


def built_in_labels() -> tuple[str, ...]:
    """The labels registered by the modules in ``BUILTIN_PROBE_MODULES``."""
    return tuple(label for label in known_probes() if resolve_probe(label).__module__ in BUILTIN_PROBE_MODULES)


def contract_path(label: str) -> Path:
    """The ``contract.py`` that holds ``label``'s entry."""
    return _TESTS / "probes" / _package(label) / "contract.py"


def find_contract(label: str) -> ProbeContract | None:
    """``label``'s contract, or None when its package's ``contract.py`` has none."""
    if not contract_path(label).is_file():
        return None
    return importlib.import_module(f"probes.{_package(label)}.contract").CONTRACTS.get(label)


def probe_contract(label: str) -> ProbeContract:
    """``label``'s contract; fails the test, naming the file to add it to, when there is none."""
    contract = find_contract(label)
    if contract is None:
        where = contract_path(label).relative_to(_TESTS.parent)
        pytest.fail(f"{label} has no contract with the shared tests: add CONTRACTS[{label!r}] to {where}")
    return contract


def _package(label: str) -> str:
    """The package under ``usersim.engine.probes`` that defines ``label``."""
    return resolve_probe(label).__module__.split(".")[3]
