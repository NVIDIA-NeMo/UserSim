# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discovery of out-of-tree extensions via packaging entry points.

Note the deliberate word choice. DataDesigner has its own entry-point system
for column generators, which it calls *plugins* and gates with
``DISABLE_DATA_DESIGNER_PLUGINS`` -- this package registers two of them. What
is discovered here is unrelated, so it is called an *extension* throughout.
Anyone running both should be able to tell from a variable name which system
they are looking at.


Every registry in this project — probes, scorers, asset domains, execution
backends, capabilities, and CLI commands — is populated the same way: the
built-ins it defines itself, merged with whatever installed distributions
advertise under the matching entry-point group. Centralizing that merge here
keeps one conflict policy and one failure policy across all of them, instead
of six subtly different ones.

Two properties the registries depend on:

**A broken extension must not take down the core.** A third-party entry point
whose import raises is reported and skipped, so one bad package cannot make
``--help`` unusable. The report is a warning rather than a debug line because
the alternative — a probe that silently fails to appear — is far harder to
diagnose than a noisy startup.

**Discovery is suppressible.** ``USERSIM_DISABLE_EXTENSIONS`` turns it off
entirely. The test suite sets this, because otherwise an extension installed
in a developer's virtualenv would change what the suite sees and results
would stop being reproducible across machines.

**A name has one owner.** When two installed packages advertise the same
entry-point name in a group, neither is loaded: picking one would make what a
run uses depend on which package happens to sort first. Every warning names
the package an entry point comes from, and ``extension_records`` keeps what
happened to each one, which ``usersim smoke`` reports.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from collections import Counter
from importlib.metadata import EntryPoint, entry_points
from typing import Any

logger = logging.getLogger("usersim.engine")

#: Environment variable that suppresses all entry-point discovery.
DISABLE_ENV_VAR = "USERSIM_DISABLE_EXTENSIONS"

#: The published extension groups. A distribution advertises one of these in
#: its ``[project.entry-points]`` table to extend the corresponding registry
#: without modifying this project.
PROBES = "usersim.probes"
SCORERS = "usersim.scorers"
ASSET_DOMAINS = "usersim.asset_domains"
BACKENDS = "usersim.backends"
CAPABILITIES = "usersim.capabilities"
COMMANDS = "usersim.commands"
TOOLSET_SOURCES = "usersim.toolset_sources"

ENTRY_POINT_GROUPS: tuple[str, ...] = (
    PROBES,
    SCORERS,
    ASSET_DOMAINS,
    BACKENDS,
    CAPABILITIES,
    COMMANDS,
    TOOLSET_SOURCES,
)


@dataclasses.dataclass(frozen=True)
class ExtensionRecord:
    """One advertised entry point, and what discovery did with it."""

    group: str
    name: str
    package: str
    target: str
    #: Why the entry point contributes nothing; empty when it loaded and was kept.
    problem: str = ""

    @property
    def source(self) -> str:
        return f"{self.package} ({self.target})"


# Loaded objects keyed by group. Only populated when discovery is enabled, so
# toggling the environment variable within a process takes effect immediately
# rather than being masked by a cached empty result.
_CACHE: dict[str, tuple[tuple[str, Any], ...]] = {}

# What happened to every entry point seen, keyed by group; filled and cleared
# with ``_CACHE``.
_RECORDS: dict[str, list[ExtensionRecord]] = {}


def extensions_disabled() -> bool:
    """Whether entry-point discovery is currently suppressed."""
    return bool(os.environ.get(DISABLE_ENV_VAR))


def clear_extension_cache() -> None:
    """Drop memoized discovery results — for tests that toggle the env var."""
    _CACHE.clear()
    _RECORDS.clear()


def _select(group: str) -> tuple[EntryPoint, ...]:
    """Entry points in ``group``, ordered by name for reproducible results."""
    return tuple(sorted(entry_points(group=group), key=lambda ep: ep.name))


def _package(ep: EntryPoint) -> str:
    """The installed distribution that advertises ``ep``, with its version."""
    dist = getattr(ep, "dist", None)
    return f"{dist.name} {dist.version}" if dist is not None else "an unknown package"


def load_extensions(group: str) -> tuple[tuple[str, Any], ...]:
    """Load every advertised extension in ``group`` as ``(name, object)``.

    Returns an empty tuple when discovery is disabled. Entry points that fail
    to import are logged and omitted rather than propagated, so a single
    broken extension degrades to "that one is missing" instead of taking the
    whole CLI down with it. A name two packages advertise is loaded from
    neither.
    """
    if group not in ENTRY_POINT_GROUPS:
        raise ValueError(f"unknown entry-point group {group!r}; known: {', '.join(ENTRY_POINT_GROUPS)}")
    if extensions_disabled():
        return ()
    if group in _CACHE:
        return _CACHE[group]

    selected = _select(group)
    advertisers = Counter(ep.name for ep in selected)
    loaded: list[tuple[str, Any]] = []
    records: list[ExtensionRecord] = []
    for ep in selected:
        record = ExtensionRecord(group=group, name=ep.name, package=_package(ep), target=ep.value)
        if advertisers[ep.name] > 1:
            others = ", ".join(_package(o) for o in selected if o.name == ep.name and o is not ep)
            record = dataclasses.replace(record, problem=f"also advertised by {others}, so neither is loaded")
            logger.warning(
                "  |-- extensions: %s entry point %r from %s is %s.", group, ep.name, record.source, record.problem
            )
        else:
            try:
                loaded.append((ep.name, ep.load()))
            except Exception as exc:
                record = dataclasses.replace(record, problem=f"failed to load: {exc}")
                logger.warning(
                    "  |-- extensions: %s entry point %r from %s failed to load and was skipped: %s",
                    group,
                    ep.name,
                    record.source,
                    exc,
                )
        records.append(record)

    result = tuple(loaded)
    _CACHE[group] = result
    _RECORDS[group] = records
    return result


def extension_source(group: str, name: str) -> str:
    """Which package, and which object in it, advertised ``name`` in ``group``."""
    for record in _RECORDS.get(group, ()):
        if record.name == name:
            return record.source
    return f"entry point {name!r}"


def note_extension_problem(group: str, name: str, problem: str) -> None:
    """Record why a loaded entry point was not used after all, such as a name a built-in holds."""
    _RECORDS[group] = [
        dataclasses.replace(r, problem=problem) if r.name == name and not r.problem else r
        for r in _RECORDS.get(group, ())
    ]


def extension_records() -> tuple[ExtensionRecord, ...]:
    """What discovery did with every entry point it has seen, in group order.

    Covers the groups loaded so far in this process. A registry notes a name
    its built-ins hold when it merges extensions in, so load the registries
    first for the complete picture.
    """
    return tuple(record for group in ENTRY_POINT_GROUPS for record in _RECORDS.get(group, ()))


def merge_extensions(
    group: str,
    builtins: dict[str, Any],
    *,
    what: str,
) -> dict[str, Any]:
    """Merge discovered extensions into ``builtins`` without displacing them.

    A built-in always wins a name collision. An extension that silently
    shadowed a shipped backend or asset domain would change the meaning of a run
    while every config file and report still named the original, so the
    collision is reported and the extension dropped.

    ``builtins`` is not mutated; the merged mapping is returned.
    """
    merged = dict(builtins)
    for name, obj in load_extensions(group):
        if name in merged:
            logger.warning(
                "  |-- extensions: %s %r from %s is already built in; ignoring it.",
                what,
                name,
                extension_source(group, name),
            )
            note_extension_problem(group, name, f"{what} {name!r} is already built in")
            continue
        merged[name] = obj
    return merged
