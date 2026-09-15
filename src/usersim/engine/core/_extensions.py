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
"""

from __future__ import annotations

import logging
import os
from importlib.metadata import EntryPoint, entry_points
from typing import Any, Dict, Tuple

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

ENTRY_POINT_GROUPS: Tuple[str, ...] = (
    PROBES,
    SCORERS,
    ASSET_DOMAINS,
    BACKENDS,
    CAPABILITIES,
    COMMANDS,
    TOOLSET_SOURCES,
)

# Loaded objects keyed by group. Only populated when discovery is enabled, so
# toggling the environment variable within a process takes effect immediately
# rather than being masked by a cached empty result.
_CACHE: Dict[str, Tuple[Tuple[str, Any], ...]] = {}


def extensions_disabled() -> bool:
    """Whether entry-point discovery is currently suppressed."""
    return bool(os.environ.get(DISABLE_ENV_VAR))


def clear_extension_cache() -> None:
    """Drop memoized discovery results — for tests that toggle the env var."""
    _CACHE.clear()


def _select(group: str) -> Tuple[EntryPoint, ...]:
    """Entry points in ``group``, ordered by name for reproducible results."""
    return tuple(sorted(entry_points(group=group), key=lambda ep: ep.name))


def load_extensions(group: str) -> Tuple[Tuple[str, Any], ...]:
    """Load every advertised extension in ``group`` as ``(name, object)``.

    Returns an empty tuple when discovery is disabled. Entry points that fail
    to import are logged and omitted rather than propagated, so a single
    broken extension degrades to "that one is missing" instead of taking the
    whole CLI down with it.
    """
    if group not in ENTRY_POINT_GROUPS:
        raise ValueError(
            f"unknown entry-point group {group!r}; "
            f"known: {', '.join(ENTRY_POINT_GROUPS)}"
        )
    if extensions_disabled():
        return ()
    if group in _CACHE:
        return _CACHE[group]

    loaded: list[Tuple[str, Any]] = []
    for ep in _select(group):
        try:
            loaded.append((ep.name, ep.load()))
        except Exception as exc:
            logger.warning(
                "  |-- extensions: %s entry point %r (from %r) failed to load "
                "and was skipped: %s",
                group, ep.name, ep.value, exc,
            )

    result = tuple(loaded)
    _CACHE[group] = result
    return result


def merge_extensions(
    group: str,
    builtins: Dict[str, Any],
    *,
    what: str,
) -> Dict[str, Any]:
    """Merge discovered extensions into ``builtins`` without displacing them.

    A built-in always wins a name collision. An extension that silently
    shadowed a shipped probe or scorer would change the meaning of a run
    while every config file and report still named the original, so the
    collision is reported and the extension dropped.

    ``builtins`` is not mutated; the merged mapping is returned.
    """
    merged = dict(builtins)
    for name, obj in load_extensions(group):
        if name in merged:
            logger.warning(
                "  |-- extensions: %s %r is already provided by this package; "
                "ignoring the one advertised under %r.",
                what, name, group,
            )
            continue
        merged[name] = obj
    return merged
