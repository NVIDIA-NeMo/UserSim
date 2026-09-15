# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Locate the shipped asset banks.

The banks are **package data** — they live at
``usersim/engine/assets/`` and are resolved through
:mod:`importlib.resources`, so they are found identically whether the
package is installed editable or from a wheel.

Resolution is an ordered **search path**, highest priority first:

1. the per-context override set by :func:`set_runtime_assets_dir`;
2. each entry of ``$USERSIM_ASSET_PATH`` (``os.pathsep``-separated);
3. ``$USERSIM_ASSETS_DIR``;
4. the packaged ``usersim/engine/assets/`` directory.

:func:`probe_assets_dir` walks that path and takes the first root that
actually holds the probe, so a package contributing one probe's banks
does not have to restate the shipped ones to be usable.

An explicit ``assets_root`` -- what ``--assets-dir`` becomes -- is not
part of the search. It pins resolution to exactly that directory. A user
who named a location wants to be told when the assets are not there,
rather than to be quietly served the packaged copies instead.

There is deliberately no CWD fallback: resolving against whatever
directory the process happened to start in makes a broken install look
healthy from the source tree.
"""

from __future__ import annotations

import os
from contextvars import ContextVar, Token
from importlib import resources
from pathlib import Path
from typing import Optional


#: Probe directories the packaged ``assets/`` tree must contain. Kept as an
#: explicit list so ``test_assets_discovery.py`` can assert it stays in
#: lockstep with ``known_probes()``: a probe added without its asset folder,
#: or a folder renamed without its probe, fails there rather than at runtime.
_ASSETS_MARKERS: tuple[str, ...] = (
    "general_open_ended",
    "general_educational",
    "tool_calling",
    "sov_ai_facts",
    "sov_ai_multilingual_parity",
    "sov_ai_dynamic",
    "safety_chat_pressure",
    "safety_agentic",
    "financial_services",
    "health_therapy_disclosure",
    "health_triage_disclosure",
    "health_decision_support_disclosure",
    "health_general_disclosure",
)

_RUNTIME_ASSETS_DIR: ContextVar[Optional[Path]] = ContextVar(
    "conversation_plugin_runtime_assets_dir",
    default=None,
)


def set_runtime_assets_dir(path: str | Path | None) -> Token[Optional[Path]]:
    """Set a per-context asset root for serialized/remote generation.

    A remote or containerised runner may need to point at a
    cluster-visible copy of the banks. The simulator config carries
    that path and the generator installs it for the duration of one row.
    """
    return _RUNTIME_ASSETS_DIR.set(Path(path) if path else None)


def reset_runtime_assets_dir(token: Token[Optional[Path]]) -> None:
    """Restore the previous per-context asset root."""
    _RUNTIME_ASSETS_DIR.reset(token)


def packaged_assets_dir() -> Path:
    """Return the ``assets/`` directory shipped inside this package.

    Uses :mod:`importlib.resources` so it resolves for both editable and
    wheel installs. Zip-imported distributions are not supported: the
    loaders below do ordinary filesystem reads, and the banks have always
    been read as real files.
    """
    return Path(str(resources.files("usersim.engine") / "assets"))


#: ``os.pathsep``-separated list of additional asset roots, searched ahead
#: of the packaged banks.
ASSET_PATH_ENV_VAR = "USERSIM_ASSET_PATH"


def asset_search_path() -> tuple[Path, ...]:
    """Asset roots in priority order, duplicates removed.

    Always ends with the packaged directory, so the shipped banks remain
    reachable no matter what a deployment prepends.
    """
    roots: list[Path] = []

    runtime_override = _RUNTIME_ASSETS_DIR.get()
    if runtime_override is not None:
        roots.append(Path(runtime_override))

    search = os.environ.get(ASSET_PATH_ENV_VAR)
    if search:
        roots.extend(Path(entry) for entry in search.split(os.pathsep) if entry)

    env_override = os.environ.get("USERSIM_ASSETS_DIR")
    if env_override:
        roots.append(Path(env_override))

    roots.append(packaged_assets_dir())
    return tuple(dict.fromkeys(roots))


def default_assets_dir() -> Path:
    """Return the highest-priority asset root.

    For the callers that need a single serializable directory -- the
    generator config and the run manifest -- rather than a path to search.
    Per-loader env overrides (``USERSIM_<PROBE>_BANK`` and friends) take
    precedence at the loader level, so a deployment can redirect a single
    bank without redirecting the whole tree.
    """
    return asset_search_path()[0]


def probe_assets_dir(
    probe: str,
    *,
    assets_root: Optional[Path] = None,
) -> Path:
    """Canonical path to a probe's asset folder.

    By convention every probe owns ``<assets_root>/<probe>/`` — this
    helper is the single source of truth for that mapping. Loaders
    compose sub-paths under it (e.g.
    ``probe_assets_dir("sov_ai_facts") / locale / "sample.yaml"``)
    instead of hard-coding the directory name as a string literal,
    so a probe rename / asset-folder rename is a single-edit
    operation and the ``_ASSETS_MARKERS == known_probes()`` invariant
    in ``test_assets_discovery.py`` keeps the registry and on-disk
    layout in lockstep.

    Args:
        probe: One of the registered probe names from
            ``usersim.engine.core.probes.known_probes()``. We do
            not validate against that registry here to avoid an
            import cycle (``_assets.py`` is imported during loader-
            module import, which happens before the probe bootstrap).
            The ``test_marker_set_matches_probe_registry`` invariant
            test guards the registry/marker relationship; the
            per-loader default-path tests in
            ``test_default_asset_paths.py`` guard the marker/on-disk
            relationship.
        assets_root: Pin resolution to exactly this directory, skipping
            the search path. This is what ``--assets-dir`` becomes: a
            user who named a location is told when the assets are not
            there rather than quietly served the packaged copies.
            When ``None``, :func:`asset_search_path` is walked and the
            first root holding ``probe`` wins.
    """
    if assets_root is not None:
        return Path(assets_root) / probe

    search = asset_search_path()
    for root in search:
        candidate = root / probe
        if candidate.is_dir():
            return candidate
    # Nothing matched. Return the highest-priority candidate so the caller
    # reports the location a user would most expect it to have been.
    return search[0] / probe
