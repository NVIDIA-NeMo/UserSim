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

A probe class may also declare ``assets_root``, the root that holds its
``<label>/`` folder, as a probe shipped in its own package does. That root
is searched for that probe after the first three entries and before the
packaged directory, so a deliberate override wins over it.

:func:`probe_assets_dir` walks that path and takes the first root that
actually holds the probe, so a package contributing one probe's banks
does not have to restate the shipped ones to be usable.

A pinned root -- what ``--assets-dir`` becomes, passed to
:func:`probe_assets_dir` as ``assets_root`` -- is not part of the search.
It pins resolution to exactly that directory, for every probe, including
one whose class declares a root. A user who named a location wants to be
told when the assets are not there, rather than to be quietly served
another copy instead.

There is deliberately no CWD fallback: resolving against whatever
directory the process happened to start in makes a broken install look
healthy from the source tree.
"""

from __future__ import annotations

import os
from contextvars import ContextVar, Token
from importlib import resources
from pathlib import Path

from usersim.engine.core import probes

_RUNTIME_ASSETS_DIR: ContextVar[Path | None] = ContextVar(
    "conversation_plugin_runtime_assets_dir",
    default=None,
)


def set_runtime_assets_dir(path: str | Path | None) -> Token[Path | None]:
    """Set a per-context asset root for serialized/remote generation.

    A remote or containerised runner may need to point at a
    cluster-visible copy of the banks. The simulator config carries
    that path and the generator installs it for the duration of one row.
    """
    return _RUNTIME_ASSETS_DIR.set(Path(path) if path else None)


def reset_runtime_assets_dir(token: Token[Path | None]) -> None:
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


def _override_roots() -> list[Path]:
    """The roots a run or a deployment sets on purpose, highest priority first."""
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

    return roots


def asset_search_path() -> tuple[Path, ...]:
    """Asset roots in priority order, duplicates removed.

    Always ends with the packaged directory, so the shipped banks remain
    reachable no matter what a deployment prepends.
    """
    return tuple(dict.fromkeys([*_override_roots(), packaged_assets_dir()]))


def declared_assets_roots() -> dict[str, Path]:
    """The ``assets_root`` each registered probe class declares, by label.

    Probes that declare none, the built-ins among them, are left out. Reads
    the registry as it stands, so a probe counts once its module is imported.
    """
    return {
        label: Path(cls.assets_root)
        for label, cls in sorted(probes._PROBE_REGISTRY.items())
        if getattr(cls, "assets_root", None) is not None
    }


def probe_asset_roots(probe: str) -> tuple[Path, ...]:
    """The roots searched for ``probe``'s assets, highest priority first.

    The search path, with the root ``probe``'s class declares placed just
    ahead of the packaged directory.
    """
    declared = declared_assets_roots().get(probe)
    roots = [*_override_roots(), *([declared] if declared is not None else []), packaged_assets_dir()]
    return tuple(dict.fromkeys(roots))


def asset_roots_in_use() -> tuple[Path, ...]:
    """Every root a lookup can read: the search path, then each declared root.

    In that order, so a setup whose probes declare no root lists exactly the
    search path.
    """
    return tuple(dict.fromkeys([*asset_search_path(), *declared_assets_roots().values()]))


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
    assets_root: Path | None = None,
) -> Path:
    """Canonical path to a probe's asset folder.

    By convention every probe owns ``<assets_root>/<probe>/`` — this
    helper is the single source of truth for that mapping. Loaders
    compose sub-paths under it (e.g.
    ``probe_assets_dir("sov_ai_facts") / locale / "sample.yaml"``)
    instead of hard-coding the directory name as a string literal,
    so a probe rename / asset-folder rename is a single-edit
    operation. ``test_assets_discovery.py`` checks that every
    registered probe resolves to a folder and that every packaged
    folder belongs to a registered probe.

    Args:
        probe: A registered probe label. Its class's ``assets_root``,
            when declared, joins the search; a label not registered yet
            is looked up on the search path alone.
        assets_root: Pin resolution to exactly this directory, skipping
            the search path. This is what ``--assets-dir`` becomes: a
            user who named a location is told when the assets are not
            there rather than quietly served the packaged copies.
            When ``None``, :func:`probe_asset_roots` is walked and the
            first root holding ``probe`` wins.
    """
    if assets_root is not None:
        return Path(assets_root) / probe

    roots = probe_asset_roots(probe)
    for root in roots:
        candidate = root / probe
        if candidate.is_dir():
            return candidate
    # Nothing matched. Report where a user would most expect it to have
    # been: the root the probe declares, else the highest-priority root.
    declared = declared_assets_roots().get(probe)
    return (declared if declared is not None else roots[0]) / probe
