# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Locate a probe's asset banks.

Each probe keeps its banks in the ``<label>/`` folder of the root its class
declares as ``assets_root``. A built-in probe declares its folder in this
package, ``usersim/engine/probes/<folder>/assets/``, read as **package data**
through :mod:`importlib.resources`, so it is found identically whether the
package is installed editable or from a wheel. A probe shipped in its own
package declares a folder inside that package.

Resolution walks an ordered **search path**, highest priority first:

1. the per-context override set by :func:`set_runtime_assets_dir`;
2. each entry of ``$USERSIM_ASSET_PATH`` (``os.pathsep``-separated);
3. ``$USERSIM_ASSETS_DIR``;

and then the probe's own root, so a deliberate override wins over it.
:func:`probe_assets_dir` takes the first root that actually holds the probe,
so a package contributing one probe's banks does not have to restate the
others to be usable.

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


#: ``os.pathsep``-separated list of additional asset roots, searched ahead
#: of each probe's own root.
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
    """The override roots in priority order, duplicates removed; empty when none is set.

    Each probe's own root comes after these, so a probe's shipped banks stay
    reachable no matter what a deployment prepends.
    """
    return tuple(dict.fromkeys(_override_roots()))


def declared_assets_roots() -> dict[str, Path]:
    """The ``assets_root`` each registered probe class declares, by label.

    Probes that declare none are left out. Reads the registry as it stands, so
    a probe counts once its module is imported.
    """
    return {
        label: Path(cls.assets_root)
        for label, cls in sorted(probes._PROBE_REGISTRY.items())
        if getattr(cls, "assets_root", None) is not None
    }


def _builtin_probes_dir() -> Path:
    """``usersim/engine/probes/``, which holds the built-in probe packages.

    Uses :mod:`importlib.resources` so it resolves for both editable and
    wheel installs. Zip-imported distributions are not supported: the
    loaders do ordinary filesystem reads.
    """
    return Path(str(resources.files("usersim.engine") / "probes"))


def _builtin_probe_root(probe: str) -> Path | None:
    """The ``assets/`` folder of the built-in package that holds ``<probe>/``, if one does.

    A package is usually named after its probe. A package that holds several probes
    keeps each one's banks in its ``assets/<probe>/``.
    """
    packages = _builtin_probes_dir()
    own = packages / probe / "assets"
    if (own / probe).is_dir():
        return own
    return next((root for root in sorted(packages.glob("*/assets")) if (root / probe).is_dir()), None)


def _declared_root(probe: str) -> Path | None:
    """The root ``probe``'s class declares, found before its module is imported if it is a built-in.

    A built-in kept in its own folder declares its package's ``assets/`` folder under
    ``usersim/engine/probes/``. Until its class is registered, the package folder that
    holds ``probe`` stands in for the declaration, so a lookup made while the probe's
    module loads, such as a prompt read at import time, or by a scorer in a process
    that never imports the probe, reaches the probe's assets.
    """
    declared = declared_assets_roots().get(probe)
    if declared is None and probe not in probes._PROBE_REGISTRY:
        return _builtin_probe_root(probe)
    return declared


def probe_asset_roots(probe: str) -> tuple[Path, ...]:
    """The roots searched for ``probe``'s assets, highest priority first.

    The search path, then the root ``probe``'s class declares.
    """
    declared = _declared_root(probe)
    return tuple(dict.fromkeys([*_override_roots(), *([declared] if declared is not None else [])]))


def asset_roots_in_use() -> tuple[Path, ...]:
    """Every root a lookup can read: the search path, then each declared root.

    In that order, so a setup whose probes declare no root lists exactly the
    search path.
    """
    return tuple(dict.fromkeys([*asset_search_path(), *declared_assets_roots().values()]))


def shipped_assets_dir(probe: str) -> Path:
    """``probe``'s own asset folder, under the root its class declares, whatever the search path holds.

    For a built-in probe, its folder in this package. For the commands that write
    or check the bank a probe ships, which an override must not redirect.

    Raises ``LookupError`` for a probe that declares no root.
    """
    declared = _declared_root(probe)
    if declared is None:
        raise LookupError(f"{probe!r} declares no assets_root, so it has no assets of its own")
    return declared / probe


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
    registered probe resolves to a folder and that every built-in
    probe's folder belongs to a registered probe.

    Args:
        probe: A probe label. Its class's ``assets_root``, when declared,
            joins the search. A label not registered yet is looked up on
            the search path, and in its own folder under
            ``usersim/engine/probes/`` when it is a built-in kept there.
        assets_root: Pin resolution to exactly this directory, skipping
            the search path. This is what ``--assets-dir`` becomes: a
            user who named a location is told when the assets are not
            there rather than quietly served another copy.
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
    # been: the root the probe declares, else the highest-priority root,
    # else where a built-in probe keeps its banks. Never a relative path,
    # which would resolve against the working directory.
    declared = _declared_root(probe)
    if declared is not None:
        return declared / probe
    if roots:
        return roots[0] / probe
    return _builtin_probes_dir() / probe / "assets" / probe
