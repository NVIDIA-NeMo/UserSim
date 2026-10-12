# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provenance helpers for simulation outcomes.

``SimulationOutcome.provenance`` captures "what was this trajectory built
against" — the pieces needed to audit a report bundle or to detect a
version-skew between a replayed panel and a refreshed persona dataset.

Conservative and cheap: no subprocess in the hot path if we can avoid
it. Git SHA is cached per process.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from functools import lru_cache
from importlib import metadata
from pathlib import Path

logger = logging.getLogger("usersim.engine")

#: The directory of the ``usersim`` package this module belongs to.
_PACKAGE_DIR = Path(__file__).resolve().parents[2]

#: Commit ids are recorded in git's 12-character short form.
_SHA_LENGTH = 12


@lru_cache(maxsize=1)
def get_code_sha() -> str | None:
    """Return the commit UserSim's code came from (12-character form), or None.

    In order: the ``USERSIM_CODE_SHA`` environment variable (for builds with no
    ``.git`` directory); the git checkout the package was loaded from, when git
    tracks it; the commit a git install recorded in ``direct_url.json`` (PEP
    610). A copy installed from a package index records no commit, and the
    repository of the working directory never counts: it is the user's, not
    UserSim's. Cached for the lifetime of the process.
    """
    env_sha = os.environ.get("USERSIM_CODE_SHA", "").strip()
    if env_sha:
        return env_sha
    return _checkout_sha(_PACKAGE_DIR) or _vcs_install_sha(own_distribution())


def own_distribution() -> metadata.Distribution | None:
    """The installed distribution that provides the ``usersim`` package.

    The distribution named ``usersim`` wins over any other that claims the
    package, such as stale metadata left by an earlier name.
    """
    names = ["usersim", *metadata.packages_distributions().get("usersim", [])]
    for name in names:
        try:
            return metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
    return None


def _checkout_sha(package_dir: Path) -> str | None:
    """HEAD of the git checkout holding ``package_dir``, if git tracks the package there.

    An installed copy inside someone else's repository (a ``.venv`` in their
    project) is untracked, so their HEAD is not mistaken for UserSim's.
    """

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(package_dir), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
        )

    try:
        if git("ls-files", "--error-unmatch", "__init__.py").returncode != 0:
            return None
        head = git("rev-parse", f"--short={_SHA_LENGTH}", "HEAD")
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.debug(f"  |-- get_code_sha: git unavailable ({e})")
        return None
    if head.returncode != 0:
        logger.debug(f"  |-- get_code_sha: git rev-parse failed: {head.stderr.strip()}")
        return None
    return head.stdout.strip() or None


def _vcs_install_sha(dist: metadata.Distribution | None) -> str | None:
    """The commit a git install recorded (PEP 610 ``vcs_info.commit_id``), if any."""
    raw = dist.read_text("direct_url.json") if dist is not None else None
    if not raw:
        return None
    try:
        commit = (json.loads(raw).get("vcs_info") or {}).get("commit_id")
    except (json.JSONDecodeError, AttributeError):
        return None
    return commit[:_SHA_LENGTH] if isinstance(commit, str) and commit else None


def get_nemotron_personas_version() -> str | None:
    """Best-effort version tag for the Nemotron-Personas dataset in use.

    Today the dataset's version is not explicitly exposed; we record
    the ``USERSIM_NEMOTRON_PERSONAS_VERSION`` environment variable if
    set, or ``None`` otherwise. The replay-skew check in
    ``LocalFileSeedSource`` loading consumes this field.
    """
    v = os.environ.get("USERSIM_NEMOTRON_PERSONAS_VERSION")
    return (v.strip() or None) if v else None
