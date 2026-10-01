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

import logging
import os
import subprocess
from functools import lru_cache

logger = logging.getLogger("usersim.engine")


@lru_cache(maxsize=1)
def get_code_sha() -> str | None:
    """Return the current git SHA (short form), or None if unavailable.

    Cached for the lifetime of the process — we do not expect the SHA
    to change mid-run. Falls back to the ``USERSIM_CODE_SHA``
    environment variable (useful in CI / container builds where a
    ``.git`` directory is not present).
    """
    env_sha = os.environ.get("USERSIM_CODE_SHA")
    if env_sha:
        return env_sha.strip() or None

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.debug(f"  |-- get_code_sha: git unavailable ({e})")
        return None

    if result.returncode != 0:
        logger.debug(f"  |-- get_code_sha: git rev-parse failed: {result.stderr.strip()}")
        return None

    sha = result.stdout.strip()
    return sha or None


def get_nemotron_personas_version() -> str | None:
    """Best-effort version tag for the Nemotron-Personas dataset in use.

    Today the dataset's version is not explicitly exposed; we record
    the ``USERSIM_NEMOTRON_PERSONAS_VERSION`` environment variable if
    set, or ``None`` otherwise. The replay-skew check in
    ``LocalFileSeedSource`` loading consumes this field.
    """
    v = os.environ.get("USERSIM_NEMOTRON_PERSONAS_VERSION")
    return (v.strip() or None) if v else None
