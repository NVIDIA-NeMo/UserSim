# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Credential loading from ``.env`` files.

Keeps API keys out of the shell history and out of the repository. Both
filenames this reads are gitignored.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger("usersim.cli.env")

#: Read in order. Later files do not override earlier ones, so the more
#: specific ``.env.local`` is read first.
ENV_FILENAMES = (".env.local", ".env")


def find_env_files(start: Path | None = None) -> list[Path]:
    """Return the ``.env`` files that apply to ``start``, nearest first.

    Walks upward so a command run from a subdirectory still finds the file at
    the project root.
    """
    here = (start or Path.cwd()).resolve()
    found: list[Path] = []
    for directory in [here, *here.parents]:
        for name in ENV_FILENAMES:
            candidate = directory / name
            if candidate.is_file():
                found.append(candidate)
        if found:
            break
    return found


def load_local_env(start: Path | None = None) -> list[Path]:
    """Load ``.env.local`` then ``.env`` into the environment.

    **Variables already set in the environment always win.** Someone who
    exported a key deliberately should not have it replaced by a stale file
    they forgot about.

    Returns the files that were read, so a caller can report them. Missing
    files are not an error: most users set variables in their shell instead.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - declared as a dependency
        logger.debug("python-dotenv not installed; skipping .env loading")
        return []

    loaded: list[Path] = []
    for path in find_env_files(start):
        load_dotenv(path, override=False)
        loaded.append(path)
    return loaded


def describe_loaded_env(paths: list[Path]) -> str:
    """Return a one-line summary naming the files read, never their contents."""
    if not paths:
        return "no .env file found; using the environment as-is"
    names = ", ".join(str(p) for p in paths)
    return f"read {names} (existing environment variables take precedence)"


def missing_keys(names: object) -> list[str]:
    """Return which of ``names`` are absent or empty in the environment."""
    return [n for n in names if not os.environ.get(str(n))]  # type: ignore[union-attr]
