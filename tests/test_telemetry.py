# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User Sim marks Data Designer's anonymous usage events as its own.

Data Designer reads ``NEMO_SESSION_PREFIX`` once, when its telemetry module is
first imported, so each case runs in a fresh interpreter: this one has already
imported both packages.
"""

from __future__ import annotations

import os
import subprocess
import sys

_PRINT_PREFIX = "import os, usersim; print(os.environ['NEMO_SESSION_PREFIX'])"


def _run(code: str, *, prefix: str | None = None) -> str:
    env = {key: value for key, value in os.environ.items() if key != "NEMO_SESSION_PREFIX"}
    if prefix is not None:
        env["NEMO_SESSION_PREFIX"] = prefix
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    return result.stdout.strip()


def test_importing_usersim_sets_the_session_prefix() -> None:
    assert _run(_PRINT_PREFIX) == "usersim-"


def test_a_prefix_the_caller_set_is_kept() -> None:
    assert _run(_PRINT_PREFIX, prefix="host-") == "host-"


def test_data_designer_prefixes_its_sessions_when_usersim_loads_it() -> None:
    """A usersim module that imports Data Designer gets the prefix in before Data Designer reads it."""
    code = (
        "import usersim.engine.generator\n"
        "from data_designer.engine.models.telemetry import SESSION_PREFIX\n"
        "print(SESSION_PREFIX)"
    )
    assert _run(code) == "usersim-"
