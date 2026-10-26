# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply the configured source scopes to a Ruff task without shell evaluation."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys


def main() -> int:
    command = [os.environ.get("UV", "uv"), "run", "ruff", *sys.argv[1:]]
    for name, default in (
        ("PY_SRC", "src scripts templates"),
        ("PY_TESTS", "tests conftest.py"),
        ("PY_E2E", "tests_e2e"),
    ):
        command += shlex.split(os.environ.get(name, default))
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
