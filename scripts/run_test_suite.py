# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Construct the ordinary pytest command shared by native and container tasks."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", action="store_true")
    args = parser.parse_args()
    command = [os.environ.get("UV", "uv"), "run", "pytest", "-n", os.environ.get("PYTEST_WORKERS", "auto")]
    if args.coverage:
        command += [f"--cov={package}" for package in shlex.split(os.environ.get("COV_PKGS", "usersim"))]
        command += [
            "--cov-report=term-missing",
            f"--cov-report=xml:{os.environ.get('COV_XML', 'coverage.xml')}",
            f"--cov-fail-under={os.environ.get('COV_FAIL_UNDER', '84')}",
        ]
    command += shlex.split(os.environ.get("PYTEST_ARGS", ""))
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
