# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the active uv executable against the Mise registry without dependencies."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    match = re.search(r'^uv\s*=\s*"([^"]+)"', (ROOT / ".mise.toml").read_text(), re.MULTILINE)
    expected = match.group(1) if match else ""
    try:
        result = subprocess.run(
            [os.environ.get("UV", "uv"), "--version"], capture_output=True, text=True, check=True, timeout=15
        )
        actual = result.stdout.split()[1]
    except (OSError, subprocess.SubprocessError, IndexError):
        actual = "missing or broken"
    if expected and actual == expected:
        return 0
    print(f"Expected uv {expected}, found {actual}. Run make bootstrap, then scripts/mise run setup.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
