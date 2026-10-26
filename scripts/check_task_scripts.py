# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate Bash task syntax, including extensionless task files."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    paths = [ROOT / "scripts/install_mise.sh", ROOT / "scripts/mise"]
    paths += [path for path in (ROOT / ".mise/tasks").rglob("*") if path.is_file() and path.suffix != ".toml"]
    for path in paths:
        subprocess.run(["bash", "-n", str(path)], check=True)
    print(f"Validated {len(paths)} Bash task/bootstrap scripts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
