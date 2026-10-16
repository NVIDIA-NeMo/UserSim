# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reject outputs and execution counts in tracked notebooks."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def main() -> int:
    files = subprocess.check_output(["git", "ls-files", "-z", "*.ipynb"]).decode().split("\0")
    dirty = []
    for path in filter(None, files):
        notebook = json.loads(Path(path).read_text())
        count = sum(len(cell.get("outputs", [])) + bool(cell.get("execution_count")) for cell in notebook["cells"])
        if count:
            dirty.append(f"{path} ({count} output/execution entries)")
    if dirty:
        print("Committed notebooks carry outputs:", *dirty, sep="\n  ")
        print("Run mise run notebooks:clean and commit the result.")
        return 1
    print(f"OK {len([path for path in files if path])} notebook(s) carry no outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
