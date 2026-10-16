# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Display release archive digests before publication approval."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    directory: Path = parser.parse_args().directory
    archives = sorted([*directory.glob("*.whl"), *directory.glob("*.tar.gz")])
    if not archives:
        parser.error(f"no release archives in {directory}")
    rows = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}" for path in archives]
    summary = "```\n" + "\n".join(rows) + "\n```\n"
    print(summary, end="")
    if destination := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(destination).open("a", encoding="utf-8") as output:
            output.write(summary)


if __name__ == "__main__":
    main()
