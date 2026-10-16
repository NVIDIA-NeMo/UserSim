# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail when a link in tracked markdown points at nothing in this checkout.

Checked: relative links, and absolute links into this repository's ``main``
branch. The README uses absolute ones so its links also work on PyPI, and they
resolve against the checkout, so moving or deleting a file still fails here.

Other external URLs are left alone: checking them needs network access, turns
an offline gate into a flaky one, and fails on anything behind auth.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: Inline links and images: `[text](target)` and `![alt](target)`.
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)")

#: Raw HTML images. Markdown renderers honour these, so a broken src is as
#: dead as a broken link -- and they are easy to miss when files move.
HTML_IMG = re.compile(r'<img\s[^>]*src="([^"]+)"')

#: URL prefixes that address a path in this repository's main branch.
REPO_URLS = (
    "https://github.com/NVIDIA-NeMo/UserSim/blob/main/",
    "https://github.com/NVIDIA-NeMo/UserSim/tree/main/",
    "https://raw.githubusercontent.com/NVIDIA-NeMo/UserSim/main/",
)

SKIP_SCHEMES = ("http://", "https://", "mailto:", "tel:", "#")


def tracked_markdown() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "*.md"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\0")
    return [REPO / f for f in out if f and not f.startswith("_")]


def link_path(path: Path, target: str) -> Path | None:
    """What ``target`` in ``path`` points at in this checkout, or None when it is not checked.

    An anchor is dropped; the file is what can be verified.
    """
    for prefix in REPO_URLS:
        if target.startswith(prefix):
            return REPO / target.removeprefix(prefix).split("#")[0]
    if target.startswith(SKIP_SCHEMES):
        return None
    return path.parent / target.split("#")[0]


def broken_links(path: Path) -> list[tuple[int, str]]:
    problems: list[tuple[int, str]] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for target in LINK.findall(line) + HTML_IMG.findall(line):
            resolved = link_path(path, target)
            if resolved is not None and not resolved.resolve().exists():
                problems.append((n, target))
    return problems


def main() -> int:
    total = 0
    for path in tracked_markdown():
        for n, target in broken_links(path):
            rel = path.relative_to(REPO)
            print(f"  {rel}:{n}: {target}")
            total += 1

    if total:
        print(f"\n✗ {total} broken link(s)")
        return 1
    print("✓ every relative link, and every link into this repository's main branch, resolves")
    return 0


if __name__ == "__main__":
    sys.exit(main())
