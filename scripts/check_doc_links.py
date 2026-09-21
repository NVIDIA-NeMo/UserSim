# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Fail when a relative link in tracked markdown points at nothing.

Three separate chunks of work broke documentation links without anything
noticing: a package rename, a directory move, and a section deletion. Each
was found by hand, weeks apart. The checker costs milliseconds.

Only relative links are resolved. External URLs are left alone -- checking
them needs network access, turns an offline gate into a flaky one, and fails
on anything behind auth.
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

#: Paths are repo-relative, so a leading `_` marks a top-level directory that
#: is not part of the distribution.
SKIP_PREFIXES = ("_",)
SKIP_SCHEMES = ("http://", "https://", "mailto:", "tel:", "#")


def tracked_markdown() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [REPO / f for f in out if not f.startswith(SKIP_PREFIXES)]


def broken_links(path: Path) -> list[tuple[int, str]]:
    problems: list[tuple[int, str]] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for target in LINK.findall(line) + HTML_IMG.findall(line):
            if target.startswith(SKIP_SCHEMES):
                continue
            # Strip an anchor; the file is what we can verify.
            resolved = (path.parent / target.split("#")[0]).resolve()
            if not resolved.exists():
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
        print(f"\n✗ {total} broken relative link(s)")
        return 1
    print("✓ every relative link in tracked markdown resolves")
    return 0


if __name__ == "__main__":
    sys.exit(main())
