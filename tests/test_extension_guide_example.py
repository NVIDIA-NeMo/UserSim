# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The extensions guide shows the example package's files exactly as they are.

``docs/extensions.md`` builds the ``my-bookings`` package file by file, and the
same package lives in ``docs/examples/my-bookings/``, which
``make check-extensions`` installs and tests against an installed User Sim.
A code block the guide marks ``<!-- example: <path> -->`` must equal that file,
its license header left out, and every file of the package that holds more
than a license header must be shown, so the guide cannot drift from code that
runs.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_GUIDE = _REPO / "docs" / "extensions.md"
_EXAMPLE = _REPO / "docs" / "examples" / "my-bookings"
_BLOCK = re.compile(r"^<!-- example: (\S+) -->\n```[a-z]*\n(.*?)^```$", re.MULTILINE | re.DOTALL)
_LICENSE_HEADER = re.compile(r"\A(?:# SPDX-[^\n]*\n)+\n?")
_BUILD_OUTPUT = ("__pycache__", ".pytest_cache", ".egg-info", "build", "dist")


def _shown() -> dict[str, str]:
    """Each file the guide shows, by its path in the package, with the block's text."""
    blocks = _BLOCK.findall(_GUIDE.read_text(encoding="utf-8"))
    assert blocks, f"{_GUIDE} marks no block with <!-- example: <path> -->"
    return dict(blocks)


def _package_files() -> dict[str, str]:
    """Each file of the example package, by its path, with its text after the license header."""
    files = {}
    for path in sorted(_EXAMPLE.rglob("*")):
        if not path.is_file() or any(part.endswith(_BUILD_OUTPUT) for part in path.parts):
            continue
        text = _LICENSE_HEADER.sub("", path.read_text(encoding="utf-8"))
        if text.strip():
            files[path.relative_to(_EXAMPLE).as_posix()] = text
    return files


def test_each_block_the_guide_marks_equals_its_file():
    shown, files = _shown(), _package_files()
    missing = sorted(set(shown) - set(files))
    assert not missing, f"the guide shows files the package does not hold: {missing}"
    differ = sorted(path for path, text in shown.items() if text != files[path])
    assert not differ, f"these blocks in {_GUIDE.name} differ from docs/examples/my-bookings/: {differ}"


def test_the_guide_shows_every_file_of_the_package():
    unshown = sorted(set(_package_files()) - set(_shown()))
    assert not unshown, f"mark a block for each in {_GUIDE.name} with <!-- example: <path> -->: {unshown}"
