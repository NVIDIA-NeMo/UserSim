# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the release archives in a dist/ directory before they are published.

Fails unless the directory holds exactly one sdist and one wheel, for the same
project and version, and:

- ``twine check --strict`` accepts both, so the metadata renders on PyPI;
- every file in the wheel comes from the sdist. ``make dist`` builds the wheel
  from the sdist, so a file only the wheel has came from somewhere else, such
  as a stale ``build/`` directory;
- the wheel ships the asset banks, and no ``embeddings.parquet`` (retrieval
  vectors stay with the embedding model that produced them) or
  ``evaluate-assets`` scorecard;
- the sdist ships no ``tests/``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

TWINE_VERSION = "7.0.0"
ASSET_SUFFIXES = (".yaml", ".yml", ".json", ".parquet")


def find_archives(dist: Path) -> tuple[Path, Path]:
    """The single sdist and single wheel in ``dist``."""
    sdists = sorted(dist.glob("*.tar.gz"))
    wheels = sorted(dist.glob("*.whl"))
    if len(sdists) != 1 or len(wheels) != 1:
        found = [p.name for p in (*sdists, *wheels)]
        raise SystemExit(f"expected one sdist and one wheel in {dist}, found {found}")
    return sdists[0], wheels[0]


def check_archives(sdist: Path, wheel: Path) -> list[str]:
    """Every problem with the pair, so one run reports them all."""
    problems: list[str] = []

    sdist_name, sdist_version = sdist.name.removesuffix(".tar.gz").rsplit("-", 1)
    wheel_name, wheel_version = wheel.name.split("-")[:2]
    if (sdist_name, sdist_version) != (wheel_name, wheel_version):
        problems.append(f"the sdist ({sdist_name} {sdist_version}) and wheel ({wheel_name} {wheel_version}) differ")

    with tarfile.open(sdist) as tar:
        # Every member sits under a "<name>-<version>/" root directory.
        sdist_files = [m.name.split("/", 1)[1] for m in tar.getmembers() if m.isfile() and "/" in m.name]
    with zipfile.ZipFile(wheel) as whl:
        wheel_files = [name for name in whl.namelist() if not name.endswith("/")]

    from_sdist = {f.removeprefix("src/") for f in sdist_files if f.startswith("src/")}
    extra = sorted(f for f in wheel_files if ".dist-info/" not in f and f not in from_sdist)
    if extra:
        problems.append(f"the wheel holds {len(extra)} file(s) the sdist doesn't, e.g. {extra[:3]}")
    if not any(f.endswith(ASSET_SUFFIXES) for f in wheel_files):
        problems.append("the wheel ships no asset banks, so every probe would fail at asset resolution")
    if any(f.endswith("embeddings.parquet") for f in wheel_files):
        problems.append("the wheel ships embeddings.parquet")
    if any(f.endswith("_eval_scorecard.parquet") or "/_scorecards/" in f for f in wheel_files):
        problems.append("the wheel ships an evaluate-assets scorecard")
    tests = [f for f in sdist_files if f.startswith("tests/")]
    if tests:
        problems.append(f"the sdist ships {len(tests)} file(s) under tests/")
    return problems


def twine_check(paths: tuple[Path, ...]) -> bool:
    """Run the pinned ``twine check --strict``, which validates the metadata PyPI renders."""
    command = ["uv", "tool", "run", "--from", f"twine=={TWINE_VERSION}", "twine", "check", "--strict"]
    return subprocess.run([*command, *map(str, paths)], check=False).returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dist", type=Path, nargs="?", default=Path("dist"))
    args = parser.parse_args(argv)

    sdist, wheel = find_archives(args.dist)
    for path in (sdist, wheel):
        print(f"  {path.name}  ({path.stat().st_size:,} bytes)")
    problems = check_archives(sdist, wheel)
    if not twine_check((sdist, wheel)):
        problems.append("twine check failed (see above)")
    if problems:
        print("✗ the release archives are not fit to publish:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print("✓ both release archives pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
