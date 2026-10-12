# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that a version tag may release the archives in dist/.

Fails unless the wheel in dist/ was built as a release version (``X.Y.Z`` or
``X.Y.ZrcN``), the tag is ``v`` followed by that version, and the commit
checked out is on ``main``. A tag ruleset limits who can push a tag, not which
commit or version it names, so the release workflow runs this on the tagged
commit before anything is published.

On success, prints ``prerelease=true`` for a release candidate and
``prerelease=false`` otherwise, and appends the same line to
``$GITHUB_OUTPUT`` when that is set, so the release workflow can mark the
release.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

#: Final releases and release candidates. Development, alpha, beta, post and
#: local versions are never released.
RELEASE_VERSION = re.compile(r"\d+\.\d+\.\d+(rc\d+)?")


def built_version(dist: Path) -> str:
    """The version of the one wheel in ``dist``, as its file name records it."""
    wheels = sorted(dist.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected one wheel in {dist}, found {[w.name for w in wheels]}; run `make dist` first")
    return wheels[0].name.split("-")[1]


def on_branch(commit: str, branch: str, repo: Path | None = None) -> bool:
    """Whether ``commit`` is ``branch``'s tip or one of its ancestors."""
    done = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, branch],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode not in (0, 1):
        raise SystemExit(f"git could not compare {commit} with {branch}: {done.stderr.strip()}")
    return done.returncode == 0


def check_tag(tag: str, version: str, *, commit: str, branch: str, repo: Path | None = None) -> list[str]:
    """Every reason ``tag`` may not release ``version`` from ``commit``, so one run reports them all."""
    problems = []
    if not RELEASE_VERSION.fullmatch(version):
        problems.append(f"the archives were built as {version}, which is not a release version (X.Y.Z or X.Y.ZrcN)")
    if tag != f"v{version}":
        problems.append(f"the tag is {tag!r}, but the archives were built as {version}, so the tag must be v{version}")
    if not on_branch(commit, branch, repo):
        problems.append(f"{commit} is not on {branch}; tag a commit on it")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tag", help="the version tag, for example v0.2.0rc1")
    parser.add_argument("dist", type=Path, nargs="?", default=Path("dist"))
    parser.add_argument("--branch", default="origin/main", help="the branch the commit must be on")
    args = parser.parse_args(argv)

    version = built_version(args.dist)
    problems = check_tag(args.tag, version, commit="HEAD", branch=args.branch)
    if problems:
        print(f"✗ {args.tag} can't release these archives:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    prerelease = "true" if "rc" in version else "false"
    print(f"✓ {args.tag} releases {version} from a commit on {args.branch}")
    print(f"prerelease={prerelease}")
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a", encoding="utf-8") as fh:
            fh.write(f"prerelease={prerelease}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
