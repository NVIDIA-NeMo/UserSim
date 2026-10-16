# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate author and co-author DCO trailers in Git commits or a hook message."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

IDENTITY = re.compile(r"[^<>\n]+ <[^<>\s@]+@[^<>\s@]+>")
REVISION = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")


def git(*args: str, input_text: str | None = None) -> str:
    return subprocess.run(
        ["git", *args],
        input=input_text,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def check_message(message: str, author: str) -> list[str]:
    """Use Git's trailer parser so a sign-off embedded in prose does not count."""
    trailers = git("-c", "trailer.separators=:", "interpret-trailers", "--parse", input_text=message)
    signoffs = set()
    required = {author}
    errors = []
    for line in trailers.splitlines():
        key, _, identity = line.partition(":")
        if key.lower() not in {"signed-off-by", "co-authored-by"}:
            continue
        identity = identity.strip()
        if not IDENTITY.fullmatch(identity):
            errors.append(f"malformed {key}: {identity!r}")
        elif key.lower() == "signed-off-by":
            signoffs.add(identity)
        else:
            required.add(identity)
    for identity in sorted(required - signoffs):
        errors.append(f"missing Signed-off-by: {identity}")
    return errors


def check_range(base: str, head: str) -> list[str]:
    if not REVISION.fullmatch(base) or not REVISION.fullmatch(head):
        raise ValueError("--base and --head must be full commit SHAs")
    if git("rev-parse", "--is-shallow-repository") == "true":
        raise ValueError("range validation requires complete history; run git fetch --unshallow first")
    commits = git("rev-list", "--reverse", f"{base}..{head}").splitlines()
    if not commits:
        return ["no commits found in the requested range"]
    errors = []
    for commit in commits:
        author = git("show", "-s", "--format=%an <%ae>", commit)
        message = git("show", "-s", "--format=%B", commit)
        errors.extend(f"{commit}: {error}" for error in check_message(message, author))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit-msg", type=Path, help="validate a commit-msg hook file")
    parser.add_argument("--base", help="PR base commit SHA")
    parser.add_argument("--head", help="PR head commit SHA")
    args = parser.parse_args()
    if bool(args.commit_msg) == bool(args.base or args.head) or bool(args.base) != bool(args.head):
        parser.error("provide --commit-msg FILE or both --base SHA and --head SHA")
    try:
        if args.commit_msg:
            author = git("var", "GIT_AUTHOR_IDENT").rsplit(">", 1)[0] + ">"
            errors = check_message(args.commit_msg.read_text(encoding="utf-8"), author)
        else:
            errors = check_range(args.base, args.head)
    except (OSError, UnicodeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"DCO validation could not complete: {exc}", file=sys.stderr)
        return 1
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        print("Use git commit -s with the commit author's identity; see CONTRIBUTING.md.", file=sys.stderr)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
