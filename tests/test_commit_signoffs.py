# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise commit policy against real temporary Git histories."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_commit_signoffs.py"
AUTHOR = "Example Author <author@example.com>"


def git(repo, *args, **kwargs):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, text=True, capture_output=True, **kwargs
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for name in ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL"):
        monkeypatch.delenv(name, raising=False)
    git(tmp_path, "init")
    git(tmp_path, "config", "user.name", "Example Author")
    git(tmp_path, "config", "user.email", "author@example.com")
    git(tmp_path, "-c", "commit.gpgsign=false", "commit", "--allow-empty", "-m", "base")
    return tmp_path


def run_check(repo, *args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=repo, capture_output=True, text=True)


@pytest.mark.parametrize(
    "trailers,valid",
    [
        (f"Signed-off-by: {AUTHOR}", True),
        ("", False),
        ("Signed-off-by: Someone Else <author@example.com>", False),
        ("Signed-off-by: Example Author <other@example.com>", False),
        ("Signed-off-by: Example Author author@example.com", False),
        (f"Signed-off-by: {AUTHOR}\nCo-authored-by: Other <other@example.com>", False),
        (
            f"Signed-off-by: {AUTHOR}\nCo-authored-by: Other <other@example.com>\nSigned-off-by: Other <other@example.com>",
            True,
        ),
        (f"Signed-off-by: {AUTHOR}\nCo-authored-by: malformed", False),
    ],
)
def test_hook_validates_identity_and_coauthors(repo, trailers, valid):
    message = repo / "message"
    message.write_text(f"feat: example\n\n{trailers}\n")
    result = run_check(repo, "--commit-msg", str(message))
    assert (result.returncode == 0) == valid, result.stderr


def test_signoff_in_prose_is_not_a_trailer(repo):
    message = repo / "message"
    message.write_text(f"feat: example\n\nSigned-off-by: {AUTHOR}\n\nThis is prose, not a trailer block.\n")
    assert run_check(repo, "--commit-msg", str(message)).returncode == 1


def test_each_commit_is_checked_including_bots(repo):
    base = git(repo, "rev-parse", "HEAD")
    git(repo, "-c", "commit.gpgsign=false", "commit", "--allow-empty", "-m", f"first\n\nSigned-off-by: {AUTHOR}")
    bot = "dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>"
    git(repo, "-c", "commit.gpgsign=false", "commit", "--allow-empty", "--author", bot, "-m", "unsigned bot")
    head = git(repo, "rev-parse", "HEAD")
    result = run_check(repo, "--base", base, "--head", head)
    assert result.returncode == 1
    assert head in result.stderr
    git(
        repo,
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--amend",
        "--allow-empty",
        "-m",
        f"signed bot\n\nSigned-off-by: {bot}",
    )
    assert run_check(repo, "--base", base, "--head", git(repo, "rev-parse", "HEAD")).returncode == 0


def test_hook_uses_author_environment(repo):
    message = repo / "message"
    message.write_text("feat: example\n\nSigned-off-by: Other <other@example.com>\n")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--commit-msg", str(message)],
        cwd=repo,
        env={**os.environ, "GIT_AUTHOR_NAME": "Other", "GIT_AUTHOR_EMAIL": "other@example.com"},
        capture_output=True,
    )
    assert result.returncode == 0


def test_empty_range_and_invalid_revision_fail(repo):
    head = git(repo, "rev-parse", "HEAD")
    assert run_check(repo, "--base", head, "--head", head).returncode == 1
    assert run_check(repo, "--base", "--all", "--head", head).returncode != 0


def test_real_commit_hook_receives_author_override(repo):
    import shlex

    hook = repo / ".git/hooks/commit-msg"
    hook.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(SCRIPT))} --commit-msg "$1"\n')
    hook.chmod(0o755)
    other = "Other Author <other@example.com>"
    git(
        repo,
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--allow-empty",
        "--author",
        other,
        "-m",
        f"other\n\nSigned-off-by: {other}",
    )
    with pytest.raises(subprocess.CalledProcessError):
        git(
            repo,
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--allow-empty",
            "--author",
            other,
            "-s",
            "-m",
            "wrong identity",
        )


def test_missing_history_fails_closed(repo):
    head = git(repo, "rev-parse", "HEAD")
    result = run_check(repo, "--base", "0" * 40, "--head", head)
    assert result.returncode == 1
    assert "could not complete" in result.stderr


@pytest.mark.parametrize("signed_intermediate", [False, True])
def test_shallow_history_cannot_hide_intermediate_commits(repo, signed_intermediate):
    base = git(repo, "rev-parse", "HEAD")
    message = "intermediate"
    if signed_intermediate:
        message += f"\n\nSigned-off-by: {AUTHOR}"
    git(repo, "-c", "commit.gpgsign=false", "commit", "--allow-empty", "-m", message)
    intermediate = git(repo, "rev-parse", "HEAD")
    git(repo, "-c", "commit.gpgsign=false", "commit", "--allow-empty", "-m", f"tip\n\nSigned-off-by: {AUTHOR}")
    head = git(repo, "rev-parse", "HEAD")

    complete = run_check(repo, "--base", base, "--head", head)
    assert complete.returncode == (0 if signed_intermediate else 1), complete.stderr
    if not signed_intermediate:
        assert intermediate in complete.stderr

    shallow = repo / "shallow"
    git(repo, "clone", "--depth=1", repo.as_uri(), str(shallow))
    git(shallow, "fetch", "--depth=1", "origin", base)
    result = run_check(shallow, "--base", base, "--head", head)
    assert result.returncode == 1
    assert "complete history" in result.stderr
    assert "git fetch --unshallow" in result.stderr

    git(shallow, "fetch", "--unshallow", "origin")
    recovered = run_check(shallow, "--base", base, "--head", head)
    assert recovered.returncode == complete.returncode, recovered.stderr
