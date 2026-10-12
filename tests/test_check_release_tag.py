# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``scripts/check_release_tag.py`` lets a tag release only the version it names, from a commit on main."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_release_tag.py"
_spec = importlib.util.spec_from_file_location("check_release_tag", _SCRIPT)
check_release_tag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_release_tag)


@pytest.fixture
def history(tmp_path: Path, git: Callable[..., str]) -> Path:
    """A repository with two commits on ``main`` and a branch ``side`` one past them; ``main`` is checked out."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "commit", "-q", "--allow-empty", "-m", "one")
    git(repo, "commit", "-q", "--allow-empty", "-m", "two")
    git(repo, "switch", "-q", "-c", "side")
    git(repo, "commit", "-q", "--allow-empty", "-m", "three")
    git(repo, "switch", "-q", "main")
    return repo


def _dist(tmp_path: Path, version: str) -> Path:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / f"usersim-{version}-py3-none-any.whl").write_bytes(b"")
    (dist / f"usersim-{version}.tar.gz").write_bytes(b"")
    return dist


class TestCheckTag:
    def test_the_tag_of_the_built_version_on_main_passes(self, history: Path) -> None:
        assert check_release_tag.check_tag("v0.2.0rc1", "0.2.0rc1", commit="main", branch="main", repo=history) == []

    def test_an_earlier_commit_on_main_passes(self, history: Path) -> None:
        assert check_release_tag.check_tag("v0.2.0rc1", "0.2.0rc1", commit="main~1", branch="main", repo=history) == []

    def test_a_tag_naming_another_version_is_refused(self, history: Path) -> None:
        problems = check_release_tag.check_tag("v0.2.0", "0.2.0rc1", commit="main", branch="main", repo=history)
        assert len(problems) == 1 and "must be v0.2.0rc1" in problems[0]

    def test_a_commit_off_main_is_refused(self, history: Path) -> None:
        problems = check_release_tag.check_tag("v0.2.0rc1", "0.2.0rc1", commit="side", branch="main", repo=history)
        assert len(problems) == 1 and "not on main" in problems[0]

    def test_a_branch_git_cannot_find_fails_closed(self, history: Path) -> None:
        with pytest.raises(SystemExit, match="could not compare"):
            check_release_tag.on_branch("HEAD", "origin/main", repo=history)

    @pytest.mark.parametrize("version", ["0.2.0rc2.dev0", "1.0.0a1", "1.0.0b1", "0.2.0.post1"])
    def test_only_a_final_release_or_a_candidate_is_released(self, history: Path, version: str) -> None:
        problems = check_release_tag.check_tag(f"v{version}", version, commit="main", branch="main", repo=history)
        assert len(problems) == 1 and "not a release version" in problems[0]


def test_the_version_comes_from_the_built_wheel(tmp_path: Path) -> None:
    assert check_release_tag.built_version(_dist(tmp_path, "0.2.0rc1")) == "0.2.0rc1"


def test_a_dist_without_one_wheel_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="run `make dist` first"):
        check_release_tag.built_version(tmp_path)


@pytest.mark.parametrize(("version", "prerelease"), [("0.2.0", "false"), ("0.2.0rc1", "true")])
def test_a_release_candidate_is_reported_as_a_pre_release(
    tmp_path: Path, history: Path, monkeypatch: pytest.MonkeyPatch, version: str, prerelease: str
) -> None:
    output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.chdir(history)
    assert check_release_tag.main([f"v{version}", str(_dist(tmp_path, version)), "--branch", "main"]) == 0
    assert output.read_text() == f"prerelease={prerelease}\n"


def test_a_refused_tag_writes_no_output(tmp_path: Path, history: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.chdir(history)
    assert check_release_tag.main(["v0.2.0", str(_dist(tmp_path, "0.2.0rc1")), "--branch", "main"]) == 1
    assert not output.exists()
