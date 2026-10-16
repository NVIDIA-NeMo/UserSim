# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Git exclusion behavior for private output and required repository inputs."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ignore_repo(tmp_path_factory):
    """Exercise the committed policy without requiring checkout Git metadata."""
    repo = tmp_path_factory.mktemp("ignore-policy")
    subprocess.run(["git", "-c", "init.templateDir=", "init", "-q", str(repo)], check=True)
    shutil.copyfile(REPO / ".gitignore", repo / ".gitignore")
    return repo


CASES = json.loads((REPO / "tests/fixtures/exclusion-policy.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["path"])
def test_git_exclusion_policy(case, ignore_repo: Path) -> None:
    result = subprocess.run(
        ["git", "-c", "core.excludesFile=/dev/null", "check-ignore", "--no-index", "-q", case["path"]],
        cwd=ignore_repo,
    )
    assert result.returncode == (0 if case["git_ignored"] else 1), case["path"]
