# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``scripts/check_dist.py`` refuses release archives that are unfit to publish."""

from __future__ import annotations

import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_dist.py"
_spec = importlib.util.spec_from_file_location("check_dist", _SCRIPT)
check_dist = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_dist)

_PACKAGE = ["usersim/__init__.py", "usersim/engine/assets/probe/bank.yaml"]


def _sdist(path: Path, files: list[str], root: str = "usersim-1.0") -> Path:
    with tarfile.open(path, "w:gz") as tar:
        for name in files:
            info = tarfile.TarInfo(f"{root}/{name}")
            tar.addfile(info, io.BytesIO(b""))
    return path


def _wheel(path: Path, files: list[str]) -> Path:
    with zipfile.ZipFile(path, "w") as whl:
        for name in [*files, "usersim-1.0.dist-info/METADATA"]:
            whl.writestr(name, "")
    return path


def _pair(tmp_path: Path, sdist: list[str] | None = None, wheel: list[str] | None = None) -> tuple[Path, Path]:
    sdist_files = sdist if sdist is not None else ["pyproject.toml", *(f"src/{f}" for f in _PACKAGE)]
    return (
        _sdist(tmp_path / "usersim-1.0.tar.gz", sdist_files),
        _wheel(tmp_path / "usersim-1.0-py3-none-any.whl", wheel if wheel is not None else _PACKAGE),
    )


def test_a_matching_pair_passes(tmp_path: Path) -> None:
    assert check_dist.check_archives(*_pair(tmp_path)) == []


@pytest.mark.parametrize(
    ("sdist", "wheel", "expected"),
    [
        (None, [*_PACKAGE, "usersim/stale.py"], "the sdist doesn't"),
        (None, ["usersim/__init__.py"], "no asset banks"),
        (
            ["pyproject.toml", *(f"src/{f}" for f in [*_PACKAGE, "usersim/engine/assets/x/embeddings.parquet"])],
            [*_PACKAGE, "usersim/engine/assets/x/embeddings.parquet"],
            "embeddings.parquet",
        ),
        (
            ["pyproject.toml", *(f"src/{f}" for f in [*_PACKAGE, "usersim/engine/assets/x/_eval_scorecard.parquet"])],
            [*_PACKAGE, "usersim/engine/assets/x/_eval_scorecard.parquet"],
            "scorecard",
        ),
        (["pyproject.toml", "tests/test_x.py", *(f"src/{f}" for f in _PACKAGE)], None, "under tests/"),
    ],
    ids=["file-not-from-the-sdist", "no-banks", "embeddings", "scorecard", "tests-in-sdist"],
)
def test_each_problem_is_reported(tmp_path: Path, sdist, wheel, expected: str) -> None:
    problems = check_dist.check_archives(*_pair(tmp_path, sdist, wheel))
    assert len(problems) == 1 and expected in problems[0]


def test_a_version_mismatch_is_reported(tmp_path: Path) -> None:
    sdist, _ = _pair(tmp_path)
    wheel = _wheel(tmp_path / "usersim-1.1-py3-none-any.whl", _PACKAGE)
    problems = check_dist.check_archives(sdist, wheel)
    assert any("differ" in p for p in problems)


def test_anything_but_one_sdist_and_one_wheel_is_refused(tmp_path: Path) -> None:
    _wheel(tmp_path / "usersim-1.0-py3-none-any.whl", _PACKAGE)
    with pytest.raises(SystemExit, match="one sdist and one wheel"):
        check_dist.find_archives(tmp_path)
