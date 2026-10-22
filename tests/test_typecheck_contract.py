# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The configured checker rejects invalid assignments through the installed tool."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml
from packaging.specifiers import SpecifierSet


@pytest.mark.parametrize(
    "text,diagnostic",
    [('value: int = "invalid"\n', "invalid-assignment"), ("type Alias = int\n", "invalid-syntax")],
)
def test_typecheck_rejects_invalid_code(tmp_path: Path, text: str, diagnostic: str) -> None:
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "invalid.py"
    source.write_text(text)
    result = subprocess.run(
        [sys.executable, "-m", "ty", "check", "--project", str(root), str(source), "--output-format", "concise"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode != 0
    assert diagnostic in result.stdout


def test_typecheck_target_matches_supported_minimum() -> None:
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text())
    supported = SpecifierSet(config["project"]["requires-python"])
    floor = next(f"3.{minor}" for minor in range(30) if f"3.{minor}" in supported)
    assert config["tool"]["ty"]["environment"]["python-version"] == floor
    hooks = yaml.safe_load((root / ".pre-commit-config.yaml").read_text())
    hook = next(hook for repo in hooks["repos"] for hook in repo["hooks"] if hook["id"] == "typecheck")
    assert f"Python {floor}" in hook["name"]
