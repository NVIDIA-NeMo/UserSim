# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task and compatibility entry points preserve argv, selectors and failures."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def recording_uv(tmp_path, monkeypatch):
    executable = tmp_path / "uv"
    record = tmp_path / "record.json"
    executable.write_text(
        f"#!{sys.executable}\nimport json, os, pathlib, sys\n"
        "if sys.argv[1:] == ['--version']:\n    print('uv 0.13.1 fixture')\n    raise SystemExit(0)\n"
        "pathlib.Path(os.environ['TASK_RECORD']).write_text(json.dumps({'argv':sys.argv[1:],"
        "'python':os.environ.get('UV_PYTHON'),'venv':os.environ.get('UV_PROJECT_ENVIRONMENT')}))\n"
        "raise SystemExit(int(os.environ.get('TASK_STATUS','0')))\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("UV", str(executable))
    monkeypatch.setenv("TASK_RECORD", str(record))
    return executable, record


@pytest.mark.parametrize("status", [0, 7])
def test_make_and_mise_preserve_quoted_pytest_arguments(recording_uv, monkeypatch, status, command_test_env):
    _, record = recording_uv
    args = "-k 'alpha or beta' --junitxml='path with spaces.xml'"
    monkeypatch.setenv("TASK_STATUS", str(status))
    records = []
    for command, extra in [
        (["make", "coverage", "PYTEST_WORKERS=2", f"PYTEST_ARGS={args}"], {}),
        ([str(ROOT / "scripts/mise"), "run", "test:ci"], {"PYTEST_WORKERS": "2", "PYTEST_ARGS": args}),
    ]:
        result = subprocess.run(
            command, cwd=ROOT, env=command_test_env(**extra), capture_output=True, check=False, timeout=30
        )
        assert (result.returncode == 0) is (status == 0)
        records.append(json.loads(record.read_text())["argv"])
    assert records[0] == records[1]
    assert records[0][-3:] == ["-k", "alpha or beta", "--junitxml=path with spaces.xml"]


@pytest.mark.parametrize("selector", ["3.11", "3.14"])
def test_setup_preserves_requested_python_and_environment(
    recording_uv, monkeypatch, tmp_path, selector, command_test_env
):
    _, record = recording_uv
    monkeypatch.setenv("UV_PYTHON", selector)
    selected = tmp_path / "selected-venv"
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", str(selected))
    subprocess.run(
        ["make", "install-ci"], cwd=ROOT, env=command_test_env(), capture_output=True, check=True, timeout=30
    )
    observed = json.loads(record.read_text())
    assert observed == {"argv": ["sync", "--locked"], "python": selector, "venv": str(selected)}


def test_extensionless_task_has_spdx_coverage(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("license_for_mise", ROOT / "scripts/update_license_headers.py")
    checker = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = checker
    spec.loader.exec_module(checker)
    task = tmp_path / ".mise/tasks/check/invalid"
    task.parent.mkdir(parents=True)
    task.write_text("#!/usr/bin/env bash\nif then\n")
    assert checker.should_process_file(task)
    assert checker.main(tmp_path, check_only=True, files=[task])[1] == 1


def test_lint_scope_preserves_empty_overrides_and_quoted_paths(recording_uv, command_test_env):
    _, record = recording_uv
    subprocess.run(
        ["make", "lint", "PY_SRC='path with spaces.py'", "PY_TESTS=", "PY_E2E="],
        cwd=ROOT,
        env=command_test_env(),
        capture_output=True,
        check=True,
        timeout=30,
    )
    assert json.loads(record.read_text())["argv"] == [
        "run",
        "ruff",
        "check",
        "--output-format=full",
        "path with spaces.py",
    ]


def test_release_tag_remains_data_through_make_and_mise(recording_uv, monkeypatch, tmp_path, command_test_env):
    _, record = recording_uv
    marker = tmp_path / "executed"
    tag = f"$(shell touch {marker})"
    monkeypatch.setenv("TAG", tag)
    subprocess.run(
        ["make", "check-release-tag"], cwd=ROOT, env=command_test_env(), capture_output=True, check=True, timeout=30
    )
    assert json.loads(record.read_text())["argv"][-2:] == [tag, "dist"]
    assert not marker.exists()
