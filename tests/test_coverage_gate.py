# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coverage enforcement uses the same overridable Make command locally and in CI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("status", [0, 9])
@pytest.mark.parametrize("overrides", [False, True])
def test_coverage_arguments_and_exit_status(tmp_path, status, overrides, monkeypatch, command_test_env):
    # A caller may itself be make coverage with report overrides. Test defaults
    # independently of its exported variables and recursive-Make argument state.
    for name, value in {
        "MAKEFLAGS": " -- COV_XML=ambient.xml PYTEST_ARGS=-k=ambient",
        "MFLAGS": "--no-print-directory",
        "MAKEOVERRIDES": "COV_XML=ambient.xml",
        "MAKELEVEL": "4",
        "COV_XML": "ambient.xml",
        "PYTEST_ARGS": "-k ambient",
        "PYTEST_WORKERS": "99",
        "COV_PKGS": "ambient",
        "COV_FAIL_UNDER": "100",
    }.items():
        monkeypatch.setenv(name, value)
    executable = tmp_path / "uv"
    recorded = tmp_path / "arguments.json"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['COVERAGE_TEST_ARGS']).write_text(json.dumps(sys.argv[1:]))\n"
        "sys.exit(int(os.environ['COVERAGE_TEST_STATUS']))\n"
    )
    executable.chmod(0o755)
    options = ["PYTEST_WORKERS=0"]
    if overrides:
        options += [
            "COV_PKGS=example another",
            "COV_XML=custom report.xml",
            "COV_FAIL_UNDER=88",
            "PYTEST_ARGS=--junitxml=junit.xml -k 'selected or other'",
        ]
    result = subprocess.run(
        ["make", "coverage", f"UV={executable}", *options],
        cwd=_ROOT,
        env=command_test_env(COVERAGE_TEST_ARGS=str(recorded), COVERAGE_TEST_STATUS=str(status)),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert (result.returncode == 0) is (status == 0)
    arguments = recorded.read_text()
    args = json.loads(arguments)
    assert args[:5] == ["run", "pytest", "-n", "0", "--cov=example" if overrides else "--cov=usersim"]
    assert f"--cov-fail-under={88 if overrides else 84}" in args
    assert f"--cov-report=xml:{'custom report.xml' if overrides else 'coverage.xml'}" in args
    if overrides:
        assert "--cov=another" in args
        assert args[-3:] == ["--junitxml=junit.xml", "-k", "selected or other"]


def test_ci_enforces_coverage_once_and_retains_failure_artifacts():
    workflow = yaml.safe_load((_ROOT / ".github/workflows/ci.yml").read_text())
    job = workflow["jobs"]["test"]
    assert job["name"] == "Tests (Python ${{ matrix.python-version }})"
    assert "3.12" in job["strategy"]["matrix"]["python-version"]
    tests = [
        step
        for step in job["steps"]
        if step.get("run", "").startswith("scripts/mise run test") and step.get("run", "") == "scripts/mise run test"
    ]
    coverage = [step for step in job["steps"] if step.get("run", "").startswith("scripts/mise run test:ci")]
    assert len(tests) == len(coverage) == 1
    assert tests[0]["if"] == "matrix.python-version != '3.12'"
    assert coverage[0]["if"] == "matrix.python-version == '3.12'"
    assert not coverage[0].get("continue-on-error", False)
    upload = next(step for step in job["steps"] if step.get("uses", "").startswith("actions/upload-artifact@"))
    assert upload["if"] == "always() && matrix.python-version == '3.12'"
    assert upload["with"]["path"] == "coverage.xml"
    assert upload["with"]["if-no-files-found"] == "error"
    assert not upload.get("continue-on-error", False)


def test_native_and_ci_bundle_membership():
    import tomllib

    tasks = tomllib.loads((_ROOT / ".mise/tasks/packaging.toml").read_text())
    native = [step["task"] for step in tasks["validate:native"]["run"]]
    assert native == ["check", "check:license:dependencies", "test:ci", "test:smoke", "check:extensions"]
    ci = tasks["validate:ci"]["run"]
    assert ci[:2] == [{"task": "validate:native"}, {"task": "check:dist"}]
    assert ci[2] == "mise run test:e2e --wheel-dir dist"
