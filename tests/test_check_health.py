# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Health checks skip only missing credentials and fail on execution errors."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_health.py"
_spec = importlib.util.spec_from_file_location("check_health", _SCRIPT)
assert _spec and _spec.loader
health = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(health)


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEALTH_TEST_KEY", raising=False)
    path = tmp_path / "models.toml"
    path.write_text(
        '[[providers]]\nname="health-test"\nendpoint="http://localhost/v1"\napi_key="HEALTH_TEST_KEY"\n'
        + "".join(
            f'[[models]]\nalias="{alias}"\nmodel="test"\nprovider="health-test"\n'
            for alias in ("user_model", "assistant_model", "judge_model", "evaluator_model")
        )
    )
    return path


def invoke(config, capsys, *args):
    code = health.main(["--models", str(config), *args])
    outcome = json.loads(capsys.readouterr().out.splitlines()[-1])
    return code, outcome


@pytest.mark.parametrize("content", [None, "[[models]", '[[models]]\nalias="user_model"'])
def test_invalid_config_fails_before_missing_credentials(config, capsys, content):
    if content is None:
        config.unlink()
    else:
        config.write_text(content)
    code, outcome = invoke(config, capsys)
    assert code == 1
    assert outcome["status"] == "failure"
    assert outcome["stage"] == "preflight"


@pytest.mark.parametrize("alias", ["user_model", "judge_model", "evaluator_model"])
def test_invalid_replay_setting_is_not_a_credential_skip(config, capsys, alias):
    config.write_text(config.read_text().replace(f'alias="{alias}"', f'alias="{alias}"\nreplay_reasoning=true'))
    code, outcome = invoke(config, capsys)
    assert code == 1
    assert outcome["status"] == "failure"
    assert outcome["stage"] == "preflight"


@pytest.mark.parametrize(
    "alias,parameter", [("assistant_model", "timeout"), ("evaluator_model", "max_parallel_requests")]
)
def test_invalid_model_parameters_fail_before_missing_credentials(config, capsys, monkeypatch, parameter, alias):
    def unexpected(*args, **kwargs):
        pytest.fail("preflight must not execute the provider pipeline")

    monkeypatch.setattr(health, "pipeline", unexpected)
    monkeypatch.setattr(health.subprocess, "run", unexpected)
    config.write_text(config.read_text().replace(f'alias="{alias}"', f'alias="{alias}"\n{parameter}=0'))
    code, outcome = invoke(config, capsys)
    assert code == 1
    assert outcome["status"] == "failure"
    assert outcome["stage"] == "preflight"


def test_missing_credential_is_explicit_skip(config, capsys, monkeypatch, tmp_path):
    def unexpected(*args, **kwargs):
        pytest.fail("missing credentials must not invoke the provider")

    monkeypatch.setattr(health.subprocess, "run", unexpected)
    summary = tmp_path / "summary.md"
    code, outcome = invoke(config, capsys, "--summary-file", str(summary))
    assert code == 0
    assert outcome["status"] == "skip"
    assert "HEALTH_TEST_KEY" in outcome["detail"]
    assert "skip" in summary.read_text()


@pytest.mark.parametrize("exported,expected", [(None, "local-key"), ("shell-key", "shell-key")])
def test_env_local_precedence_and_complete_pipeline(config, capsys, monkeypatch, tmp_path, exported, expected):
    (tmp_path / ".env.local").write_text("HEALTH_TEST_KEY=local-key\n")
    (tmp_path / ".env").write_text("HEALTH_TEST_KEY=base-key\n")
    if exported:
        monkeypatch.setenv("HEALTH_TEST_KEY", exported)
    seen = []

    def command(argv, *, check, timeout):
        import os
        import sys

        assert argv[:3] == [sys.executable, "-m", "usersim.cli"]
        assert os.environ["HEALTH_TEST_KEY"] == expected
        assert timeout == 17
        seen.append(argv[3])
        out = Path(argv[argv.index("--out") + 1])
        if argv[3] == "simulate":
            (out / "run=test-run").mkdir(parents=True)
        elif argv[3] == "report":
            report = out / "run=test-run" / "index.html"
            report.parent.mkdir(parents=True)
            report.write_text("report")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(health.subprocess, "run", command)
    code, outcome = invoke(config, capsys, "--timeout", "17")
    assert code == 0
    assert outcome["status"] == "pass"
    assert seen == ["simulate", "eval", "report"]
    assert expected not in json.dumps(outcome)


@pytest.mark.parametrize(
    "stage,error",
    [
        ("simulate", "nonzero"),
        ("eval", "nonzero"),
        ("report", "nonzero"),
        ("simulate", "timeout"),
        ("simulate", "missing-tool"),
    ],
)
def test_command_errors_fail(config, capsys, monkeypatch, stage, error):
    monkeypatch.setenv("HEALTH_TEST_KEY", "key")

    def command(argv, **kwargs):
        if argv[3] == stage:
            if error == "timeout":
                raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
            if error == "missing-tool":
                raise FileNotFoundError("usersim")
            raise subprocess.CalledProcessError(9, argv)
        if argv[3] == "simulate":
            (Path(argv[argv.index("--out") + 1]) / "run=test-run").mkdir(parents=True)

    monkeypatch.setattr(health.subprocess, "run", command)
    code, outcome = invoke(config, capsys)
    assert code == 1
    assert outcome["status"] == "failure"
    assert outcome["stage"] == stage


@pytest.mark.parametrize("stage", ["simulate", "report"])
def test_successful_commands_must_produce_outputs(config, capsys, monkeypatch, stage):
    monkeypatch.setenv("HEALTH_TEST_KEY", "key")

    def command(argv, **kwargs):
        if argv[3] == "simulate" and stage != "simulate":
            (Path(argv[argv.index("--out") + 1]) / "run=test-run").mkdir(parents=True)

    monkeypatch.setattr(health.subprocess, "run", command)
    code, outcome = invoke(config, capsys)
    assert code == 1
    assert outcome["status"] == "failure"
    assert outcome["stage"] == stage


def test_import_failure_is_not_a_skip(config, capsys, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "dotenv", None)
    code, outcome = invoke(config, capsys)
    assert code == 1
    assert outcome["status"] == "failure"
    assert "ModuleNotFoundError" in outcome["detail"]


def test_workflow_preserves_failures():
    import yaml

    workflow = yaml.safe_load((_SCRIPT.parents[1] / ".github/workflows/health-checks.yml").read_text())
    job = workflow["jobs"]["live"]
    assert not job.get("continue-on-error", False)
    assert job["strategy"]["fail-fast"] is False
    assert all(not step.get("continue-on-error", False) for step in job["steps"])
    assert "pipefail" in workflow["defaults"]["run"]["shell"].split()
