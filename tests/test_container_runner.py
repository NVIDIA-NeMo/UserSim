# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Container orchestration preserves failures, limits and host isolation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "container_runner", Path(__file__).parents[1] / "scripts/run_container_tests.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_failure_keeps_diagnostics(tmp_path):
    log = tmp_path / "failure.log"
    assert runner.run_logged([sys.executable, "-c", "print('diagnostic'); raise SystemExit(7)"], log, timeout=5) == 7
    assert "diagnostic" in log.read_text()


@pytest.mark.parametrize("status", [0, 9, 124])
def test_run_limits_mounts_cleanup_and_exit_status(monkeypatch, tmp_path, status):
    commands = []
    configurations = []

    def verify_client(command, environment):
        assert command[:2] == ["docker", "--config"]
        config = Path(command[2])
        assert config.is_dir(), "client configuration expired before Docker operation"
        assert list(config.iterdir()) == [], "stored Docker settings entered isolated config"
        assert command[3:5] == ["--host", "unix:///var/run/docker.sock"]
        assert environment["DOCKER_CONFIG"] == str(config)
        configurations.append(config)

    def fake_logged(command, log, **kwargs):
        verify_client(command, kwargs.get("env", {}))
        commands.append((command, kwargs["env"]))
        return status

    cleanup = []

    def fake_cleanup(command, **kwargs):
        verify_client(command, kwargs.get("env", {}))
        cleanup.append((command, kwargs["env"]))
        return runner.subprocess.CompletedProcess(command, 0, "removed", "")

    for selector in list(runner.os.environ):
        if selector.startswith(("DOCKER_", "BUILDX_", "BUILDKIT_")):
            monkeypatch.delenv(selector)
    # Stored default settings must not select a different daemon than the build.
    home = tmp_path / "home"
    (home / ".docker").mkdir(parents=True)
    (home / ".docker/config.json").write_text('{"currentContext":"remote-canary"}')
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(runner.Path, "is_socket", lambda self: True)
    monkeypatch.setattr(runner, "run_logged", fake_logged)
    monkeypatch.setattr(runner.subprocess, "run", fake_cleanup)
    assert runner.main(["run", "--artifacts", str(tmp_path)]) == status
    command, environment = commands[0]
    assert "--network=none" in command
    assert "--cpus=2" in command
    assert "--memory=8g" in command
    assert "--memory-swap=8g" in command
    assert "/tmp:rw,exec,nosuid,size=1g" in command
    assert command.count("--mount") == 1
    attempt = next(tmp_path.glob("run-*"))
    assert f"type=bind,source={attempt},target=/artifacts" in command
    assert "--env-file" not in command
    assert cleanup == [(command[:5] + ["rm", "-f", command[command.index("--name") + 1]], environment)]
    assert cleanup[0][1] is environment
    assert configurations[0] == configurations[1]
    assert all(not config.exists() for config in configurations)


def test_logged_timeout_preserves_partial_output(tmp_path):
    log = tmp_path / "partial.log"
    command = [sys.executable, "-c", "import time; print('before timeout', flush=True); time.sleep(10)"]
    assert runner.run_logged(command, log, timeout=1) == 124
    assert "before timeout" in log.read_text()


@pytest.mark.parametrize("statuses, expected", [([0, 0], 0), ([5, 0], 5), ([0, 6], 6), ([124, 0], 124)])
def test_inside_runs_both_shared_targets_and_propagates_failure(monkeypatch, tmp_path, statuses, expected):
    calls = []
    environments = []
    results = iter(statuses)

    def fake_logged(command, log, **kwargs):
        calls.append((command, log))
        environments.append(kwargs["env"])
        return next(results)

    monkeypatch.setenv("COVERAGE_FILE", "original")
    monkeypatch.setattr(runner, "run_logged", fake_logged)
    assert runner.inside(tmp_path) == expected
    assert [command[-2:] for command, _ in calls] == [["run", "test:ci"], ["run", "test:e2e"]]
    assert all(command[0] == str(runner.ROOT / "scripts/mise") for command, _ in calls)
    assert [log.name for _, log in calls] == ["unit.log", "e2e.log"]

    assert "E2E_ARGS" not in environments[0]
    assert environments[0]["PYTEST_WORKERS"] == "2"
    assert environments[0]["COV_XML"] == str(tmp_path / "coverage.xml")
    assert all(log.parent == tmp_path for _, log in calls)
    assert not list(tmp_path.glob("run-*"))
    assert "--run-only" in environments[1]["E2E_ARGS"]


def test_build_requires_successful_audit(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(runner, "audit_context", lambda *a, **kw: 7)
    monkeypatch.setattr(runner, "run_logged", lambda *a, **kw: calls.append(a))
    assert runner.main(["build", "--artifacts", str(tmp_path)]) == 7
    assert calls == []


@pytest.mark.parametrize("value", ["tcp://example.invalid:2375", "unix:///tmp/remote.sock"])
def test_audit_rejects_endpoint_before_socket_probe(monkeypatch, tmp_path, value):
    monkeypatch.setenv("DOCKER_HOST", value)
    monkeypatch.setattr(runner.Path, "is_socket", lambda self: pytest.fail("socket probe preceded admission"))
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **kw: pytest.fail("Docker invoked"))
    assert runner.main(["audit", "--artifacts", str(tmp_path)]) == 1


def test_audit_requires_local_socket(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "LOCAL_SOCKET", tmp_path / "missing.sock")
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **kw: pytest.fail("Docker invoked"))
    assert runner.main(["audit", "--artifacts", str(tmp_path)]) == 1


@pytest.mark.parametrize(
    "builder_status, export_status, leak, missing, expected",
    [
        (7, 0, False, False, 7),
        (0, 9, False, False, 9),
        (0, 0, True, False, 1),
        (0, 0, False, True, 1),
        (0, 0, False, False, 0),
    ],
)
def test_audit_export_is_required_and_checked(
    monkeypatch, tmp_path, builder_status, export_status, leak, missing, expected
):
    monkeypatch.setattr(runner, "local_client", lambda config: (["docker"], {}))

    def fake_logged(command, log, **kwargs):
        if "inspect" in command:
            log.write_text("Driver: docker\n")
            return builder_status
        exported = Path(command[command.index("--output") + 1].split("dest=", 1)[1])
        context = Path(command[-1])
        for case in runner.json.loads((runner.ROOT / "tests/fixtures/exclusion-policy.json").read_text()):
            if case["docker_included"] and not missing or not case["docker_included"] and leak:
                target = exported / case["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text((context / case["path"]).read_text())
        return export_status

    monkeypatch.setattr(runner, "run_logged", fake_logged)
    assert runner.main(["audit", "--artifacts", str(tmp_path)]) == expected


def test_audit_rejects_nonlocal_builder_driver(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "local_client", lambda config: (["docker"], {}))

    def fake_logged(command, log, **kwargs):
        assert "inspect" in command, "Context transfer preceded builder admission"
        log.write_text("Driver: remote\n")
        return 0

    monkeypatch.setattr(runner, "run_logged", fake_logged)
    assert runner.main(["audit", "--artifacts", str(tmp_path)]) == 1


def test_client_pins_transport_and_isolates_config(monkeypatch, tmp_path):
    for name in runner.os.environ:
        if name.startswith(("DOCKER_", "BUILDX_", "BUILDKIT_")):
            monkeypatch.delenv(name)
    monkeypatch.setattr(runner.Path, "is_socket", lambda self: True)
    monkeypatch.setenv("DOCKER_HOST", "")
    client, environment = runner.local_client(tmp_path / "client")
    assert client == ["docker", "--config", str(tmp_path / "client"), "--host", "unix:///var/run/docker.sock"]
    assert environment["DOCKER_CONFIG"] == str(tmp_path / "client")
    assert "DOCKER_HOST" not in environment


def test_run_missing_local_socket_invokes_no_docker(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(runner, "LOCAL_SOCKET", tmp_path / "missing.sock")
    monkeypatch.setattr(runner, "run_logged", lambda *a, **kw: calls.append(a) or 0)
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: calls.append(a))
    assert runner.main(["run", "--artifacts", str(tmp_path)]) == 1
    assert calls == []


@pytest.mark.parametrize(
    "selector",
    [
        "DOCKER_HOST",
        "DOCKER_CONTEXT",
        "DOCKER_CONFIG",
        "DOCKER_TLS_VERIFY",
        "DOCKER_CERT_PATH",
        "BUILDX_BUILDER",
        "BUILDX_CONFIG",
        "BUILDKIT_HOST",
        "DOCKER_BUILDKIT",
        "BUILDX_BAKE_FILE",
        "BUILDKIT_PROGRESS",
    ],
)
def test_each_inherited_selector_is_rejected_before_socket_probe(monkeypatch, tmp_path, selector):
    monkeypatch.setenv(selector, "hostile-selector")
    monkeypatch.setattr(runner.Path, "is_socket", lambda self: pytest.fail("socket probe preceded admission"))
    with pytest.raises(ValueError, match=selector):
        runner.local_client(tmp_path / "client")
    assert not (tmp_path / "client").exists()


@pytest.mark.parametrize("action", ["audit", "build", "run"])
def test_action_wiring_rejects_remote_transport_without_any_command(monkeypatch, tmp_path, action):
    monkeypatch.setenv("DOCKER_HOST", "tcp://example.invalid:2375")
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: pytest.fail("Docker invoked"))
    monkeypatch.setattr(runner, "run_logged", lambda *args, **kwargs: pytest.fail("Docker invoked"))
    assert runner.main([action, "--artifacts", str(tmp_path)]) == 1


@pytest.mark.parametrize(
    "outcome,expected,message",
    [
        ("removed", 0, "container removed"),
        ("already-absent", 0, "Status: already-absent"),
        ("already-absent-zero", 0, "Status: already-absent"),
        ("denied-zero", 1, "removal permission denied"),
        ("wrong-absent", 1, "different-name"),
        ("mixed-absent", 1, "removal permission denied"),
        ("denied", 1, "partial cleanup output"),
        ("timeout", 1, "partial output"),
        ("socket-error", 1, "cleanup socket unavailable"),
    ],
)
def test_each_cleanup_outcome_retains_diagnostics(monkeypatch, tmp_path, capsys, outcome, expected, message):
    client, environment, name = (
        ["docker", "--host", "unix:///var/run/docker.sock"],
        {"DOCKER_CONFIG": "isolated"},
        "synthetic-container",
    )

    def execute(command, **kwargs):
        assert command == client + ["rm", "-f", name]
        assert kwargs["env"] is environment
        if outcome == "timeout":
            raise runner.subprocess.TimeoutExpired(command, 30, output=b"partial output", stderr=b"cleanup timeout")
        if outcome == "socket-error":
            raise OSError("cleanup socket unavailable")
        absent = f"Error response from daemon: No such container: {name}\n"
        responses = {
            "removed": (0, "container removed\n", ""),
            "already-absent": (1, "", absent),
            "already-absent-zero": (0, "", absent),
            "denied-zero": (0, "partial cleanup output", "removal permission denied"),
            "wrong-absent": (1, "", "Error response from daemon: No such container: different-name\n"),
            "mixed-absent": (1, "", absent + "removal permission denied"),
            "denied": (1, "partial cleanup output", "removal permission denied"),
        }
        return runner.subprocess.CompletedProcess(command, *responses[outcome])

    monkeypatch.setattr(runner.subprocess, "run", execute)
    log = tmp_path / "cleanup.log"
    assert runner.cleanup_container(client, environment, name, log) == expected
    assert message in log.read_text()
    if expected:
        assert name in capsys.readouterr().err


@pytest.mark.parametrize("primary,expected", [(0, 1), (9, 9), (124, 124)])
def test_cleanup_failure_precedence_is_explicit(monkeypatch, tmp_path, primary, expected):
    monkeypatch.setattr(runner, "local_client", lambda config: (["docker"], {}))
    monkeypatch.setattr(runner, "run_logged", lambda *args, **kwargs: primary)
    monkeypatch.setattr(runner, "cleanup_container", lambda *args: 1)
    assert runner.main(["run", "--artifacts", str(tmp_path)]) == expected


@pytest.mark.parametrize("action", ["audit", "build", "run"])
def test_fresh_attempts_keep_later_failures_separate(monkeypatch, tmp_path, action):
    root = tmp_path / "results"
    root.mkdir()
    (root / "operator-note.txt").write_text("unrelated")
    attempts = []
    codes = iter([0, 7])

    def audit(destination, **kwargs):
        attempts.append(destination)
        code = next(codes)
        (destination / ("failure.log" if code else "audit-result.json")).write_text("partial" if code else "passed")
        return code

    def logged(command, log, **kwargs):
        if action == "run":
            attempts.append(log.parent)
            code = next(codes)
            if not code:
                (log.parent / "unit.xml").write_text("passed")
        else:
            code = 0
        log.write_text("partial" if code else "passed")
        return code

    monkeypatch.setattr(runner, "audit_context", audit)
    monkeypatch.setattr(runner, "local_client", lambda config: (["docker"], {}))
    monkeypatch.setattr(runner, "run_logged", logged)
    monkeypatch.setattr(runner, "cleanup_container", lambda *args: 0)
    assert runner.main([action, "--artifacts", str(root)]) == 0
    assert runner.main([action, "--artifacts", str(root)]) == 7
    assert len(attempts) == 2 and attempts[0] != attempts[1]
    success = {"audit": "audit-result.json", "build": "build.log", "run": "unit.xml"}[action]
    assert (attempts[0] / success).read_text() == "passed"
    assert not (attempts[1] / success).exists()
    assert (root / "operator-note.txt").read_text() == "unrelated"


@pytest.mark.parametrize("kind", ["symlink", "source", "source-ancestor"])
def test_output_root_admission_preserves_rejected_paths(monkeypatch, tmp_path, kind):
    source = tmp_path / "checkout"
    source.mkdir()
    (source / "tracked.txt").write_text("unchanged")
    monkeypatch.setattr(runner, "ROOT", source)
    root = source
    if kind == "symlink":
        root = tmp_path / "link"
        root.symlink_to(source, target_is_directory=True)
    elif kind == "source-ancestor":
        root = tmp_path
    with pytest.raises(ValueError):
        runner.attempt_directory(root)
    assert (source / "tracked.txt").read_text() == "unchanged"
    assert not list(tmp_path.rglob("run-*"))
