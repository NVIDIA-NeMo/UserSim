# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Installed-test setup must honor explicit archives and reject ambiguous modes."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_e2e.py"


@pytest.mark.parametrize("flag", ["--wheel", "--wheel-dir"])
def test_run_only_rejects_archive_selection(flag, tmp_path):
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--run-only", flag, str(tmp_path / "nonexistent")],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 2
    assert "--run-only cannot be combined" in result.stderr
    assert not result.stdout


@pytest.mark.parametrize("count", [0, 2])
def test_archive_directory_must_have_exactly_one_wheel(count, tmp_path):
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    for index in range(count):
        (wheel_dir / f"example-{index}.whl").touch()
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--wheel-dir",
            str(wheel_dir),
            "--work-dir",
            str(tmp_path / "work"),
            "--artifacts",
            str(tmp_path / "artifacts"),
            "--uv",
            "must-not-run-uv",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 2
    assert f"expects exactly one wheel, found {count}" in result.stderr
    assert not result.stdout, "invalid archive selection must fail before setup or build commands"


@pytest.mark.parametrize("status", [0, 9])
def test_make_uses_selected_wheel_and_preserves_failure(status, tmp_path, command_test_env):
    uv = tmp_path / "fake-uv"
    recorded = tmp_path / "arguments.json"
    version = re.search(r'^uv = "([^"]+)"', (ROOT / ".mise.toml").read_text(), re.MULTILINE).group(1)
    uv.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        f"if sys.argv[1:] == ['--version']: print('uv {version}'); sys.exit(0)\n"
        "pathlib.Path(os.environ['E2E_RECORDED_ARGS']).write_text(json.dumps({'argv':sys.argv[1:],'wheel':os.environ.get('WHEEL_DIR'),'args':os.environ.get('E2E_ARGS'),'uv':os.environ.get('UV')}))\n"
        f"sys.exit({status})\n"
    )
    uv.chmod(0o755)
    wheels = tmp_path / "release wheels"
    result = subprocess.run(
        ["make", "test-e2e", f"UV={uv}", f"WHEEL_DIR={wheels}", "E2E_ARGS=--artifacts output/e2e-custom"],
        cwd=ROOT,
        env=command_test_env(E2E_RECORDED_ARGS=str(recorded)),
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert (result.returncode == 0) is (status == 0), result.stderr
    arguments = json.loads(recorded.read_text())
    assert arguments["argv"] == ["run", "--frozen", "python", "scripts/run_e2e.py"]
    assert arguments["wheel"] == str(wheels)
    assert arguments["uv"] == str(uv)
    assert arguments["args"] == "--artifacts output/e2e-custom"


def test_ci_runs_installed_e2e_and_retains_diagnostics():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    job = workflow["jobs"]["e2e"]
    assert job["timeout-minutes"] <= 30
    commands = [step["run"] for step in job["steps"] if "run" in step]
    assert commands == [
        "bash scripts/install_mise.sh",
        "scripts/mise run setup:tools",
        "scripts/mise run setup:ci",
        "scripts/mise run test:e2e",
    ]
    e2e = next(step for step in job["steps"] if step.get("run") == "scripts/mise run test:e2e")
    assert e2e["env"]["E2E_ARGS"] == "--artifacts test-results/e2e"
    upload = next(step for step in job["steps"] if step.get("uses", "").startswith("actions/upload-artifact@"))
    assert upload["if"] == "failure()"
    assert upload["with"]["path"] == "test-results/e2e/"
    assert not any(step.get("continue-on-error") for step in job["steps"])


def test_explicit_wheel_conflicts_with_environment_directory(tmp_path):
    import os
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_e2e.py"), "--wheel", str(tmp_path / "selected.whl"), "--prepare-only"],
        env={**os.environ, "WHEEL_DIR": str(tmp_path / "old")},
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 2
    assert "conflicts with the WHEEL_DIR" in result.stderr


def test_final_explicit_directory_defeats_ambient_archive_selectors(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("e2e_identity", RUNNER)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    checkout = tmp_path / "checkout"
    default_artifacts = checkout / "output/e2e"
    default_artifacts.mkdir(parents=True)
    prior = {
        "operator-evidence.txt": b"untouched prior result",
    }
    for name, content in prior.items():
        (default_artifacts / name).write_bytes(content)
    monkeypatch.setattr(runner, "ROOT", checkout)
    validated = tmp_path / "validated"
    other = tmp_path / "other"
    for directory in (validated, other):
        directory.mkdir()
        (directory / "example.whl").touch()
    selected = []
    monkeypatch.setenv("WHEEL_DIR", str(other))
    monkeypatch.setenv("E2E_ARGS", f"--wheel-dir {other}")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(RUNNER),
            "--wheel-dir",
            str(validated),
            "--work-dir",
            str(tmp_path / "work"),
            "--artifacts",
            str(tmp_path / "artifacts"),
            "--prepare-only",
        ],
    )
    monkeypatch.setattr(
        runner, "prepare", lambda work, args, parser: selected.append(runner.archive(work, args, parser))
    )
    for _ in range(2):
        assert runner.main() == 0
        assert {p.name: p.read_bytes() for p in default_artifacts.iterdir()} == prior
        assert list((checkout / "output").iterdir()) == [default_artifacts]
    assert selected == [(validated / "example.whl").resolve()] * 2


@pytest.mark.parametrize("filename,run_only", [(".env", False), (".env.local", True)])
def test_external_work_directory_stops_ancestor_dotenv_loading(filename, run_only, tmp_path, monkeypatch):
    import importlib.util

    from usersim.cli._env import load_local_env

    spec = importlib.util.spec_from_file_location("e2e_dotenv", RUNNER)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    ancestor = tmp_path / "external"
    work = ancestor / "persistent-work"
    work.mkdir(parents=True)
    (ancestor / filename).write_text("USERSIM_E2E_ANCESTOR_CANARY=unexpected\n")
    (work / "installed.json").write_text(json.dumps({"assets": "unused"}))
    monkeypatch.delenv("USERSIM_E2E_ANCESTOR_CANARY", raising=False)
    monkeypatch.delenv("E2E_ARGS", raising=False)
    monkeypatch.delenv("WHEEL_DIR", raising=False)
    checked = []

    def check_cli_directories(work):
        for case in ("workflow", "tool-choice"):
            cwd = work / "cases" / case
            cwd.mkdir(parents=True, exist_ok=True)
            monkeypatch.chdir(cwd)
            load_local_env()
            assert "USERSIM_E2E_ANCESTOR_CANARY" not in os.environ
            checked.append(case)

    def prepare(work, args, parser):
        check_cli_directories(work)

    def run_suite(work, artifacts, env):
        check_cli_directories(work)
        return 0

    monkeypatch.setattr(runner, "prepare", prepare)
    monkeypatch.setattr(runner, "run_suite", run_suite)
    argv = [str(RUNNER), "--work-dir", str(work), "--artifacts", str(tmp_path / "artifacts")]
    if run_only:
        argv.append("--run-only")
    monkeypatch.setattr(sys, "argv", argv)
    assert runner.main() == 0
    assert checked == ["workflow", "tool-choice"] * (1 if run_only else 2)


def test_install_receipt_records_wheel_bytes(tmp_path, monkeypatch):
    import hashlib
    import importlib.util

    spec = importlib.util.spec_from_file_location("e2e_receipt", RUNNER)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    wheel = tmp_path / "example.whl"
    wheel.write_bytes(b"selected wheel bytes")
    installed = {"module": str(tmp_path / "runtime/lib/usersim/__init__.py"), "assets": "packaged-assets"}
    monkeypatch.setattr(runner, "command", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *args, **kwargs: json.dumps(installed))
    runner.record_install(tmp_path, wheel, {})
    receipt = json.loads((tmp_path / "installed.json").read_text())
    assert receipt["wheel"] == str(wheel)
    assert receipt["wheel_sha256"] == hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert receipt["module"] == installed["module"]
    assert receipt["assets"] == installed["assets"]


def test_run_only_exports_original_install_receipt(tmp_path, monkeypatch):
    import importlib.util
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location("e2e_export", RUNNER)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    work = tmp_path / "work"
    artifacts = tmp_path / "artifacts"
    work.mkdir()
    artifacts.mkdir()
    receipt = {"assets": "installed-assets", "wheel": "/old/build.whl", "wheel_sha256": "original-digest"}
    (work / "installed.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    assert runner.run_suite(work, artifacts, {}) == 0
    assert json.loads((artifacts / "installed.json").read_text()) == receipt


@pytest.fixture
def e2e_runner(monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("e2e_harness_runner", RUNNER)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.delenv("E2E_ARGS", raising=False)
    monkeypatch.delenv("WHEEL_DIR", raising=False)
    return runner


@pytest.fixture
def prepared_case(tmp_path, monkeypatch, e2e_runner):
    runner = e2e_runner
    work, root = tmp_path / "work", tmp_path / "results"
    work.mkdir()
    root.mkdir()
    (root / "operator-note.txt").write_text("unrelated file")
    (work / "installed.json").write_text(json.dumps({"assets": "assets", "wheel_sha256": "original-digest"}))

    def invoke(*flags):
        monkeypatch.setattr(sys, "argv", [str(RUNNER), "--work-dir", str(work), "--artifacts", str(root), *flags])
        return runner.main()

    return runner, work, root, invoke


@pytest.mark.parametrize("failure", ["nonzero", "timeout"])
def test_fresh_attempts_isolate_failure_from_previous_success(failure, prepared_case, monkeypatch):
    from types import SimpleNamespace

    runner, work, root, invoke = prepared_case
    statuses = iter([0, 7])

    def execute(args, **kwargs):
        cases = Path(kwargs["env"]["E2E_ARTIFACTS"])
        code = next(statuses)
        if not code:
            (cases / "scores.parquet").write_bytes(b"passing scores")
            Path(args[args.index("--junitxml") + 1]).write_text('<testsuite failures="0"/>')
        else:
            (cases / "failure.log").write_text("partial failure")
            if failure == "timeout":
                raise subprocess.TimeoutExpired(args, 900)
        return SimpleNamespace(returncode=code)

    monkeypatch.setattr(runner.subprocess, "run", execute)
    assert invoke("--run-only") == 0
    first = next(root.glob("run-*"))
    (work / "installed.json").write_text(json.dumps({"assets": "assets", "wheel_sha256": "new-digest"}))
    assert invoke("--run-only") == (1 if failure == "timeout" else 7)
    second = next(p for p in root.glob("run-*") if p != first)
    assert (first / "scores.parquet").read_bytes() == b"passing scores"
    assert not (second / "scores.parquet").exists()
    assert not (second / "junit.xml").exists()
    assert (second / "failure.log").read_text() == "partial failure"
    assert json.loads((second / "installed.json").read_text())["wheel_sha256"] == "new-digest"
    assert (root / "operator-note.txt").read_text() == "unrelated file"


def test_preparation_state_has_no_result_directory_and_invalidates_failed_readiness(prepared_case, monkeypatch):
    runner, work, root, invoke = prepared_case
    with monkeypatch.context() as first:
        first.setattr(runner, "prepare", lambda *args: None)
        assert invoke("--prepare-only") == 0
    assert list(root.iterdir()) == [root / "operator-note.txt"]

    def interrupted_environment(work):
        assert not (work / "installed.json").exists()
        raise RuntimeError("interrupted preparation")

    monkeypatch.setattr(runner, "environment", interrupted_environment)
    with pytest.raises(RuntimeError, match="interrupted preparation"):
        invoke()
    attempt = next(root.glob("run-*"))
    assert list(attempt.iterdir()) == []
    with pytest.raises(SystemExit) as failure:
        invoke("--run-only")
    assert failure.value.code == 2
    assert list(root.glob("run-*")) == [attempt]


@pytest.mark.parametrize("kind", ["symlink", "dangling", "directory", "fifo", "hardlink"])
def test_owned_file_admission_rejects_each_invalid_kind_unchanged(kind, prepared_case, tmp_path):
    runner, work, root, _ = prepared_case
    target, entry = tmp_path / "outside", work / ".env"
    if kind != "dangling":
        target.write_bytes(b"external bytes")
    if kind in {"symlink", "dangling"}:
        entry.symlink_to(target)
    elif kind == "directory":
        entry.mkdir()
    elif kind == "fifo":
        os.mkfifo(entry)
    else:
        os.link(target, entry)
    before = (work / "installed.json").read_bytes()
    with pytest.raises(ValueError):
        runner.admit_paths(work, root)
    assert (work / "installed.json").read_bytes() == before
    assert not list(root.glob("run-*"))
    assert target.read_bytes() == b"external bytes" if kind != "dangling" else not target.exists()


@pytest.mark.parametrize(
    "name",
    [
        ".env",
        ".env.local",
        "runtime-requirements.txt",
        "installed.json",
        "home",
        "tokenizers",
        "dist",
        "runtime",
        "harness",
        "suite",
        "cases",
    ],
)
def test_every_owned_slot_is_protected(name, prepared_case, tmp_path):
    runner, work, root, _ = prepared_case
    target = tmp_path / "outside"
    target.write_bytes(b"external bytes")
    entry = work / name
    entry.unlink(missing_ok=True)
    entry.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        runner.admit_paths(work, root)
    assert target.read_bytes() == b"external bytes"
    assert not list(root.glob("run-*"))


def test_directory_kind_and_export_hardlink_are_rejected(prepared_case, tmp_path):
    runner, work, root, _ = prepared_case
    (work / "home").write_text("not a directory")
    with pytest.raises(ValueError, match="directory"):
        runner.admit_paths(work, root)
    (work / "home").unlink()
    outside = tmp_path / "outside"
    outside.write_bytes(b"external export")
    os.link(outside, work / "runtime-requirements.txt")
    with pytest.raises(ValueError, match="singly linked"):
        runner.admit_paths(work, root)
    assert outside.read_bytes() == b"external export"


@pytest.mark.parametrize("mode", ["--prepare-only", "--run-only"])
def test_main_rejects_unsafe_work_before_any_attempt(mode, prepared_case, tmp_path, monkeypatch):
    runner, work, root, invoke = prepared_case
    outside = tmp_path / "outside"
    outside.write_bytes(b"external bytes")
    (work / ".env.local").symlink_to(outside)
    monkeypatch.setattr(runner, "prepare", lambda *args: pytest.fail("unsafe preparation admitted"))
    with pytest.raises(ValueError, match="symlink"):
        invoke(mode)
    assert outside.read_bytes() == b"external bytes"
    assert not list(root.glob("run-*"))


@pytest.mark.parametrize(
    "location",
    [
        "work",
        "work-child",
        "work-ancestor",
        "source",
        "source-ancestor",
        "normalized-source",
        "work-source",
        "artifact-symlink",
        "lexical-symlink",
    ],
)
def test_root_admission_rejects_overlap_and_lexical_links(location, tmp_path, monkeypatch, e2e_runner):
    runner = e2e_runner
    source, work, root = tmp_path / "checkout", tmp_path / "work", tmp_path / "results"
    source.mkdir()
    work.mkdir()
    (source / "tracked.txt").write_text("unchanged")
    (tmp_path / "existing").mkdir()
    monkeypatch.setattr(runner, "ROOT", source)
    if location == "work":
        root = work
    elif location == "work-child":
        root = work / "child"
    elif location == "work-ancestor":
        root = tmp_path
    elif location == "source":
        root = source
    elif location == "source-ancestor":
        work = tmp_path
    elif location == "normalized-source":
        work = tmp_path / "existing/../checkout/work"
    elif location == "work-source":
        work = source
    elif location == "artifact-symlink":
        root = tmp_path / "output-link"
        root.symlink_to(source, target_is_directory=True)
    else:
        (tmp_path / "link").symlink_to(tmp_path / "existing", target_is_directory=True)
        work = tmp_path / "link/../work"
    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    with pytest.raises(ValueError):
        runner.admit_paths(work, root)
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*")) == before
    assert (source / "tracked.txt").read_text() == "unchanged"


@pytest.mark.parametrize("receipt", [None, "not-json", "[]", "{}", '{"assets": null}'])
def test_run_only_readiness_failure_leaves_existing_results_unchanged(receipt, prepared_case):
    _, work, root, invoke = prepared_case
    identity = work / "installed.json"
    identity.unlink() if receipt is None else identity.write_text(receipt)
    with pytest.raises(SystemExit) as failure:
        invoke("--run-only")
    assert failure.value.code == 2
    assert list(root.iterdir()) == [root / "operator-note.txt"]


@pytest.mark.parametrize(
    "name", ["runtime", "harness", "suite", "cases", ".env", ".env.local", "runtime-requirements.txt", "installed.json"]
)
def test_selected_input_must_survive_work_mutation(name, prepared_case):
    import argparse

    runner, work, root, _ = prepared_case
    if name in {"runtime", "harness", "suite", "cases"}:
        directory = work / name
        directory.mkdir()
        wheel = directory / "selected.whl"
    else:
        wheel = work / name
    wheel.write_bytes(b"original wheel")
    args = argparse.Namespace(wheel=wheel, wheel_dir=None, run_only=False)
    with pytest.raises(ValueError, match="wheel"):
        runner.validate_run_inputs(work, args, argparse.ArgumentParser())
    assert wheel.read_bytes() == b"original wheel"
    assert not list(root.glob("run-*"))


def test_existing_output_inputs_and_normal_venv_links_remain_usable(prepared_case, monkeypatch):
    runner, work, root, invoke = prepared_case
    (work / "runtime/bin").mkdir(parents=True)
    (work / "runtime/bin/python").symlink_to(sys.executable)
    wheel = root / "previous-result.whl"
    wheel.write_bytes(b"original wheel")
    selected = []
    monkeypatch.setattr(
        runner, "prepare", lambda work, args, parser: selected.append(runner.archive(work, args, parser))
    )
    assert invoke("--prepare-only", "--wheel", str(wheel)) == 0
    assert selected == [wheel.resolve()]
    assert wheel.read_bytes() == b"original wheel"
    assert (work / "runtime/bin/python").is_symlink()
    assert not list(root.glob("run-*"))


def test_safe_parent_alias_uses_canonical_installed_import(tmp_path, monkeypatch, e2e_runner):
    runner = e2e_runner
    (tmp_path / "existing").mkdir()
    work = tmp_path / "existing/../work"
    work.mkdir()
    (work / "runtime").mkdir()
    wheel = tmp_path / "original.whl"
    wheel.write_bytes(b"original wheel")
    installed = {"module": str((work / "runtime/lib/usersim/__init__.py").resolve()), "assets": "assets"}
    monkeypatch.setattr(runner, "command", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *args, **kwargs: json.dumps(installed))
    runner.record_install(work, wheel, {})
    monkeypatch.setattr(runner, "prepare", lambda work, *args: runner.record_install(work, wheel, {}))
    monkeypatch.setattr(
        sys, "argv", [str(RUNNER), "--prepare-only", "--work-dir", str(work), "--artifacts", str(tmp_path / "results")]
    )
    assert runner.main() == 0
    assert json.loads((work / "installed.json").read_text())["module"] == installed["module"]
