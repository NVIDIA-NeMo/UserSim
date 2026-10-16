# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare and run installed-wheel CLI tests outside the source checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def command(args: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=cwd, env=env, check=True, timeout=600)


def environment(work: Path) -> dict[str, str]:
    # Runtime configuration is deliberately independent of the invoking shell.
    env = {name: os.environ[name] for name in ("PATH", "LANG", "LC_ALL", "TMPDIR") if name in os.environ}
    env.update(
        {
            "HOME": str(work / "home"),
            "TIKTOKEN_CACHE_DIR": str(work / "tokenizers"),
            "USERSIM_DISABLE_EXTENSIONS": "1",
            "USERSIM_MOVE_REASONING_EFFORT": "off",
            "DO_NOT_TRACK": "1",
            "NEMO_TELEMETRY_ENABLED": "false",
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "OTEL_SDK_DISABLED": "true",
            "E2E_API_KEY": "synthetic-local-key",
            "PYTHONNOUSERSITE": "1",
        }
    )
    Path(env["HOME"]).mkdir(exist_ok=True)
    return env


def selected_wheel(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Path | None:
    if args.wheel_dir:
        wheels = sorted(args.wheel_dir.resolve().glob("*.whl"))
        if len(wheels) != 1:
            parser.error(f"--wheel-dir expects exactly one wheel, found {len(wheels)} in {args.wheel_dir}")
        wheel = wheels[0].resolve(strict=True)
    elif args.wheel:
        wheel = args.wheel.resolve(strict=True)
    else:
        return None
    if not wheel.is_file():
        parser.error(f"selected wheel must be a regular file: {wheel}")
    return wheel


def archive(work: Path, args: argparse.Namespace, parser: argparse.ArgumentParser) -> Path:
    wheel = selected_wheel(args, parser)
    if wheel is not None:
        return wheel
    dist = work / "dist"
    dist.mkdir(exist_ok=True)
    command([args.uv, "build", "--wheel", "--out-dir", str(dist)], cwd=ROOT)
    wheels = list(dist.glob("*.whl"))
    if len(wheels) != 1:
        parser.error(f"expected exactly one wheel in {dist}, found {len(wheels)}")
    return wheels[0]


def record_install(work: Path, wheel: Path, env: dict[str, str]) -> None:
    runtime, suite = work / "runtime", work / "suite"
    python = runtime / "bin/python"
    # tiktoken downloads encoding data lazily. Download once during setup,
    # before the socket guard and container's network isolation take effect.
    command([str(python), "-c", "import tiktoken; tiktoken.get_encoding('cl100k_base')"], cwd=work, env=env)
    if suite.exists():
        shutil.rmtree(suite)
    shutil.copytree(ROOT / "tests_e2e", suite)
    metadata = subprocess.check_output(
        [
            str(python),
            "-c",
            "import json, usersim; from data_designer.engine.models.telemetry import TELEMETRY_ENABLED; assert not TELEMETRY_ENABLED; from usersim.engine.core._assets import packaged_assets_dir; print(json.dumps({'module': usersim.__file__, 'assets': str(packaged_assets_dir())}))",
        ],
        cwd=work,
        env=env,
        text=True,
        timeout=30,
    )
    installed = json.loads(metadata.strip().splitlines()[-1])
    if not Path(installed["module"]).resolve().is_relative_to(runtime.resolve()):
        raise RuntimeError(f"UserSim did not import from the runtime environment: {installed}")
    with wheel.open("rb") as stream:
        installed.update(wheel=str(wheel), wheel_sha256=hashlib.file_digest(stream, "sha256").hexdigest())
    (work / "installed.json").write_text(json.dumps(installed), encoding="utf-8")


def prepare(work: Path, args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    (work / "installed.json").unlink(missing_ok=True)
    runtime, harness = work / "runtime", work / "harness"
    python = runtime / "bin/python"
    env = environment(work)
    wheel = archive(work, args, parser)
    # Export locked runtime dependencies without the editable project or dev tools.
    requirements = work / "runtime-requirements.txt"
    command(
        [args.uv, "export", "--frozen", "--no-dev", "--no-emit-project", "--quiet", "--output-file", str(requirements)],
        cwd=ROOT,
    )
    for target in (runtime, harness):
        if target.exists():
            shutil.rmtree(target)
        command([args.uv, "venv", "--python", sys.executable, str(target)], cwd=work)
    command([args.uv, "pip", "sync", "--python", str(python), str(requirements)], cwd=work)
    command([args.uv, "pip", "install", "--python", str(python), "--no-deps", str(wheel)], cwd=work)
    command(
        [args.uv, "pip", "install", "--python", str(harness / "bin/python"), "pytest==9.0.2", "pyarrow==24.0.0"],
        cwd=work,
    )
    record_install(work, wheel, env)


def admit_paths(work: Path, artifacts: Path) -> tuple[Path, Path]:
    for path in (work, artifacts):
        if any(part.is_symlink() for part in (path, *path.parents)):
            raise ValueError("e2e work and artifact paths must not use symlinks")
    work, artifacts, source = work.resolve(), artifacts.resolve(), ROOT.resolve()
    if work == source or source in work.parents or work in source.parents:
        raise ValueError("e2e work directory must not overlap the checkout root or its ancestors")
    if artifacts == source or artifacts in source.parents:
        raise ValueError("artifact destination must not overlap the checkout root or its ancestors")
    if artifacts == work or work in artifacts.parents or artifacts in work.parents:
        raise ValueError("artifact destination must not overlap the e2e work directory")
    directories = [
        work,
        artifacts,
        *(work / name for name in ("home", "tokenizers", "dist", "runtime", "harness", "suite", "cases")),
    ]
    files = [work / name for name in (".env", ".env.local", "runtime-requirements.txt", "installed.json")]
    for path in [*directories, *files]:
        if path.is_symlink():
            raise ValueError(f"owned e2e work entry must not use symlinks: {path}")
        if path.exists():
            if path in directories and not path.is_dir():
                raise ValueError(f"owned e2e work entry must be a directory: {path}")
            if path in files and (not path.is_file() or path.stat().st_nlink != 1):
                raise ValueError(f"owned e2e work entry must be a singly linked regular file: {path}")
    return work, artifacts


def validate_run_inputs(work: Path, args: argparse.Namespace, parser: argparse.ArgumentParser) -> Path | None:
    wheel = selected_wheel(args, parser)
    if wheel is not None:
        targets = (work / name for name in ("runtime", "harness", "suite", "cases"))
        overwritten = (work / name for name in (".env", ".env.local", "runtime-requirements.txt", "installed.json"))
        if wheel in overwritten:
            raise ValueError(f"selected wheel would be overwritten by e2e setup: {wheel}")
        for target in targets:
            if wheel.is_relative_to(target):
                raise ValueError(f"selected wheel would be removed by e2e setup or execution: {wheel}")
    if args.run_only:
        receipt = work / "installed.json"
        if not receipt.is_file():
            parser.error("--run-only requires a prepared --work-dir")
        try:
            installed = json.loads(receipt.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            parser.error("--run-only requires a valid installed.json readiness receipt")
        if not isinstance(installed, dict) or not isinstance(installed.get("assets"), str) or not installed["assets"]:
            parser.error("--run-only readiness receipt requires an installed assets path")
    return wheel


def attempt_directory(work: Path, root: Path) -> Path:
    work, root = admit_paths(work, root)
    root.mkdir(parents=True, exist_ok=True)
    artifacts = Path(tempfile.mkdtemp(prefix="run-", dir=root))
    print(f"E2e attempt artifacts: {artifacts}", flush=True)
    return artifacts


def run_suite(work: Path, artifacts: Path, env: dict[str, str]) -> int:
    runtime, harness, suite = work / "runtime", work / "harness", work / "suite"
    shutil.copyfile(work / "installed.json", artifacts / "installed.json")
    cases = work / "cases"
    if cases.exists():
        shutil.rmtree(cases)
    cases.mkdir()
    env.update(
        {
            "E2E_RUNTIME": str(runtime),
            "E2E_ASSETS": json.loads((work / "installed.json").read_text())["assets"],
            "E2E_ARTIFACTS": str(cases),
            "PYTHONPATH": str(suite / "network_guard"),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        }
    )
    try:
        result = subprocess.run(
            [
                str(harness / "bin/python"),
                "-m",
                "pytest",
                "-c",
                str(suite / "pytest.ini"),
                str(suite),
                "-v",
                "--junitxml",
                str(artifacts / "junit.xml"),
            ],
            cwd=work,
            env=env,
            check=False,
            timeout=900,
        )
    except subprocess.TimeoutExpired:
        print("Installed CLI suite exceeded 900 seconds", file=sys.stderr)
        returncode = 1
    else:
        returncode = result.returncode
    finally:
        shutil.copytree(cases, artifacts, dirs_exist_ok=True)
    print(f"E2e work directory: {work}\nArtifacts: {artifacts}")
    return returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uv", default=os.environ.get("UV", "uv"), help="uv executable used for setup.")
    wheel_source = parser.add_mutually_exclusive_group()
    wheel_source.add_argument("--wheel", type=Path, help="Use an existing wheel instead of building one.")
    wheel_source.add_argument(
        "--wheel-dir", type=Path, help="Use the single existing wheel in this directory; never build."
    )
    parser.add_argument(
        "--work-dir", type=Path, help="Persistent external work directory (default: a new temporary one)."
    )
    parser.add_argument(
        "--artifacts", type=Path, default=ROOT / "output/e2e", help="Root for fresh per-attempt output."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--prepare-only", action="store_true", help="Install environments and tokenizer cache; do not run tests."
    )
    mode.add_argument(
        "--run-only", action="store_true", help="Run prepared environments without installing or downloading."
    )
    args = parser.parse_args()
    if args.run_only and (args.wheel or args.wheel_dir):
        parser.error("--run-only cannot be combined with --wheel or --wheel-dir; prepare that wheel first")
    work = (
        args.work_dir.absolute() if args.work_dir else Path(tempfile.gettempdir()) / f"usersim-e2e-{uuid.uuid4().hex}"
    )
    artifacts = args.artifacts.absolute()
    work, artifacts = admit_paths(work, artifacts)
    wheel = validate_run_inputs(work, args, parser)
    if wheel is not None:
        args.wheel, args.wheel_dir = wheel, None
    if not args.prepare_only:
        artifacts = attempt_directory(work, artifacts)
    work.mkdir(parents=True, exist_ok=True)
    # The CLI searches upward from each case cwd; stop at our owned boundary.
    for filename in (".env", ".env.local"):
        (work / filename).write_text("", encoding="utf-8")
    if not args.run_only:
        prepare(work, args, parser)
    if args.prepare_only:
        print(f"Prepared {work}. Run again with --work-dir {work} --run-only.")
        return 0
    return run_suite(work, artifacts, environment(work))


if __name__ == "__main__":
    raise SystemExit(main())
