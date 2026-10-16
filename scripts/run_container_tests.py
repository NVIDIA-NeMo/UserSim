# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and run CPU validation without exposing the host checkout or credentials."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_logged(
    command: list[str], log: Path, *, timeout: int, cwd: Path = ROOT, env: dict[str, str] | None = None
) -> int:
    """Preserve diagnostics even when a command fails or times out."""
    print("Running:", " ".join(command), flush=True)
    with log.open("w") as stream:
        with subprocess.Popen(
            command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        ) as process:
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Stop descendants as well as Mise; leave no timed-out suite running.
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                print(f"Timed out after {timeout}s; see {log}", file=sys.stderr)
                return 124
    print(f"Exit {code}; log: {log}", flush=True)
    return code


def attempt_directory(root: Path) -> Path:
    if any(part.is_symlink() for part in (root, *root.parents)):
        raise ValueError("container artifact root must not use symlinks")
    root = root.resolve()
    if root == ROOT or root in ROOT.parents:
        raise ValueError("container artifact root must not overlap the checkout root or its ancestors")
    root.mkdir(parents=True, exist_ok=True)
    artifacts = Path(tempfile.mkdtemp(prefix="run-", dir=root))
    print(f"Container attempt artifacts: {artifacts}", flush=True)
    return artifacts


def cleanup_container(client: list[str], environment: dict[str, str], name: str, log: Path) -> int:
    """Record bounded cleanup diagnostics without replacing the primary exit status."""
    command = client + ["rm", "-f", name]
    code, status, stdout, stderr = 1, "failed", "", ""
    try:
        result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=30, check=False)
        code, stdout, stderr = result.returncode, result.stdout, result.stderr
        if (
            code in {0, 1}
            and not stdout.strip()
            and stderr.strip() == f"Error response from daemon: No such container: {name}"
        ):
            status = "already-absent"
        elif code == 0 and not stderr.strip():
            status = "removed"
    except subprocess.TimeoutExpired as error:
        code, status = 124, "timed-out"
        stdout, stderr = error.stdout or "", error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        stderr += "\nCleanup timed out after 30s"
    except OSError as error:
        stderr = str(error)
    try:
        log.write_text(
            f"Container: {name}\nStatus: {status}\nExit code: {code}\nstdout:\n{stdout}\nstderr:\n{stderr}\n"
        )
    except OSError as error:
        print(f"Could not retain cleanup diagnostics in {log}: {error}", file=sys.stderr)
        status = "failed"
    if status in {"removed", "already-absent"}:
        return 0
    print(
        f"Cleanup failed for {name} on unix://{LOCAL_SOCKET}; it may still be running. "
        f"Manual recovery required; see {log}",
        file=sys.stderr,
    )
    return 1


def inside(artifacts: Path = Path("/artifacts")) -> int:
    """Use the same Mise contracts as native validation."""
    artifacts.mkdir(exist_ok=True)
    common: dict[str, str] = {**os.environ, "COVERAGE_FILE": str(artifacts / ".coverage")}
    unit: dict[str, str] = {
        **common,
        "PYTEST_WORKERS": "2",
        "PYTEST_ARGS": f"--junitxml={artifacts / 'unit.xml'}",
        "COV_XML": str(artifacts / "coverage.xml"),
    }
    unit.pop("E2E_ARGS", None)
    unit.pop("WHEEL_DIR", None)
    e2e: dict[str, str] = {
        **common,
        "E2E_ARGS": shlex.join(["--work-dir", "/opt/e2e", "--run-only", "--artifacts", str(artifacts / "e2e")]),
    }
    codes = [
        run_logged([str(ROOT / "scripts/mise"), "run", task], artifacts / f"{name}.log", timeout=1500, env=environment)
        for name, task, environment in (("unit", "test:ci", unit), ("e2e", "test:e2e", e2e))
    ]
    return next((code for code in codes if code), 0)


LOCAL_SOCKET = Path("/var/run/docker.sock")


def local_client(config: Path) -> tuple[list[str], dict[str, str]]:
    """Admit only the local socket before running even a Docker probe."""
    selectors = sorted(
        name for name, value in os.environ.items() if value and name.startswith(("DOCKER_", "BUILDX_", "BUILDKIT_"))
    )
    if selectors:
        raise ValueError("Unset inherited Docker/BuildKit selectors: " + ", ".join(selectors))
    if not LOCAL_SOCKET.is_socket():
        raise ValueError(f"Required local Docker Unix socket unavailable: {LOCAL_SOCKET}")
    config.mkdir(exist_ok=True)
    environment = {
        name: value for name, value in os.environ.items() if not name.startswith(("DOCKER_", "BUILDX_", "BUILDKIT_"))
    }
    environment["DOCKER_CONFIG"] = str(config)
    endpoint = f"unix://{LOCAL_SOCKET}"
    return ["docker", "--config", str(config), "--host", endpoint], environment


def audit_context(artifacts: Path, *, timeout: int) -> int:
    """Export synthetic canaries using Docker's actual exclusion semantics."""
    with tempfile.TemporaryDirectory(prefix="usersim-context-audit-") as temporary:
        work = Path(temporary)
        client, environment = local_client(work / "docker-client")
        settings = {
            "endpoint": f"unix://{LOCAL_SOCKET}",
            "builder": "default",
            "driver": "docker",
            "config": str(work / "docker-client"),
            "inherited_selectors": "rejected",
        }
        (artifacts / "audit-settings.json").write_text(json.dumps(settings, indent=2) + "\n")
        # With an empty client config and explicit --host, the default docker
        # driver has no stored remote context or builder to select.
        code = run_logged(
            client + ["buildx", "inspect", "default"],
            artifacts / "audit-builder.log",
            timeout=min(timeout, 30),
            env=environment,
        )
        if code:
            return code
        inspection = (artifacts / "audit-builder.log").read_text()
        if not any(line.split() == ["Driver:", "docker"] for line in inspection.splitlines()):
            raise ValueError("Context audit requires the default docker driver")
        context, exported = work / "context", work / "exported"
        context.mkdir()
        shutil.copyfile(ROOT / ".dockerignore", context / ".dockerignore")
        # A scratch-only recipe avoids resolving or pulling the normal base image.
        (context / "Dockerfile").write_text("FROM scratch AS context-audit\nCOPY . /\n")
        cases = json.loads((ROOT / "tests/fixtures/exclusion-policy.json").read_text())
        if (
            not cases
            or not any(case["docker_included"] for case in cases)
            or not any(not case["docker_included"] for case in cases)
        ):
            raise ValueError("Context audit requires positive and negative examples")
        for case in cases:
            target = context / case["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("synthetic context canary\n")
        code = run_logged(
            client
            + [
                "build",
                "--builder",
                "default",
                "--network=none",
                "--target",
                "context-audit",
                "--output",
                f"type=local,dest={exported}",
                str(context),
            ],
            artifacts / "audit-export.log",
            timeout=min(timeout, 90),
            env=environment,
        )
        if code:
            return code
        failures = [case["path"] for case in cases if (exported / case["path"]).is_file() != case["docker_included"]]
        receipt = {"settings": settings, "checked": cases, "failures": failures}
        (artifacts / "audit-result.json").write_text(json.dumps(receipt, indent=2) + "\n")
        if failures:
            raise ValueError("Docker context policy mismatch: " + ", ".join(failures))
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "build", "run", "inside"))
    parser.add_argument("--image", default="usersim-tests:local")
    parser.add_argument(
        "--artifacts", type=Path, default=ROOT / "test-results/container", help="Root for fresh per-attempt output."
    )
    parser.add_argument("--timeout", type=int, default=3600)
    arguments = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args([*arguments, *shlex.split(os.environ.get("CONTAINER_ARGS", ""))])
    if args.action == "inside":
        return inside()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    artifacts = attempt_directory(args.artifacts.absolute())
    if args.action in ("audit", "build"):
        try:
            code = audit_context(artifacts, timeout=args.timeout)
            if code or args.action == "audit":
                return code
            with tempfile.TemporaryDirectory(prefix="usersim-docker-client-") as temporary:
                client, environment = local_client(Path(temporary) / "config")
                return run_logged(
                    client
                    + [
                        "build",
                        "--builder",
                        "default",
                        "--target",
                        "tests",
                        "-f",
                        "containers/Dockerfile.test",
                        "-t",
                        args.image,
                        ".",
                    ],
                    artifacts / "build.log",
                    timeout=args.timeout,
                    env=environment,
                )
        except (OSError, ValueError) as error:
            print(f"Container audit/build failed: {error}", file=sys.stderr)
            return 1
    try:
        with tempfile.TemporaryDirectory(prefix="usersim-docker-client-") as temporary:
            client, environment = local_client(Path(temporary) / "config")
            name = f"usersim-tests-{uuid.uuid4().hex[:12]}"
            command = client + [
                "run",
                "--rm",
                "--name",
                name,
                "--network=none",
                "--cpus=2",
                "--memory=8g",
                "--memory-swap=8g",
                "--pids-limit=512",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--tmpfs",
                "/tmp:rw,exec,nosuid,size=1g",
                "--mount",
                f"type=bind,source={artifacts},target=/artifacts",
                args.image,
            ]
            primary = 1
            try:
                primary = run_logged(command, artifacts / "container.log", timeout=args.timeout, env=environment)
            except (OSError, ValueError, subprocess.TimeoutExpired) as error:
                print(f"Container run failed for {name}: {error}", file=sys.stderr)
            finally:
                # A killed docker client can leave its container running after timeout.
                cleanup = cleanup_container(client, environment, name, artifacts / "cleanup.log")
            return primary or cleanup
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"Container run/cleanup failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
