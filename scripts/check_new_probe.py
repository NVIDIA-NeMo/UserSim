# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that a new built-in probe, added the documented way, passes the shared tests.

Copies the repository's files into a scratch folder and adds a probe there built
from ``templates/extension_package``: its package under
``src/usersim/engine/probes/``, its tests and ``contract.py`` under
``tests/probes/``, one line in ``BUILTIN_PACKAGES`` and one row in the
``docs/probes.md`` catalog. It then runs the whole test suite in that copy.

The suite must pass, apart from the new probe's trajectory pin, which a probe's
author sets from the test's failure message. That test must fail and name the
probe's ``contract.py``. So a shared test that lists the probes by hand, or that
a probe built from the template can't pass, fails this check rather than the
first contributor who adds a probe.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABEL = "zz_ci_probe"
PIN_TEST = f"tests/engine/core/test_episode_input.py::test_trajectory_id_is_pinned_per_probe[{LABEL}]"
CONTRACT = f"tests/probes/{LABEL}/contract.py"


def copy_repository(dest: Path) -> None:
    """Copy the tracked and untracked-but-not-ignored files, as they are in the working tree."""
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout.decode()
    for name in filter(None, listed.split("\0")):
        source = ROOT / name
        if source.is_file():
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def _rename(text: str) -> str:
    """The template's names, as the new probe's: ``my_probe`` becomes the label, ``my_`` its prefix."""
    text = text.replace("my_probe", LABEL).replace("MyProbe", "ZzCiProbe")
    return re.sub(r"\bmy_", "zz_ci_", text)


def add_probe(repo: Path) -> None:
    """Add the template's probe as a built-in, following ``docs/engine/AUTHORING_A_PROBE.md``."""
    template = repo / "templates" / "extension_package"
    package = repo / "src" / "usersim" / "engine" / "probes" / LABEL
    shutil.copytree(template / "src" / "my_probe", package)
    (package / "assets" / "my_probe").rename(package / "assets" / LABEL)
    for path in package.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".yaml"}:
            path.write_text(_rename(path.read_text(encoding="utf-8")), encoding="utf-8")

    tests = repo / "tests" / "probes" / LABEL
    tests.mkdir()
    (tests / "__init__.py").write_text("", encoding="utf-8")
    test_text = _rename((template / "tests" / "test_my_probe.py").read_text(encoding="utf-8"))
    # In this repository the probe is a package under usersim.engine.probes, and
    # BUILTIN_PACKAGES registers it, so there are no entry points to test.
    test_text = re.sub(rf"^from {LABEL}\.", f"from usersim.engine.probes.{LABEL}.", test_text, flags=re.MULTILINE)
    test_text = test_text.replace("from importlib.metadata import entry_points\n", "")
    test_text, removed = re.subn(r"def test_the_entry_points_\w+\(\):\n(?:    .*\n)+\n\n", "", test_text)
    if removed != 1:
        raise SystemExit("the template's tests no longer have one entry-point test to leave out; update this script")
    (tests / "test_probe.py").write_text(test_text, encoding="utf-8")
    (tests / "contract.py").write_text(
        "from _probe_contract import ParityCase, ProbeContract\n\n"
        "CONTRACTS = {\n"
        f'    "{LABEL}": ProbeContract(trajectory_id="0000000000000000", succeeds_on=None, parity=(ParityCase(1001),)),\n'
        "}\n",
        encoding="utf-8",
    )

    roster = repo / "src" / "usersim" / "engine" / "core" / "builtin_packages.py"
    text, added = re.subn(
        r"(BUILTIN_PACKAGES: tuple\[str, \.\.\.\] = \(\n(?:    .*\n)*?)\)", rf'\1    "{LABEL}",\n)', roster.read_text()
    )
    if added != 1:
        raise SystemExit(f"could not find the BUILTIN_PACKAGES tuple in {roster.relative_to(repo)}")
    roster.write_text(text, encoding="utf-8")

    catalog = repo / "docs" / "probes.md"
    lines = catalog.read_text(encoding="utf-8").splitlines(keepends=True)
    header = lines.index("| Family | Probe | Catches |\n")
    end = next(i for i in range(header + 2, len(lines)) if not lines[i].startswith("|"))
    lines.insert(end, f"| General | `{LABEL}` | A probe added from the extension template |\n")
    catalog.write_text("".join(lines), encoding="utf-8")


def run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k not in {"VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"}}
    return subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", default="auto", help="pytest-xdist workers for the suite (default: auto)")
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="usersim-new-probe-") as scratch:
        repo = Path(scratch)
        print(f"→ copying the repository to {repo} and adding the built-in probe {LABEL!r}")
        copy_repository(repo)
        add_probe(repo)
        # Some tests read the git index, as a checkout has one.
        for command in (["git", "init", "-q"], ["git", "add", "-A"]):
            subprocess.run(command, cwd=repo, check=True)

        print("→ installing the copy's environment")
        synced = run(["uv", "sync", "--locked", "--quiet"], repo)
        if synced.returncode:
            print(synced.stdout + synced.stderr, file=sys.stderr)
            return 1

        print("→ running the test suite with the new probe")
        suite = run(
            ["uv", "run", "pytest", "-n", args.workers, "-q", "-p", "no:cacheprovider", "--deselect", PIN_TEST], repo
        )
        if suite.returncode:
            print(suite.stdout[-20000:] + suite.stderr[-5000:], file=sys.stderr)
            print("✗ a probe added from the template fails the shared tests (see above)", file=sys.stderr)
            return 1

        print("→ checking that its trajectory pin fails and says where to set it")
        pin = run(["uv", "run", "pytest", "-q", "-p", "no:cacheprovider", PIN_TEST], repo)
        if pin.returncode == 0 or CONTRACT not in pin.stdout:
            print(pin.stdout[-5000:] + pin.stderr[-2000:], file=sys.stderr)
            print(f"✗ the new probe's trajectory pin should fail and name {CONTRACT}", file=sys.stderr)
            return 1

    print("✓ a probe added from the template passes the shared tests; only its trajectory pin is left to set")
    return 0


if __name__ == "__main__":
    sys.exit(main())
