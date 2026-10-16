# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bootstrap binds the installer signature to its pinned key and selected path."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RELEASE_KEY = "24853EC9F655CE80B48E6C3A8B81C9D17413A06D"


@pytest.fixture(scope="module")
def signing_material(tmp_path_factory):
    if not shutil.which("gpg"):
        pytest.skip("Mise bootstrap requires GPG")
    work = tmp_path_factory.mktemp("mise-signatures")
    home = work / "gnupg"
    home.mkdir(mode=0o700)
    command = ["gpg", "--homedir", str(home), "--batch", "--pinentry-mode", "loopback", "--passphrase", ""]
    keys = []
    for name in ("Pinned", "Other"):
        subprocess.run(
            [*command, "--quick-generate-key", f"{name} <{name.lower()}@example.test>", "ed25519", "sign", "0"],
            check=True,
            capture_output=True,
            timeout=30,
        )
        output = subprocess.check_output([*command, "--with-colons", "--list-keys", name], timeout=10).decode()
        keys.append(next(line.split(":")[9] for line in output.splitlines() if line.startswith("fpr:")))
    (work / "keys.asc").write_bytes(subprocess.check_output([*command, "--armor", "--export"], timeout=10))
    installer = work / "installer.sh"
    installer.write_text(
        "#!/bin/sh\n"
        'printf "#!/bin/sh\\nprintf \'2026.8.5 fixture\\\\n\'\\n" > "$MISE_INSTALL_PATH"\n'
        'chmod +x "$MISE_INSTALL_PATH"\n'
        'touch "$BOOTSTRAP_MARKER"\n'
    )
    for index, key in enumerate(keys):
        subprocess.run(
            [*command, "--local-user", key, "--output", str(work / f"installer-{index}.sig"), "--sign", str(installer)],
            check=True,
            capture_output=True,
            timeout=10,
        )
    yield work, keys[0]
    subprocess.run(["gpgconf", "--homedir", str(home), "--kill", "all"], check=False, capture_output=True, timeout=10)


@pytest.mark.parametrize("signer,success", [(0, True), (1, False)])
def test_mixed_key_bundle_requires_the_pinned_installer_signer(tmp_path, signing_material, signer, success):
    material, pinned = signing_material
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (tmp_path / ".mise.toml").write_text('min_version = "2026.8.5"\n')
    # The fixture owns both private keys. Substitute only the public-key pin in
    # its isolated copy so verification still runs through real GPG.
    script = (ROOT / "scripts/install_mise.sh").read_text().replace(RELEASE_KEY, pinned)
    (scripts / "install_mise.sh").write_text(script)
    curl = tmp_path / "curl"
    curl.write_text(
        f"#!{sys.executable}\nimport os, shutil, sys\n"
        "name = 'keys.asc' if 'by-fingerprint' in sys.argv[-1] else os.environ['SIGNED_FIXTURE']\n"
        f"shutil.copyfile({str(material)!r} + '/' + name, sys.argv[sys.argv.index('-o') + 1])\n"
    )
    curl.chmod(0o755)
    marker = tmp_path / "executed"
    destination = tmp_path / "custom-bin/mise"
    result = subprocess.run(
        ["bash", str(scripts / "install_mise.sh")],
        env={
            **os.environ,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "MISE_INSTALL_PATH": str(destination),
            "BOOTSTRAP_MARKER": str(marker),
            "SIGNED_FIXTURE": f"installer-{signer}.sig",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert (result.returncode == 0) is success, result.stderr
    assert marker.exists() is success
    if not success:
        assert "signer does not match" in result.stderr


def test_custom_install_path_is_used_by_the_task_wrapper(tmp_path):
    executable = tmp_path / "mise"
    executable.write_text('#!/bin/sh\nprintf "custom-mise %s\\n" "$1"\n')
    executable.chmod(0o755)
    env = {**os.environ, "MISE_INSTALL_PATH": str(executable)}
    env.pop("MISE_BIN", None)
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/mise"), "--version"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert result.stdout == "custom-mise --version\n"
