# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the NGC CLI bootstrap helpers + ``usersim setup-ngc``.

All tests run offline. The real NGC release zip is replaced with a
fake fixture (a synthetic ``ngc-cli/`` directory containing a stub
launcher that prints ``NGC CLI 0.0.0-test``); ``urllib.request.urlopen``
is monkeypatched to return that fake zip from disk so no network call
is made.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
import textwrap
import zipfile
from pathlib import Path

import pytest

from usersim.cli import _ngc


# ---------------------------------------------------------------------------
# Fake NGC release fixture
# ---------------------------------------------------------------------------


def _build_fake_ngc_zip(dest: Path) -> str:
    """Build a tiny ``ngccli_*.zip`` fixture; return its sha256."""
    payload_root = dest.parent / "_payload"
    if payload_root.exists():
        shutil.rmtree(payload_root)
    ngc_dir = payload_root / "ngc-cli"
    ngc_dir.mkdir(parents=True)

    launcher = ngc_dir / "ngc"
    launcher.write_text(
        textwrap.dedent(
            """\
            #!/bin/bash
            # Stub NGC launcher used in tests. The real launcher relies on its
            # sibling ``ngc-cli.python`` interpreter; this stub just answers
            # --version so ngc_available()'s probe passes.
            if [ "$1" = "--version" ]; then
                echo "NGC CLI 0.0.0-test"
                exit 0
            fi
            echo "stub-ngc: $@" 1>&2
            exit 1
            """
        )
    )
    launcher.chmod(0o755)
    (ngc_dir / "ngc-cli.python").write_bytes(b"stub-interpreter")
    (ngc_dir / "README.txt").write_text("stub")

    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in payload_root.rglob("*"):
            zf.write(path, path.relative_to(payload_root))

    return _ngc._sha256_file(dest)


# ---------------------------------------------------------------------------
# _resolve_release / arch normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "system, machine, expected_zip",
    [
        ("Linux", "x86_64", "ngccli_linux.zip"),
        ("Linux", "aarch64", "ngccli_arm64.zip"),
        ("Linux", "arm64", "ngccli_arm64.zip"),  # alias normalization
    ],
)
def test_resolve_release_supported(monkeypatch, system, machine, expected_zip):
    monkeypatch.setattr(_ngc.platform, "system", lambda: system)
    monkeypatch.setattr(_ngc.platform, "machine", lambda: machine)
    zip_filename, sha = _ngc._resolve_release()
    assert zip_filename == expected_zip
    assert len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)


@pytest.mark.parametrize(
    "system, machine",
    [
        ("Darwin", "arm64"),  # Mac is .pkg, not zip — refuse
        ("Darwin", "x86_64"),
        ("Windows", "AMD64"),
        ("Linux", "i686"),
        ("FreeBSD", "amd64"),
    ],
)
def test_resolve_release_unsupported(monkeypatch, system, machine):
    monkeypatch.setattr(_ngc.platform, "system", lambda: system)
    monkeypatch.setattr(_ngc.platform, "machine", lambda: machine)
    with pytest.raises(NotImplementedError) as excinfo:
        _ngc._resolve_release()
    assert "manual" in str(excinfo.value).lower() or "ngc.nvidia.com" in str(excinfo.value).lower()


def test_resolved_release_returns_url_and_sha(monkeypatch):
    monkeypatch.setattr(_ngc.platform, "system", lambda: "Linux")
    monkeypatch.setattr(_ngc.platform, "machine", lambda: "x86_64")
    zip_filename, url, sha = _ngc.resolved_release()
    assert zip_filename == "ngccli_linux.zip"
    assert url.startswith("https://api.ngc.nvidia.com/")
    assert _ngc.NGC_VERSION in url
    assert zip_filename in url
    assert len(sha) == 64


# ---------------------------------------------------------------------------
# _require_venv
# ---------------------------------------------------------------------------


def test_require_venv_returns_prefix_in_venv(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.setattr(sys, "base_prefix", "/some/other/system/prefix")
    assert _ngc._require_venv() == Path(tmp_path)


def test_require_venv_refuses_outside_venv(monkeypatch):
    monkeypatch.setattr(sys, "prefix", "/usr")
    monkeypatch.setattr(sys, "base_prefix", "/usr")
    with pytest.raises(RuntimeError, match="virtualenv"):
        _ngc._require_venv()


# ---------------------------------------------------------------------------
# ensure_ngc_cli — happy path with mocked download
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _patched_download(monkeypatch, fake_zip: Path, expected_sha: str):
    """Monkeypatch the network download + the per-arch sha pin so the install
    logic uses our fixture zip instead of fetching from NGC."""

    def fake_urlopen(url, *args, **kwargs):  # noqa: ARG001
        return open(fake_zip, "rb")

    monkeypatch.setattr(_ngc.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(_ngc.platform, "system", lambda: "Linux")
    monkeypatch.setattr(_ngc.platform, "machine", lambda: "x86_64")
    monkeypatch.setitem(_ngc._RELEASES, ("Linux", "x86_64"), ("ngccli_linux.zip", expected_sha))
    yield


def test_ensure_ngc_cli_installs_when_missing(monkeypatch, tmp_path):
    fake_zip = tmp_path / "ngccli_linux.zip"
    expected_sha = _build_fake_ngc_zip(fake_zip)

    target_dir = tmp_path / "venv" / "bin"
    target_dir.parent.mkdir()

    monkeypatch.setattr(_ngc, "ngc_available", lambda: False)
    pre_path = "/usr/bin:/bin"
    monkeypatch.setenv("PATH", pre_path)

    with _patched_download(monkeypatch, fake_zip, expected_sha):
        result_path = _ngc.ensure_ngc_cli(target_dir=target_dir)

    # Symlink at <venv>/bin/ngc -> ../opt/ngc-cli/ngc.
    assert result_path == target_dir / "ngc"
    assert result_path.is_symlink()
    assert os.readlink(result_path) == str(Path("..") / "opt" / "ngc-cli" / "ngc")

    # Real launcher exists, is executable, and runs.
    payload_launcher = target_dir.parent / "opt" / "ngc-cli" / "ngc"
    assert payload_launcher.is_file()
    assert payload_launcher.stat().st_mode & 0o111

    # PATH was prepended for the current process.
    assert os.environ["PATH"].split(os.pathsep)[0] == str(target_dir)
    assert "/usr/bin" in os.environ["PATH"]  # original entries preserved


def test_ensure_ngc_cli_noop_when_present(monkeypatch, tmp_path):
    """When ``ngc_available()`` returns True we must not download."""
    monkeypatch.setattr(_ngc, "ngc_available", lambda: True)

    fake_ngc = tmp_path / "ngc"
    fake_ngc.write_text("#!/bin/sh\nexit 0\n")
    fake_ngc.chmod(0o755)
    monkeypatch.setattr(_ngc.shutil, "which", lambda name: str(fake_ngc) if name == "ngc" else None)

    def boom(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("urlopen should not be called when ngc is already present")

    monkeypatch.setattr(_ngc.urllib.request, "urlopen", boom)

    result = _ngc.ensure_ngc_cli(target_dir=tmp_path / "unused")
    assert result == fake_ngc


def test_ensure_ngc_cli_force_reinstalls(monkeypatch, tmp_path):
    """``force=True`` skips the early-return even when ngc is on PATH."""
    fake_zip = tmp_path / "ngccli_linux.zip"
    expected_sha = _build_fake_ngc_zip(fake_zip)
    target_dir = tmp_path / "venv" / "bin"

    monkeypatch.setattr(_ngc, "ngc_available", lambda: True)  # pretend already installed
    with _patched_download(monkeypatch, fake_zip, expected_sha):
        _ngc.ensure_ngc_cli(force=True, target_dir=target_dir)

    assert (target_dir / "ngc").is_symlink()


def test_ensure_ngc_cli_raises_on_sha_mismatch(monkeypatch, tmp_path):
    fake_zip = tmp_path / "ngccli_linux.zip"
    _build_fake_ngc_zip(fake_zip)

    target_dir = tmp_path / "venv" / "bin"
    monkeypatch.setattr(_ngc, "ngc_available", lambda: False)

    bogus_sha = "0" * 64
    with _patched_download(monkeypatch, fake_zip, bogus_sha):
        with pytest.raises(RuntimeError, match="sha256 mismatch"):
            _ngc.ensure_ngc_cli(target_dir=target_dir)

    # Nothing was symlinked into the venv.
    assert not (target_dir / "ngc").exists()


def test_ensure_ngc_cli_replaces_existing_symlink(monkeypatch, tmp_path):
    """Re-running install should replace the prior symlink atomically (--force path)."""
    fake_zip = tmp_path / "ngccli_linux.zip"
    expected_sha = _build_fake_ngc_zip(fake_zip)
    target_dir = tmp_path / "venv" / "bin"
    target_dir.mkdir(parents=True)

    # Pre-existing stale symlink at the target.
    stale = target_dir / "ngc"
    os.symlink("/nonexistent", stale)
    assert stale.is_symlink()

    monkeypatch.setattr(_ngc, "ngc_available", lambda: False)
    with _patched_download(monkeypatch, fake_zip, expected_sha):
        _ngc.ensure_ngc_cli(target_dir=target_dir)

    assert os.readlink(stale) == str(Path("..") / "opt" / "ngc-cli" / "ngc")


def test_ensure_ngc_cli_fast_path_finds_venv_install(monkeypatch, tmp_path):
    """If <sys.prefix>/bin/ngc exists but is not on PATH (Jupyter kernel
    launched outside the activated venv), we must add it to PATH and
    reuse it instead of re-downloading."""
    fake_venv = tmp_path / "venv"
    venv_bin = fake_venv / "bin"
    venv_bin.mkdir(parents=True)
    fake_ngc = venv_bin / "ngc"
    fake_ngc.write_text("#!/bin/sh\nexit 0\n")
    fake_ngc.chmod(0o755)

    monkeypatch.setattr(sys, "prefix", str(fake_venv))
    monkeypatch.setattr(sys, "base_prefix", "/some/other/system/prefix")

    # Initial probe fails because PATH doesn't have venv/bin yet.
    pre_path = "/usr/bin:/bin"
    monkeypatch.setenv("PATH", pre_path)

    call_count = {"available": 0}

    def fake_available():
        call_count["available"] += 1
        # First call: kernel PATH doesn't see ngc. Subsequent calls
        # (after we prepend venv_bin to PATH): see it.
        return str(venv_bin) in os.environ.get("PATH", "")

    monkeypatch.setattr(_ngc, "ngc_available", fake_available)

    def boom(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("urlopen should not be called when venv install exists")

    monkeypatch.setattr(_ngc.urllib.request, "urlopen", boom)

    result = _ngc.ensure_ngc_cli()
    assert result == fake_ngc
    assert os.environ["PATH"].split(os.pathsep)[0] == str(venv_bin)
    assert call_count["available"] >= 2  # initial probe + post-prepend probe


def test_ensure_ngc_cli_fast_path_falls_through_when_binary_broken(monkeypatch, tmp_path):
    """If <sys.prefix>/bin/ngc exists but ngc --version fails (corrupted
    install), we must fall through to the full download path."""
    fake_venv = tmp_path / "venv"
    venv_bin = fake_venv / "bin"
    venv_bin.mkdir(parents=True)
    broken_ngc = venv_bin / "ngc"
    broken_ngc.write_text("not a real binary")
    broken_ngc.chmod(0o755)

    monkeypatch.setattr(sys, "prefix", str(fake_venv))
    monkeypatch.setattr(sys, "base_prefix", "/some/other/system/prefix")
    monkeypatch.setattr(_ngc, "ngc_available", lambda: False)  # both probes fail

    fake_zip = tmp_path / "ngccli_linux.zip"
    expected_sha = _build_fake_ngc_zip(fake_zip)

    with _patched_download(monkeypatch, fake_zip, expected_sha):
        result = _ngc.ensure_ngc_cli(target_dir=venv_bin)

    # Full install ran: payload extracted, symlink replaced.
    assert result.is_symlink()
    assert (fake_venv / "opt" / "ngc-cli" / "ngc").is_file()


def test_setup_ngc_handles_url_error(monkeypatch, capsys):
    """A network failure in `_download` must surface as a friendly CLI
    error with proxy hint, not a raw urllib traceback."""
    from usersim.cli import main

    monkeypatch.setattr(_ngc, "ngc_available", lambda: False)
    monkeypatch.setattr(_ngc.platform, "system", lambda: "Linux")
    monkeypatch.setattr(_ngc.platform, "machine", lambda: "x86_64")
    # Force venv detection so we get past _require_venv.
    monkeypatch.setattr(sys, "prefix", "/tmp/fake_venv_for_test")
    monkeypatch.setattr(sys, "base_prefix", "/some/other/system/prefix")

    import urllib.error as urlerror

    def fake_urlopen(url, *args, **kwargs):  # noqa: ARG001
        raise urlerror.URLError("Name or service not known")

    monkeypatch.setattr(_ngc.urllib.request, "urlopen", fake_urlopen)

    rc = main(["setup-ngc"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "network error" in err.lower()
    assert "proxy" in err.lower()  # hint visible to user
    assert "Traceback" not in err


def test_setup_ngc_handles_unsupported_platform(monkeypatch, capsys):
    """NotImplementedError from _resolve_release must surface as rc=2 with
    the manual-install URL, not a raw traceback."""
    from usersim.cli import main

    monkeypatch.setattr(_ngc, "ngc_available", lambda: False)
    monkeypatch.setattr(_ngc.platform, "system", lambda: "Windows")
    monkeypatch.setattr(_ngc.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(sys, "prefix", "/tmp/fake_venv_for_test")
    monkeypatch.setattr(sys, "base_prefix", "/some/other/system/prefix")

    rc = main(["setup-ngc"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "Windows" in err
    assert "ngc.nvidia.com" in err
    assert "Traceback" not in err


def test_setup_ngc_no_key_succeeds_with_optional_note(monkeypatch, capsys):
    """Without an NGC key, setup-ngc still succeeds (rc 0) and prints an
    optional next-step note — not an error-toned discovery failure."""
    from pathlib import Path

    from usersim.cli import main

    monkeypatch.delenv("NGC_CLI_API_KEY", raising=False)
    monkeypatch.delenv("NGC_API_KEY", raising=False)
    monkeypatch.setattr(_ngc, "ensure_ngc_cli", lambda *a, **k: Path("/venv/bin/ngc"))

    # Discovery must not run without a key.
    def _no_discovery():
        raise AssertionError("ensure_ngc_org must not run without an NGC key")

    monkeypatch.setattr(_ngc, "ensure_ngc_org", _no_discovery)

    rc = main(["setup-ngc"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "NGC CLI ready" in captured.out
    assert "Next step" in captured.out
    # The error-toned line must not appear anywhere.
    assert "cannot discover org" not in (captured.out + captured.err)


# ---------------------------------------------------------------------------
# ensure_ngc_org — auto-discovery via `ngc org list`
# ---------------------------------------------------------------------------


def test_parse_first_org_extracts_name_column():
    csv_output = textwrap.dedent(
        """\
        Name,Display Name
        my-personal-org,My Personal Org
        nvidia-team,NVIDIA Team
        """
    )
    assert _ngc._parse_first_org(csv_output) == "my-personal-org"


def test_parse_first_org_handles_quoted_display_names():
    csv_output = textwrap.dedent(
        """\
        Name,Display Name
        my-org-id,"Last, First Org"
        """
    )
    assert _ngc._parse_first_org(csv_output) == "my-org-id"


def test_parse_first_org_returns_none_on_empty():
    assert _ngc._parse_first_org("") is None
    assert _ngc._parse_first_org("Name\n") is None


def test_parse_first_org_falls_back_to_first_column_if_schema_shifts():
    """If the header doesn't have a 'name' column, we fall back to col 0."""
    csv_output = textwrap.dedent(
        """\
        OrgIdentifier,Display
        my-org,My Org
        """
    )
    assert _ngc._parse_first_org(csv_output) == "my-org"


def test_ensure_ngc_org_uses_existing_env_var(monkeypatch):
    monkeypatch.setenv("NGC_CLI_ORG", "preset-org")

    def boom(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("subprocess.run should not be called when NGC_CLI_ORG is set")

    monkeypatch.setattr(_ngc.subprocess, "run", boom)
    assert _ngc.ensure_ngc_org() == "preset-org"
    assert os.environ["NGC_CLI_ORG"] == "preset-org"


def test_ensure_ngc_org_discovers_via_ngc_org_list(monkeypatch):
    monkeypatch.delenv("NGC_CLI_ORG", raising=False)
    monkeypatch.delenv("NGC_ORG", raising=False)
    monkeypatch.setenv("NGC_CLI_API_KEY", "nvapi-fake")
    monkeypatch.setattr(_ngc.shutil, "which", lambda name: "/path/ngc" if name == "ngc" else None)

    fake_csv = "Name,Display Name\nauto-discovered-org,Some Display Name\n"

    def fake_run(cmd, **kwargs):
        assert cmd[0:3] == ["ngc", "org", "list"]
        result = subprocess_module.CompletedProcess(args=cmd, returncode=0, stdout=fake_csv, stderr="")
        return result

    import subprocess as subprocess_module

    monkeypatch.setattr(_ngc.subprocess, "run", fake_run)

    org = _ngc.ensure_ngc_org()
    assert org == "auto-discovered-org"
    assert os.environ["NGC_CLI_ORG"] == "auto-discovered-org"
    assert os.environ.get("NGC_ORG") == "auto-discovered-org"


def test_ensure_ngc_org_returns_none_when_no_api_key(monkeypatch, capsys):
    monkeypatch.delenv("NGC_CLI_ORG", raising=False)
    monkeypatch.delenv("NGC_ORG", raising=False)
    monkeypatch.delenv("NGC_CLI_API_KEY", raising=False)
    monkeypatch.delenv("NGC_API_KEY", raising=False)
    monkeypatch.setattr(_ngc.shutil, "which", lambda name: "/path/ngc" if name == "ngc" else None)

    org = _ngc.ensure_ngc_org()
    assert org is None
    err = capsys.readouterr().err
    assert "NGC_CLI_API_KEY not set" in err
    assert "api-keys" in err  # points at where to generate the key
    # The message must not drag NVIDIA_API_KEY into NGC auth — separate
    # credentials. Guard against re-introducing that conflation.
    assert "NVIDIA_API_KEY" not in err


def test_ensure_ngc_org_ignores_nvidia_api_key(monkeypatch, capsys):
    """NVIDIA_API_KEY (inference) is unrelated to NGC auth and must not satisfy it."""
    monkeypatch.delenv("NGC_CLI_ORG", raising=False)
    monkeypatch.delenv("NGC_ORG", raising=False)
    monkeypatch.delenv("NGC_CLI_API_KEY", raising=False)
    monkeypatch.delenv("NGC_API_KEY", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-inference-key")
    monkeypatch.setattr(_ngc.shutil, "which", lambda name: "/path/ngc" if name == "ngc" else None)

    called = False

    def fake_run(cmd, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("org list must not run without an NGC credential")

    monkeypatch.setattr(_ngc.subprocess, "run", fake_run)

    org = _ngc.ensure_ngc_org()
    assert org is None
    assert called is False
    # NVIDIA_API_KEY must not be copied into the NGC var.
    assert os.environ.get("NGC_CLI_API_KEY") is None


def test_ensure_ngc_org_missing_org_hint_is_specific(monkeypatch, capsys):
    """The 'Missing org' failure surfaces the fresh-host `ngc config set` hint."""
    monkeypatch.delenv("NGC_CLI_ORG", raising=False)
    monkeypatch.delenv("NGC_ORG", raising=False)
    monkeypatch.setenv("NGC_CLI_API_KEY", "nvapi-fake")
    monkeypatch.setattr(_ngc.shutil, "which", lambda name: "/path/ngc" if name == "ngc" else None)

    import subprocess as subprocess_module

    def fake_run(cmd, **kwargs):
        raise subprocess_module.CalledProcessError(
            returncode=1,
            cmd=cmd,
            output="",
            stderr="Missing org - If Authenticated, org is also required.\n",
        )

    monkeypatch.setattr(_ngc.subprocess, "run", fake_run)

    org = _ngc.ensure_ngc_org()
    assert org is None
    err = capsys.readouterr().err
    assert "ngc config set" in err
    # The clean fresh-host message, not the raw exception / command array.
    assert "no NGC org configured" in err
    assert "Command '[" not in err


def test_ensure_ngc_org_returns_none_when_ngc_missing(monkeypatch, capsys):
    monkeypatch.delenv("NGC_CLI_ORG", raising=False)
    monkeypatch.delenv("NGC_ORG", raising=False)
    monkeypatch.setenv("NGC_CLI_API_KEY", "nvapi-fake")
    monkeypatch.setattr(_ngc.shutil, "which", lambda name: None)

    org = _ngc.ensure_ngc_org()
    assert org is None
    err = capsys.readouterr().err
    assert "NGC CLI not on PATH" in err


def test_ensure_ngc_org_returns_none_when_ngc_org_list_fails(monkeypatch, capsys):
    monkeypatch.delenv("NGC_CLI_ORG", raising=False)
    monkeypatch.delenv("NGC_ORG", raising=False)
    monkeypatch.setenv("NGC_CLI_API_KEY", "nvapi-fake")
    monkeypatch.setattr(_ngc.shutil, "which", lambda name: "/path/ngc" if name == "ngc" else None)

    import subprocess as subprocess_module

    def fake_run(cmd, **kwargs):
        raise subprocess_module.CalledProcessError(returncode=1, cmd=cmd, output="", stderr="bad api key")

    monkeypatch.setattr(_ngc.subprocess, "run", fake_run)

    org = _ngc.ensure_ngc_org()
    assert org is None
    err = capsys.readouterr().err
    assert "ngc config set" in err


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_setup_ngc_check_only_returns_0_when_present(monkeypatch, capsys):
    from usersim.cli import main

    monkeypatch.setattr(_ngc, "ngc_available", lambda: True)
    monkeypatch.setattr(_ngc.shutil, "which", lambda name: "/some/path/ngc" if name == "ngc" else None)

    rc = main(["setup-ngc", "--check-only"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "/some/path/ngc" in out


def test_setup_ngc_check_only_returns_1_when_missing(monkeypatch, capsys):
    from usersim.cli import main

    monkeypatch.setattr(_ngc, "ngc_available", lambda: False)
    rc = main(["setup-ngc", "--check-only"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "NOT available" in err


def test_setup_ngc_dry_run_prints_url_and_sha(monkeypatch, capsys):
    from usersim.cli import main

    monkeypatch.setattr(_ngc.platform, "system", lambda: "Linux")
    monkeypatch.setattr(_ngc.platform, "machine", lambda: "x86_64")

    def boom(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("urlopen should not be called in --dry-run")

    monkeypatch.setattr(_ngc.urllib.request, "urlopen", boom)

    rc = main(["setup-ngc", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert _ngc.NGC_VERSION in out
    assert "ngccli_linux.zip" in out
    assert "https://api.ngc.nvidia.com/" in out


def test_setup_ngc_dry_run_reports_unsupported(monkeypatch, capsys):
    from usersim.cli import main

    monkeypatch.setattr(_ngc.platform, "system", lambda: "Windows")
    monkeypatch.setattr(_ngc.platform, "machine", lambda: "AMD64")

    rc = main(["setup-ngc", "--dry-run"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "Windows" in err
