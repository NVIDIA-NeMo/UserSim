# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NGC CLI bootstrap helpers.

The Nemotron-Personas datasets that this project's persona-sampling
pipeline consumes are hosted on NGC. ``data-designer download personas``
shells out to the ``ngc`` binary and silently no-ops with a "NGC CLI not
found" banner when the binary is missing (see
``data_designer.cli.utils.check_ngc_cli_available`` in the installed
``data-designer`` package). On a machine where the binary is not on
``$PATH``, the persona download therefore reports success while
downloading nothing, once per locale.

This module is the cross-platform installer that ``usersim setup-ngc``
and the notebook auto-install path both call. It pins a known-good NGC
CLI version (currently ``4.8.2``) along with a per-arch sha256 digest
so the install is reproducible: download → verify → extract into
``<venv>/opt/ngc-cli/`` → symlink ``<venv>/bin/ngc`` at the launcher.

To bump the pin: open
``https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/<new-version>/files``,
copy the per-zip ``sha256_base64`` (decode to hex) or run
``shasum -a 256`` on a freshly downloaded zip, and update
:data:`NGC_VERSION` and :data:`_RELEASES`.

Scope: Linux x86_64 and Linux aarch64 only. Upstream ships macOS as
``.pkg`` installers requiring ``sudo installer -pkg``, which cannot land
in a virtualenv without re-implementing pkg payload extraction, so this
errors out with a pointer to the manual install instead. Same for Windows.
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger("usersim.cli._ngc")


NGC_VERSION = "4.8.2"
"""Pinned NGC CLI version. Bump procedure documented in module docstring."""

_BASE_URL = "https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions"

_MANUAL_INSTALL_URL = "https://org.ngc.nvidia.com/setup/installers/cli"

_RELEASES: dict[tuple[str, str], tuple[str, str]] = {
    # (system, machine) -> (zip_filename, sha256_hex)
    # SHAs taken from the NGC CLI 4.8.2 release notes:
    # https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/4.8.2/files
    ("Linux", "x86_64"): ("ngccli_linux.zip", "8f12ed29aaa49be96e059439f993a990b953a709b152550f062d5190e21afafa"),
    ("Linux", "aarch64"): ("ngccli_arm64.zip", "0243568105b0472dc2dd96fb985c59eb5f0a49913025988b7c4eec68a5c0b5dc"),
}

_VENV_BIN_DIR = "bin"
_PAYLOAD_REL_PATH = Path("opt") / "ngc-cli"

# Current and legacy NGC CLI API keys
_NGC_KEY_ENV_VARS = ("NGC_CLI_API_KEY", "NGC_API_KEY")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def has_ngc_key() -> bool:
    """True if any accepted NGC API-key env var is set to a non-empty value."""
    return any(os.environ.get(var) for var in _NGC_KEY_ENV_VARS)


def ngc_available() -> bool:
    """Return True iff ``ngc`` is on ``$PATH`` and ``ngc --version`` exits 0.

    Mirrors ``data_designer.cli.utils.check_ngc_cli_available`` so that a
    True return here implies the persona-download path will not bail.
    """
    if shutil.which("ngc") is None:
        return False
    try:
        subprocess.run(
            ["ngc", "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


def ensure_ngc_cli(
    *,
    force: bool = False,
    target_dir: Path | None = None,
) -> Path:
    """Idempotently install NGC CLI; return the resolved ``ngc`` binary path.

    Behavior:

    * If ``ngc`` is already on ``$PATH`` (or recoverable by prepending the
      venv's ``bin/`` to ``$PATH``) and ``force`` is False, this is a
      no-op and returns the existing path. The recover-via-PATH-prepend
      branch matters for Jupyter kernels launched outside an activated
      venv: a prior ``usersim setup-ngc`` run dropped the binary at
      ``<sys.prefix>/bin/ngc`` but the kernel's ``$PATH`` doesn't see it.
    * Otherwise downloads the version-pinned zip for the host
      ``(platform.system(), platform.machine())`` pair, verifies the sha256,
      extracts the ``ngc-cli/`` payload to ``<target_dir.parent>/opt/ngc-cli/``,
      and creates a relative symlink ``<target_dir>/ngc -> ../opt/ngc-cli/ngc``.
    * Prepends ``target_dir`` to ``os.environ["PATH"]`` so the installed
      binary is visible to subsequent ``subprocess`` calls in the same
      Python process (matters for the notebook's persona-download cell
      when the kernel was launched outside the activated venv).
    * Verifies the install with ``ngc --version`` and raises
      ``RuntimeError`` on probe failure.

    ``target_dir`` defaults to ``<sys.prefix>/bin`` when running inside an
    activated venv (``.venv/bin/`` for this project). Outside a venv the
    function raises a ``RuntimeError`` pointing at ``source .venv/bin/activate``;
    pass ``target_dir`` explicitly to override (mainly for tests).
    """
    if not force and ngc_available():
        path = shutil.which("ngc")
        assert path is not None  # ngc_available() returned True
        logger.info("NGC CLI already installed at %s", path)
        return Path(path)

    # Fast-path: a prior `usersim setup-ngc` run left a usable binary at
    # <sys.prefix>/bin/ngc but the current $PATH doesn't see it (Jupyter
    # kernel launched outside the activated venv). Adding it to PATH is
    # cheap and idempotent — if the probe still fails, we fall through to
    # the full download.
    if not force and sys.prefix != sys.base_prefix:
        venv_bin = Path(sys.prefix) / _VENV_BIN_DIR
        candidate = venv_bin / "ngc"
        if candidate.exists():
            _prepend_path(venv_bin)
            if ngc_available():
                logger.info("NGC CLI already installed at %s (added to PATH)", candidate)
                return candidate

    if target_dir is None:
        target_dir = _require_venv() / _VENV_BIN_DIR
    target_dir = Path(target_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    zip_filename, expected_sha256 = _resolve_release()
    url = f"{_BASE_URL}/{NGC_VERSION}/files/{zip_filename}"

    payload_dir = target_dir.parent / _PAYLOAD_REL_PATH
    launcher_link = target_dir / "ngc"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        zip_path = tmp_path / zip_filename
        # Print to stderr so the user sees something during the ~60 MB
        # blocking download; logger.info-only would silence this for
        # callers (notebook + plain CLI invocation) without -v.
        print(
            f"Downloading NGC CLI {NGC_VERSION} ({zip_filename}, ~60 MB) from {url} ...",
            file=sys.stderr,
            flush=True,
        )
        logger.info("downloading NGC CLI %s from %s", NGC_VERSION, url)
        _download(url, zip_path)
        actual_sha = _sha256_file(zip_path)
        if actual_sha != expected_sha256:
            raise RuntimeError(
                "NGC CLI zip sha256 mismatch — refusing to install. "
                f"expected={expected_sha256} actual={actual_sha} url={url}"
            )

        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

        extracted_payload = extract_dir / "ngc-cli"
        if not extracted_payload.is_dir():
            raise RuntimeError(
                f"unexpected NGC CLI zip layout — no top-level 'ngc-cli/' dir "
                f"under {extract_dir}; got {[p.name for p in extract_dir.iterdir()]}"
            )

        # Replace any prior payload (handles --force / version bumps).
        if payload_dir.exists():
            shutil.rmtree(payload_dir)
        payload_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(extracted_payload, payload_dir)

    launcher = payload_dir / "ngc"
    if not launcher.is_file():
        raise RuntimeError(f"NGC CLI launcher not found at {launcher} after extract")
    launcher.chmod(0o755)

    # Atomic-replace the symlink. We use a relative target so the venv
    # remains relocatable up to its parent (matches `ln -sf` from the
    # upstream NGC reference install script).
    relative_target = Path("..") / _PAYLOAD_REL_PATH / "ngc"
    if launcher_link.exists() or launcher_link.is_symlink():
        launcher_link.unlink()
    os.symlink(relative_target, launcher_link)

    # Make the install visible to the current process. .venv/bin is
    # usually on PATH already (since `source .venv/bin/activate`
    # prepended it), but a Jupyter kernel started outside the venv may
    # not have it.
    _prepend_path(target_dir)

    _verify_install()
    logger.info("NGC CLI %s installed at %s", NGC_VERSION, launcher_link)
    return launcher_link


def resolved_release() -> tuple[str, str, str]:
    """Public wrapper around :func:`_resolve_release` for the ``--dry-run`` printer.

    Returns ``(zip_filename, url, expected_sha256)``.
    """
    zip_filename, sha = _resolve_release()
    url = f"{_BASE_URL}/{NGC_VERSION}/files/{zip_filename}"
    return zip_filename, url, sha


def ensure_ngc_org() -> str | None:
    """Auto-discover the user's NGC org and export it as ``NGC_CLI_ORG``.

    NGC CLI 4.x rejects authenticated commands without an explicit org
    context (``Missing org - If Authenticated, org is also required.``)
    and explicitly rejects the ``no-org`` sentinel when an API key is
    set. So even a public-catalog download like
    ``ngc registry resource download-version nvidia/nemotron-personas/...``
    won't run unless ``NGC_CLI_ORG`` (or ``~/.ngc/config``) names a real
    org the API key has access to.

    Strategy: if ``NGC_CLI_ORG`` is already set we trust it; otherwise we
    shell out to ``ngc org list --format csv`` (which uses the
    ``NGC_CLI_API_KEY`` we already set) and pick the first org. The
    chosen value is exported via ``os.environ["NGC_CLI_ORG"]`` so any
    subsequent ``subprocess.run`` inherits it.

    Returns the org name, or ``None`` if discovery failed (no API key,
    NGC CLI missing, or `ngc org list` returned no rows). On failure we
    print an actionable message to stderr — the caller can choose to
    error or proceed.
    """
    existing = os.environ.get("NGC_CLI_ORG") or os.environ.get("NGC_ORG")
    if existing:
        os.environ["NGC_CLI_ORG"] = existing
        os.environ.setdefault("NGC_ORG", existing)
        return existing

    if shutil.which("ngc") is None:
        print(
            "NGC CLI not on PATH; cannot discover org. Run `usersim setup-ngc` first.",
            file=sys.stderr,
        )
        return None

    if not has_ngc_key():
        print(
            "NGC_CLI_API_KEY not set; cannot discover org. Can be generated at "
            "https://org.ngc.nvidia.com/account/api-keys. Then run:\n"
            '    export NGC_CLI_API_KEY="<your-ngc-key>"\n'
            "    usersim setup-ngc\n"
            "See docs/personas.md for details.",
            file=sys.stderr,
        )
        return None

    try:
        # `--format csv` emits header + rows; we take the first data row's
        # first column. NGC CLI 4.x prints the column name as something
        # like "Org Name" then a list. We're defensive about whitespace
        # and BOM-style noise.
        result = subprocess.run(
            ["ngc", "org", "list", "--format_type", "csv"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        stderr = (getattr(exc, "stderr", "") or "").strip()
        # Clean instructions for "Missing org" failure
        if "Missing org" in stderr:
            print(
                "Cannot discover NGC org: this host has no NGC org configured. "
                "Run `ngc config set` once and pick your org, or export "
                "NGC_CLI_ORG=<your-org>, then re-run.",
                file=sys.stderr,
            )
            return None
        detail = stderr or f"{type(exc).__name__} running `ngc org list`"
        print(
            f"Could not list NGC orgs: {detail}. Run `ngc config set` once on "
            "this host to configure your org, or export NGC_CLI_ORG=<your-org>.",
            file=sys.stderr,
        )
        return None

    org = _parse_first_org(result.stdout)
    if not org:
        print(
            "`ngc org list` returned no orgs. The API key may be invalid "
            "or have no org membership; run `ngc config set` on this host "
            "and pick an org interactively.",
            file=sys.stderr,
        )
        return None

    os.environ["NGC_CLI_ORG"] = org
    os.environ.setdefault("NGC_ORG", org)
    logger.info("NGC_CLI_ORG auto-discovered as %r", org)
    print(f"NGC_CLI_ORG auto-discovered as {org!r}", file=sys.stderr)
    return org


def _parse_first_org(csv_output: str) -> str | None:
    """Parse the first org name from ``ngc org list --format csv`` output.

    The NGC CLI csv format includes a header row and quotes fields with
    commas. We use stdlib :mod:`csv` for safety against quoted display
    names with embedded commas.
    """
    import csv as _csv
    import io as _io

    reader = _csv.reader(_io.StringIO(csv_output))
    rows = [r for r in reader if r and any(cell.strip() for cell in r)]
    if len(rows) < 2:  # need header + at least one row
        return None
    header = [c.strip().lower() for c in rows[0]]
    # Prefer the column literally named "name" (the canonical org-id);
    # fall back to first column if the schema shifts.
    try:
        name_idx = header.index("name")
    except ValueError:
        name_idx = 0
    first = rows[1][name_idx].strip()
    return first or None


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _resolve_release() -> tuple[str, str]:
    """Return ``(zip_filename, sha256_hex)`` for the current host.

    Normalises the few aliases ``platform.machine()`` returns inconsistently:

    * Linux ``arm64`` → ``aarch64`` (most distros report ``aarch64`` but
      a small minority report ``arm64``; the binary is the same).

    On macOS / Windows / unsupported Linux arches, raises
    ``NotImplementedError`` with a pointer to the manual installer.
    On Apple Silicon Macs running x86_64 Python under Rosetta,
    ``platform.machine()`` returns ``x86_64`` — that's the right answer
    for picking a binary that matches the host Python's arch, but this
    branch is moot for us because we error out on macOS regardless.
    """
    system = platform.system()
    machine = platform.machine()

    if system == "Linux" and machine == "arm64":
        machine = "aarch64"

    key = (system, machine)
    if key not in _RELEASES:
        raise NotImplementedError(
            f"NGC CLI auto-install is not supported on {system} / {machine}. "
            f"Install manually from {_MANUAL_INSTALL_URL} and re-run. "
            f"(macOS ships as a sudo .pkg installer; Windows as an .exe.)"
        )
    return _RELEASES[key]


def _require_venv() -> Path:
    """Refuse to install into a non-venv Python's prefix."""
    if sys.prefix == sys.base_prefix:
        raise RuntimeError(
            "usersim setup-ngc must run inside the project's virtualenv "
            "so the binary lands under .venv/bin/ instead of a system path. "
            "Run `source .venv/bin/activate` (after `uv sync`) and retry."
        )
    return Path(sys.prefix)


def _download(url: str, dest: Path) -> None:
    """Stream a URL to disk. Honours http_proxy / https_proxy env vars via urllib."""
    with urllib.request.urlopen(url) as response, open(dest, "wb") as fh:
        shutil.copyfileobj(response, fh)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _prepend_path(directory: Path) -> None:
    """Idempotently prepend ``directory`` to ``$PATH`` for the current process."""
    current = os.environ.get("PATH", "")
    parts = current.split(os.pathsep) if current else []
    str_dir = str(directory)
    if parts and parts[0] == str_dir:
        return
    parts = [str_dir] + [p for p in parts if p != str_dir]
    os.environ["PATH"] = os.pathsep.join(parts)


def _verify_install() -> None:
    """Run ``ngc --version``; raise with stderr if the post-install probe fails."""
    try:
        result = subprocess.run(
            ["ngc", "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        raise RuntimeError(f"NGC CLI install verification failed: {exc}. stderr={stderr!r}") from exc
    logger.info("ngc --version: %s", (result.stdout or "").strip())
