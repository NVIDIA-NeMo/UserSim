# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``usersim setup-ngc`` — install NGC CLI into the project's venv.

The Nemotron-Personas datasets that the simulator's persona panels are
sampled from live on NGC. ``data-designer download personas`` shells out
to the ``ngc`` binary and silently no-ops if it isn't on ``$PATH``,
which is what cluster runs of ``notebooks/01_simulate.ipynb`` were
hitting (eight identical "NGC CLI not found" banners, one per locale,
no actual download).

Running this command after ``uv sync`` puts a pinned, sha256-verified
NGC CLI release under ``.venv/opt/ngc-cli/`` and symlinks it into
``.venv/bin/ngc``. Re-runs are no-ops; ``--force`` re-downloads. The
notebook persona-download cell auto-invokes the same helper
(:func:`cli._ngc.ensure_ngc_cli`) when ``ngc`` is missing, so cluster
runs that bypass the documented setup step self-heal on first use.

This subcommand only installs the binary. NGC auth (``ngc config set``
or ``$NGC_CLI_API_KEY``) is left to the user; see ``docs/personas.md``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import urllib.error

logger = logging.getLogger("usersim.cli.setup_ngc")


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "setup-ngc",
        help="⚙️  Install NGC CLI into the project's venv (idempotent).",
        description=(
            "Download a pinned, sha256-verified NGC CLI release into "
            ".venv/opt/ngc-cli/ and symlink it to .venv/bin/ngc. NGC CLI "
            "is required by `data-designer download personas`, which the "
            "simulator's persona-panel step relies on."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Reinstall even if `ngc` is already on PATH.",
    )
    p.add_argument(
        "--check-only",
        action="store_true",
        help=("Don't install — just check whether `ngc` is available. Exits 0 if available, 1 if missing."),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=("Print the resolved release (URL + sha256 + target path) and exit without downloading."),
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    from usersim.cli._ngc import (
        NGC_VERSION,
        ensure_ngc_cli,
        ngc_available,
        resolved_release,
    )

    if args.check_only:
        if ngc_available():
            import shutil

            print(f"ngc available at {shutil.which('ngc')}")
            return 0
        print("ngc NOT available", file=sys.stderr)
        return 1

    if args.dry_run:
        try:
            zip_filename, url, expected_sha = resolved_release()
        except NotImplementedError as exc:
            print(f"usersim setup-ngc — dry run\n  ERROR: {exc}", file=sys.stderr)
            return 2
        print("usersim setup-ngc — dry run")
        print(f"  version:    {NGC_VERSION}")
        print(f"  zip:        {zip_filename}")
        print(f"  url:        {url}")
        print(f"  sha256:     {expected_sha}")
        print("  target:     <venv>/bin/ngc -> ../opt/ngc-cli/ngc")
        print(f"  ngc on path: {ngc_available()}")
        return 0

    try:
        path = ensure_ngc_cli(force=args.force)
    except NotImplementedError as exc:
        print(f"usersim setup-ngc: {exc}", file=sys.stderr)
        return 2
    except urllib.error.URLError as exc:
        # DNS / 5xx / 4xx / connection refused / timeout. A proxy is the
        # usual culprit and the raw urllib traceback does not say so, so
        # hint at the proxy environment variables.
        print(
            f"usersim setup-ngc: network error fetching NGC CLI release: {exc}. "
            f"If you're behind a corporate proxy, export http_proxy / https_proxy "
            f"and retry; urllib honours both.",
            file=sys.stderr,
        )
        return 1
    except (RuntimeError, OSError) as exc:
        print(f"usersim setup-ngc failed: {exc}", file=sys.stderr)
        return 1
    print(f"NGC CLI ready at {path}")

    # Org discovery needs an NGC key. Without one the install (this command's
    # job) is already done — print an optional next-step note rather than an
    # error-toned "cannot discover org" that reads like a failure.
    from usersim.cli._ngc import ensure_ngc_org, has_ngc_key

    if has_ngc_key():
        ensure_ngc_org()
    else:
        print(
            "Next step: to download Nemotron-Personas, set NGC_CLI_API_KEY "
            "(generate at https://org.ngc.nvidia.com/account/api-keys) and "
            "re-run. Not needed for `usersim smoke`. See docs/personas.md."
        )
    return 0
