# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Repo-wide pytest configuration.

Lives at the repo root so it applies to both test roots declared in
``pyproject.toml`` (``tests``, which holds both the CLI and engine suites).
"""

from __future__ import annotations

import os
import socket

import pytest

_ALLOWED_FAMILIES = {getattr(socket, "AF_UNIX", None)} - {None}

# Applied by pytest-env from the [tool.pytest.ini_options] env block. Checked
# rather than set here because a root conftest runs after any conftest that
# already imported a registry, which is too late to keep discovery out.
_HERMETIC_ENV_VAR = "USERSIM_DISABLE_EXTENSIONS"

_BLOCKED_MESSAGE = (
    "Outbound network access is blocked during tests. Something under test "
    "tried to open a socket -- patch it at the boundary instead (see "
    "tests/conftest.py::deterministic_token_encoder for the tiktoken case). "
    "If a test genuinely needs a socket, unblock it explicitly rather than "
    "removing this guard."
)


class BlockedNetworkCall(BaseException):
    """Raised when a test attempts an outbound connection.

    Deliberately not an ``Exception``. Production code retries failed model
    calls and falls back on failed ones, both via ``except Exception``, so an
    ordinary exception here gets absorbed: the test spends seconds in retry
    backoff and then passes down a fallback path, reporting success for
    whatever it meant to assert. Inheriting from ``BaseException`` means an
    unmocked boundary stops the test at the point it happens.
    """


def _require_hermetic_env() -> None:
    """Abort the run if the suite is not hermetic, naming the cause.

    Without ``pytest-env`` installed, pytest emits only
    ``PytestConfigWarning: Unknown config option: env`` and three tests then
    fail for reasons that look unrelated to a missing package. Failing here
    costs one clear message instead.
    """
    if os.environ.get(_HERMETIC_ENV_VAR):
        return
    raise pytest.UsageError(
        f"{_HERMETIC_ENV_VAR} is not set, so an extension installed in this "
        "environment could register itself into the probe, scorer and domain "
        "registries and change what the suite sees. It is normally set by "
        "pytest-env from the [tool.pytest.ini_options] env block in "
        "pyproject.toml, so the usual cause is that the dev dependency group "
        "is not installed. Run `uv sync` (or `make install-dev`) and retry."
    )


def _require_async_plugin(config) -> None:
    """Abort the run if async tests cannot be collected, naming the cause.

    Without ``pytest-asyncio`` every ``async def`` test is skipped with a
    warning rather than failed, so a suite that never ran the async paths
    still reports green. Failing here costs one clear message instead.
    """
    if config.pluginmanager.hasplugin("asyncio"):
        return
    raise pytest.UsageError(
        "pytest-asyncio is not installed, so every async test would be "
        "skipped with a warning and the suite would report green without "
        "having run them. It is part of the dev dependency group, so the "
        "usual cause is that the group is not installed. Run `uv sync` "
        "(or `make install-dev`) and retry."
    )


def pytest_configure(config) -> None:
    """Fail loudly on outbound connections instead of depending on egress.

    One unguarded ``tiktoken.get_encoding`` call in ``reporting/_reasoning.py``
    took out 43 tests across five files on any host without egress to
    ``openaipublic.blob.core.windows.net``, and the troubleshooting docs
    misattributed it to a pandas/pyarrow version conflict. Nothing in this
    suite legitimately reaches the network -- ``test_setup_ngc`` patches
    ``urllib.request.urlopen`` and ``test_model_catalog`` patches
    ``httpx.post`` -- so the only thing a socket can mean here is an
    unmocked boundary.

    ``connect`` is patched rather than ``socket.socket`` itself so the class
    stays a class and ``isinstance`` checks in third-party code keep working.
    AF_UNIX is left open because subprocess and multiprocessing IPC use it.
    """
    _require_hermetic_env()
    _require_async_plugin(config)

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def guarded_connect(self, address):
        if self.family in _ALLOWED_FAMILIES:
            return real_connect(self, address)
        raise BlockedNetworkCall(f"{_BLOCKED_MESSAGE} (address={address!r})")

    def guarded_connect_ex(self, address):
        if self.family in _ALLOWED_FAMILIES:
            return real_connect_ex(self, address)
        raise BlockedNetworkCall(f"{_BLOCKED_MESSAGE} (address={address!r})")

    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
