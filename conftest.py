# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repo-wide pytest configuration.

Lives at the repo root so it applies to both test roots declared in
``pyproject.toml`` (``tests``, which holds both the CLI and engine suites).
"""

from __future__ import annotations

import socket

_ALLOWED_FAMILIES = {getattr(socket, "AF_UNIX", None)} - {None}

_BLOCKED_MESSAGE = (
    "Outbound network access is blocked during tests. Something under test "
    "tried to open a socket -- patch it at the boundary instead (see "
    "tests/conftest.py::deterministic_token_encoder for the tiktoken case). "
    "If a test genuinely needs a socket, unblock it explicitly rather than "
    "removing this guard."
)


class BlockedNetworkCall(RuntimeError):
    """Raised when a test attempts an outbound connection."""


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
