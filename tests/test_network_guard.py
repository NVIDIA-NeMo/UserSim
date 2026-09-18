# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The repo-wide network guard must actually be armed.

Without this, the guard in the root ``conftest.py`` could be removed or
silently stop applying and nothing would notice until a test started
depending on network egress again -- which is the failure that took out 43
tests across five files.
"""

from __future__ import annotations

import json
import socket

import pytest

# TEST-NET-3 (RFC 5737): reserved for documentation and guaranteed not to be
# routable. Using a literal address keeps DNS out of the picture, so the guard
# is what fails rather than name resolution.
_UNROUTABLE = ("203.0.113.1", 80)

# The guard lives in the root conftest, which cannot be imported here because
# `conftest` resolves to this directory's one.
_GUARD_NAME = "BlockedNetworkCall"


def _assert_guard_raised(excinfo) -> None:
    """The guard must be a BaseException, not an Exception.

    Both the model-call retry loop and the probe fallbacks catch
    ``Exception``. An ordinary exception here is therefore absorbed: the test
    spends the whole backoff budget retrying and then passes down a fallback
    path, reporting success for whatever it set out to assert.
    """
    assert type(excinfo.value).__name__ == _GUARD_NAME
    assert not isinstance(excinfo.value, Exception), (
        "the network guard is an Exception again, so retry and fallback paths will swallow it and unmocked boundaries will go unnoticed"
    )


def test_outbound_connect_is_blocked() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(BaseException, match="network access is blocked") as excinfo:
            sock.connect(_UNROUTABLE)
        _assert_guard_raised(excinfo)
    finally:
        sock.close()


def test_outbound_connect_ex_is_blocked() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(BaseException, match="network access is blocked") as excinfo:
            sock.connect_ex(_UNROUTABLE)
        _assert_guard_raised(excinfo)
    finally:
        sock.close()


def test_reasoning_estimate_degrades_when_no_tokenizer(monkeypatch) -> None:
    """A missing tokenizer must yield ``unavailable``, not crash the report.

    ``tiktoken.get_encoding`` fetches its vocabulary over HTTPS, so an
    air-gapped host cannot build an encoder. That has to route into the
    ``unavailable`` tier the estimator already defines for aliases with no
    visible content, so ``usersim report`` still completes.
    """
    from usersim.reporting import _reasoning

    monkeypatch.setattr(_reasoning, "_encoder", lambda name: None)
    tokens, source = _reasoning.estimate_reasoning_for_row(
        json.dumps([{"role": "assistant", "content": "hello there"}]),
        "assistant_model",
        output_tokens=1000,
    )
    assert (tokens, source) == (0, "unavailable")
