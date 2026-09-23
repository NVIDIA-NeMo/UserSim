# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

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


def _guard_names(raised: BaseException) -> set[str]:
    """Collect the exception type names, unwrapping any group.

    An async client runs its connection attempt inside a task group, which
    re-raises whatever escaped wrapped in a ``BaseExceptionGroup``. The guard
    is still in there and still not an ``Exception``, so the check has to look
    through the wrapper rather than at the outermost type.
    """
    names = {type(raised).__name__}
    if isinstance(raised, BaseExceptionGroup):
        for inner in raised.exceptions:
            names |= _guard_names(inner)
    return names


def _assert_guard_raised(excinfo) -> None:
    """The guard must be a BaseException, not an Exception.

    Both the model-call retry loop and the probe fallbacks catch
    ``Exception``. An ordinary exception here is therefore absorbed: the test
    spends the whole backoff budget retrying and then passes down a fallback
    path, reporting success for whatever it set out to assert.
    """
    assert _GUARD_NAME in _guard_names(excinfo.value), f"expected the guard somewhere in {_guard_names(excinfo.value)}"
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


async def test_outbound_connect_is_blocked_from_a_coroutine() -> None:
    """The guard patches the socket class, so it applies on any loop too."""
    import asyncio

    with pytest.raises(BaseException, match="network access is blocked") as excinfo:
        await asyncio.open_connection(*_UNROUTABLE)
    _assert_guard_raised(excinfo)


async def test_outbound_connect_is_blocked_for_an_async_http_client() -> None:
    """The client wraps the failure in a group; the guard survives inside it."""
    import httpx

    with pytest.raises(BaseException) as excinfo:
        async with httpx.AsyncClient() as client:
            await client.get(f"http://{_UNROUTABLE[0]}/")
    _assert_guard_raised(excinfo)


def test_gather_is_never_used_in_a_form_that_absorbs_the_guard() -> None:
    """``return_exceptions=True`` collects BaseException instead of raising.

    The guard is a ``BaseException`` precisely so nothing swallows it. That
    call form puts it in a results list, which restores the silent-absorption
    this whole guard exists to prevent.
    """
    import ast
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    offenders: list[str] = []
    for directory in ("src", "tests"):
        for path in (repo_root / directory).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                for keyword in node.keywords:
                    if (
                        keyword.arg == "return_exceptions"
                        and isinstance(keyword.value, ast.Constant)
                        and keyword.value.value is True
                    ):
                        offenders.append(f"{path.relative_to(repo_root)}:{node.lineno}")
    assert not offenders, (
        f"gather(..., return_exceptions=True) collects BaseException rather than "
        f"propagating it, so the network guard stops stopping tests: {offenders}"
    )


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
