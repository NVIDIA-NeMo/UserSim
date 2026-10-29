# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``usersim.testing.pytest_plugin`` gives any test suite that opts in the harness this repo's suite runs under.

Each test runs a separate test project, the way an extension's suite would load
the plugin.
"""

from __future__ import annotations

from importlib.metadata import distribution

import pytest

_OPT_IN = 'pytest_plugins = ["usersim.testing.pytest_plugin"]\n'

_HARNESS_TESTS = """
import socket

import pandas as pd
import pytest

from usersim.testing.pytest_plugin import BlockedNetworkCall


def test_outbound_connect_is_blocked():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(BlockedNetworkCall):
            sock.connect(("203.0.113.1", 80))
    finally:
        sock.close()


def test_persona_sampler_reads_the_synthetic_dataset(synthetic_persona_dataset):
    import data_designer.interface.data_designer as dd

    assert dd.MANAGED_ASSETS_PATH == synthetic_persona_dataset
    rows = pd.read_parquet(synthetic_persona_dataset / "datasets" / "en_US.parquet")
    assert len(rows) == 4


def test_tokenizer_is_the_word_splitter():
    from usersim.reporting import _reasoning

    assert _reasoning._encoder("o200k_base").encode("three short words") == [0, 0, 0]


async def test_async_tests_run():
    assert True
"""


def test_a_project_that_opts_in_gets_the_whole_harness(pytester: pytest.Pytester) -> None:
    """Run in a fresh interpreter, so nothing this suite already set up can pass for the plugin's own work."""
    pytester.makeini("[pytest]\nasyncio_mode = auto\n")
    pytester.makeconftest(_OPT_IN)
    pytester.makepyfile(test_harness=_HARNESS_TESTS)

    result = pytester.runpytest_subprocess("-p", "no:cacheprovider", timeout=300)

    result.assert_outcomes(passed=4)


def test_a_run_without_discovery_disabled_stops_before_collecting(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("USERSIM_DISABLE_EXTENSIONS", raising=False)
    pytester.makeini("[pytest]\n")
    pytester.makeconftest(_OPT_IN)
    pytester.makepyfile("def test_never_collected():\n    pass\n")

    result = pytester.runpytest()

    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*USERSIM_DISABLE_EXTENSIONS is not set*"])


def test_a_run_without_pytest_asyncio_stops_before_collecting(pytester: pytest.Pytester) -> None:
    pytester.makeini("[pytest]\n")
    pytester.makeconftest(_OPT_IN)
    pytester.makepyfile("def test_never_collected():\n    pass\n")

    result = pytester.runpytest("-p", "no:asyncio")

    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*pytest-asyncio is not installed*"])


def test_installing_usersim_loads_no_pytest_plugin() -> None:
    """The harness blocks sockets, so it applies only where a suite asks for it."""
    plugins = [entry.name for entry in distribution("usersim").entry_points if entry.group == "pytest11"]
    assert plugins == []
