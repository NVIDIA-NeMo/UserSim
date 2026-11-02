# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The hermetic test harness, as a pytest plugin a test suite opts into.

Load it from the top-level ``conftest.py`` of a test suite::

    pytest_plugins = ["usersim.testing.pytest_plugin"]

Installing ``usersim`` does not load it: there is no ``pytest11`` entry point,
so a project's tests change only once it opts in. With the plugin loaded, a
test run:

- fails any outbound network connection with ``BlockedNetworkCall``;
- stops before collecting anything unless ``USERSIM_DISABLE_EXTENSIONS`` is
  set, so the extensions installed in the environment cannot register
  themselves and change what the tests see;
- stops unless pytest-asyncio is installed, which runs ``async def`` tests;
- stubs the tokenizer behind the reasoning-token estimate, which otherwise
  downloads its vocabulary on first use;
- points Data Designer's persona sampler at four synthetic ``en_US`` personas,
  so sampling needs no downloaded persona dataset.

This project's own suite loads the same plugin from its root ``conftest.py``,
so built-in probes and extensions are tested under the same rules.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

__all__ = ["BlockedNetworkCall", "deterministic_token_encoder", "synthetic_persona_dataset"]

_ALLOWED_FAMILIES = {getattr(socket, "AF_UNIX", None)} - {None}

_HERMETIC_ENV_VAR = "USERSIM_DISABLE_EXTENSIONS"

_BLOCKED_MESSAGE = (
    "Outbound network access is blocked during tests. Something under test "
    "tried to open a socket -- patch it at the boundary instead (see "
    "usersim.testing.pytest_plugin.deterministic_token_encoder for the tiktoken "
    "case). If a test genuinely needs a socket, unblock it explicitly rather "
    "than removing this guard."
)

_ORIGINAL_CONNECT = pytest.StashKey[tuple[Any, Any]]()


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
    """Stop the run if extension discovery is on, naming the cause.

    With discovery on, the tests that a stray extension breaks fail in ways
    that look unrelated to it; one message up front names the cause instead.
    Checked rather than set: a plugin or conftest loaded earlier may already
    have looked up a registry, and setting the variable after that is too
    late to keep discovery out.
    """
    if os.environ.get(_HERMETIC_ENV_VAR):
        return
    raise pytest.UsageError(
        f"{_HERMETIC_ENV_VAR} is not set, so an extension installed in this "
        "environment could register itself into the probe, scorer and domain "
        "registries and change what the tests see. Set it for the test run, "
        'for example with pytest-env: env = ["USERSIM_DISABLE_EXTENSIONS=1"] '
        "under [tool.pytest.ini_options]. If that setting is already there, "
        "pytest-env is not installed. In a UserSim checkout, `uv sync` (or "
        "`make install-dev`) installs it."
    )


def _require_async_plugin(config: pytest.Config) -> None:
    """Stop the run if async tests cannot run, naming the cause.

    Without pytest-asyncio, pytest before 8.4 skips every ``async def`` test
    with a warning and later versions fail each one, so either way the async
    paths go untested. ``assert_scorer_conforms`` is itself async.
    """
    if config.pluginmanager.hasplugin("asyncio"):
        return
    raise pytest.UsageError(
        "pytest-asyncio is not installed, so no async test can run. Install it "
        "with the other test dependencies and retry. In a UserSim checkout, "
        "`uv sync` (or `make install-dev`) installs it."
    )


def pytest_configure(config: pytest.Config) -> None:
    """Check the environment, then fail every outbound connection.

    A test reaches a model, a download or any other service only through a
    boundary it mocks, so an outbound connection means a boundary was missed.
    ``connect`` is patched rather than ``socket.socket`` itself so the class
    stays a class and ``isinstance`` checks in third-party code keep working.
    AF_UNIX is left open because subprocess and multiprocessing IPC use it.
    """
    _require_hermetic_env()
    _require_async_plugin(config)

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    config.stash[_ORIGINAL_CONNECT] = (real_connect, real_connect_ex)

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


def pytest_unconfigure(config: pytest.Config) -> None:
    """Restore the socket methods this run replaced."""
    original = config.stash.get(_ORIGINAL_CONNECT, None)
    if original is not None:
        socket.socket.connect, socket.socket.connect_ex = original


class _WordSplitEncoder:
    """Deterministic stand-in for a ``tiktoken`` encoding."""

    @staticmethod
    def encode(text: str) -> list[int]:
        return [0] * len(text.split())


@pytest.fixture(autouse=True, scope="session")
def deterministic_token_encoder() -> Iterator[None]:
    """Stub the tokenizer behind the reasoning-token estimate for the whole session.

    ``tiktoken.get_encoding`` downloads its vocabulary on first use: tiktoken
    ships the tokenizer, not the vocabulary. Tests of reports and reasoning
    estimates check how token counts are aggregated, not tiktoken's counts,
    and a word splitter gives the same answer on every host.

    Autouse, so a new test cannot depend on network access through it.
    """
    import usersim.reporting._reasoning as reasoning

    original = reasoning._encoder
    reasoning._encoder = lambda name: _WordSplitEncoder()
    try:
        yield
    finally:
        reasoning._encoder = original


# First name, last name, sex, age, education level, occupation, city. Values
# use the en_US dataset's own vocabulary, and every row is in Texas.
_SYNTHETIC_PERSONAS = (
    ("Sarah", "Johnson", "Female", 42, "bachelors", "elementary_or_middle_school_teacher", "Austin"),
    ("Daniel", "Reyes", "Male", 29, "some_college", "electrician", "San Antonio"),
    ("Grace", "Liu", "Female", 67, "graduate", "pharmacist", "Houston"),
    ("Marcus", "Hill", "Male", 51, "high_school", "driver_sales_worker_or_truck_driver", "Dallas"),
)
_OCEAN_TRAITS = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")


def _synthetic_persona_rows() -> list[dict[str, Any]]:
    return [
        {
            "uuid": f"synthetic-persona-{index}",
            "first_name": first_name,
            "last_name": last_name,
            "sex": sex,
            "age": age,
            "education_level": education_level,
            "occupation": occupation,
            "city": city,
            "region": "TX",
            "country": "USA",
            "persona": f"{first_name} {last_name} works as a {occupation.replace('_', ' ')} in {city}, Texas.",
            **{
                trait: {"description": "Synthetic trait.", "label": "average", "t_score": 50} for trait in _OCEAN_TRAITS
            },
        }
        for index, (first_name, last_name, sex, age, education_level, occupation, city) in enumerate(
            _SYNTHETIC_PERSONAS, start=1
        )
    ]


@pytest.fixture(autouse=True, scope="session")
def synthetic_persona_dataset(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Point Data Designer's persona sampler at a small synthetic dataset.

    The sampler reads the Nemotron-Personas datasets, a separate download
    that a fresh checkout does not have. Tests of the sampler graph do not
    depend on the dataset's contents, so a few rows serve as well as
    millions, and every shipped probe accepts every row. Yields the managed
    assets root; the rows are in ``datasets/en_US.parquet`` beneath it.

    Autouse, so a new test cannot depend on the downloaded datasets.
    """
    import pandas as pd

    root = tmp_path_factory.mktemp("managed-assets")
    (root / "datasets").mkdir()
    pd.DataFrame(_synthetic_persona_rows()).to_parquet(root / "datasets" / "en_US.parquet", index=False)
    with pytest.MonkeyPatch.context() as patcher:
        # UserSim constructs DataDesigner without managed_assets_path, so the
        # sampler reads this module default.
        patcher.setattr("data_designer.interface.data_designer.MANAGED_ASSETS_PATH", root)
        yield root
