# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entry-point discovery: isolation, determinism, and the test-suite switch."""

from __future__ import annotations

import logging
from importlib.metadata import EntryPoint

import pytest

from usersim.engine.core import _extensions
from usersim.engine.core._extensions import (
    DISABLE_ENV_VAR,
    ENTRY_POINT_GROUPS,
    PROBES,
    clear_extension_cache,
    load_extensions,
    merge_extensions,
    extensions_disabled,
)


def _fake_entry_points(monkeypatch, *specs: tuple[str, str], group: str = PROBES):
    """Advertise ``(name, "module:attr")`` pairs under ``group``."""
    eps = [EntryPoint(name=n, value=v, group=group) for n, v in specs]
    monkeypatch.setattr(
        _extensions,
        "entry_points",
        lambda group: [e for e in eps if e.group == group],
    )
    clear_extension_cache()


@pytest.fixture
def discovery_enabled(monkeypatch):
    """Turn discovery on for one test, restoring the suite's default after."""
    monkeypatch.delenv(DISABLE_ENV_VAR, raising=False)
    clear_extension_cache()
    yield
    clear_extension_cache()


class TestSuiteIsHermetic:
    """An extension installed in a developer's venv must not reach the suite.

    Without this the probe, scorer, and domain registries would differ
    between machines and the suite would stop being reproducible.
    """

    def test_discovery_is_disabled_under_test(self) -> None:
        assert extensions_disabled(), (
            f"{DISABLE_ENV_VAR} must be set for the suite; check the "
            "[tool.pytest.ini_options] env block in pyproject.toml"
        )

    def test_disabled_discovery_finds_nothing(self, monkeypatch) -> None:
        """Even with an entry point present, nothing loads while disabled."""
        _fake_entry_points(monkeypatch, ("evil", "json:dumps"))
        assert load_extensions(PROBES) == ()

    def test_the_switch_is_not_cached(self, monkeypatch) -> None:
        """Toggling within a process takes effect, so tests can opt in."""
        _fake_entry_points(monkeypatch, ("p", "json:dumps"))
        assert load_extensions(PROBES) == ()
        monkeypatch.delenv(DISABLE_ENV_VAR, raising=False)
        assert [n for n, _ in load_extensions(PROBES)] == ["p"]


class TestDiscovery:
    def test_loads_advertised_objects(self, monkeypatch, discovery_enabled) -> None:
        import json

        _fake_entry_points(monkeypatch, ("dumps", "json:dumps"))
        assert load_extensions(PROBES) == (("dumps", json.dumps),)

    def test_order_is_by_name_not_install_order(
        self,
        monkeypatch,
        discovery_enabled,
    ) -> None:
        """Registry contents must not depend on pip install ordering."""
        _fake_entry_points(
            monkeypatch,
            ("zeta", "json:dumps"),
            ("alpha", "json:loads"),
        )
        assert [n for n, _ in load_extensions(PROBES)] == ["alpha", "zeta"]

    def test_broken_extension_is_skipped_not_fatal(
        self,
        monkeypatch,
        discovery_enabled,
        caplog,
    ) -> None:
        """One bad package must not make the CLI unusable."""
        _fake_entry_points(
            monkeypatch,
            ("broken", "no_such_module_xyz:thing"),
            ("working", "json:dumps"),
        )
        with caplog.at_level(logging.WARNING, logger="usersim.engine"):
            loaded = load_extensions(PROBES)
        assert [n for n, _ in loaded] == ["working"]
        # Silence here would mean a probe that never appears and no way to
        # find out why.
        assert any("broken" in r.getMessage() for r in caplog.records)

    def test_unknown_group_is_rejected(self) -> None:
        """A typo'd group would otherwise return () and look like 'none installed'."""
        with pytest.raises(ValueError, match="unknown entry-point group"):
            load_extensions("usersim.probez")

    def test_published_groups_are_namespaced(self) -> None:
        assert all(g.startswith("usersim.") for g in ENTRY_POINT_GROUPS)
        assert len(set(ENTRY_POINT_GROUPS)) == len(ENTRY_POINT_GROUPS)


class TestMergePolicy:
    def test_extension_extends_builtins(self, monkeypatch, discovery_enabled) -> None:
        _fake_entry_points(monkeypatch, ("extra", "json:dumps"))
        merged = merge_extensions(PROBES, {"builtin": object()}, what="probe")
        assert set(merged) == {"builtin", "extra"}

    def test_builtin_wins_a_collision(
        self,
        monkeypatch,
        discovery_enabled,
        caplog,
    ) -> None:
        """Shadowing a shipped probe would change a run's meaning silently."""
        import json

        shipped = object()
        _fake_entry_points(monkeypatch, ("tool_calling", "json:dumps"))
        with caplog.at_level(logging.WARNING, logger="usersim.engine"):
            merged = merge_extensions(
                PROBES,
                {"tool_calling": shipped},
                what="probe",
            )
        assert merged["tool_calling"] is shipped
        assert merged["tool_calling"] is not json.dumps
        assert any("tool_calling" in r.getMessage() for r in caplog.records)

    def test_builtins_are_not_mutated(self, monkeypatch, discovery_enabled) -> None:
        _fake_entry_points(monkeypatch, ("extra", "json:dumps"))
        builtins: dict = {"builtin": object()}
        merge_extensions(PROBES, builtins, what="probe")
        assert set(builtins) == {"builtin"}
