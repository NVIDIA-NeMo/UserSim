# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``usersim.engine.core._assets`` — resolution of the
packaged asset banks.

The banks are package data under ``usersim/engine/assets/``, resolved
through :mod:`importlib.resources`.

The invariants here fall into two groups: that resolution works and honours
its overrides, and that the probe registry and the on-disk layout stay in
lockstep.

Loader-level tests live in the per-bank test files (``test_agentic_bank.py``,
``test_pressure_bank.py``, …) and exercise these helpers indirectly via the
shipped sample.
"""

from __future__ import annotations

import os
from pathlib import Path

from usersim.engine.core._assets import (
    default_assets_dir,
    packaged_assets_dir,
    probe_assets_dir,
    reset_runtime_assets_dir,
    set_runtime_assets_dir,
)


class TestPackagedResolution:
    def test_resolves_inside_the_package(self) -> None:
        path = packaged_assets_dir()
        assert path.is_dir(), f"{path} is not a directory"
        assert path.parent.name == "engine", f"assets must be package data, resolved {path}"

    def test_independent_of_cwd(self, tmp_path: Path, monkeypatch) -> None:
        """Resolution must not depend on where the process was started."""
        baseline = default_assets_dir()
        monkeypatch.chdir(tmp_path)
        assert default_assets_dir() == baseline

    def test_no_cwd_fallback_even_with_a_decoy(self, tmp_path: Path, monkeypatch) -> None:
        """A directory named ``assets`` in the CWD must not be picked up."""
        decoy = tmp_path / "assets"
        (decoy / "sov_ai_facts").mkdir(parents=True)
        monkeypatch.chdir(tmp_path)
        assert default_assets_dir() != decoy
        assert default_assets_dir() == packaged_assets_dir()


class TestOverrides:
    def test_env_var_takes_precedence(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("USERSIM_ASSETS_DIR", str(tmp_path))
        assert default_assets_dir() == tmp_path

    def test_runtime_context_beats_env(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("USERSIM_ASSETS_DIR", str(tmp_path / "env"))
        ctx = tmp_path / "ctx"
        token = set_runtime_assets_dir(ctx)
        try:
            assert default_assets_dir() == ctx
        finally:
            reset_runtime_assets_dir(token)
        assert default_assets_dir() == tmp_path / "env"

    def test_runtime_override_is_restored(self, tmp_path: Path) -> None:
        baseline = default_assets_dir()
        token = set_runtime_assets_dir(tmp_path)
        reset_runtime_assets_dir(token)
        assert default_assets_dir() == baseline

    def test_explicit_root_wins_in_probe_assets_dir(self, tmp_path: Path) -> None:
        assert probe_assets_dir("sov_ai_facts", assets_root=tmp_path) == (tmp_path / "sov_ai_facts")


class TestRegistryAndLayoutInvariants:
    """Every registered probe resolves to an asset folder, and every packaged
    folder belongs to a registered probe. Importing the generator triggers the
    probe bootstrap that populates the registry."""

    @staticmethod
    def _labels() -> set[str]:
        import usersim.engine.generator  # noqa: F401
        from usersim.engine.core.probes import known_probes

        return set(known_probes())

    def test_every_registered_probe_resolves_to_a_folder(self) -> None:
        unresolved = {
            label: probe_assets_dir(label) for label in self._labels() if not probe_assets_dir(label).is_dir()
        }
        assert not unresolved, f"registered probes without an asset folder: {unresolved}"

    def test_every_built_in_probe_ships_its_folder_in_the_package(self) -> None:
        missing = sorted(label for label in self._labels() if not (packaged_assets_dir() / label).is_dir())
        assert not missing, f"built-in probes whose assets are not under {packaged_assets_dir()}: {missing}"

    def test_packaged_tree_has_no_stray_top_level_entries(self) -> None:
        """Everything shipped under assets/ should be a probe folder. Catches
        a doc image or scratch file riding into the wheel, and a folder left
        behind by a renamed probe."""
        labels = self._labels()
        stray = [
            p.name for p in packaged_assets_dir().iterdir() if p.name not in labels and not p.name.startswith("__")
        ]
        assert not stray, f"unexpected entries in packaged assets/: {stray}"


class TestAssetSearchPath:
    """Assets resolve along an ordered path, so a package can contribute one
    probe's banks without restating the shipped ones."""

    def test_packaged_root_is_always_last(self, monkeypatch, tmp_path) -> None:
        from usersim.engine.core._assets import (
            asset_search_path,
            packaged_assets_dir,
        )

        monkeypatch.setenv("USERSIM_ASSET_PATH", str(tmp_path))
        assert asset_search_path()[-1] == packaged_assets_dir()

    def test_entries_are_searched_in_order(self, monkeypatch, tmp_path) -> None:
        from usersim.engine.core._assets import asset_search_path

        first, second = tmp_path / "a", tmp_path / "b"
        monkeypatch.setenv(
            "USERSIM_ASSET_PATH",
            f"{first}{os.pathsep}{second}",
        )
        path = asset_search_path()
        assert path[0] == first and path[1] == second

    def test_duplicates_collapse(self, monkeypatch, tmp_path) -> None:
        from usersim.engine.core._assets import asset_search_path

        monkeypatch.setenv(
            "USERSIM_ASSET_PATH",
            f"{tmp_path}{os.pathsep}{tmp_path}",
        )
        assert len(asset_search_path()) == len(set(asset_search_path()))

    def test_contributed_root_wins_for_the_probe_it_has(
        self,
        monkeypatch,
        tmp_path,
    ) -> None:
        from usersim.engine.core._assets import probe_assets_dir

        contributed = tmp_path / "extra"
        (contributed / "financial_services").mkdir(parents=True)
        monkeypatch.setenv("USERSIM_ASSET_PATH", str(contributed))
        assert probe_assets_dir("financial_services").parent == contributed

    def test_shipped_banks_stay_reachable(self, monkeypatch, tmp_path) -> None:
        """Prepending a root must not hide the probes it does not carry."""
        from usersim.engine.core._assets import (
            packaged_assets_dir,
            probe_assets_dir,
        )

        contributed = tmp_path / "extra"
        (contributed / "financial_services").mkdir(parents=True)
        monkeypatch.setenv("USERSIM_ASSET_PATH", str(contributed))
        assert probe_assets_dir("tool_calling").parent == packaged_assets_dir()

    def test_explicit_root_is_pinned_not_searched(self, tmp_path) -> None:
        """A user who named a directory is told when assets are absent
        rather than quietly served the packaged copies."""
        from usersim.engine.core._assets import probe_assets_dir

        resolved = probe_assets_dir("tool_calling", assets_root=tmp_path)
        assert resolved == tmp_path / "tool_calling"
        assert not resolved.exists()

    def test_unresolvable_probe_reports_the_top_root(
        self,
        monkeypatch,
        tmp_path,
    ) -> None:
        from usersim.engine.core._assets import probe_assets_dir

        monkeypatch.setenv("USERSIM_ASSET_PATH", str(tmp_path))
        assert probe_assets_dir("no_such_probe").parent == tmp_path
