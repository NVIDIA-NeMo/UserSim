# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``usersim.engine.core._assets`` — resolution of the
packaged asset banks.

The banks are package data under ``usersim/engine/assets/``, resolved
through :mod:`importlib.resources`.

The invariants here fall into two groups: that resolution works and honours
its overrides and a probe's declared ``assets_root``, and that the probe
registry and the on-disk layout stay in lockstep.

Loader-level tests live in the per-bank test files (``test_agentic_bank.py``,
``test_pressure_bank.py``, …) and exercise these helpers indirectly via the
shipped sample.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import usersim
from usersim.engine.core._assets import (
    asset_roots_in_use,
    asset_search_path,
    default_assets_dir,
    packaged_assets_dir,
    probe_asset_roots,
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
        """In the packaged banks or in the probe's own folder, so a plain install has it."""
        package = Path(usersim.__file__).parent
        outside = {
            label: probe_assets_dir(label)
            for label in self._labels()
            if not probe_assets_dir(label).is_relative_to(package)
        }
        assert not outside, f"built-in probes whose assets are outside {package}: {outside}"

    def test_a_built_in_that_declares_a_root_keeps_it_in_its_own_folder(self) -> None:
        """Its banks sit beside its code, at ``engine/probes/<label>/assets/<label>/``."""
        from usersim.engine.core.probes import resolve_probe

        declared = {label: resolve_probe(label) for label in self._labels() if resolve_probe(label).assets_root}
        assert declared, "no built-in probe declares an assets_root"
        for label, cls in declared.items():
            folder = Path(sys.modules[cls.__module__].__file__).parent
            assert folder.name == label and Path(cls.assets_root) == folder / "assets", label

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


class TestDeclaredAssetsRoot:
    """A probe whose assets ship outside ``usersim`` names their root on its
    class, so it resolves without anyone setting ``USERSIM_ASSET_PATH``."""

    def test_a_declared_root_serves_its_probe(self, register_test_probe, tmp_path) -> None:
        (tmp_path / "ext_probe").mkdir()
        register_test_probe("ext_probe", assets_root=tmp_path)

        assert probe_assets_dir("ext_probe") == tmp_path / "ext_probe"

    def test_it_is_searched_after_the_overrides_and_before_the_packaged_banks(
        self, register_test_probe, monkeypatch, tmp_path
    ) -> None:
        override = tmp_path / "override"
        monkeypatch.setenv("USERSIM_ASSET_PATH", str(override))
        register_test_probe("ext_probe", assets_root=tmp_path / "declared")

        assert probe_asset_roots("ext_probe") == (override, tmp_path / "declared", packaged_assets_dir())

    def test_a_deliberate_override_wins_over_the_declared_root(
        self, register_test_probe, monkeypatch, tmp_path
    ) -> None:
        override, declared = tmp_path / "override", tmp_path / "declared"
        (override / "ext_probe").mkdir(parents=True)
        (declared / "ext_probe").mkdir(parents=True)
        monkeypatch.setenv("USERSIM_ASSET_PATH", str(override))
        register_test_probe("ext_probe", assets_root=declared)

        assert probe_assets_dir("ext_probe") == override / "ext_probe"

    def test_a_pinned_root_ignores_the_declared_root(self, register_test_probe, tmp_path) -> None:
        (tmp_path / "declared" / "ext_probe").mkdir(parents=True)
        register_test_probe("ext_probe", assets_root=tmp_path / "declared")

        assert probe_assets_dir("ext_probe", assets_root=tmp_path / "pinned") == tmp_path / "pinned" / "ext_probe"

    def test_missing_assets_are_reported_under_the_declared_root(
        self, register_test_probe, monkeypatch, tmp_path
    ) -> None:
        monkeypatch.setenv("USERSIM_ASSET_PATH", str(tmp_path / "override"))
        register_test_probe("ext_probe", assets_root=tmp_path / "declared")

        assert probe_assets_dir("ext_probe") == tmp_path / "declared" / "ext_probe"

    def test_a_declared_root_serves_only_its_own_probe(self, register_test_probe, tmp_path) -> None:
        """A package's root may hold other folders; only its probe reads from it."""
        (tmp_path / "tool_calling").mkdir()
        register_test_probe("ext_probe", assets_root=tmp_path)

        assert probe_assets_dir("tool_calling") == packaged_assets_dir() / "tool_calling"

    def test_roots_in_use_add_declared_roots_after_the_search_path(self, register_test_probe, tmp_path) -> None:
        register_test_probe("ext_probe", assets_root=tmp_path)

        roots, search = asset_roots_in_use(), asset_search_path()
        assert roots[: len(search)] == search
        assert tmp_path in roots[len(search) :]


class TestProbeAssetsDir:
    """``BaseProbe.assets_dir()`` is the folder a probe's own bank loader reads."""

    def test_a_built_in_probe_reads_its_packaged_folder(self) -> None:
        from usersim.engine.core.probes import resolve_probe

        assert resolve_probe("sov_ai_facts").assets_dir() == packaged_assets_dir() / "sov_ai_facts"

    def test_a_probe_reads_the_folder_under_its_declared_root(self, register_test_probe, tmp_path) -> None:
        (tmp_path / "ext_probe").mkdir()
        probe = register_test_probe("ext_probe", assets_root=tmp_path)

        assert probe.assets_dir() == tmp_path / "ext_probe"

    def test_a_deliberate_override_reaches_the_probe(self, register_test_probe, monkeypatch, tmp_path) -> None:
        override, declared = tmp_path / "override", tmp_path / "declared"
        (override / "ext_probe").mkdir(parents=True)
        (declared / "ext_probe").mkdir(parents=True)
        monkeypatch.setenv("USERSIM_ASSET_PATH", str(override))
        probe = register_test_probe("ext_probe", assets_root=declared)

        assert probe.assets_dir() == override / "ext_probe"


class TestPinnedPackagedRoot:
    """Pinning the packaged directory pins the assets that ship with User Sim."""

    @staticmethod
    def _package(monkeypatch, tmp_path: Path) -> Path:
        """A stand-in ``usersim`` package whose packaged banks are at ``engine/assets/``."""
        import usersim.engine.core._assets as assets

        package = tmp_path / "usersim"
        (package / "engine" / "assets").mkdir(parents=True)
        monkeypatch.setattr(assets, "packaged_assets_dir", lambda: package / "engine" / "assets")
        return package

    def test_it_reaches_a_probe_whose_banks_are_in_its_own_folder_in_the_package(
        self, register_test_probe, monkeypatch, tmp_path
    ) -> None:
        package = self._package(monkeypatch, tmp_path)
        own = package / "engine" / "probes" / "ext_probe" / "assets"
        (own / "ext_probe").mkdir(parents=True)
        register_test_probe("ext_probe", assets_root=own)

        assert probe_assets_dir("ext_probe", assets_root=package / "engine" / "assets") == own / "ext_probe"

    def test_it_stays_strict_for_a_root_outside_the_package(self, register_test_probe, monkeypatch, tmp_path) -> None:
        package = self._package(monkeypatch, tmp_path)
        (tmp_path / "extension" / "ext_probe").mkdir(parents=True)
        register_test_probe("ext_probe", assets_root=tmp_path / "extension")

        pinned = package / "engine" / "assets"
        assert probe_assets_dir("ext_probe", assets_root=pinned) == pinned / "ext_probe"

    def test_any_other_pinned_root_stays_strict(self, register_test_probe, monkeypatch, tmp_path) -> None:
        package = self._package(monkeypatch, tmp_path)
        own = package / "engine" / "probes" / "ext_probe" / "assets"
        (own / "ext_probe").mkdir(parents=True)
        register_test_probe("ext_probe", assets_root=own)

        assert probe_assets_dir("ext_probe", assets_root=tmp_path / "mine") == tmp_path / "mine" / "ext_probe"


class TestBeforeTheProbeIsImported:
    """A built-in kept in its own folder is found before its class is registered."""

    def test_its_banks_and_prompts_are_found_in_a_fresh_process(self):
        # A prompt read while the probe's module loads, or a scorer in a process
        # that never imports the probe, looks the assets up with the probe unregistered.
        code = (
            "from usersim.engine.core import probes\n"
            "from usersim.engine.core.prompt_loader import load_prompt_file\n"
            "from usersim.engine.core.seeds import load_seeds\n"
            "assert not probes._PROBE_REGISTRY, 'a probe was registered before the lookup'\n"
            "print(len(load_seeds('en_US', 'general_educational', 'subjects')))\n"
            "print(sorted(load_prompt_file('general_open_ended')))\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

        assert result.returncode == 0, result.stderr
        count, keys = result.stdout.splitlines()
        assert int(count) > 0
        assert "user_agent_system_prompt" in keys
