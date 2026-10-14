# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for core/provenance.py and probe-module metadata constants.

Scope:
- ``get_code_sha`` returns a string (or None) and is callable repeatedly.
- ``USERSIM_CODE_SHA`` env var overrides git probing.
- ``get_nemotron_personas_version`` honours the env override.
- Every shipped probe module declares ``PROBE_FAMILY`` /
  ``PROBE_VARIANTS`` / ``PROMPT_VERSION``.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from importlib import metadata
from pathlib import Path

import pytest

from usersim.engine.core import provenance as provenance_module


def _repo(git: Callable[..., str], tmp_path: Path, tracked: list[str], untracked: tuple[str, ...] = ()) -> Path:
    repo = tmp_path / "repo"
    for rel in (*tracked, *untracked):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text("")
    git(repo, "init", "-q")
    git(repo, "add", *tracked)
    git(repo, "commit", "-q", "-m", "init")
    return repo


class TestCheckoutSha:
    def test_a_checkout_that_tracks_the_package_reports_its_head(self, tmp_path: Path, git: Callable[..., str]) -> None:
        repo = _repo(git, tmp_path, ["usersim/__init__.py"])
        assert provenance_module._checkout_sha(repo / "usersim") == git(repo, "rev-parse", "--short=12", "HEAD")

    def test_an_installed_copy_inside_another_repository_reports_nothing(
        self, tmp_path: Path, git: Callable[..., str]
    ) -> None:
        site = ".venv/lib/site-packages/usersim"
        repo = _repo(git, tmp_path, ["app.py"], untracked=(f"{site}/__init__.py",))
        assert provenance_module._checkout_sha(repo / site) is None

    def test_a_copy_outside_any_repository_reports_nothing(self, tmp_path: Path) -> None:
        package = tmp_path / "site-packages" / "usersim"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        assert provenance_module._checkout_sha(package) is None

    def test_without_git_it_reports_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, git: Callable[..., str]
    ) -> None:
        repo = _repo(git, tmp_path, ["usersim/__init__.py"])
        monkeypatch.setenv("PATH", str(tmp_path / "no-bin"))
        assert provenance_module._checkout_sha(repo / "usersim") is None


class TestVcsInstallSha:
    @staticmethod
    def _dist(tmp_path: Path, direct_url: dict | None) -> metadata.Distribution:
        info = tmp_path / "usersim-0.0.0.dist-info"
        info.mkdir()
        (info / "METADATA").write_text("Metadata-Version: 2.1\nName: usersim\nVersion: 0.0.0\n")
        if direct_url is not None:
            (info / "direct_url.json").write_text(json.dumps(direct_url))
        return metadata.PathDistribution(info)

    def test_a_git_install_reports_the_commit_it_recorded(self, tmp_path: Path) -> None:
        commit = "0123456789abcdef0123456789abcdef01234567"
        dist = self._dist(
            tmp_path, {"url": "https://example.com/u.git", "vcs_info": {"vcs": "git", "commit_id": commit}}
        )
        assert provenance_module._vcs_install_sha(dist) == commit[:12]

    @pytest.mark.parametrize(
        "direct_url",
        [None, {"url": "file:///src/usersim", "dir_info": {"editable": True}}],
        ids=["from-an-index", "editable"],
    )
    def test_other_installs_report_nothing(self, tmp_path: Path, direct_url: dict | None) -> None:
        assert provenance_module._vcs_install_sha(self._dist(tmp_path, direct_url)) is None

    def test_no_distribution_reports_nothing(self) -> None:
        assert provenance_module._vcs_install_sha(None) is None


class TestOwnDistribution:
    @staticmethod
    def _installed(monkeypatch: pytest.MonkeyPatch, names: set[str], claimants: list[str]) -> None:
        """Pretend ``names`` are installed and ``claimants`` claim the ``usersim`` package, in that order."""

        def distribution(name: str):
            if name not in names:
                raise metadata.PackageNotFoundError(name)
            return name

        monkeypatch.setattr(provenance_module.metadata, "distribution", distribution)
        monkeypatch.setattr(provenance_module.metadata, "packages_distributions", lambda: {"usersim": claimants})

    def test_the_distribution_named_usersim_wins_over_stale_claimants(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._installed(monkeypatch, {"usersim", "nemo-usersim"}, ["nemo-usersim", "usersim"])
        assert provenance_module.own_distribution() == "usersim"

    def test_another_claimant_is_used_when_usersim_is_not_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._installed(monkeypatch, {"nemo-usersim"}, ["nemo-usersim"])
        assert provenance_module.own_distribution() == "nemo-usersim"

    def test_nothing_installed_gives_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._installed(monkeypatch, set(), [])
        assert provenance_module.own_distribution() is None


class TestCodeSha:
    def test_the_working_directorys_repository_is_never_recorded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, git: Callable[..., str]
    ) -> None:
        """Running inside one's own repository must not record that repository's commit as UserSim's."""
        repo = _repo(git, tmp_path, ["app.py"])
        monkeypatch.delenv("USERSIM_CODE_SHA", raising=False)
        monkeypatch.chdir(repo)
        provenance_module.get_code_sha.cache_clear()
        try:
            assert provenance_module.get_code_sha() != git(repo, "rev-parse", "--short=12", "HEAD")
        finally:
            provenance_module.get_code_sha.cache_clear()

    def test_returns_string_or_none(self) -> None:
        # Clear the lru_cache so we reliably probe fresh on this call.
        provenance_module.get_code_sha.cache_clear()
        sha = provenance_module.get_code_sha()
        assert sha is None or isinstance(sha, str)

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("USERSIM_CODE_SHA", "env-sha-1234")
        provenance_module.get_code_sha.cache_clear()
        assert provenance_module.get_code_sha() == "env-sha-1234"

    def test_env_override_blank_falls_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("USERSIM_CODE_SHA", "   ")
        provenance_module.get_code_sha.cache_clear()
        # Blank env should be treated as unset; fall through to git or None.
        sha = provenance_module.get_code_sha()
        assert sha is None or (isinstance(sha, str) and len(sha) > 0)


class TestNemotronPersonasVersion:
    def test_default_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("USERSIM_NEMOTRON_PERSONAS_VERSION", raising=False)
        assert provenance_module.get_nemotron_personas_version() is None

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("USERSIM_NEMOTRON_PERSONAS_VERSION", "2026-03")
        assert provenance_module.get_nemotron_personas_version() == "2026-03"


@pytest.mark.parametrize(
    "module_path",
    [
        "usersim.engine.probes.general_open_ended.generator",
        "usersim.engine.probes.general_educational.generator",
        "usersim.engine.probes.tool_calling.generator",
    ],
)
class TestProbeMetadata:
    """Every shipped probe declares the three metadata constants."""

    def test_has_probe_family(self, module_path: str) -> None:
        mod = importlib.import_module(module_path)
        fam = getattr(mod, "PROBE_FAMILY", None)
        assert isinstance(fam, str) and fam

    def test_has_probe_variants(self, module_path: str) -> None:
        # Migrated probes declare ``PROBE_VARIANTS`` as a list at
        # source time but ``@register_probe`` coerces to tuple to
        # signal immutability — accept either sequence type.
        mod = importlib.import_module(module_path)
        variants = getattr(mod, "PROBE_VARIANTS", None)
        assert isinstance(variants, (list, tuple)) and variants
        assert all(isinstance(v, str) and v for v in variants)

    def test_has_prompt_version(self, module_path: str) -> None:
        mod = importlib.import_module(module_path)
        pv = getattr(mod, "PROMPT_VERSION", None)
        assert isinstance(pv, str) and pv
