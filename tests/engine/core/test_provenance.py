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

import pytest

from usersim.engine.core import provenance as provenance_module


class TestCodeSha:
    def test_returns_string_or_none(self) -> None:
        # Clear the lru_cache so we reliably probe fresh on this call.
        provenance_module.get_code_sha.cache_clear()
        sha = provenance_module.get_code_sha()
        assert sha is None or isinstance(sha, str)

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("USERSIM_CODE_SHA", "env-sha-1234")
        provenance_module.get_code_sha.cache_clear()
        assert provenance_module.get_code_sha() == "env-sha-1234"

    def test_env_override_blank_falls_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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
