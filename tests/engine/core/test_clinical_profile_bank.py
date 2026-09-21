# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Tests for the clinical-profile bank loader (``core.clinical_profile_bank``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from usersim.engine.core.clinical_profile_bank import (
    DEFAULT_PROFILE_ID,
    ClinicalProfileBankError,
    clinical_profile_bank_path_for,
    default_clinical_profile_bank_path,
    load_clinical_profile_bank,
    load_clinical_profile_bank_for_client,
    reset_clinical_profile_bank_cache,
)

_VALID = """
schema_version: v0.1
client: health_therapy_disclosure
bank_id: therapy_synthetic
bank_version: v0.1.0
entries:
  - id: __default__
    placeholder: true
    profile:
      risk:
        level: moderate
  - id: 11111111-1111-1111-1111-111111111111
    placeholder: false
    profile:
      risk:
        level: high
"""


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "bank.yaml"
    p.write_text(text, encoding="utf-8")
    return p


@pytest.fixture(autouse=True)
def _reset_cache():
    reset_clinical_profile_bank_cache()
    yield
    reset_clinical_profile_bank_cache()


class TestLoad:
    def test_loads_valid_bank(self, tmp_path):
        bank = load_clinical_profile_bank(_write(tmp_path, _VALID))
        assert bank.client == "health_therapy_disclosure"
        assert bank.bank_version == "v0.1.0"
        assert len(bank.profiles) == 2

    def test_by_id_and_default_fallback(self, tmp_path):
        bank = load_clinical_profile_bank(_write(tmp_path, _VALID))
        assert bank.by_id(DEFAULT_PROFILE_ID).placeholder is True
        real = bank.by_id("11111111-1111-1111-1111-111111111111")
        assert real is not None and real.profile["risk"]["level"] == "high"
        assert bank.by_id("nope") is None

    def test_stamps_bank_fields_on_entries(self, tmp_path):
        bank = load_clinical_profile_bank(_write(tmp_path, _VALID))
        for prof in bank.profiles:
            assert prof.bank_id == "therapy_synthetic"
            assert prof.bank_version == "v0.1.0"
            assert prof.client == "health_therapy_disclosure"

    def test_missing_file(self, tmp_path):
        with pytest.raises(ClinicalProfileBankError, match="file not found"):
            load_clinical_profile_bank(tmp_path / "absent.yaml")


class TestValidation:
    @pytest.mark.parametrize("field", ["schema_version", "client", "bank_id", "bank_version"])
    def test_missing_required_top_level_field(self, tmp_path, field):
        text = "\n".join(l for l in _VALID.splitlines() if not l.startswith(f"{field}:"))
        with pytest.raises(ClinicalProfileBankError, match=field):
            load_clinical_profile_bank(_write(tmp_path, text))

    def test_unsupported_schema_version(self, tmp_path):
        text = _VALID.replace("schema_version: v0.1", "schema_version: v9.9")
        with pytest.raises(ClinicalProfileBankError, match="unsupported schema_version"):
            load_clinical_profile_bank(_write(tmp_path, text))

    def test_empty_entries_rejected(self, tmp_path):
        text = "schema_version: v0.1\nclient: c\nbank_id: b\nbank_version: v1\nentries: []\n"
        with pytest.raises(ClinicalProfileBankError, match="non-empty list"):
            load_clinical_profile_bank(_write(tmp_path, text))

    def test_non_bool_placeholder_rejected(self, tmp_path):
        text = _VALID.replace("placeholder: true", "placeholder: maybe")
        with pytest.raises(ClinicalProfileBankError, match="placeholder must be a bool"):
            load_clinical_profile_bank(_write(tmp_path, text))

    def test_non_mapping_profile_rejected(self, tmp_path):
        text = """
schema_version: v0.1
client: c
bank_id: b
bank_version: v1
entries:
  - id: x
    profile: "not a mapping"
"""
        with pytest.raises(ClinicalProfileBankError, match="profile must be a mapping"):
            load_clinical_profile_bank(_write(tmp_path, text))

    def test_duplicate_id_rejected(self, tmp_path):
        text = """
schema_version: v0.1
client: c
bank_id: b
bank_version: v1
entries:
  - id: dup
    profile: {}
  - id: dup
    profile: {}
"""
        with pytest.raises(ClinicalProfileBankError, match="duplicate profile id"):
            load_clinical_profile_bank(_write(tmp_path, text))


class TestPathResolutionAndCache:
    def test_env_override_wins(self, tmp_path, monkeypatch):
        custom = _write(tmp_path, _VALID)
        monkeypatch.setenv("MY_CUSTOM_PROFILES", str(custom))
        p = clinical_profile_bank_path_for("health_therapy_disclosure", "MY_CUSTOM_PROFILES")
        assert p == custom

    def test_default_path_used_when_no_override(self):
        p = clinical_profile_bank_path_for("health_therapy_disclosure", "UNSET_VAR_XYZ")
        assert p == default_clinical_profile_bank_path("health_therapy_disclosure")

    def test_unknown_client_rejected(self):
        with pytest.raises(ClinicalProfileBankError, match="unknown client probe label"):
            default_clinical_profile_bank_path("not_a_client")

    def test_loader_caches_by_client(self, tmp_path, monkeypatch):
        custom = _write(tmp_path, _VALID)
        monkeypatch.setenv("MY_CUSTOM_PROFILES", str(custom))
        first = load_clinical_profile_bank_for_client("health_therapy_disclosure", "MY_CUSTOM_PROFILES")
        second = load_clinical_profile_bank_for_client("health_therapy_disclosure", "MY_CUSTOM_PROFILES")
        assert first is second  # cached instance
