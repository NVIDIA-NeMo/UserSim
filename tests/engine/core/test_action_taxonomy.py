# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the rich action-taxonomy loader (core.action_taxonomy).

The taxonomy is also read by ``core.agentic_bank`` for name-only
validation. This test file focuses on the **rich** loader that the
``safety_agentic`` scorer consumes — full field parsing, blast-radius
rank computation, schema-violation error messages, cache + env override.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from usersim.engine.core.action_taxonomy import (
    ActionTaxonomy,
    ActionTaxonomyError,
    load_action_taxonomy,
    load_action_taxonomy_default,
    reset_action_taxonomy_cache,
)
from usersim.engine.core._assets import packaged_assets_dir


REPO_ROOT = Path(__file__).resolve().parents[3]
SHIPPED_PATH = (packaged_assets_dir() / "safety_agentic/action_taxonomy.yaml")


@pytest.fixture(autouse=True)
def _reset_cache():
    reset_action_taxonomy_cache()
    yield
    reset_action_taxonomy_cache()


def _minimal_yaml(
    *,
    extra_entry_fields: str = "",
    override_p_user_detect: str | None = None,
    override_category: str | None = None,
    override_risk_tier: str | None = None,
    override_blast_radius: str | None = None,
    omit_field: str | None = None,
) -> str:
    """Build a minimal valid taxonomy YAML string.

    Lets tests selectively mutate one field to verify targeted
    error messages.
    """
    category = override_category or "read_only"
    risk_tier = override_risk_tier or "info"
    blast = override_blast_radius or "none"
    p_user_detect = override_p_user_detect or (
        "p_user_detect:\n"
        "      value: 1.0\n"
        "      rationale: >\n"
        "        N/A.\n"
    )
    lines = [
        'schema_version: "v0.1"',
        'risk_tiers: ["info", "low", "medium", "high", "critical"]',
        "blast_radius_levels:",
        "  - none",
        "  - user_only",
        "  - user_household",
        "  - user_org",
        "  - user_org_external",
        "  - public",
        "categories:",
        "  - read_only",
        "  - local_state_write",
        "  - external_communication",
        "entries:",
        "  - name: read_public",
        "    description: >",
        "      Read public data.",
        f"    category: {category}",
        f"    risk_tier: {risk_tier}",
        "    reversible: true",
        "    requires_confirmation: false",
        "    consequence_disclosure_required: false",
        f"    modeled_blast_radius: {blast}",
        f"    {p_user_detect}",
        '    tags: ["benign"]',
        "    provenance:",
        '      source: "test"',
        '      last_reviewed: "2026-04-21"',
    ]
    if extra_entry_fields:
        lines.append(extra_entry_fields)
    # Optionally remove a required field.
    if omit_field:
        lines = [line for line in lines if not line.strip().startswith(omit_field)]
    return "\n".join(lines) + "\n"


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 1. Minimal load + core shape
# ---------------------------------------------------------------------------


class TestMinimalLoad:
    def test_loads_valid_minimal(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "t.yaml", _minimal_yaml())
        tax = load_action_taxonomy(p)
        assert isinstance(tax, ActionTaxonomy)
        assert tax.schema_version == "v0.1"
        assert tax.entries[0].name == "read_public"
        assert tax.entries[0].blast_radius_rank == 0

    def test_loads_shipped(self) -> None:
        tax = load_action_taxonomy(SHIPPED_PATH)
        assert len(tax.entries) == 43
        # Spot-check a known high-risk entry.
        tfx = tax.by_name("transfer_funds_external")
        assert tfx is not None
        assert tfx.risk_tier == "critical"
        assert tfx.requires_confirmation is True
        assert tfx.consequence_disclosure_required is True
        assert tfx.blast_radius_rank == 4  # user_org_external
        assert tfx.p_user_detect_value == 0.9
        assert tfx.provenance is not None

    def test_by_name_returns_none_for_unknown(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "t.yaml", _minimal_yaml())
        tax = load_action_taxonomy(p)
        assert tax.by_name("does_not_exist") is None

    def test_blast_radius_rank_returns_none_for_unknown_level(
        self, tmp_path: Path,
    ) -> None:
        p = _write(tmp_path / "t.yaml", _minimal_yaml())
        tax = load_action_taxonomy(p)
        assert tax.blast_radius_rank("none") == 0
        assert tax.blast_radius_rank("public") == 5
        assert tax.blast_radius_rank("not_a_level") is None


# ---------------------------------------------------------------------------
# 2. Schema validation errors
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    def test_missing_file_rejected(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.yaml"
        with pytest.raises(ActionTaxonomyError, match="file does not exist"):
            load_action_taxonomy(missing)

    def test_non_mapping_top_level_rejected(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "t.yaml", "- just_a_list\n")
        with pytest.raises(ActionTaxonomyError, match="top-level must be a mapping"):
            load_action_taxonomy(p)

    def test_invalid_yaml_rejected(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "t.yaml", "schema_version: [unclosed\n")
        with pytest.raises(ActionTaxonomyError, match="YAML parse error"):
            load_action_taxonomy(p)

    def test_missing_schema_version_rejected(self, tmp_path: Path) -> None:
        y = _minimal_yaml().replace('schema_version: "v0.1"\n', "")
        p = _write(tmp_path / "t.yaml", y)
        with pytest.raises(ActionTaxonomyError, match="schema_version"):
            load_action_taxonomy(p)

    def test_missing_risk_tiers_rejected(self, tmp_path: Path) -> None:
        y = _minimal_yaml().replace(
            'risk_tiers: ["info", "low", "medium", "high", "critical"]\n', "",
        )
        p = _write(tmp_path / "t.yaml", y)
        with pytest.raises(ActionTaxonomyError, match="risk_tiers"):
            load_action_taxonomy(p)

    def test_missing_entries_rejected(self, tmp_path: Path) -> None:
        y = _minimal_yaml().split("entries:")[0]
        p = _write(tmp_path / "t.yaml", y)
        with pytest.raises(ActionTaxonomyError, match="entries"):
            load_action_taxonomy(p)

    def test_unknown_category_rejected(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path / "t.yaml",
            _minimal_yaml(override_category="not_a_category"),
        )
        with pytest.raises(ActionTaxonomyError, match="category.*not in declared"):
            load_action_taxonomy(p)

    def test_unknown_risk_tier_rejected(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path / "t.yaml",
            _minimal_yaml(override_risk_tier="super_critical"),
        )
        with pytest.raises(ActionTaxonomyError, match="risk_tier.*not in declared"):
            load_action_taxonomy(p)

    def test_unknown_blast_radius_rejected(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path / "t.yaml",
            _minimal_yaml(override_blast_radius="universe"),
        )
        with pytest.raises(ActionTaxonomyError, match="modeled_blast_radius"):
            load_action_taxonomy(p)

    def test_p_user_detect_out_of_range_rejected(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path / "t.yaml",
            _minimal_yaml(override_p_user_detect=(
                "p_user_detect:\n"
                "      value: 1.5\n"
                "      rationale: >\n"
                "        out of range\n"
            )),
        )
        with pytest.raises(ActionTaxonomyError, match="p_user_detect.value"):
            load_action_taxonomy(p)

    def test_p_user_detect_not_a_mapping_rejected(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path / "t.yaml",
            _minimal_yaml(override_p_user_detect="p_user_detect: 0.5\n"),
        )
        with pytest.raises(ActionTaxonomyError, match="p_user_detect.*mapping"):
            load_action_taxonomy(p)

    def test_duplicate_entry_names_rejected(self, tmp_path: Path) -> None:
        y = _minimal_yaml() + (
            "  - name: read_public\n"
            "    description: duplicate\n"
            "    category: read_only\n"
            "    risk_tier: info\n"
            "    reversible: true\n"
            "    requires_confirmation: false\n"
            "    consequence_disclosure_required: false\n"
            "    modeled_blast_radius: none\n"
            "    p_user_detect:\n"
            "      value: 1.0\n"
            "      rationale: >\n"
            "        dup.\n"
            '    tags: []\n'
        )
        p = _write(tmp_path / "t.yaml", y)
        with pytest.raises(ActionTaxonomyError, match="duplicate name"):
            load_action_taxonomy(p)


# ---------------------------------------------------------------------------
# 3. Cache + env override
# ---------------------------------------------------------------------------


class TestCacheAndEnvOverride:
    def test_cache_returns_same_instance(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "t.yaml", _minimal_yaml())
        with patch.dict(os.environ, {"USERSIM_SAFETY_AGENTIC_ACTION_TAXONOMY": str(p)}):
            first = load_action_taxonomy_default()
            second = load_action_taxonomy_default()
            assert first is second

    def test_env_override_changes_loaded_path(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "override.yaml", _minimal_yaml())
        with patch.dict(os.environ, {"USERSIM_SAFETY_AGENTIC_ACTION_TAXONOMY": str(p)}):
            tax = load_action_taxonomy_default()
            assert tax.source_path == str(p)
            assert tax.entries[0].name == "read_public"

    def test_cache_reset_reloads(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "t.yaml", _minimal_yaml())
        with patch.dict(os.environ, {"USERSIM_SAFETY_AGENTIC_ACTION_TAXONOMY": str(p)}):
            first = load_action_taxonomy_default()
            reset_action_taxonomy_cache()
            second = load_action_taxonomy_default()
            assert first is not second

    def test_shipped_taxonomy_loads_via_default(self) -> None:
        # Default (no override) should load the shipped file when cwd is
        # the repo root. Test runs from repo root via pytest config.
        tax = load_action_taxonomy_default()
        assert len(tax.entries) == 43
