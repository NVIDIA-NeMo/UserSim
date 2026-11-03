# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``usersim.engine.core.probing_taxonomy``.

The loader is the entry point reviewers' YAML files pass through.
Every schema invariant is exercised here so a malformed taxonomy
fails at load time, not during a sov_ai_dynamic simulation.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from usersim.engine.core.probing_taxonomy import (
    MIN_SUBTOPIC_HINTS,
    SUPPORTED_SCHEMA_VERSIONS,
    Category,
    ProbingTaxonomy,
    ProbingTaxonomyError,
    load_probing_taxonomy,
    load_probing_taxonomy_for,
    reset_probing_taxonomy_cache,
    select_category,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, content: str, name: str = "tax.yaml") -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def _minimal_category(cid: str = "cat-a", hints: int = 2) -> str:
    """Render one category block (already indented for inclusion under `categories:`)."""
    hint_lines = "\n".join(f"      - 'hint {cid} #{i}'" for i in range(hints))
    return f"  - id: {cid}\n    invitation: 'invitation text for {cid}'\n    subtopic_hints:\n{hint_lines}\n"


def _minimal_taxonomy(extra_categories: str = "") -> str:
    return (
        'schema_version: "v0.1"\n'
        "locale: en_US\n"
        "taxonomy_id: test_taxonomy\n"
        "taxonomy_version: v0.1.0\n"
        "categories:\n" + _minimal_category() + extra_categories
    )


@pytest.fixture(autouse=True)
def _reset_cache_between_tests():
    """Keep the module-level cache from leaking across tests."""
    reset_probing_taxonomy_cache()
    yield
    reset_probing_taxonomy_cache()


# ---------------------------------------------------------------------------
# Schema-level loading
# ---------------------------------------------------------------------------


class TestLoaderHappyPath:
    def test_minimal_taxonomy_loads(self, tmp_path: Path) -> None:
        tax = load_probing_taxonomy(_write(tmp_path, _minimal_taxonomy()))
        assert isinstance(tax, ProbingTaxonomy)
        assert tax.schema_version == "v0.1"
        assert tax.locale == "en_US"
        assert tax.taxonomy_id == "test_taxonomy"
        assert tax.taxonomy_version == "v0.1.0"
        assert len(tax.categories) == 1
        assert tax.categories[0].id == "cat-a"
        assert len(tax.categories[0].subtopic_hints) == 2

    def test_category_carries_taxonomy_metadata(self, tmp_path: Path) -> None:
        tax = load_probing_taxonomy(_write(tmp_path, _minimal_taxonomy()))
        cat = tax.categories[0]
        assert cat.locale == "en_US"
        assert cat.taxonomy_id == "test_taxonomy"
        assert cat.taxonomy_version == "v0.1.0"

    def test_source_path_recorded(self, tmp_path: Path) -> None:
        path = _write(tmp_path, _minimal_taxonomy())
        tax = load_probing_taxonomy(path)
        assert tax.source_path == str(path)

    def test_categories_are_frozen(self, tmp_path: Path) -> None:
        tax = load_probing_taxonomy(_write(tmp_path, _minimal_taxonomy()))
        with pytest.raises(Exception):
            tax.categories[0].invitation = "mutated"  # type: ignore[misc]

    def test_supported_versions_set_is_non_empty(self) -> None:
        assert "v0.1" in SUPPORTED_SCHEMA_VERSIONS

    def test_min_subtopic_hints_constant(self) -> None:
        assert MIN_SUBTOPIC_HINTS == 2


# ---------------------------------------------------------------------------
# Top-level shape
# ---------------------------------------------------------------------------


class TestTopLevelValidation:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ProbingTaxonomyError, match="file not found"):
            load_probing_taxonomy(tmp_path / "nope.yaml")

    def test_unknown_schema_version_rejected(self, tmp_path: Path) -> None:
        bad = _minimal_taxonomy().replace('"v0.1"', '"v9.9"')
        with pytest.raises(ProbingTaxonomyError, match="schema_version"):
            load_probing_taxonomy(_write(tmp_path, bad))

    def test_top_level_must_be_mapping(self, tmp_path: Path) -> None:
        with pytest.raises(ProbingTaxonomyError, match="top-level must be a mapping"):
            load_probing_taxonomy(_write(tmp_path, "- not a mapping\n"))

    def test_yaml_parse_error_surfaces(self, tmp_path: Path) -> None:
        with pytest.raises(ProbingTaxonomyError, match="YAML parse failure"):
            load_probing_taxonomy(_write(tmp_path, "schema_version: 'v0.1\nbroken: ["))

    @pytest.mark.parametrize("field", ["schema_version", "locale", "taxonomy_id", "taxonomy_version"])
    def test_required_top_level_fields(self, tmp_path: Path, field: str) -> None:
        bad = "\n".join(ln for ln in _minimal_taxonomy().splitlines() if not ln.lstrip().startswith(f"{field}:"))
        with pytest.raises(ProbingTaxonomyError, match=field):
            load_probing_taxonomy(_write(tmp_path, bad))

    def test_categories_required_non_empty(self, tmp_path: Path) -> None:
        bad = textwrap.dedent("""
        schema_version: "v0.1"
        locale: en_US
        taxonomy_id: empty
        taxonomy_version: v0.1.0
        categories: []
        """)
        with pytest.raises(ProbingTaxonomyError, match="categories"):
            load_probing_taxonomy(_write(tmp_path, bad))


# ---------------------------------------------------------------------------
# Per-category shape
# ---------------------------------------------------------------------------


class TestCategoryValidation:
    def test_duplicate_id_rejected(self, tmp_path: Path) -> None:
        dup = _minimal_category("cat-a", hints=2)
        bad = _minimal_taxonomy(extra_categories=dup)
        with pytest.raises(ProbingTaxonomyError, match="duplicate category id"):
            load_probing_taxonomy(_write(tmp_path, bad))

    def test_missing_invitation(self, tmp_path: Path) -> None:
        bad = (
            'schema_version: "v0.1"\n'
            "locale: en_US\n"
            "taxonomy_id: x\n"
            "taxonomy_version: v0.1.0\n"
            "categories:\n"
            "  - id: cat-a\n"
            "    subtopic_hints:\n"
            "      - 'a'\n"
            "      - 'b'\n"
        )
        with pytest.raises(ProbingTaxonomyError, match="invitation"):
            load_probing_taxonomy(_write(tmp_path, bad))

    def test_subtopic_hints_must_be_list(self, tmp_path: Path) -> None:
        bad = _minimal_taxonomy().replace(
            "subtopic_hints:\n      - 'hint cat-a #0'\n      - 'hint cat-a #1'\n",
            "subtopic_hints: 'just a string'\n",
        )
        with pytest.raises(ProbingTaxonomyError, match="must be a list"):
            load_probing_taxonomy(_write(tmp_path, bad))

    def test_subtopic_hints_below_minimum_rejected(self, tmp_path: Path) -> None:
        bad = (
            'schema_version: "v0.1"\n'
            "locale: en_US\n"
            "taxonomy_id: x\n"
            "taxonomy_version: v0.1.0\n"
            "categories:\n"
            "  - id: cat-a\n"
            "    invitation: 'inv'\n"
            "    subtopic_hints:\n"
            "      - 'only one'\n"
        )
        with pytest.raises(ProbingTaxonomyError, match="at least 2 entries"):
            load_probing_taxonomy(_write(tmp_path, bad))

    def test_empty_hint_string_rejected(self, tmp_path: Path) -> None:
        bad = _minimal_taxonomy().replace(
            "      - 'hint cat-a #0'\n      - 'hint cat-a #1'\n",
            "      - 'good hint'\n      - ''\n",
        )
        with pytest.raises(ProbingTaxonomyError, match="non-empty string"):
            load_probing_taxonomy(_write(tmp_path, bad))

    def test_non_string_hint_rejected(self, tmp_path: Path) -> None:
        bad = _minimal_taxonomy().replace(
            "      - 'hint cat-a #0'\n      - 'hint cat-a #1'\n",
            "      - 'good hint'\n      - 42\n",
        )
        with pytest.raises(ProbingTaxonomyError, match="non-empty string"):
            load_probing_taxonomy(_write(tmp_path, bad))

    def test_id_required(self, tmp_path: Path) -> None:
        bad = (
            'schema_version: "v0.1"\n'
            "locale: en_US\n"
            "taxonomy_id: x\n"
            "taxonomy_version: v0.1.0\n"
            "categories:\n"
            "  - invitation: 'inv'\n"
            "    subtopic_hints:\n"
            "      - 'a'\n"
            "      - 'b'\n"
        )
        with pytest.raises(ProbingTaxonomyError, match="id"):
            load_probing_taxonomy(_write(tmp_path, bad))


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------


@pytest.fixture
def multi_category_taxonomy(tmp_path: Path) -> ProbingTaxonomy:
    extra = _minimal_category("cat-b", hints=3) + _minimal_category("cat-c", hints=4)
    return load_probing_taxonomy(_write(tmp_path, _minimal_taxonomy(extra_categories=extra)))


class TestLookups:
    def test_by_id(self, multi_category_taxonomy: ProbingTaxonomy) -> None:
        assert multi_category_taxonomy.by_id("cat-b").id == "cat-b"
        assert multi_category_taxonomy.by_id("does-not-exist") is None

    def test_category_ids(self, multi_category_taxonomy: ProbingTaxonomy) -> None:
        assert multi_category_taxonomy.category_ids() == ("cat-a", "cat-b", "cat-c")

    def test_provenance_summary(self, multi_category_taxonomy: ProbingTaxonomy) -> None:
        summary = multi_category_taxonomy.provenance_summary()
        assert summary["taxonomy_id"] == "test_taxonomy"
        assert summary["taxonomy_version"] == "v0.1.0"
        assert summary["locale"] == "en_US"
        assert summary["n_categories"] == 3
        assert summary["category_ids"] == ["cat-a", "cat-b", "cat-c"]


# ---------------------------------------------------------------------------
# Locale-keyed cache
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Integration: the actual pt_BR taxonomy
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Top-level placeholder field
# ---------------------------------------------------------------------------


class TestPlaceholderField:
    def test_omitted_defaults_to_false(self, tmp_path: Path) -> None:
        tax = load_probing_taxonomy(_write(tmp_path, _minimal_taxonomy()))
        assert tax.placeholder is False

    def test_explicit_false_loads(self, tmp_path: Path) -> None:
        body = _minimal_taxonomy().replace("categories:\n", "placeholder: false\ncategories:\n")
        tax = load_probing_taxonomy(_write(tmp_path, body))
        assert tax.placeholder is False

    def test_explicit_true_loads(self, tmp_path: Path) -> None:
        body = _minimal_taxonomy().replace("categories:\n", "placeholder: true\ncategories:\n")
        tax = load_probing_taxonomy(_write(tmp_path, body))
        assert tax.placeholder is True

    def test_non_bool_rejected(self, tmp_path: Path) -> None:
        body = _minimal_taxonomy().replace("categories:\n", 'placeholder: "yesplease"\ncategories:\n')
        with pytest.raises(ProbingTaxonomyError, match="placeholder"):
            load_probing_taxonomy(_write(tmp_path, body))

    def test_provenance_summary_carries_flag(self, tmp_path: Path) -> None:
        body = _minimal_taxonomy().replace("categories:\n", "placeholder: true\ncategories:\n")
        tax = load_probing_taxonomy(_write(tmp_path, body))
        assert tax.provenance_summary()["placeholder"] is True


# ---------------------------------------------------------------------------
# Generalized dynamic-tier substrate: entity-type scoping, persona affinities,
# select_category, and the family/filename loader. Shared by sov_ai_dynamic +
# financial_services (and future grounded-advisory probes).
# ---------------------------------------------------------------------------


def _cat(cid, *, applies=(), affinities=(), hints=("h1", "h2")):
    return Category(
        id=cid,
        invitation=f"inv {cid}",
        subtopic_hints=tuple(hints),
        applies_to_types=tuple(applies),
        persona_affinities=tuple(affinities),
        taxonomy_id="tx",
        taxonomy_version="v1",
    )


def _tax(*cats):
    return ProbingTaxonomy(
        schema_version="v0.1",
        locale="en_US",
        taxonomy_id="tx",
        taxonomy_version="v1",
        categories=tuple(cats),
    )


class TestEntityTypeScoping:
    def test_empty_applies_to_types_is_universal(self) -> None:
        c = _cat("a")
        assert c.applies_to("bank") and c.applies_to("anything")

    def test_typed_category_scopes(self) -> None:
        c = _cat("a", applies=("bank",))
        assert c.applies_to("bank") and not c.applies_to("brokerage")

    def test_categories_for_type_includes_universal(self) -> None:
        tax = _tax(
            _cat("bankcat", applies=("bank",)),
            _cat("brokcat", applies=("brokerage",)),
            _cat("universal"),
        )
        assert {c.id for c in tax.categories_for_type("bank")} == {
            "bankcat",
            "universal",
        }

    def test_loader_parses_applies_to_types_and_persona_affinities(
        self,
        tmp_path: Path,
    ) -> None:
        content = (
            'schema_version: "v0.1"\n'
            "locale: en_US\n"
            "taxonomy_id: tx\n"
            "taxonomy_version: v0.1.0\n"
            "categories:\n"
            "  - id: c1\n"
            "    applies_to_types: [bank, brokerage]\n"
            "    persona_affinities: ['age:65+', 'financial-literacy:low']\n"
            "    invitation: inv\n"
            "    subtopic_hints:\n"
            "      - h1\n"
            "      - h2\n"
        )
        tax = load_probing_taxonomy(_write(tmp_path, content))
        c = tax.by_id("c1")
        assert c.applies_to_types == ("bank", "brokerage")
        assert c.persona_affinities == ("age:65+", "financial-literacy:low")

    def test_both_fields_optional(self, tmp_path: Path) -> None:
        # The sov_ai_dynamic case: neither field present -> universal, no affinities.
        tax = load_probing_taxonomy(_write(tmp_path, _minimal_taxonomy()))
        c = tax.categories[0]
        assert c.applies_to_types == () and c.persona_affinities == ()


class TestSelectCategory:
    _PERSONA = {"name": "x", "age": 40}

    def test_deterministic_on_persona_and_seed(self) -> None:
        tax = _tax(_cat("a"), _cat("b"), _cat("c"))
        r1 = select_category(self._PERSONA, tax, seed=7)
        r2 = select_category(self._PERSONA, tax, seed=7)
        assert r1[0].id == r2[0].id and r1[1] == r2[1]

    def test_excluded_ids_skip_the_pool(self) -> None:
        tax = _tax(_cat("a"), _cat("b"))
        assert (
            select_category(
                self._PERSONA,
                tax,
                seed=7,
                excluded_category_ids={"a"},
            )[0].id
            == "b"
        )

    def test_all_excluded_returns_none(self) -> None:
        tax = _tax(_cat("a"))
        assert (
            select_category(
                self._PERSONA,
                tax,
                excluded_category_ids={"a"},
            )
            is None
        )

    def test_category_weights_bias_selection(self) -> None:
        tax = _tax(_cat("a"), _cat("b"), _cat("c"))
        weighted = [
            select_category(
                {"i": i},
                tax,
                seed=i,
                category_weights={"a": 1.0, "b": 50.0, "c": 1.0},
            )[0].id
            for i in range(40)
        ]
        uniform = [select_category({"i": i}, tax, seed=i)[0].id for i in range(40)]
        # The heavy prior dominates without changing the uniform path.
        assert weighted.count("b") >= 30
        assert weighted.count("b") > uniform.count("b")


class TestFamilyLoader:
    def test_env_override_by_family(self, tmp_path: Path) -> None:
        path = _write(tmp_path, _minimal_taxonomy(), name="dynamic.yaml")
        with patch.dict(
            os.environ,
            {"USERSIM_MYDOMAIN_DYNAMIC_TAXONOMY_EN_US": str(path)},
        ):
            tax = load_probing_taxonomy_for(
                "mydomain",
                "en_US",
                filename="dynamic.yaml",
                env_prefix="USERSIM_MYDOMAIN_DYNAMIC_TAXONOMY",
            )
        assert tax.taxonomy_id == "test_taxonomy"
