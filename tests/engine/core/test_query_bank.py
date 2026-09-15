# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``core/query_bank.py`` — cross-locale query bank loader.

Three layers, mirroring the test patterns in ``test_fact_bank.py`` and
``test_probing_taxonomy.py``:

1. **Schema validation** — every required field, every cross-product
   invariant (rendering ⨯ locale), every error path.
2. **Lookup helpers** — `by_id`, `by_domain`, `supporting_locale`,
   `matching_persona_tags`.
3. **Cache + env override + shipped sample bank** — process-local
   cache behaves correctly, env override picks up an alternate path,
   the shipped sample bank loads cleanly (an end-to-end check
   that the schema and the sample stay in sync).
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from usersim.engine.core.query_bank import (
    QueryBank,
    QueryBankError,
    default_query_bank_path,
    load_query_bank,
    load_query_bank_default,
    reset_query_bank_cache,
)
from usersim.engine.core.locale import SHIPPED_LOCALES
from usersim.engine.core._assets import packaged_assets_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_bank_yaml(
    *,
    bank_id: str = "test_bank",
    bank_version: str = "v0.1.0",
    extra_entries: str = "",
    locales: tuple[str, ...] = ("en_US", "pt_BR"),
    domains: tuple[str, ...] = ("health",),
) -> str:
    """Build a minimal valid bank as a YAML string. Tests append /
    mutate from this base so they don't all re-write the boilerplate.

    Constructed directly (not via ``textwrap.dedent``) because dedent
    can't handle f-string interpolations with different per-line
    indentation.
    """
    lines: list[str] = []
    lines.append('schema_version: "v0.1"')
    lines.append(f'bank_id: "{bank_id}"')
    lines.append(f'bank_version: "{bank_version}"')
    lines.append("locales:")
    for loc in locales:
        lines.append(f"  - {loc}")
    lines.append("domains:")
    for dom in domains:
        lines.append(f"  - {dom}")
    lines.append("entries:")
    lines.append("  - id: Q-TEST-001")
    lines.append("    domain: health")
    lines.append("    placeholder: true")
    lines.append("    difficulty: easy")
    lines.append(
        '    persona_tags: ["interest:health", "age:any", "education:any"]'
    )
    lines.append("    concern: |")
    lines.append("      Test concern describing what failure mode this query is")
    lines.append("      meant to surface.")
    lines.append("    renderings:")
    for loc in locales:
        lines.append(f"      {loc}: |")
        lines.append(f"        Sample question in {loc}.")
    lines.append("    provenance:")
    lines.append('      source: "placeholder"')
    lines.append('      last_reviewed: "2026-04-20"')
    if extra_entries:
        lines.append(extra_entries.rstrip("\n"))
    return "\n".join(lines) + "\n"


def _write_bank(tmp_path: Path, yaml_text: str, name: str = "bank.yaml") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / name
    p.write_text(yaml_text, encoding="utf-8")
    return p


@pytest.fixture(autouse=True)
def _reset_cache_between_tests():
    reset_query_bank_cache()
    yield
    reset_query_bank_cache()


# ---------------------------------------------------------------------------
# Happy path: minimal valid bank
# ---------------------------------------------------------------------------


class TestMinimalBankLoad:
    def test_loads_minimal_bank(self, tmp_path: Path) -> None:
        path = _write_bank(tmp_path, _minimal_bank_yaml())
        bank = load_query_bank(path)
        assert isinstance(bank, QueryBank)
        assert bank.bank_id == "test_bank"
        assert bank.bank_version == "v0.1.0"
        assert bank.locales == ("en_US", "pt_BR")
        assert bank.domains == ("health",)
        assert len(bank.queries) == 1

    def test_query_carries_bank_provenance(self, tmp_path: Path) -> None:
        path = _write_bank(tmp_path, _minimal_bank_yaml())
        bank = load_query_bank(path)
        q = bank.queries[0]
        assert q.bank_id == "test_bank"
        assert q.bank_version == "v0.1.0"

    def test_query_renderings_indexed_by_locale(self, tmp_path: Path) -> None:
        path = _write_bank(tmp_path, _minimal_bank_yaml())
        bank = load_query_bank(path)
        q = bank.queries[0]
        assert q.rendering_for("en_US").startswith("Sample question in en_US")
        assert q.rendering_for("pt_BR").startswith("Sample question in pt_BR")
        assert q.rendering_for("ja_JP") is None  # not in this bank

    def test_provenance_summary(self, tmp_path: Path) -> None:
        path = _write_bank(tmp_path, _minimal_bank_yaml())
        bank = load_query_bank(path)
        s = bank.provenance_summary()
        assert s["bank_id"] == "test_bank"
        assert s["n_queries"] == 1
        assert s["n_placeholders"] == 1


# ---------------------------------------------------------------------------
# Schema-version + top-level field validation
# ---------------------------------------------------------------------------


class TestTopLevelValidation:
    def test_unknown_schema_version_rejected(self, tmp_path: Path) -> None:
        path = _write_bank(
            tmp_path,
            _minimal_bank_yaml().replace('"v0.1"', '"v9.99"'),
        )
        with pytest.raises(QueryBankError, match="not supported"):
            load_query_bank(path)

    def test_missing_bank_id_rejected(self, tmp_path: Path) -> None:
        text = _minimal_bank_yaml().replace('bank_id: "test_bank"\n', "")
        path = _write_bank(tmp_path, text)
        with pytest.raises(QueryBankError, match="bank_id"):
            load_query_bank(path)

    def test_empty_locales_rejected(self, tmp_path: Path) -> None:
        text = _minimal_bank_yaml().replace(
            "locales:\n  - en_US\n  - pt_BR",
            "locales: []",
        )
        path = _write_bank(tmp_path, text)
        with pytest.raises(QueryBankError, match="locales"):
            load_query_bank(path)

    def test_duplicate_locale_rejected(self, tmp_path: Path) -> None:
        text = _minimal_bank_yaml().replace(
            "locales:\n  - en_US\n  - pt_BR",
            "locales:\n  - en_US\n  - en_US",
        )
        path = _write_bank(tmp_path, text)
        with pytest.raises(QueryBankError, match="duplicate"):
            load_query_bank(path)

    def test_empty_domains_rejected(self, tmp_path: Path) -> None:
        text = _minimal_bank_yaml().replace(
            "domains:\n  - health",
            "domains: []",
        )
        path = _write_bank(tmp_path, text)
        with pytest.raises(QueryBankError, match="domains"):
            load_query_bank(path)

    def test_empty_entries_rejected(self, tmp_path: Path) -> None:
        path = _write_bank(
            tmp_path,
            _minimal_bank_yaml().split("entries:")[0] + "entries: []\n",
        )
        with pytest.raises(QueryBankError, match="entries"):
            load_query_bank(path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(QueryBankError, match="not found"):
            load_query_bank(tmp_path / "absent.yaml")

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("not: valid: yaml:\n  - [\n", encoding="utf-8")
        with pytest.raises(QueryBankError, match="YAML parse failure"):
            load_query_bank(path)


# ---------------------------------------------------------------------------
# Entry-level validation
# ---------------------------------------------------------------------------


class TestEntryValidation:
    def test_duplicate_id_rejected(self, tmp_path: Path) -> None:
        # Append a second entry with the same id. Entry indentation
        # must match what _minimal_bank_yaml uses (2 spaces under
        # `entries:`).
        extra = "\n".join([
            "  - id: Q-TEST-001",
            "    domain: health",
            "    placeholder: true",
            "    difficulty: easy",
            '    persona_tags: ["age:any"]',
            "    concern: |",
            "      Same id as the first.",
            "    renderings:",
            "      en_US: |",
            "        duplicate",
            "      pt_BR: |",
            "        duplicada",
            "    provenance:",
            '      source: "placeholder"',
            '      last_reviewed: "2026-04-20"',
        ])
        path = _write_bank(tmp_path, _minimal_bank_yaml(extra_entries=extra))
        with pytest.raises(QueryBankError, match="duplicate query id"):
            load_query_bank(path)

    def test_unknown_domain_rejected(self, tmp_path: Path) -> None:
        text = _minimal_bank_yaml().replace(
            "domain: health",
            "domain: not-in-the-bank",
        )
        path = _write_bank(tmp_path, text)
        with pytest.raises(QueryBankError, match="not in the bank's "):
            load_query_bank(path)

    def test_invalid_difficulty_rejected(self, tmp_path: Path) -> None:
        text = _minimal_bank_yaml().replace(
            "difficulty: easy",
            "difficulty: super-hard",
        )
        path = _write_bank(tmp_path, text)
        with pytest.raises(QueryBankError, match="difficulty"):
            load_query_bank(path)

    def test_missing_concern_rejected(self, tmp_path: Path) -> None:
        text = _minimal_bank_yaml().replace(
            "    concern: |\n"
            "      Test concern describing what failure mode this query is\n"
            "      meant to surface.",
            "    concern: ''",
        )
        path = _write_bank(tmp_path, text)
        with pytest.raises(QueryBankError, match="concern"):
            load_query_bank(path)

    def test_empty_persona_tags_rejected(self, tmp_path: Path) -> None:
        text = _minimal_bank_yaml().replace(
            'persona_tags: ["interest:health", "age:any", "education:any"]',
            "persona_tags: []",
        )
        path = _write_bank(tmp_path, text)
        with pytest.raises(QueryBankError, match="persona_tags"):
            load_query_bank(path)

    def test_placeholder_must_be_bool(self, tmp_path: Path) -> None:
        text = _minimal_bank_yaml().replace(
            "placeholder: true",
            "placeholder: yes-please",
        )
        path = _write_bank(tmp_path, text)
        with pytest.raises(QueryBankError, match="placeholder"):
            load_query_bank(path)

    def test_non_placeholder_with_placeholder_source_rejected(
        self, tmp_path: Path,
    ) -> None:
        text = _minimal_bank_yaml().replace(
            "placeholder: true",
            "placeholder: false",
        )
        # Note: the default minimal bank uses source: "placeholder".
        # Keeping placeholder=False with that source should fail.
        path = _write_bank(tmp_path, text)
        with pytest.raises(QueryBankError, match="'placeholder' is only allowed"):
            load_query_bank(path)

    def test_non_iso_date_rejected(self, tmp_path: Path) -> None:
        text = _minimal_bank_yaml().replace(
            'last_reviewed: "2026-04-20"',
            'last_reviewed: "20-04-2026"',
        )
        path = _write_bank(tmp_path, text)
        with pytest.raises(QueryBankError, match="last_reviewed"):
            load_query_bank(path)


# ---------------------------------------------------------------------------
# Renderings cross-product
# ---------------------------------------------------------------------------


class TestRenderingsCrossProduct:
    def test_missing_locale_rendering_rejected(self, tmp_path: Path) -> None:
        # Bank declares en_US + pt_BR + ja_JP, but the entry only has
        # en_US + pt_BR renderings — should fail unless ja_JP is in
        # renderings_opted_out.
        bank_yaml = textwrap.dedent("""
            schema_version: "v0.1"
            bank_id: "test_bank"
            bank_version: "v0.1.0"
            locales:
              - en_US
              - pt_BR
              - ja_JP
            domains:
              - health
            entries:
              - id: Q-TEST-001
                domain: health
                placeholder: true
                difficulty: easy
                persona_tags: ["age:any"]
                concern: |
                  Test concern.
                renderings:
                  en_US: "Hello"
                  pt_BR: "Olá"
                provenance:
                  source: "placeholder"
                  last_reviewed: "2026-04-20"
            """).strip()
        path = _write_bank(tmp_path, bank_yaml)
        with pytest.raises(
            QueryBankError, match="missing rendering for locale"
        ):
            load_query_bank(path)

    def test_opt_out_skips_locale_check(self, tmp_path: Path) -> None:
        bank_yaml = textwrap.dedent("""
            schema_version: "v0.1"
            bank_id: "test_bank"
            bank_version: "v0.1.0"
            locales:
              - en_US
              - pt_BR
              - ja_JP
            domains:
              - public-services
            entries:
              - id: Q-USTAX-001
                domain: public-services
                placeholder: true
                difficulty: easy
                persona_tags: ["age:any"]
                concern: |
                  Question that's US-specific and has no meaningful
                  rendering in pt_BR or ja_JP.
                renderings:
                  en_US: "When is the US federal tax deadline?"
                renderings_opted_out:
                  - pt_BR
                  - ja_JP
                provenance:
                  source: "placeholder"
                  last_reviewed: "2026-04-20"
            """).strip()
        path = _write_bank(tmp_path, bank_yaml)
        bank = load_query_bank(path)
        q = bank.queries[0]
        assert q.supports_locale("en_US")
        assert not q.supports_locale("pt_BR")
        assert not q.supports_locale("ja_JP")
        assert q.renderings_opted_out == ("pt_BR", "ja_JP")

    def test_locale_in_both_renderings_and_opt_out_rejected(
        self, tmp_path: Path,
    ) -> None:
        bank_yaml = textwrap.dedent("""
            schema_version: "v0.1"
            bank_id: "test_bank"
            bank_version: "v0.1.0"
            locales:
              - en_US
              - pt_BR
            domains:
              - health
            entries:
              - id: Q-TEST-001
                domain: health
                placeholder: true
                difficulty: easy
                persona_tags: ["age:any"]
                concern: |
                  test
                renderings:
                  en_US: "Hi"
                  pt_BR: "Oi"
                renderings_opted_out:
                  - pt_BR
                provenance:
                  source: "placeholder"
                  last_reviewed: "2026-04-20"
            """).strip()
        path = _write_bank(tmp_path, bank_yaml)
        with pytest.raises(QueryBankError, match="appears in BOTH"):
            load_query_bank(path)

    def test_opt_out_must_be_in_bank_locales(self, tmp_path: Path) -> None:
        bank_yaml = textwrap.dedent("""
            schema_version: "v0.1"
            bank_id: "test_bank"
            bank_version: "v0.1.0"
            locales:
              - en_US
              - pt_BR
            domains:
              - health
            entries:
              - id: Q-TEST-001
                domain: health
                placeholder: true
                difficulty: easy
                persona_tags: ["age:any"]
                concern: |
                  test
                renderings:
                  en_US: "Hi"
                  pt_BR: "Oi"
                renderings_opted_out:
                  - de_DE
                provenance:
                  source: "placeholder"
                  last_reviewed: "2026-04-20"
            """).strip()
        path = _write_bank(tmp_path, bank_yaml)
        with pytest.raises(QueryBankError, match="catches typos"):
            load_query_bank(path)

    def test_rendering_locale_must_be_in_bank_locales(self, tmp_path: Path) -> None:
        bank_yaml = textwrap.dedent("""
            schema_version: "v0.1"
            bank_id: "test_bank"
            bank_version: "v0.1.0"
            locales:
              - en_US
              - pt_BR
            domains:
              - health
            entries:
              - id: Q-TEST-001
                domain: health
                placeholder: true
                difficulty: easy
                persona_tags: ["age:any"]
                concern: |
                  test
                renderings:
                  en_US: "Hi"
                  pt_BR: "Oi"
                  de_DE: "Hallo"
                provenance:
                  source: "placeholder"
                  last_reviewed: "2026-04-20"
            """).strip()
        path = _write_bank(tmp_path, bank_yaml)
        with pytest.raises(
            QueryBankError, match="not in the bank's declared locales"
        ):
            load_query_bank(path)


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


class TestLookups:
    def test_by_id(self, tmp_path: Path) -> None:
        path = _write_bank(tmp_path, _minimal_bank_yaml())
        bank = load_query_bank(path)
        q = bank.by_id("Q-TEST-001")
        assert q is not None and q.id == "Q-TEST-001"
        assert bank.by_id("nonexistent") is None

    def test_by_domain(self, tmp_path: Path) -> None:
        path = _write_bank(tmp_path, _minimal_bank_yaml())
        bank = load_query_bank(path)
        assert len(bank.by_domain("health")) == 1
        assert len(bank.by_domain("public-services")) == 0

    def test_supporting_locale(self, tmp_path: Path) -> None:
        path = _write_bank(tmp_path, _minimal_bank_yaml())
        bank = load_query_bank(path)
        assert len(bank.supporting_locale("en_US")) == 1
        assert len(bank.supporting_locale("ja_JP")) == 0  # not in this bank

    def test_query_ids(self, tmp_path: Path) -> None:
        path = _write_bank(tmp_path, _minimal_bank_yaml())
        bank = load_query_bank(path)
        assert bank.query_ids() == ("Q-TEST-001",)


# ---------------------------------------------------------------------------
# Persona-tag matcher
# ---------------------------------------------------------------------------


class TestPersonaTagMatching:
    def test_wildcard_matches_anyone(self, tmp_path: Path) -> None:
        path = _write_bank(tmp_path, _minimal_bank_yaml())
        bank = load_query_bank(path)
        # Q-TEST-001 has age:any + education:any + interest:health.
        # A persona with the right interest matches via wildcards.
        ms = bank.matching_persona_tags(
            ["interest:health", "age:18-44", "education:tertiary"]
        )
        assert len(ms) == 1

    def test_missing_required_interest_no_match(self, tmp_path: Path) -> None:
        path = _write_bank(tmp_path, _minimal_bank_yaml())
        bank = load_query_bank(path)
        ms = bank.matching_persona_tags(
            ["interest:music", "age:18-44", "education:tertiary"]
        )
        assert len(ms) == 0

    def test_persona_wildcard_satisfies_specific_tag(
        self, tmp_path: Path,
    ) -> None:
        # Bank entry requires age:45-64. A persona carrying age:any
        # (a wildcard from the persona side, used in tests) should
        # satisfy.
        bank_yaml = textwrap.dedent("""
            schema_version: "v0.1"
            bank_id: "test_bank"
            bank_version: "v0.1.0"
            locales:
              - en_US
            domains:
              - health
            entries:
              - id: Q-AGE-001
                domain: health
                placeholder: true
                difficulty: easy
                persona_tags: ["age:45-64", "education:any"]
                concern: |
                  test
                renderings:
                  en_US: "test"
                provenance:
                  source: "placeholder"
                  last_reviewed: "2026-04-20"
            """).strip()
        path = _write_bank(tmp_path, bank_yaml)
        bank = load_query_bank(path)
        ms = bank.matching_persona_tags(["age:any", "education:any"])
        assert len(ms) == 1

    def test_specific_tag_not_satisfied_returns_no_match(
        self, tmp_path: Path,
    ) -> None:
        bank_yaml = textwrap.dedent("""
            schema_version: "v0.1"
            bank_id: "test_bank"
            bank_version: "v0.1.0"
            locales:
              - en_US
            domains:
              - health
            entries:
              - id: Q-AGE-001
                domain: health
                placeholder: true
                difficulty: easy
                persona_tags: ["age:45-64", "education:any"]
                concern: |
                  test
                renderings:
                  en_US: "test"
                provenance:
                  source: "placeholder"
                  last_reviewed: "2026-04-20"
            """).strip()
        path = _write_bank(tmp_path, bank_yaml)
        bank = load_query_bank(path)
        ms = bank.matching_persona_tags(["age:18-44", "education:tertiary"])
        assert len(ms) == 0


# ---------------------------------------------------------------------------
# Cache + env override + default bank
# ---------------------------------------------------------------------------


class TestCacheAndDefault:
    def test_load_once_cached(self, tmp_path: Path) -> None:
        path = _write_bank(tmp_path, _minimal_bank_yaml())
        with patch.dict(os.environ, {"USERSIM_SOV_AI_MULTILINGUAL_PARITY_BANK": str(path)}):
            b1 = load_query_bank_default()
            b2 = load_query_bank_default()
        assert b1 is b2

    def test_env_override_picks_up_alternate_path(self, tmp_path: Path) -> None:
        path = _write_bank(
            tmp_path,
            _minimal_bank_yaml(bank_id="alt_bank"),
            name="alt.yaml",
        )
        with patch.dict(os.environ, {"USERSIM_SOV_AI_MULTILINGUAL_PARITY_BANK": str(path)}):
            bank = load_query_bank_default()
        assert bank.bank_id == "alt_bank"

    def test_reset_drops_cache(self, tmp_path: Path) -> None:
        a = _write_bank(
            tmp_path / "a", _minimal_bank_yaml(bank_id="a"),
        )
        b = _write_bank(
            tmp_path / "b", _minimal_bank_yaml(bank_id="b"), name="bank.yaml",
        )
        with patch.dict(os.environ, {"USERSIM_SOV_AI_MULTILINGUAL_PARITY_BANK": str(a)}):
            assert load_query_bank_default().bank_id == "a"
        reset_query_bank_cache()
        with patch.dict(os.environ, {"USERSIM_SOV_AI_MULTILINGUAL_PARITY_BANK": str(b)}):
            assert load_query_bank_default().bank_id == "b"

    def test_default_path_resolves_to_assets(self) -> None:
        p = default_query_bank_path()
        assert p.parts[-3:] == ("assets", "sov_ai_multilingual_parity", "sample.yaml")


# ---------------------------------------------------------------------------
# Shipped sample bank — end-to-end check that schema + content stay in sync
# ---------------------------------------------------------------------------


class TestShippedSampleBank:
    @pytest.fixture
    def sample_bank(self) -> QueryBank:
        return load_query_bank((packaged_assets_dir() / "sov_ai_multilingual_parity/sample.yaml"))

    def test_loads_clean(self, sample_bank: QueryBank) -> None:
        assert sample_bank.bank_id == "sample_v1"
        assert sample_bank.bank_version == "v0.6.0"

    def test_all_supported_locales(self, sample_bank: QueryBank) -> None:
        assert set(sample_bank.locales) == set(SHIPPED_LOCALES)

    def test_every_query_has_rendering_for_every_locale(
        self, sample_bank: QueryBank,
    ) -> None:
        # The shipped sample doesn't use renderings_opted_out — every
        # query should support every locale. Catches accidental drift.
        for q in sample_bank.queries:
            for loc in sample_bank.locales:
                assert q.supports_locale(loc), (
                    f"shipped sample query {q.id} has no rendering for {loc}"
                )

    def test_reviewed_queries_have_native_renderings(self, sample_bank: QueryBank) -> None:
        # Generated bracket-label rows are intentionally placeholders;
        # reviewed rows must not carry language-label scaffolding.
        for q in sample_bank.queries:
            if q.placeholder:
                continue
            assert all(not text.lstrip().startswith("[") for text in q.renderings.values())

    def test_every_query_has_a_concern(self, sample_bank: QueryBank) -> None:
        # Concern is the load-bearing reviewer-facing field — the
        # shipped sample must demonstrate every entry carries a useful
        # concern, since reviewers will copy this format.
        for q in sample_bank.queries:
            assert len(q.concern) >= 50, (
                f"{q.id} concern is too short ({len(q.concern)} chars) — "
                "should be a real description of what failure mode the "
                "query surfaces"
            )

    def test_domains_cover_expected_set(self, sample_bank: QueryBank) -> None:
        # Catch accidental domain renames; the sample is documented
        # as covering the beta matched-pair domain matrix.
        assert set(sample_bank.domains) == {
            "health", "public-services", "civic", "tech", "finance",
            "education", "employment", "housing-consumer",
        }

    def test_provenance_summary_smoke(self, sample_bank: QueryBank) -> None:
        s = sample_bank.provenance_summary()
        assert s["n_queries"] >= 1
        assert s["n_placeholders"] == 40
        assert s["bank_id"] == "sample_v1"
