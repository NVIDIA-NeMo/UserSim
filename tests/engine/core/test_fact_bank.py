# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``usersim.engine.core.fact_bank``.

The loader is the entry-point that subject-matter reviewers' YAML
files pass through. Every schema invariant is exercised here so a
malformed bank fails at load time, not deep inside a simulator run.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from usersim.engine.core.fact_bank import (
    ALLOWED_DIFFICULTIES,
    ALLOWED_QUESTION_TYPES,
    FactBank,
    FactBankError,
    SUPPORTED_SCHEMA_VERSIONS,
    load_fact_bank,
)
from usersim.engine.core._assets import packaged_assets_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, content: str) -> Path:
    """Write YAML to a temp file. Content is taken verbatim — callers are
    responsible for using consistent indentation."""
    p = tmp_path / "bank.yaml"
    p.write_text(content, encoding="utf-8")
    return p


def _minimal_factual_recall_entry(id_: str = "X-001") -> str:
    """One factual_recall entry, indented for inclusion under `entries:`."""
    return (
        f"  - id: {id_}\n"
        f"    category: cat-a\n"
        f"    question_type: factual_recall\n"
        f"    placeholder: true\n"
        f'    question: "Is the sky red?"\n'
        f"    false_premises:\n"
        f'      - "Sky is red (it is blue)"\n'
        f'    ground_truth: "The sky appears blue due to Rayleigh scattering."\n'
        f'    persona_tags: ["interest:geography"]\n'
        f"    difficulty: easy\n"
        f"    provenance:\n"
        f'      source: "placeholder"\n'
        f'      last_reviewed: "2026-04-17"\n'
    )


def _minimal_bank(extra_entries: str = "") -> str:
    """A complete bank with one entry plus optional appended `entries`."""
    return (
        'schema_version: "v0.1"\n'
        "locale: en_US\n"
        "bank_id: test_bank\n"
        "bank_version: v0.1.0\n"
        "categories: [cat-a]\n"
        "entries:\n" + _minimal_factual_recall_entry() + extra_entries
    )


# ---------------------------------------------------------------------------
# Schema-level loading
# ---------------------------------------------------------------------------


class TestLoaderHappyPath:
    def test_minimal_bank_loads(self, tmp_path: Path) -> None:
        bank = load_fact_bank(_write(tmp_path, _minimal_bank()))
        assert isinstance(bank, FactBank)
        assert bank.schema_version == "v0.1"
        assert bank.locale == "en_US"
        assert bank.bank_id == "test_bank"
        assert bank.bank_version == "v0.1.0"
        assert bank.categories == ("cat-a",)
        assert len(bank.facts) == 1

    def test_fact_carries_bank_metadata(self, tmp_path: Path) -> None:
        bank = load_fact_bank(_write(tmp_path, _minimal_bank()))
        f = bank.facts[0]
        assert f.locale == "en_US"
        assert f.bank_id == "test_bank"
        assert f.bank_version == "v0.1.0"

    def test_source_path_recorded(self, tmp_path: Path) -> None:
        path = _write(tmp_path, _minimal_bank())
        bank = load_fact_bank(path)
        assert bank.source_path == str(path)

    def test_facts_are_frozen(self, tmp_path: Path) -> None:
        bank = load_fact_bank(_write(tmp_path, _minimal_bank()))
        with pytest.raises(Exception):
            bank.facts[0].question = "mutated"  # type: ignore[misc]

    def test_supported_versions_set_is_non_empty(self) -> None:
        assert "v0.1" in SUPPORTED_SCHEMA_VERSIONS

    def test_question_types_set(self) -> None:
        assert ALLOWED_QUESTION_TYPES == frozenset({"factual_recall", "completion"})

    def test_difficulties_set(self) -> None:
        assert ALLOWED_DIFFICULTIES == frozenset({"easy", "medium", "hard"})


# ---------------------------------------------------------------------------
# Schema-version + top-level shape
# ---------------------------------------------------------------------------


class TestTopLevelValidation:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FactBankError, match="file not found"):
            load_fact_bank(tmp_path / "nope.yaml")

    def test_unknown_schema_version_rejected(self, tmp_path: Path) -> None:
        bank_yaml = _minimal_bank().replace('"v0.1"', '"v9.9"')
        with pytest.raises(FactBankError, match="schema_version"):
            load_fact_bank(_write(tmp_path, bank_yaml))

    def test_top_level_must_be_mapping(self, tmp_path: Path) -> None:
        with pytest.raises(FactBankError, match="top-level must be a mapping"):
            load_fact_bank(_write(tmp_path, "- not a mapping\n"))

    def test_yaml_parse_error_surfaces(self, tmp_path: Path) -> None:
        with pytest.raises(FactBankError, match="YAML parse failure"):
            load_fact_bank(_write(tmp_path, "schema_version: 'v0.1\nbroken: ["))

    @pytest.mark.parametrize("field", ["schema_version", "locale", "bank_id", "bank_version"])
    def test_required_top_level_fields(self, tmp_path: Path, field: str) -> None:
        # Strip the field by replacing its line with a comment.
        bank_yaml = _minimal_bank()
        # Remove the requested field's line wholesale.
        new_lines = [ln for ln in bank_yaml.splitlines() if not ln.lstrip().startswith(f"{field}:")]
        with pytest.raises(FactBankError, match=field):
            load_fact_bank(_write(tmp_path, "\n".join(new_lines)))

    def test_categories_required_non_empty(self, tmp_path: Path) -> None:
        bank_yaml = _minimal_bank().replace("categories: [cat-a]", "categories: []")
        with pytest.raises(FactBankError, match="categories"):
            load_fact_bank(_write(tmp_path, bank_yaml))

    def test_entries_required_non_empty(self, tmp_path: Path) -> None:
        # Minimal bank with empty entries.
        bank_yaml = textwrap.dedent("""
        schema_version: "v0.1"
        locale: en_US
        bank_id: empty
        bank_version: v0.1.0
        categories: [cat-a]
        entries: []
        """)
        with pytest.raises(FactBankError, match="entries"):
            load_fact_bank(_write(tmp_path, bank_yaml))


# ---------------------------------------------------------------------------
# Per-entry shape
# ---------------------------------------------------------------------------


class TestEntryValidation:
    def test_duplicate_id_rejected(self, tmp_path: Path) -> None:
        dup = _minimal_factual_recall_entry("X-001")  # same id
        bank_yaml = _minimal_bank(extra_entries=dup)
        with pytest.raises(FactBankError, match="duplicate fact id"):
            load_fact_bank(_write(tmp_path, bank_yaml))

    def test_unknown_category_rejected(self, tmp_path: Path) -> None:
        bank_yaml = _minimal_bank().replace("category: cat-a", "category: cat-zzz", 1)
        with pytest.raises(FactBankError, match="cat-zzz.*not in bank categories"):
            load_fact_bank(_write(tmp_path, bank_yaml))

    def test_invalid_question_type_rejected(self, tmp_path: Path) -> None:
        bank_yaml = _minimal_bank().replace("question_type: factual_recall", "question_type: nonsense")
        with pytest.raises(FactBankError, match="question_type"):
            load_fact_bank(_write(tmp_path, bank_yaml))

    def test_invalid_difficulty_rejected(self, tmp_path: Path) -> None:
        bank_yaml = _minimal_bank().replace("difficulty: easy", "difficulty: trivial")
        with pytest.raises(FactBankError, match="difficulty"):
            load_fact_bank(_write(tmp_path, bank_yaml))

    def test_persona_tags_must_be_non_empty(self, tmp_path: Path) -> None:
        bank_yaml = _minimal_bank().replace('persona_tags: ["interest:geography"]', "persona_tags: []")
        with pytest.raises(FactBankError, match="persona_tags"):
            load_fact_bank(_write(tmp_path, bank_yaml))

    def test_persona_tag_format_enforced(self, tmp_path: Path) -> None:
        bank_yaml = _minimal_bank().replace(
            'persona_tags: ["interest:geography"]',
            'persona_tags: ["just-a-string"]',
        )
        with pytest.raises(FactBankError, match="persona_tag"):
            load_fact_bank(_write(tmp_path, bank_yaml))

    def test_unknown_persona_tag_prefix_rejected(self, tmp_path: Path) -> None:
        bank_yaml = _minimal_bank().replace(
            'persona_tags: ["interest:geography"]',
            'persona_tags: ["zodiac:libra"]',
        )
        with pytest.raises(FactBankError, match="zodiac:libra"):
            load_fact_bank(_write(tmp_path, bank_yaml))

    def test_placeholder_must_be_bool(self, tmp_path: Path) -> None:
        bank_yaml = _minimal_bank().replace("placeholder: true", "placeholder: yesplease")
        with pytest.raises(FactBankError, match="placeholder must be a bool"):
            load_fact_bank(_write(tmp_path, bank_yaml))


# ---------------------------------------------------------------------------
# question_type-specific invariants
# ---------------------------------------------------------------------------


class TestCompletionEntries:
    def test_completion_requires_graceful_unknown(self, tmp_path: Path) -> None:
        bank_yaml = textwrap.dedent("""
        schema_version: "v0.1"
        locale: en_US
        bank_id: test
        bank_version: v0.1.0
        categories: [poetry]
        entries:
          - id: P-001
            category: poetry
            question_type: completion
            placeholder: true
            question: "Continue: 'Mary had a little...'"
            ground_truth: "lamb, its fleece was white as snow."
            persona_tags: ["interest:literature"]
            difficulty: easy
            provenance:
              source: "placeholder"
              last_reviewed: "2026-04-17"
        """)
        with pytest.raises(FactBankError, match="graceful_unknown"):
            load_fact_bank(_write(tmp_path, bank_yaml))

    def test_completion_rejects_false_premises(self, tmp_path: Path) -> None:
        bank_yaml = textwrap.dedent("""
        schema_version: "v0.1"
        locale: en_US
        bank_id: test
        bank_version: v0.1.0
        categories: [poetry]
        entries:
          - id: P-001
            category: poetry
            question_type: completion
            placeholder: true
            question: "Continue: 'Mary had a little...'"
            false_premises: ["completions don't have premises"]
            ground_truth: "lamb, its fleece was white as snow."
            graceful_unknown: "I don't recall this verse precisely."
            persona_tags: ["interest:literature"]
            difficulty: easy
            provenance:
              source: "placeholder"
              last_reviewed: "2026-04-17"
        """)
        with pytest.raises(FactBankError, match="false_premises"):
            load_fact_bank(_write(tmp_path, bank_yaml))

    def test_completion_loads_when_well_formed(self, tmp_path: Path) -> None:
        bank_yaml = textwrap.dedent("""
        schema_version: "v0.1"
        locale: en_US
        bank_id: test
        bank_version: v0.1.0
        categories: [poetry]
        entries:
          - id: P-001
            category: poetry
            question_type: completion
            placeholder: true
            question: "Continue: 'Mary had a little...'"
            ground_truth: "lamb, its fleece was white as snow."
            graceful_unknown: "I don't recall this verse precisely."
            persona_tags: ["interest:literature"]
            difficulty: easy
            provenance:
              source: "placeholder"
              last_reviewed: "2026-04-17"
        """)
        bank = load_fact_bank(_write(tmp_path, bank_yaml))
        assert bank.facts[0].graceful_unknown == "I don't recall this verse precisely."
        assert bank.facts[0].false_premises == ()


class TestFactualRecallEntries:
    def test_false_premises_can_be_empty(self, tmp_path: Path) -> None:
        bank_yaml = _minimal_bank().replace(
            '    false_premises:\n      - "Sky is red (it is blue)"\n',
            "    false_premises: []\n",
        )
        bank = load_fact_bank(_write(tmp_path, bank_yaml))
        assert bank.facts[0].false_premises == ()


# ---------------------------------------------------------------------------
# Provenance discipline
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_placeholder_can_omit_references(self, tmp_path: Path) -> None:
        # The minimal sample is exactly that; loading shouldn't raise.
        bank = load_fact_bank(_write(tmp_path, _minimal_bank()))
        assert bank.facts[0].provenance.references == ()
        assert bank.facts[0].provenance.source == "placeholder"

    def test_non_placeholder_requires_real_source(self, tmp_path: Path) -> None:
        bank_yaml = _minimal_bank().replace("placeholder: true", "placeholder: false")
        # Source is still 'placeholder' string — should be rejected.
        with pytest.raises(FactBankError, match="real provenance.source"):
            load_fact_bank(_write(tmp_path, bank_yaml))

    def test_non_placeholder_requires_at_least_one_reference(self, tmp_path: Path) -> None:
        bank_yaml = (
            _minimal_bank()
            .replace(
                "placeholder: true",
                "placeholder: false",
            )
            .replace(
                'source: "placeholder"',
                'source: "Britannica entry on the sky"',
            )
        )
        # No references list at all → must fail.
        with pytest.raises(FactBankError, match="references"):
            load_fact_bank(_write(tmp_path, bank_yaml))

    def test_non_placeholder_with_references_loads(self, tmp_path: Path) -> None:
        bank_yaml = (
            _minimal_bank()
            .replace(
                "    placeholder: true\n",
                "    placeholder: false\n",
            )
            .replace(
                '      source: "placeholder"\n      last_reviewed: "2026-04-17"\n',
                (
                    '      source: "Britannica entry on Rayleigh scattering"\n'
                    '      last_reviewed: "2026-04-17"\n'
                    "      references:\n"
                    '        - "https://www.britannica.com/science/Rayleigh-scattering"\n'
                ),
            )
        )
        bank = load_fact_bank(_write(tmp_path, bank_yaml))
        assert bank.facts[0].provenance.references == ("https://www.britannica.com/science/Rayleigh-scattering",)

    def test_iso_date_format_enforced(self, tmp_path: Path) -> None:
        bank_yaml = _minimal_bank().replace(
            'last_reviewed: "2026-04-17"',
            'last_reviewed: "April 17 2026"',
        )
        with pytest.raises(FactBankError, match="YYYY-MM-DD"):
            load_fact_bank(_write(tmp_path, bank_yaml))

    def test_invalid_calendar_date_rejected(self, tmp_path: Path) -> None:
        bank_yaml = _minimal_bank().replace(
            'last_reviewed: "2026-04-17"',
            'last_reviewed: "2026-13-99"',
        )
        with pytest.raises(FactBankError, match="not a real calendar date"):
            load_fact_bank(_write(tmp_path, bank_yaml))


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------


@pytest.fixture
def multi_entry_bank(tmp_path: Path) -> FactBank:
    yaml = textwrap.dedent("""
    schema_version: "v0.1"
    locale: pt_BR
    bank_id: multi
    bank_version: v0.2.0
    categories: [geo, music]
    entries:
      - id: G-001
        category: geo
        question_type: factual_recall
        placeholder: true
        question: "Where does the Amazon flow?"
        false_premises: []
        ground_truth: "The Atlantic."
        persona_tags: ["interest:geography", "region:southeast"]
        difficulty: easy
        provenance:
          source: "placeholder"
          last_reviewed: "2026-04-17"

      - id: G-002
        category: geo
        question_type: factual_recall
        placeholder: true
        question: "Where is the Pantanal?"
        false_premises: []
        ground_truth: "Mato Grosso (do Sul)."
        persona_tags: ["interest:geography", "region:any"]
        difficulty: medium
        provenance:
          source: "placeholder"
          last_reviewed: "2026-04-17"

      - id: M-001
        category: music
        question_type: completion
        placeholder: false
        question: "Continue 'Garota de Ipanema'..."
        ground_truth: "que vem e que passa..."
        graceful_unknown: "I do not know."
        persona_tags: ["interest:music", "region:southeast"]
        difficulty: easy
        provenance:
          source: "Vinícius/Jobim 1962 sheet music"
          last_reviewed: "2026-04-17"
          references:
            - "https://example.com/garota-de-ipanema"
    """)
    p = tmp_path / "multi.yaml"
    p.write_text(yaml, encoding="utf-8")
    return load_fact_bank(p)


class TestLookups:
    def test_by_id(self, multi_entry_bank: FactBank) -> None:
        assert multi_entry_bank.by_id("G-001").id == "G-001"
        assert multi_entry_bank.by_id("nope") is None

    def test_by_category(self, multi_entry_bank: FactBank) -> None:
        geo = multi_entry_bank.by_category("geo")
        assert {f.id for f in geo} == {"G-001", "G-002"}
        assert multi_entry_bank.by_category("does-not-exist") == ()

    def test_by_tag(self, multi_entry_bank: FactBank) -> None:
        ge = multi_entry_bank.by_tag("interest:geography")
        assert {f.id for f in ge} == {"G-001", "G-002"}

    def test_by_tag_misses_for_unknown_tag(self, multi_entry_bank: FactBank) -> None:
        assert multi_entry_bank.by_tag("interest:cinema") == ()

    def test_placeholder_partition(self, multi_entry_bank: FactBank) -> None:
        ph = multi_entry_bank.placeholder_only()
        non_ph = multi_entry_bank.non_placeholder()
        assert {f.id for f in ph} == {"G-001", "G-002"}
        assert {f.id for f in non_ph} == {"M-001"}

    def test_placeholder_fraction(self, multi_entry_bank: FactBank) -> None:
        # 2 of 3 placeholders.
        assert multi_entry_bank.placeholder_fraction() == pytest.approx(2 / 3)

    def test_provenance_summary(self, multi_entry_bank: FactBank) -> None:
        summary = multi_entry_bank.provenance_summary()
        assert summary["bank_id"] == "multi"
        assert summary["bank_version"] == "v0.2.0"
        assert summary["locale"] == "pt_BR"
        assert summary["n_facts"] == 3
        assert summary["n_placeholder"] == 2
        assert set(summary["distinct_sources"]) == {
            "placeholder",
            "Vinícius/Jobim 1962 sheet music",
        }


class TestPersonaTagMatching:
    def test_concrete_tag_intersection(self, multi_entry_bank: FactBank) -> None:
        # A persona interested in music + living in southeast picks up:
        #   - G-001 (concrete intersection on region:southeast)
        #   - M-001 (concrete intersection on interest:music + region:southeast)
        #   - G-002 (its region:any wildcard matches the persona's region:southeast)
        out = multi_entry_bank.matching_persona_tags(["interest:music", "region:southeast"])
        assert {f.id for f in out} == {"G-001", "G-002", "M-001"}

    def test_persona_wildcard_matches_any_with_prefix(self, multi_entry_bank: FactBank) -> None:
        # Persona with 'region:any' should pull in every entry that has
        # any region:* tag.
        out = multi_entry_bank.matching_persona_tags(["region:any"])
        # G-001 has region:southeast, G-002 has region:any, M-001 has region:southeast
        assert {f.id for f in out} == {"G-001", "G-002", "M-001"}

    def test_fact_wildcard_matches_any_persona_with_prefix(self, multi_entry_bank: FactBank) -> None:
        # Persona with region:nordeste should pull G-002 (region:any on fact)
        # but neither G-001 nor M-001 (concrete southeast on facts).
        out = multi_entry_bank.matching_persona_tags(["region:nordeste"])
        assert {f.id for f in out} == {"G-002"}

    def test_empty_input_returns_empty(self, multi_entry_bank: FactBank) -> None:
        assert multi_entry_bank.matching_persona_tags([]) == ()

    def test_iteration_order_is_file_order(self, multi_entry_bank: FactBank) -> None:
        # by_category preserves file order; same for matching_persona_tags
        # results to keep simulator behaviour deterministic.
        out = multi_entry_bank.matching_persona_tags(["interest:geography", "interest:music"])
        # File order is G-001, G-002, M-001 — preserved.
        assert [f.id for f in out] == ["G-001", "G-002", "M-001"]


# ---------------------------------------------------------------------------
# Integration: the actual pt_BR sample bank
# ---------------------------------------------------------------------------


class TestPtBrSampleBank:
    """Loads the placeholder bank shipped at assets/sov_ai_facts/pt_BR/sample.yaml."""

    @pytest.fixture
    def bank(self) -> FactBank:
        # __file__ is at conversation-plugin/tests/test_fact_bank.py;
        # parents[3] is the repo root.
        return load_fact_bank((packaged_assets_dir() / "sov_ai_facts/pt_BR/sample.yaml"))

    def test_loads(self, bank: FactBank) -> None:
        assert bank.locale == "pt_BR"
        assert bank.bank_id == "pt_BR_sample"
        assert bank.schema_version == "v0.1"

    def test_60_entries_10_categories(self, bank: FactBank) -> None:
        assert len(bank.facts) == 60
        assert len(bank.categories) == 10
        for cat in bank.categories:
            assert len(bank.by_category(cat)) == 6, f"category {cat} expected to have 6 entries"

    def test_all_entries_placeholder_until_real_refs(self, bank: FactBank) -> None:
        # Per the schema's promotion convention, an entry promoted to
        # ``placeholder: false`` MUST carry a real (non-stub) reference
        # citation. The current bank carries authored content but only
        # boilerplate "URLs pending native/SME audit" stubs in its
        # ``provenance.references`` fields, so all 60 entries remain
        # ``placeholder: true``. A reviewer attaching real Wikipedia /
        # government / academic citations re-flips entries to
        # ``placeholder: false`` and bumps ``bank_version``; this test
        # then needs an updated invariant matching the new mix. See
        # rationale.
        assert bank.placeholder_fraction() == 1.0
        assert len(bank.non_placeholder()) == 0

    def test_completion_entries_have_graceful_unknowns(self, bank: FactBank) -> None:
        completions = [f for f in bank.facts if f.question_type == "completion"]
        assert len(completions) == 8
        assert all(f.graceful_unknown for f in completions)

    def test_factual_recall_entries_have_false_premises(self, bank: FactBank) -> None:
        # Every shipped factual_recall entry deliberately embeds at least one
        # false premise — that's the whole point of this category. The
        # pt_BR DIGI-B006 entry was migrated from completion → factual_recall
        # during the AI-authored beta pass since "Brazil poem about digital
        # services" is not a coherent completion topic, so we ship 52
        # factual + 8 completion in pt_BR.
        factual = [f for f in bank.facts if f.question_type == "factual_recall"]
        assert len(factual) == 52
        for f in factual:
            assert len(f.false_premises) >= 1, f"{f.id} declares no false premises but is factual_recall"

    def test_every_entry_carries_persona_tags(self, bank: FactBank) -> None:
        for f in bank.facts:
            assert len(f.persona_tags) >= 1, f"{f.id} has no persona_tags"

    def test_non_placeholder_entries_have_no_todo_or_placeholder_text(
        self,
        bank: FactBank,
    ) -> None:
        forbidden = (
            "reviewer todo",
            "todo do revisor",
            "esta entrada é placeholder",
            "wrong national agency",
        )
        for f in bank.non_placeholder():
            text = "\n".join([f.question, f.ground_truth, f.graceful_unknown or ""]).lower()
            assert not any(token in text for token in forbidden), f.id
