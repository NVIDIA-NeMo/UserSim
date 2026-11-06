# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``core.persona_tags``: the tags a persona matches bank entries by."""

from __future__ import annotations

from typing import Any

import pytest

from usersim.engine.core.persona_tags import persona_to_tags


@pytest.fixture
def pt_br_persona_southeast_teacher() -> dict[str, Any]:
    return {
        "first_name": "Mariana",
        "last_name": "Silva",
        "age": 37,
        "state": "SP",
        "state_abbrev": "SP",
        "city": "São Paulo",
        "education_level": "Ensino superior (graduação)",
        "occupation": "Professora de história em escola pública",
    }


@pytest.fixture
def pt_br_persona_northeast_retiree() -> dict[str, Any]:
    return {
        "first_name": "José",
        "last_name": "Santos",
        "age": 72,
        "state": "PE",
        "city": "Recife",
        "education_level": "Ensino fundamental incompleto",
        "occupation": "Aposentado",
    }


# ---------------------------------------------------------------------------
# Persona → tag mapping
# ---------------------------------------------------------------------------


class TestPersonaToTags:
    def test_brazilian_southeast_teacher_picks_right_tags(
        self, pt_br_persona_southeast_teacher: dict[str, Any]
    ) -> None:
        tags = persona_to_tags(pt_br_persona_southeast_teacher, "pt_BR")
        assert "region:southeast" in tags
        assert "region:any" in tags
        assert "age:25-44" in tags
        assert "age:any" in tags
        assert "education:tertiary" in tags
        assert "occupation-family:teacher" in tags
        assert "interest:literature" in tags or "interest:history" in tags

    def test_brazilian_northeast_retiree(self, pt_br_persona_northeast_retiree: dict[str, Any]) -> None:
        tags = persona_to_tags(pt_br_persona_northeast_retiree, "pt_BR")
        assert "region:northeast" in tags
        assert "age:65+" in tags
        assert "occupation-family:retired" in tags
        assert "education:primary" in tags

    @pytest.mark.parametrize(
        "state_code, expected_region",
        [
            ("SP", "region:southeast"),
            ("RJ", "region:southeast"),
            ("BA", "region:northeast"),
            ("PE", "region:northeast"),
            ("RS", "region:south"),
            ("PR", "region:south"),
            ("AM", "region:north"),
            ("GO", "region:central-west"),
            ("DF", "region:central-west"),
        ],
    )
    def test_all_br_macro_regions(self, state_code: str, expected_region: str) -> None:
        persona = {"state_abbrev": state_code, "age": 30, "occupation": "Engineer"}
        tags = persona_to_tags(persona, "pt_BR")
        assert expected_region in tags

    @pytest.mark.parametrize(
        "region, expected_nation",
        [
            ("London", "region:england"),
            ("Yorkshire and The Humber", "region:england"),
            ("South West", "region:england"),
            ("Scotland", "region:scotland"),
            ("Wales", "region:wales"),
            ("Northern Ireland", "region:northern-ireland"),
        ],
    )
    def test_uk_regions_map_to_their_nation(self, region: str, expected_nation: str) -> None:
        tags = persona_to_tags({"region": region, "age": 30}, "en_GB")
        assert [t for t in tags if t.startswith("region:")] == sorted(["region:any", expected_nation])

    def test_an_unrecognised_uk_region_adds_no_nation(self) -> None:
        tags = persona_to_tags({"region": "Atlantis", "age": 30}, "en_GB")
        assert [t for t in tags if t.startswith("region:")] == ["region:any"]

    def test_uk_nations_are_only_derived_for_en_gb(self) -> None:
        tags = persona_to_tags({"region": "Scotland", "age": 30}, "en_US")
        assert "region:scotland" not in tags

    @pytest.mark.parametrize(
        "region, expected_land",
        [
            ("Bayern", "region:bayern"),
            ("Baden-Württemberg", "region:baden-wuerttemberg"),
            ("Thüringen", "region:thueringen"),
            ("Nordrhein-Westfalen", "region:nordrhein-westfalen"),
            ("Sachsen-Anhalt", "region:sachsen-anhalt"),
            ("Berlin", "region:berlin"),
        ],
    )
    def test_german_laender_map_to_their_land(self, region: str, expected_land: str) -> None:
        tags = persona_to_tags({"region": region, "age": 30}, "de_DE")
        assert [t for t in tags if t.startswith("region:")] == sorted(["region:any", expected_land])

    def test_an_unrecognised_german_region_adds_no_land(self) -> None:
        tags = persona_to_tags({"region": "Atlantis", "age": 30}, "de_DE")
        assert [t for t in tags if t.startswith("region:")] == ["region:any"]

    def test_laender_are_only_derived_for_de_de(self) -> None:
        tags = persona_to_tags({"region": "Bayern", "age": 30}, "en_US")
        assert "region:bayern" not in tags

    @pytest.mark.parametrize(
        "education_level, expected",
        [
            ("No qualifications", "education:primary"),
            ("GCSE level or below", "education:secondary"),
            ("Apprenticeship", "education:secondary"),
            ("A level or equivalent", "education:secondary"),
            ("Degree or other higher education", "education:tertiary"),
        ],
    )
    def test_uk_education_levels_map_to_a_level(self, education_level: str, expected: str) -> None:
        tags = persona_to_tags({"education_level": education_level, "age": 30}, "en_GB")
        assert [t for t in tags if t.startswith("education:")] == [expected]

    @pytest.mark.parametrize(
        "education_level, locale",
        [
            ("Technical diploma or certificate not equal to degree", "en_IN"),
            ("9th_12th_no_diploma", "en_US"),
            ("Other Diploma", "en_SG"),
        ],
    )
    def test_one_locales_education_keys_leave_anothers_levels_alone(self, education_level: str, locale: str) -> None:
        """Values a UK "degree" or German "diplom" key would read as tertiary."""
        tags = persona_to_tags({"education_level": education_level, "age": 30}, locale)
        assert "education:tertiary" not in tags

    @pytest.mark.parametrize(
        "education_level, expected",
        [
            ("Ohne beruflichen Bildungsabschluss", "education:secondary"),
            ("Lehre, Berufsausbildung im dualen System", "education:secondary"),
            ("Fachschulabschluss", "education:tertiary"),
            ("Bachelor, Master, Diplom, Promotion", "education:tertiary"),
        ],
    )
    def test_german_education_levels_map_to_a_level(self, education_level: str, expected: str) -> None:
        tags = persona_to_tags({"education_level": education_level, "age": 30}, "de_DE")
        assert [t for t in tags if t.startswith("education:")] == [expected]

    @pytest.mark.parametrize(
        "occupation, family",
        [
            ("Sozialpädagogische Assistentin/Kinderpflegerin", "teacher"),
            ("Altenpflegerin", "healthcare"),
            ("Gesundheits- und Krankenpflegerin", "healthcare"),
            ("Kinderkrankenschwester", "healthcare"),
            ("Ambulante Pflegerin", "healthcare"),
            ("Arzthelferin", "healthcare"),
            ("Ingenieur für Architektur", "engineering"),
            ("Ingenieur für Landespflege/Landschaftsplanung/-architektur", "engineering"),
            ("Helfer in der Landwirtschaft", "agriculture"),
            ("Bildende Künstlerin", "arts"),
        ],
    )
    def test_german_occupations_map_to_a_family(self, occupation: str, family: str) -> None:
        tags = persona_to_tags({"occupation": occupation, "age": 30}, "de_DE")
        assert f"occupation-family:{family}" in tags

    @pytest.mark.parametrize(
        "occupation",
        [
            "Raumpflegerin",
            "Baumpfleger",
            "Tierpflegerin Fachrichtung Zoo",
            "Fachverkäuferin für Kosmetik und Körperpflege",
            "Rechtspflegerin",
            "Pflegemutter im Kinderdorf",
        ],
    )
    def test_german_care_words_outside_nursing_add_no_family(self, occupation: str) -> None:
        tags = persona_to_tags({"occupation": occupation, "age": 30}, "de_DE")
        assert not [t for t in tags if t.startswith("occupation-family:")]

    def test_unknown_locale_still_returns_fallback_tags(self) -> None:
        persona = {"age": 40, "occupation": "Teacher"}
        tags = persona_to_tags(persona, "xx_YY")
        # Never empty: baseline fallbacks ensure there's always something
        # for the matcher to intersect.
        assert "region:any" in tags
        assert "age:any" in tags
        # Age bin still works (locale-agnostic).
        assert "age:25-44" in tags

    def test_missing_age_is_handled_gracefully(self) -> None:
        tags = persona_to_tags({"state_abbrev": "SP", "occupation": "Teacher"}, "pt_BR")
        # No age-bin tag (other than age:any fallback) — no crash.
        assert not any(t.startswith("age:") and t != "age:any" for t in tags)
        assert "age:any" in tags

    def test_education_substring_matching(self) -> None:
        for raw, expected in [
            ("Ensino fundamental completo", "education:primary"),
            ("High school diploma", "education:secondary"),
            ("Master's degree", "education:tertiary"),
            ("Pós-graduação lato sensu", "education:tertiary"),
        ]:
            tags = persona_to_tags({"education_level": raw, "age": 30}, "pt_BR")
            assert expected in tags, f"{raw!r} → expected {expected!r}, got {tags}"

    def test_occupation_substring_matching_multilingual(self) -> None:
        # English + Portuguese variants of the same concept should
        # produce the same occupation-family tag.
        for raw in ("Doctor of medicine", "Médico cardiologista"):
            tags = persona_to_tags({"occupation": raw, "age": 40}, "pt_BR")
            assert "occupation-family:healthcare" in tags
            assert "interest:health" in tags
