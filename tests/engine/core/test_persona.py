# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for core/persona.py formatting."""

from usersim.engine.core.persona import (
    format_persona_for_prompt,
    religion_language_context,
)


class TestFormatPersona:
    def test_includes_name(self, full_persona):
        text = format_persona_for_prompt(full_persona)
        assert "Maria Silva" in text

    def test_includes_demographics(self, full_persona):
        text = format_persona_for_prompt(full_persona)
        assert "Age: 35" in text
        assert "Software Engineer" in text

    def test_handles_minimal_persona(self, minimal_persona):
        text = format_persona_for_prompt(minimal_persona)
        assert "Taro Tanaka" in text
        assert "72" in text

    def test_handles_locale_fields(self):
        # Nemotron-Personas universal location schema: ``city`` /
        # ``district`` / ``region`` / ``country``.
        persona = {
            "first_name": "Jean",
            "last_name": "Dupont",
            "age": 45,
            "district": "Lyon",
            "region": "Rhône",
            "country": "France",
        }
        text = format_persona_for_prompt(persona)
        assert "Jean Dupont" in text
        # Location string built from district + region + country
        # in small→large order, deduped.
        assert "Location: Lyon, Rhône, France" in text

    def test_singapore_persona_renders_district_and_country(self):
        # City-state: ``en_SG`` populates only ``district`` (subzone,
        # e.g. "Bukit Panjang") + ``country``; ``city`` and
        # ``region`` are both null. Pin that the district surfaces
        # rather than the row collapsing to country-only.
        persona = {
            "first_name": "Wei",
            "last_name": "Tan",
            "age": 33,
            "district": "Bukit Panjang",
            "country": "Singapore",
        }
        text = format_persona_for_prompt(persona)
        assert "Location: Bukit Panjang, Singapore" in text


class TestReligionAndLanguages:
    """India-dataset fields, and the sentinels they arrive with.

    Nemotron-Personas writes ``"-"`` for a language slot the person does
    not have — 74% of rows for ``second_language``, 93% for
    ``third_language`` — so the unused-slot case is the common one, not
    the edge case. A null cell that has been through pandas arrives as
    float NaN, which ``str()`` renders as the literal "nan"; truthy, so
    it would survive a bare presence check and reach the prompt as a
    fact about the person.
    """

    BASE = {"first_name": "Ravi", "last_name": "Kumar", "age": 34}

    def _lines(self, **fields) -> list:
        text = format_persona_for_prompt({**self.BASE, **fields})
        return [
            ln for ln in text.splitlines()
            if ln.startswith(("Religion:", "Languages:"))
        ]

    def test_religion_renders(self):
        assert self._lines(religion="Hindu") == ["Religion: Hindu"]

    def test_languages_render_in_order(self):
        assert self._lines(
            first_language="Hindi", second_language="English",
        ) == ["Languages: Hindi, English"]

    def test_unused_language_slots_are_dropped(self):
        assert self._lines(
            first_language="Malayalam", second_language="-",
            third_language="-",
        ) == ["Languages: Malayalam"]

    def test_religion_dash_is_dropped(self):
        """Same sentinel, same treatment — the filter is shared, not
        duplicated per field."""
        assert self._lines(religion="-") == []

    def test_nan_never_reaches_the_prompt(self):
        """A NaN rendered as "nan" would read as the person's religion."""
        nan = float("nan")
        assert self._lines(religion=nan) == []
        assert self._lines(first_language=nan) == []
        assert self._lines(
            first_language="Hindi", second_language="-", third_language=nan,
        ) == ["Languages: Hindi"]

    def test_nan_is_dropped_for_every_field_not_just_these(self):
        """The NaN guard lives in the shared accessor, so it covers
        fields this MR does not touch."""
        text = format_persona_for_prompt({
            "first_name": "Ravi", "last_name": "Kumar",
            "age": float("nan"), "occupation": float("nan"),
        })
        assert "nan" not in text.lower()

    def test_dataset_no_religion_value_is_preserved(self):
        """"Other" is how these datasets spell "no religion" (en_IN
        "Other" / hi_Deva_IN "अन्य" / hi_Latn_IN "Anya"), so it must
        render. Only the literal placeholder words are dropped."""
        assert self._lines(religion="Other") == ["Religion: Other"]

    def test_placeholder_words_are_dropped_case_insensitively(self):
        """"none" / "nan" as literal cell text mean an unfilled cell, not
        a fact -- a CSV round-trip is the usual way they appear."""
        for text in ("none", "None", "NONE", "nan", "NaN", " - ", "  "):
            assert self._lines(religion=text) == [], text

    def test_placeholder_words_inside_prose_survive(self):
        """The match is on the whole stripped cell, so prose that merely
        contains one of the words is untouched."""
        assert self._lines(religion="None practising") == [
            "Religion: None practising"
        ]

    def test_absent_fields_render_nothing(self):
        assert self._lines() == []

    def test_india_shaped_persona_renders_all_four(self):
        """One realistic India row, all four outputs at once.

        The languages line is asserted whole rather than by substring:
        slot order is semantic — ``first_language`` is the mother tongue,
        so "Malayalam, English" and "English, Malayalam" describe
        different people. If the comprehension building it ever became a
        set, ordering would be lost and a substring check would still
        pass.

        The two prose facets render through the facet loop rather than
        the demographics block, so they need asserting separately.
        """
        text = format_persona_for_prompt({
            "first_name": "Ravi",
            "last_name": "Kumar",
            "age": 34,
            "city": "Kochi",
            "country": "India",
            "religion": "Hindu",
            "first_language": "Malayalam",
            "second_language": "English",
            "third_language": "-",
            "religious_background": "Observes the major festivals",
            "linguistic_background": "Malayalam at home, English at work",
        })
        assert "Religion: Hindu" in text
        assert "Languages: Malayalam, English" in text
        assert "Religious Background: Observes the major festivals." in text
        assert (
            "Linguistic Background: Malayalam at home, English at work."
            in text
        )

    def test_sentinel_on_a_prose_facet_is_dropped(self):
        """The facet loop is a separate path from the demographics block,
        so the placeholder filter needs pinning on both."""
        text = format_persona_for_prompt({
            **self.BASE,
            "religious_background": "-",
            "linguistic_background": float("nan"),
        })
        assert "Religious Background" not in text
        assert "Linguistic Background" not in text
        assert "nan" not in text.lower()

    def test_non_india_persona_gains_no_new_labels(self, full_persona_non_indic):
        """A persona from a dataset without these columns must render
        exactly as before — none of the four labels appear.

        Checks the prose facets as well as the demographics lines: those
        two go through the facet loop rather than the demographics
        block, so ``_lines`` above does not see them. Real fixtures
        rather than a minimal dict, because a persona with no facets at
        all never exercises that loop.
        """
        text = format_persona_for_prompt(full_persona_non_indic)
        for label in (
            "Religion:",
            "Languages:",
            "Religious Background:",
            "Linguistic Background:",
        ):
            assert label not in text, label


class TestReligionLanguageAuditContext:
    def test_matches_the_normalized_prompt_fields(self):
        context = religion_language_context({
            "religion": "Hindu",
            "first_language": "Malayalam",
            "second_language": "English",
            "third_language": "-",
            "religious_background": "Observes the major festivals",
            "linguistic_background": "Malayalam at home, English at work",
        })
        assert context == {
            "religion": "Hindu",
            "languages": ["Malayalam", "English"],
            "religious_background": "Observes the major festivals",
            "linguistic_background": "Malayalam at home, English at work",
        }

    def test_sentinels_are_absent_from_the_audit_view(self):
        context = religion_language_context({
            "religion": "-",
            "first_language": float("nan"),
            "second_language": "none",
            "religious_background": " ",
            "linguistic_background": None,
        })
        assert context == {
            "religion": None,
            "languages": [],
            "religious_background": None,
            "linguistic_background": None,
        }

    def test_non_india_persona_has_a_stable_empty_shape(self):
        assert religion_language_context({"first_name": "Maria"}) == {
            "religion": None,
            "languages": [],
            "religious_background": None,
            "linguistic_background": None,
        }


class TestNameExemption:
    """Name fields exempt "nan" from the placeholder filter.

    `_INVALID_VALUES` reads "nan" as an unfilled cell, which is right for
    a religion or a language slot and wrong for a name: "Nan" is a real
    given name. Filtering it renames the person rather than tidying a
    blank, and the row still renders — just as somebody else.

    The exemption is per-field and covers the WORD only. A genuinely
    empty cell is rejected before `exempt` is consulted, so a real float
    NaN cannot slip through as the text "nan".
    """

    def test_exempt_word_is_kept_as_a_name(self):
        assert format_persona_for_prompt(
            {"first_name": "Nan", "last_name": "Goldin"}
        ).startswith("Name: Nan Goldin")

    def test_real_null_name_is_still_dropped(self):
        for empty in (None, float("nan"), "", "   "):
            assert format_persona_for_prompt(
                {"first_name": empty, "last_name": "Rao"}
            ).startswith("Name: Rao"), repr(empty)

    def test_unexempted_placeholders_still_filter_on_names(self):
        """Only "nan" is exempt — "-" and "none" are not."""
        for text in ("-", "none"):
            assert format_persona_for_prompt(
                {"first_name": text, "last_name": "Rao"}
            ).startswith("Name: Rao"), text

    def test_exemption_does_not_leak_to_other_fields(self):
        """Same row: the name survives, religion and languages do not."""
        text = format_persona_for_prompt({
            "first_name": "Nan", "last_name": "Goldin",
            "religion": "nan", "first_language": "nan",
        })
        assert "Name: Nan Goldin" in text
        assert "Religion" not in text
        assert "Languages" not in text

    def test_both_names_missing_falls_back(self):
        assert format_persona_for_prompt({"age": 30}).startswith("Name: Unknown")
