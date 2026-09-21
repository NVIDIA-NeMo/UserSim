# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Persona formatting utilities for conversation simulation."""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any

#: Cell values that mean "nothing here" rather than a fact about the
#: person. ``"-"`` is what the dataset writes for a slot the person does
#: not have (74% of rows for ``second_language``, 93% for
#: ``third_language``); ``"nan"`` is what ``str()`` makes of a null cell
#: that pandas upcast to float NaN, and is truthy, so without this it
#: would reach the prompt as ``Religion: nan``.
_INVALID_VALUES = ("", "-", "nan", "none")

#: Entries of :data:`_INVALID_VALUES` that name fields keep. "Nan" is a
#: real given name, so filtering it renames the person rather than tidying
#: an unfilled cell -- and the row still renders, just as somebody else.
_NAME_EXEMPT = ("nan",)


def _is_present_or_valid(value: Any, exempt: tuple[str, ...] = ()) -> bool:
    """True when a persona cell holds a value worth rendering.

    Matched case-insensitively against the whole stripped value, so a
    dash or the word "none" *inside* prose is left alone -- only a field
    that is nothing but one of :data:`_INVALID_VALUES` is dropped.

    ``exempt`` lists entries of :data:`_INVALID_VALUES` this field should
    keep anyway, for columns where the word is real content rather than a
    placeholder. It never rescues a genuinely empty cell: ``None`` and float
    ``NaN`` are rejected before ``exempt`` is consulted, so an exempted "nan"
    keeps the *string* without also letting a null through as the text "nan".
    """
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    text = str(value).strip().lower()
    if text in exempt:
        return True
    return text not in _INVALID_VALUES


def _field_text(
    persona: dict[str, Any],
    key: str,
    exempt: tuple[str, ...] = (),
) -> str:
    """Return one normalized persona field, or ``""`` when it is absent."""
    value = persona.get(key)
    return str(value) if _is_present_or_valid(value, exempt) else ""


def religion_language_context(persona: dict[str, Any]) -> dict[str, Any]:
    """Structured audit view of religion/language fields shown to the user agent.

    The raw persona is dropped from trajectory output, but these protected,
    content-determining fields now affect the simulated user's prompt. Persisting
    their normalized view makes that influence inspectable without requiring the
    source persona dataset to be present.
    """
    return {
        "religion": _field_text(persona, "religion") or None,
        "languages": [
            value
            for value in (
                _field_text(persona, "first_language"),
                _field_text(persona, "second_language"),
                _field_text(persona, "third_language"),
            )
            if value
        ],
        "religious_background": (_field_text(persona, "religious_background") or None),
        "linguistic_background": (_field_text(persona, "linguistic_background") or None),
    }


def format_persona_for_prompt(persona: dict[str, Any]) -> str:
    """Build a structured persona text block for user agent prompts.

    Outputs the person's name first, followed by demographics, then labeled
    sections for each available persona facet (in shuffled order to reduce
    positional bias).
    """

    def f(key: str, exempt: tuple[str, ...] = ()) -> str:
        """The field's text, or ``""`` when the cell holds nothing."""
        return _field_text(persona, key, exempt)

    # Read ``age`` directly. Deriving it from a birth date would make the
    # rendered persona change with the calendar while ``trajectory_id``, which
    # excludes age, stayed the same, so two runs would disagree under one id.
    age_str = f("age")

    name_parts = [
        f("first_name", _NAME_EXEMPT),
        f("last_name", _NAME_EXEMPT),
    ]
    name = " ".join(p for p in name_parts if p) or "Unknown"

    demographics: list[str] = []
    if age_str:
        demographics.append(f"Age: {age_str}")
    if f("sex"):
        demographics.append(f"Sex: {f('sex')}")
    if f("marital_status"):
        demographics.append(f"Marital Status: {f('marital_status')}")
    # Build the Location string from the four universal Nemotron-
    # Personas fields (``city`` / ``district`` / ``region`` /
    # ``country``) in small→large order, deduping and skipping
    # empties. See the matching block in
    # ``cli/_pipeline.py::_add_persona_expressions`` for the full
    # per-locale schema audit.
    loc_parts: list[str] = []
    seen: set[str] = set()
    for key in ("city", "district", "region", "country"):
        v = f(key)
        if v and v not in seen:
            loc_parts.append(v)
            seen.add(v)
    if loc_parts:
        demographics.append(f"Location: {', '.join(loc_parts)}")
    # Religion and spoken languages ship on the India persona dataset only, so
    # these are presence-gated rather than locale-gated: absent everywhere else,
    # they simply do not render, and a locale that adds them later is picked up
    # with no code change. Both are load-bearing for India runs — religion and
    # mother tongue are the attributes a culturally-grounded request turns on,
    # and without them the user agent infers them from the name and city.
    protected_context = religion_language_context(persona)
    religion = protected_context["religion"]
    if religion:
        demographics.append(f"Religion: {religion}")
    languages = protected_context["languages"]
    if languages:
        demographics.append(f"Languages: {', '.join(languages)}")

    if f("occupation"):
        demographics.append(f"Occupation: {f('occupation')}")
    if f("education_level"):
        demographics.append(f"Education Level: {f('education_level')}")
    if f("bachelors_field"):
        demographics.append(f"Bachelors Field: {f('bachelors_field')}")

    sections: list[str] = []
    sections.append(f"Name: {name}")
    if demographics:
        sections.append("Demographics:\n" + "\n".join(demographics))

    facet_labels = [
        ("persona", "Overview"),
        ("professional_persona", "Professional"),
        ("finance_persona", "Finance"),
        ("healthcare_persona", "Healthcare"),
        ("sports_persona", "Sports"),
        ("arts_persona", "Arts"),
        ("travel_persona", "Travel"),
        ("culinary_persona", "Culinary"),
        ("skills_and_expertise", "Skills & Expertise"),
        ("hobbies_and_interests", "Hobbies & Interests"),
        ("career_goals_and_ambitions", "Career Goals"),
        ("cultural_background", "Cultural Background"),
        # India-dataset prose facets; absent elsewhere, so they self-skip.
        ("religious_background", "Religious Background"),
        ("linguistic_background", "Linguistic Background"),
    ]
    rng = random.Random(int(hashlib.sha256(name.encode()).hexdigest(), 16) & 0xFFFFFFFF)
    rng.shuffle(facet_labels)
    for key, label in facet_labels:
        val = f(key)
        if val:
            sections.append(f"{label}: {val.rstrip('.') + '.'}")

    return "\n\n".join(sections)
