# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every locale's simulated user is told which language to write in, in its prompt and on every turn."""

from __future__ import annotations

import pytest

from usersim.engine.core.behavioral import get_conversation_language
from usersim.engine.core.locale import INDIA_VARIANT_LOCALES, SHIPPED_LOCALES
from usersim.engine.core.user_language import _language_requirement, language_instruction
from usersim.extension import ConversationSimulatorConfig
from usersim.testing import ScriptedModel, run_episode

ALL_LOCALES = sorted({*SHIPPED_LOCALES, *INDIA_VARIANT_LOCALES})
ENGLISH_LOCALES = ("en_US", "en_GB", "en_SG", "en_IN")


@pytest.mark.parametrize("locale", ALL_LOCALES)
def test_every_locale_names_its_language(locale: str) -> None:
    language = get_conversation_language(locale)
    instruction = language_instruction(language, locale)
    assert f"write your messages in {language}." in instruction
    assert _language_requirement(language, locale).startswith(f"in {language}.")


@pytest.mark.parametrize("locale", ENGLISH_LOCALES)
def test_an_english_locale_allows_idioms_but_not_another_language(locale: str) -> None:
    language = get_conversation_language(locale)
    for text in (language_instruction(language, locale), _language_requirement(language, locale)):
        assert "borrowed word" in text
        assert "phrases or sentences in another language" in text
        assert "in Latin letters" in text
        assert "romanize" not in text


def test_english_without_a_locale_gets_the_english_directive() -> None:
    assert language_instruction("English") == language_instruction("English", "en_US")


@pytest.mark.parametrize(
    ("locale", "directive"),
    [
        ("hi_Deva_IN", "native Devanagari script"),
        ("ja_JP", "native Japanese (hiragana, katakana, and kanji) script"),
        ("ta_Taml_IN", "native Tamil script"),
        ("ta_Latn_IN", "romanized Tamil"),
        ("hi_Latn_IN", "romanized Hindi"),
    ],
)
def test_a_locale_with_a_script_rule_repeats_it_on_every_turn(locale: str, directive: str) -> None:
    language = get_conversation_language(locale)
    assert directive in language_instruction(language, locale)
    assert directive in _language_requirement(language, locale)


def test_a_latin_script_language_gets_no_script_rule() -> None:
    assert _language_requirement("Portuguese", "pt_BR") == "in Portuguese."


class _AsksAgain(ScriptedModel):
    """A user who keeps asking, so the loop writes follow-up prompts."""

    CONTENT = {**ScriptedModel.CONTENT, "summary": "no"}


async def test_each_follow_up_prompt_states_the_language_requirement() -> None:
    run = await run_episode(
        "general_open_ended",
        config=ConversationSimulatorConfig(name="conversation_messages", locale="en_US", max_turns=3),
        persona={"first_name": "Ana", "last_name": "Souza", "age": 35, "city": "Austin"},
        data={"theme": {"type": "travel", "description": "A weekend trip to Austin"}},
        model_factory=_AsksAgain,
    )
    follow_ups = [prompt for prompt in run.prompts if prompt["role"] == "user"][1:]
    assert follow_ups
    requirement = _language_requirement("English", "en_US")
    for prompt in follow_ups:
        text = prompt["messages"][-1]["content"]
        assert f"Write the user's next message {requirement}" in text
        assert f"Remember: write {requirement}" in text
