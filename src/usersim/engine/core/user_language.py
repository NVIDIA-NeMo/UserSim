# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What language and script the simulated user writes in, and how a user turn is checked.

``language_instruction`` is the block a probe adds to the user model's prompt;
``_script_directive`` makes it script-explicit for native-script and romanized
locales. ``_is_user_script_compliant`` is the deterministic script check the
loop runs on every user turn, and ``_language_judge_clause`` is the language
condition folded into the user-turn judge gate. ``_is_non_ascii_locale`` marks
the token-dense scripts whose replies get a larger token budget.
"""

from __future__ import annotations

import re

from usersim.engine.core.language_detection import script_compliance_fraction
from usersim.engine.core.locale import (
    INDIA_VARIANT_LOCALES,
    expected_language_name,
    expected_script_ranges,
)

_NON_ASCII_LOCALES = frozenset(
    {
        "ja_JP",
        "zh_CN",
        "zh_TW",
        "ko_KR",
        "hi_Deva_IN",
        "ar_SA",
        "th_TH",
        "he_IL",
    }
)


def _is_non_ascii_locale(locale: str) -> bool:
    """Return True for locales whose scripts require more tokens per semantic unit."""
    return locale in _NON_ASCII_LOCALES


# Human-readable native-script labels for the non-Latin shipped locales.
# Used to make the user-model language instruction script-explicit so the
# simulator stops accepting romanized / transliterated turns. Latin-script
# locales are absent here (no script directive needed).
_SCRIPT_LABEL_BY_LOCALE = {
    "hi_Deva_IN": "Devanagari",
    "ja_JP": "Japanese (hiragana, katakana, and kanji)",
    "ko_KR": "Hangul",
}

# Locales whose script is Latin but whose language is NOT English, so the
# deterministic script check (and lingua) cannot verify them — the inline
# judge does it instead via a folded language-conformance clause. Romanized
# Hindi is the first; the set is the extension point for any future
# "typed-on-an-English-keyboard" locale.
ROMANIZED_LOCALES = frozenset({"hi_Latn_IN"})

# Per-romanized-locale prompt directive (steers the user-model to produce the
# right romanization so the judge rarely has to reject) and judge clause.
_ROMANIZED_DIRECTIVE_BY_LOCALE = {
    "hi_Latn_IN": (
        " You MUST write in romanized Hindi using the Latin/Roman alphabet "
        "(e.g. 'aap kaise hain') — do NOT use Devanagari, and do NOT switch "
        "to English."
    ),
}
_ROMANIZED_JUDGE_CLAUSE_BY_LOCALE = {
    "hi_Latn_IN": (
        "\n\nADDITIONALLY: the USER message MUST be written in Hindi using "
        "the Latin/Roman alphabet (romanized Hindi, e.g. 'aap kaise hain'), "
        "NOT Devanagari and NOT English. If it is not, give the rating "
        "failure and include the token [LANGUAGE] in your explanation."
    ),
}


# ── India language-variant extensions (derived from the registry) ──────────
# Rather than duplicate the locale list, extend the maps above from
# ``INDIA_VARIANT_LOCALES`` so adding/removing a variant is a one-spot change
# in ``core/locale.py``. Native-script variants get the token bump + a
# script-explicit directive; romanized twins get the romanized directive +
# folded judge clause (same shape as ``hi_Latn_IN``).
_INDIA_NATIVE_LOCALES = {loc: v for loc, v in INDIA_VARIANT_LOCALES.items() if not v.romanized}
_INDIA_ROMANIZED_LOCALES = {loc: v for loc, v in INDIA_VARIANT_LOCALES.items() if v.romanized}

# Native Indic/Arabic scripts are token-dense like the other non-ASCII locales.
_NON_ASCII_LOCALES = frozenset(_NON_ASCII_LOCALES | set(_INDIA_NATIVE_LOCALES))

# Script-explicit directive label for each native-script variant.
_SCRIPT_LABEL_BY_LOCALE.update({loc: v.script_label for loc, v in _INDIA_NATIVE_LOCALES.items() if v.script_label})

# Romanized twins ride the LLM-judge language clause (lingua can't detect them).
ROMANIZED_LOCALES = frozenset(ROMANIZED_LOCALES | set(_INDIA_ROMANIZED_LOCALES))

_ROMANIZED_DIRECTIVE_BY_LOCALE.update(
    {
        loc: (
            f" You MUST write in romanized {v.language_display} using the "
            "Latin/Roman alphabet — do NOT use the native script, and do NOT "
            "switch to English."
        )
        for loc, v in _INDIA_ROMANIZED_LOCALES.items()
    }
)

_ROMANIZED_JUDGE_CLAUSE_BY_LOCALE.update(
    {
        loc: (
            f"\n\nADDITIONALLY: the USER message MUST be written in "
            f"{v.language_display} using the Latin/Roman alphabet (romanized "
            f"{v.language_display}), NOT the native script and NOT English. If it "
            "is not, give the rating failure and include the token [LANGUAGE] in "
            "your explanation."
        )
        for loc, v in _INDIA_ROMANIZED_LOCALES.items()
    }
)


def _script_directive(locale: str | None) -> str:
    """Return a script-explicit clause for non-Latin and romanized locales, else ''."""
    if not locale:
        return ""
    label = _SCRIPT_LABEL_BY_LOCALE.get(locale)
    if label:
        return f" You MUST use the native {label} script; do NOT romanize or transliterate into Latin letters."
    return _ROMANIZED_DIRECTIVE_BY_LOCALE.get(locale, "")


def _language_judge_clause(locale: str) -> str:
    """Judge-gate clause appended for romanized locales, else ''.

    Folded into the existing user-judge gate prompt (no extra LLM call).
    Phrased as plain instruction referencing the existing rating/explanation
    fields by name so ``_parse_judge_response``'s first-match regex is
    unaffected; a ``[LANGUAGE]`` token in the explanation lets the loop
    attribute the failure to language vs a generic role violation.
    """
    return _ROMANIZED_JUDGE_CLAUSE_BY_LOCALE.get(locale, "")


#: What an English-locale user may and may not mix into English. A user in
#: India or Singapore writes local idioms and borrowed words, but whole phrases
#: in another language are another locale's conversation (hi_Latn_IN, for one).
_ENGLISH_ALLOWANCE = (
    "Local idioms and the occasional borrowed word are fine, but do NOT write "
    "phrases or sentences in another language, whether in its own script or "
    "in Latin letters."
)


def _is_english(language: str, locale: str | None) -> bool:
    """Whether the user writes English: by the locale's registered language, else by the label."""
    name = expected_language_name(locale) if locale else None
    if name is not None:
        return name == "ENGLISH"
    return language.endswith("English")


def _language_requirement(language: str, locale: str | None = None) -> str:
    """The language a user turn must be in, as words to follow "write" ("in Portuguese.").

    The loop's per-turn instructions end with it, so every locale's user hears
    the same requirement on every turn as in its system prompt.
    """
    if _is_english(language, locale):
        return f"in {language}. {_ENGLISH_ALLOWANCE}"
    return f"in {language}.{_script_directive(locale)}"


def language_instruction(language: str, locale: str | None = None) -> str:
    """Build the language instruction block for prompts.

    Every locale gets one. An English locale's says English, with local idioms
    and borrowed words allowed; another language's says that language, made
    script-explicit (forbidding romanization) when ``locale`` maps to a
    non-Latin script or a romanized one.
    """
    if _is_english(language, locale):
        return f"IMPORTANT: You MUST write your messages in {language}. {_ENGLISH_ALLOWANCE}"
    return (
        f"IMPORTANT: You MUST write your messages in {language}. "
        f"The entire conversation should be in {language}."
        f"{_script_directive(locale)}"
    )


def _count_letters(text: str) -> int:
    """Number of alphabetic codepoints in ``text`` (script-neutral count)."""
    return sum(1 for ch in text if ch.isalpha())


#: A run of printable ASCII that holds a digit: a key, a password, an order
#: number or a flight code, which a user writes the same way in every language.
_CODE = re.compile(r"[\x21-\x7e]*[0-9][\x21-\x7e]*")


def _is_user_script_compliant(
    text: str,
    locale: str,
    *,
    min_script: float,
    min_letters: int,
) -> bool:
    """Deterministic check that a USER turn is in the locale's expected script.

    Returns True (compliant) when:
      - the locale has no registered expected script (nothing to enforce), or
      - the turn has fewer than ``min_letters`` alphabetic codepoints (too
        short for the script signal to be reliable — bare numbers, "OK"), or
      - the fraction of letters in the expected script is >= ``min_script``.

    Codes, runs of printable ASCII that hold a digit (a key, a password, an
    order number), are left out before counting, so a turn that hands one
    over is judged by the words around it.

    No model call and no lingua dependency: pure Unicode-range arithmetic.
    For Latin-script locales this is a no-op (Latin text scores ~1.0), so
    English / French / Portuguese runs are unaffected; for ``hi_Deva_IN`` it
    rejects romanized / transliterated Hindi.
    """
    ranges = expected_script_ranges(locale)
    if not ranges:
        return True
    text = _CODE.sub(" ", text)
    if _count_letters(text) < min_letters:
        return True
    return script_compliance_fraction(text, ranges) >= min_script
