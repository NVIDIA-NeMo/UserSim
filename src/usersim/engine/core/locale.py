# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single source of truth for locale → language + expected-script mappings.

The simulator already implicitly resolves locale to a `language` string
that gets pinned in assistant system prompts (`"Respond in Portuguese"`
etc.). The deterministic evaluator scorers (`language_compliance`,
`response_shape`, `refusal_basics`) need the same mapping but as
machine-readable identifiers — language enum names from the lingua
library, and Unicode script ranges for script-compliance checks.

This module exposes both as plain Python data so consumers can use
either without taking a hard import dependency on lingua. The lingua
detector itself is built lazily in
:mod:`usersim.engine.core.language_detection` so the scorer's
~170 MB language-model cost only loads when a scorer that needs it
actually runs.

Adding a new locale = adding it to ``SHIPPED_LOCALES`` plus one row in
each locale-indexed map below. Tests check key parity so a half-added
locale is caught at CI time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SHIPPED_LOCALES: tuple[str, ...] = (
    "en_US",
    "en_IN",
    "en_SG",
    "pt_BR",
    "fr_FR",
    "ja_JP",
    "hi_Deva_IN",
    "hi_Latn_IN",
    "ko_KR",
)


# ---------------------------------------------------------------------------
# Unicode script ranges
# ---------------------------------------------------------------------------
#
# Each script is a tuple of ``(lo_codepoint, hi_codepoint)`` inclusive
# pairs from the Unicode block table. Public so the script-dominance
# helper can iterate over them by name.

# Latin: basic + Latin-1 + Latin Extended-A + Latin Extended-B (covers
# every accented vowel that French / Portuguese / Spanish use).
LATIN_RANGES: tuple[tuple[int, int], ...] = (
    (0x0041, 0x005A),  # A-Z
    (0x0061, 0x007A),  # a-z
    (0x00C0, 0x024F),  # Latin-1 Supplement + Latin Extended-A + Extended-B
)

DEVANAGARI_RANGES: tuple[tuple[int, int], ...] = (
    (0x0900, 0x097F),  # Devanagari block
)

HANGUL_RANGES: tuple[tuple[int, int], ...] = (
    (0x1100, 0x11FF),  # Hangul Jamo
    (0x3130, 0x318F),  # Hangul Compatibility Jamo
    (0xAC00, 0xD7AF),  # Hangul Syllables
)

HIRAGANA_RANGES: tuple[tuple[int, int], ...] = ((0x3040, 0x309F),)

KATAKANA_RANGES: tuple[tuple[int, int], ...] = ((0x30A0, 0x30FF),)

# CJK Unified Ideographs (Han / 漢字 / kanji). Doesn't include CJK Ext A/B/...
# which are mostly historical / specialty kanji rarely seen in modern text;
# adding them later is a one-line change.
HAN_RANGES: tuple[tuple[int, int], ...] = ((0x4E00, 0x9FFF),)

# ── Indic + Arabic scripts (used by the India language-variant layer) ──────
# One Unicode block per script. Devanagari (already above) is shared by
# Hindi / Marathi / Nepali; the rest are one-language-dominant in India.

BENGALI_RANGES: tuple[tuple[int, int], ...] = (
    (0x0980, 0x09FF),  # Bengali (also Assamese)
)

GURMUKHI_RANGES: tuple[tuple[int, int], ...] = (
    (0x0A00, 0x0A7F),  # Gurmukhi (Punjabi)
)

GUJARATI_RANGES: tuple[tuple[int, int], ...] = (
    (0x0A80, 0x0AFF),  # Gujarati
)

ODIA_RANGES: tuple[tuple[int, int], ...] = (
    (0x0B00, 0x0B7F),  # Odia / Oriya
)

TAMIL_RANGES: tuple[tuple[int, int], ...] = (
    (0x0B80, 0x0BFF),  # Tamil
)

TELUGU_RANGES: tuple[tuple[int, int], ...] = (
    (0x0C00, 0x0C7F),  # Telugu
)

KANNADA_RANGES: tuple[tuple[int, int], ...] = (
    (0x0C80, 0x0CFF),  # Kannada
)

MALAYALAM_RANGES: tuple[tuple[int, int], ...] = (
    (0x0D00, 0x0D7F),  # Malayalam
)

ARABIC_RANGES: tuple[tuple[int, int], ...] = (
    (0x0600, 0x06FF),  # Arabic block (Urdu is written in Arabic script)
)

# ── Scripts no supported locale expects ───────────────────────────────────
# No locale declares these, so they never appear in
# ``LOCALE_TO_EXPECTED_SCRIPT_RANGES``. They are named anyway because they
# turn up as INTRUDERS: a degraded model emits stray Cyrillic/Hebrew/Greek
# glyphs mid-word in Indic prose, and the script-integrity check can only
# report "Cyrillic" instead of the useless "other" if the bucket has a name.

CYRILLIC_RANGES: tuple[tuple[int, int], ...] = (
    (0x0400, 0x04FF),  # Cyrillic
)

HEBREW_RANGES: tuple[tuple[int, int], ...] = (
    (0x0590, 0x05FF),  # Hebrew
)

GREEK_RANGES: tuple[tuple[int, int], ...] = (
    (0x0370, 0x03FF),  # Greek and Coptic
)


# ---------------------------------------------------------------------------
# Locale → expected language (lingua enum NAME, not the enum object)
# ---------------------------------------------------------------------------
#
# Stored as strings so this module has zero hard imports. The detector
# module materializes these to ``lingua.Language`` enum objects on first
# use.

LOCALE_TO_LANGUAGE_NAME: dict[str, str | None] = {
    "en_US": "ENGLISH",
    "en_IN": "ENGLISH",
    "en_SG": "ENGLISH",
    "pt_BR": "PORTUGUESE",
    "fr_FR": "FRENCH",
    "ja_JP": "JAPANESE",
    "hi_Deva_IN": "HINDI",
    # Romanized Hindi: deliberately ``None``. lingua cannot detect romanized
    # Hindi (it reads as English), so the deterministic language scorers
    # ``_noop``/skip on ``None`` rather than emit wrong signals; in-sim
    # language enforcement uses the LLM judge clause (see core/simulation.py
    # ROMANIZED_LOCALES) and assistant-side conformance rides the evaluator's
    # ``language_appropriateness`` judge axis.
    "hi_Latn_IN": None,
    "ko_KR": "KOREAN",
}

LOCALE_TO_LANGUAGE_DISPLAY: dict[str, str] = {
    "en_US": "English",
    "en_IN": "Indian English",
    "en_SG": "English",
    "pt_BR": "Portuguese",
    "fr_FR": "French",
    "ja_JP": "Japanese",
    "hi_Deva_IN": "Hindi Devanagari",
    "hi_Latn_IN": "Hindi Latin",
    "ko_KR": "Korean",
}


# ---------------------------------------------------------------------------
# Locale → expected script ranges
# ---------------------------------------------------------------------------
#
# A script is "expected" if its Unicode block is the canonical writing
# system for the locale's primary language. Multiple scripts can be
# co-canonical (Japanese mixes Hiragana + Katakana + Han routinely; this
# module treats all three as expected for ``ja_JP``).
#
# Latin loanwords and numerals appearing inside non-Latin text (e.g.,
# the word "computer" inside a Hindi response, or "100" inside a Japanese
# response) are handled at scoring time by the
# :func:`script_compliance_fraction` helper, which only counts letter
# codepoints — digits and punctuation are script-neutral.

LOCALE_TO_EXPECTED_SCRIPT_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "en_US": LATIN_RANGES,
    "en_IN": LATIN_RANGES,
    "en_SG": LATIN_RANGES,
    "pt_BR": LATIN_RANGES,
    "fr_FR": LATIN_RANGES,
    "ja_JP": HIRAGANA_RANGES + KATAKANA_RANGES + HAN_RANGES,
    "hi_Deva_IN": DEVANAGARI_RANGES,
    "hi_Latn_IN": LATIN_RANGES,
    "ko_KR": HANGUL_RANGES,
}


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------


def supported_locales() -> tuple[str, ...]:
    """Locales for which this module knows both an expected language
    and an expected script. Tests assert this matches every locale that
    ships a fact bank or probing taxonomy."""
    return tuple(sorted(SHIPPED_LOCALES))


def expected_language_name(locale: str) -> str | None:
    """Return the lingua language enum NAME (e.g. ``'ENGLISH'``) the
    assistant should respond in for ``locale``, or ``None`` if the
    locale is not registered (or is a locale lingua cannot detect, e.g.
    romanized twins). Falls through to the India variant table."""
    if locale in LOCALE_TO_LANGUAGE_NAME:
        return LOCALE_TO_LANGUAGE_NAME[locale]
    v = INDIA_VARIANT_LOCALES.get(locale)
    if v is not None:
        return v.lingua_name
    return None


def expected_language_display(locale: str) -> str | None:
    """Return a human-readable language label suitable for prompts.

    Falls through to the India variant table for variant locales."""
    if locale in LOCALE_TO_LANGUAGE_DISPLAY:
        return LOCALE_TO_LANGUAGE_DISPLAY[locale]
    v = INDIA_VARIANT_LOCALES.get(locale)
    if v is not None:
        return v.language_display
    return None


def expected_script_ranges(locale: str) -> tuple[tuple[int, int], ...]:
    """Return the Unicode codepoint ranges that count as "expected
    script" for ``locale``. Empty tuple for unregistered locales.

    Falls through to the India variant table for variant locales."""
    if locale in LOCALE_TO_EXPECTED_SCRIPT_RANGES:
        return LOCALE_TO_EXPECTED_SCRIPT_RANGES[locale]
    v = INDIA_VARIANT_LOCALES.get(locale)
    if v is not None:
        return v.script_ranges
    return ()


# Named script-bucket map used by the script-dominance helper. Public so
# callers can also enumerate the buckets directly.
SCRIPT_BUCKETS: dict[str, tuple[tuple[int, int], ...]] = {
    "latin": LATIN_RANGES,
    "devanagari": DEVANAGARI_RANGES,
    "hangul": HANGUL_RANGES,
    "hiragana": HIRAGANA_RANGES,
    "katakana": KATAKANA_RANGES,
    "han": HAN_RANGES,
    "bengali": BENGALI_RANGES,
    "gurmukhi": GURMUKHI_RANGES,
    "gujarati": GUJARATI_RANGES,
    "odia": ODIA_RANGES,
    "tamil": TAMIL_RANGES,
    "telugu": TELUGU_RANGES,
    "kannada": KANNADA_RANGES,
    "malayalam": MALAYALAM_RANGES,
    "arabic": ARABIC_RANGES,
    # Expected by no locale; present so an intrusion can be named (see above).
    "cyrillic": CYRILLIC_RANGES,
    "hebrew": HEBREW_RANGES,
    "greek": GREEK_RANGES,
}


# ---------------------------------------------------------------------------
# India language-variant layer (stop-gap)
# ---------------------------------------------------------------------------
#
# These conversation locales reuse an existing India persona dataset + the
# India asset banks, but conduct the conversation in another Indian language
# (native script) or its romanized Latin transliteration. They are DELIBERATELY
# kept out of ``SHIPPED_LOCALES`` (which is the "asset-complete" set every
# probe must ship banks/prompt-packs for). Instead, the presence-aware
# resolvers below prefer a variant's own asset when it exists and otherwise
# fall back to ``base_locale`` (en_IN), so dropping in real per-language assets
# later automatically takes the native path with no code change.
#
# Hindi already ships as two first-class locales (``hi_Deva_IN`` /
# ``hi_Latn_IN``), so it is not repeated here.


@dataclass(frozen=True)
class IndiaVariant:
    """Metadata for one India language-variant conversation locale.

    - ``base_locale`` — the shipped India locale whose persona dataset +
      asset banks + prompt packs are reused (``en_IN`` today).
    - ``language_display`` — human/prompt-facing language name (fed to the
      user-agent language instruction and to reports).
    - ``lingua_name`` — the ``lingua.Language`` enum NAME for the
      deterministic language scorer, or ``None`` when lingua cannot detect
      the language (romanized twins; or native scripts lingua lacks —
      Kannada/Malayalam/Odia/Nepali). ``None`` => the deterministic language
      scorer skips, but in-sim native-script enforcement still applies.
    - ``script_ranges`` — expected Unicode script ranges for the user turn.
    - ``script_label`` — human name of the native script for the user-agent
      "use the native <X> script" directive (``None`` for romanized twins,
      which use the romanized directive instead).
    - ``romanized`` — True for the Latin-transliteration twin.
    """

    base_locale: str
    language_display: str
    lingua_name: str | None
    script_ranges: tuple[tuple[int, int], ...]
    script_label: str | None
    romanized: bool


# Compact spec for the 11 additional Indian languages in scope; each expands
# into a native-script locale + its romanized-Latin twin. Fields:
# (native_locale, latin_locale, language, lingua_name_or_None, script_ranges,
#  native_script_label). ``lingua_name`` is set only where the installed
# lingua build actually has the language (verified: Bengali/Gujarati/Hindi/
# Marathi/Punjabi/Tamil/Telugu/Urdu; NOT Kannada/Malayalam/Odia/Nepali).
# Hindi already ships as first-class ``hi_Deva_IN`` / ``hi_Latn_IN``.
_INDIA_VARIANT_SPEC: tuple[tuple[str, str, str, str | None, tuple[tuple[int, int], ...], str], ...] = (
    ("bn_Beng_IN", "bn_Latn_IN", "Bengali", "BENGALI", BENGALI_RANGES, "Bengali"),
    ("or_Orya_IN", "or_Latn_IN", "Odia", None, ODIA_RANGES, "Odia"),
    ("ta_Taml_IN", "ta_Latn_IN", "Tamil", "TAMIL", TAMIL_RANGES, "Tamil"),
    ("te_Telu_IN", "te_Latn_IN", "Telugu", "TELUGU", TELUGU_RANGES, "Telugu"),
    ("ur_Arab_IN", "ur_Latn_IN", "Urdu", "URDU", ARABIC_RANGES, "Arabic (Urdu)"),
    ("ml_Mlym_IN", "ml_Latn_IN", "Malayalam", None, MALAYALAM_RANGES, "Malayalam"),
    # Marathi: lingua HAS a Marathi model, but Marathi shares the Devanagari
    # script AND much vocabulary with the shipped Hindi locale, so including it
    # in the detector misclassifies short Hindi text as Marathi (and vice
    # versa) — degrading a first-class locale for a preview one. So we keep
    # Marathi lingua_name=None (script-only enforcement, like Nepali) rather
    # than run unreliable Hindi/Marathi language-ID.
    ("mr_Deva_IN", "mr_Latn_IN", "Marathi", None, DEVANAGARI_RANGES, "Devanagari"),
    ("gu_Gujr_IN", "gu_Latn_IN", "Gujarati", "GUJARATI", GUJARATI_RANGES, "Gujarati"),
    ("ne_Deva_IN", "ne_Latn_IN", "Nepali", None, DEVANAGARI_RANGES, "Devanagari"),
    ("kn_Knda_IN", "kn_Latn_IN", "Kannada", None, KANNADA_RANGES, "Kannada"),
    ("pa_Guru_IN", "pa_Latn_IN", "Punjabi", "PUNJABI", GURMUKHI_RANGES, "Gurmukhi"),
)

_INDIA_BASE_LOCALE = "en_IN"


def _build_india_variant_locales() -> dict[str, IndiaVariant]:
    out: dict[str, IndiaVariant] = {}
    for native, latin, language, lingua, ranges, script_label in _INDIA_VARIANT_SPEC:
        out[native] = IndiaVariant(
            base_locale=_INDIA_BASE_LOCALE,
            language_display=language,
            lingua_name=lingua,
            script_ranges=ranges,
            script_label=script_label,
            romanized=False,
        )
        # Romanized twin: lingua cannot detect it (reads as English) so
        # lingua_name=None + LATIN_RANGES; language enforcement rides the
        # romanized directive + user-judge language clause in core/simulation.py.
        out[latin] = IndiaVariant(
            base_locale=_INDIA_BASE_LOCALE,
            language_display=language,
            lingua_name=None,
            script_ranges=LATIN_RANGES,
            script_label=None,
            romanized=True,
        )
    return out


INDIA_VARIANT_LOCALES: dict[str, IndiaVariant] = _build_india_variant_locales()


#: Per-corpus language metadata for persona sampling. Keyed by the PERSONA
#: DATASET locale, not the conversation locale, because everything here is a
#: property of the parquet being sampled rather than of the conversation.
#:
#: - ``language_column`` — the column holding the person's language.
#: - ``language_name_remap`` — how THIS corpus spells a language, for the
#:   ones it does not spell the way the locale's display name does. Absent
#:   when the corpus uses the plain Latin name, so most entries omit it.
#:
#: A corpus with NO language column has no entry at all. Only the India
#: datasets carry one -- see DataDesigner's
#: ``sampling_gen/entities/dataset_based_person_fields.py``, which lists
#: first/second/third_language under "India locales (en_IN, hi_Deva_IN,
#: hi_Latn_IN)". Constraining on the column elsewhere would ask the sampler
#: to filter on a field that does not exist rather than harmlessly doing
#: nothing, so callers gate on membership here and a mixed-locale panel
#: works without the caller special-casing rows.
#:
#: Carrying the column does NOT make a corpus language-appropriate.
#: ``hi_Deva_IN`` and ``hi_Latn_IN`` ship their own parquet, but each is the
#: same pan-India population as ``en_IN`` merely re-rendered -- measured over
#: all three at 1M rows, Hindi is the first language of 41.7% and absent
#: entirely from 45.1%, identically in each.
#:
#: A future native ``ta_Taml_IN`` parquet adds its own entry here rather
#: than changing any lookup.
PERSONA_CORPUS_LANGUAGE: dict[str, dict[str, Any]] = {
    "en_IN": {"language_column": "first_language"},
    "hi_Latn_IN": {
        "language_column": "first_language",
        # Latin script, but transliterated Hindi rather than English: this
        # corpus writes "Angrezi", not "English". The other twelve
        # conversable languages happen to match their English spelling.
        "language_name_remap": {"English": "Angrezi"},
    },
    "hi_Deva_IN": {
        "language_column": "first_language",
        # Devanagari spellings, derived by pairing this corpus row-by-row
        # against ``en_IN``: the two are the same population re-rendered,
        # so row N is the same person in both. Covers the languages a run
        # can actually converse in (the India variants plus English); the
        # corpus carries ~120 more that no shipped locale can request.
        "language_name_remap": {
            "Bengali": "बंगाली",
            "English": "अंग्रेज़ी",
            "Gujarati": "गुजराती",
            "Hindi": "हिंदी",
            "Kannada": "कन्नडा",
            "Malayalam": "मलयालम",
            "Marathi": "मराठी",
            "Nepali": "नेपाली",
            "Odia": "ओडिया",
            "Punjabi": "पंजाबी",
            "Tamil": "तमिल",
            "Telugu": "तेलुगू",
            "Urdu": "उर्दू",
        },
    },
}


def india_variant(locale: str) -> "IndiaVariant | None":
    """Return the :class:`IndiaVariant` for ``locale``, or ``None``."""
    return INDIA_VARIANT_LOCALES.get(locale)


def _variant_base(locale: str) -> str | None:
    """Return the India base locale for a registered variant, else ``None``."""
    v = INDIA_VARIANT_LOCALES.get(locale)
    return v.base_locale if v is not None else None


def persona_dataset_locale(locale: str, *, datasets_dir: "Any | None" = None) -> str:
    """Which persona-dataset locale to sample for a conversation ``locale``.

    Presence-aware: for a registered variant, return the variant itself if
    ``<datasets_dir>/<locale>.parquet`` exists (a native persona dataset has
    landed), otherwise the variant's ``base_locale`` (``en_IN``). Non-variant
    locales are returned unchanged (so an unknown/typo locale still fails
    fast downstream in DD).
    """
    base = _variant_base(locale)
    if base is None:
        return locale
    if datasets_dir is not None:
        from pathlib import Path

        if (Path(datasets_dir) / f"{locale}.parquet").exists():
            return locale
    return base


def asset_locale(locale: str, *, exists: "Any | None" = None) -> str:
    """Which locale to resolve asset banks / taxonomies against.

    Presence-aware: for a registered variant, return the variant itself when
    ``exists(locale)`` is truthy (a native bank/taxonomy has landed),
    otherwise the ``base_locale`` (``en_IN``). ``exists`` is an optional
    predicate injected by the caller (so this module stays filesystem-free).
    Non-variant locales are returned unchanged.
    """
    base = _variant_base(locale)
    if base is None:
        return locale
    if exists is not None and exists(locale):
        return locale
    return base


def prompt_pack_locale(locale: str, *, has: "Any | None" = None) -> str:
    """Which locale to resolve a per-probe prompt pack against.

    Presence-aware: for a registered variant, return the variant itself when
    ``has(locale)`` is truthy (a native prompt pack has been authored),
    otherwise the ``base_locale`` (``en_IN``). Non-variant locales unchanged.
    """
    base = _variant_base(locale)
    if base is None:
        return locale
    if has is not None and has(locale):
        return locale
    return base
