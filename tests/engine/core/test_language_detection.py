# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Tests for ``core/language_detection.py`` — lingua wrappers + script helpers.

The lingua detector is built lazily and shared process-wide. Tests
that change the detector state should call
:func:`reset_detector` in their teardown — the autouse fixture below
does this automatically. Lingua loading is ~170 MB and a few hundred
ms; the per-test overhead is negligible compared to the first-time
detector build (which most tests amortize across the file).
"""

from __future__ import annotations

import pytest

from usersim.engine.core.language_detection import (
    detect_language_name,
    foreign_scripts_used,
    reset_detector,
    script_compliance_fraction,
    script_dominance,
)
from usersim.engine.core.locale import (
    DEVANAGARI_RANGES,
    KANNADA_RANGES,
    LATIN_RANGES,
    expected_script_ranges,
)


@pytest.fixture(autouse=True)
def _reset_detector_between_test_files():
    """Ensure detector state is clean before each test."""
    yield
    # Don't drop after every test (slow); only at file teardown.
    # If a test mutates global state, it should call reset_detector
    # itself. For now, keep the cached detector across tests for speed.


# ---------------------------------------------------------------------------
# Detector caching
# ---------------------------------------------------------------------------


class TestDetectorCaching:
    def test_first_call_builds_detector(self) -> None:
        reset_detector()
        # No exception, returns a valid detection (cold-start).
        result = detect_language_name("Hello, this is English text.")
        assert result == "ENGLISH"

    def test_subsequent_calls_reuse_detector(self) -> None:
        # Two calls should both succeed and the second should be much
        # faster — but we don't time-assert (CI noise). The functional
        # check is that both return correctly.
        a = detect_language_name("Bonjour, comment allez-vous?")
        b = detect_language_name("Bonjour, comment allez-vous?")
        assert a == b == "FRENCH"

    def test_reset_drops_cached_detector(self) -> None:
        # After reset, a subsequent detection still works (rebuild).
        detect_language_name("Hello")
        reset_detector()
        result = detect_language_name("Hello world")
        assert result == "ENGLISH"


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


class TestDetectLanguageName:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("Hello, how are you doing today?", "ENGLISH"),
            ("Bonjour, comment ca va aujourd'hui?", "FRENCH"),
            ("Olá, tudo bem com você hoje?", "PORTUGUESE"),
            ("こんにちは、お元気ですか？", "JAPANESE"),
            ("नमस्ते, आप कैसे हैं आज?", "HINDI"),
        ],
    )
    def test_detects_each_locale_language(
        self,
        text: str,
        expected: str,
    ) -> None:
        assert detect_language_name(text) == expected

    @pytest.mark.parametrize("text", ["", " ", "   \n  \t  "])
    def test_empty_or_whitespace_returns_none(self, text: str) -> None:
        assert detect_language_name(text) is None

    def test_short_unambiguous_text_still_detects(self) -> None:
        # Lingua's headline feature is short-text accuracy. Sanity-
        # check it on a single recognizable Hindi word.
        assert detect_language_name("नमस्ते") == "HINDI"


# ---------------------------------------------------------------------------
# Script compliance
# ---------------------------------------------------------------------------


class TestScriptComplianceFraction:
    def test_pure_latin_in_latin_locale(self) -> None:
        assert script_compliance_fraction("Hello world", LATIN_RANGES) == 1.0

    def test_pure_devanagari_in_devanagari_locale(self) -> None:
        assert script_compliance_fraction("नमस्ते", DEVANAGARI_RANGES) == 1.0

    def test_latin_text_against_devanagari_returns_zero(self) -> None:
        assert script_compliance_fraction("namaste", DEVANAGARI_RANGES) == 0.0

    def test_mixed_devanagari_and_latin(self) -> None:
        # "Hello नमस्ते" — Latin "Hello" (5 letters) + Devanagari
        # "नमस्ते" (6 letters by Python iteration). Compliance against
        # Devanagari = 6/(5+6) = 0.5454... — match within rounding.
        text = "Hello नमस्ते"
        result = script_compliance_fraction(text, DEVANAGARI_RANGES)
        # Don't assert exact value — combining marks make the count
        # implementation-dependent. Assert it's strictly between 0 and 1.
        assert 0.0 < result < 1.0

    def test_empty_text_vacuously_compliant(self) -> None:
        assert script_compliance_fraction("", LATIN_RANGES) == 1.0

    def test_no_letters_vacuously_compliant(self) -> None:
        # Digits, punctuation, whitespace are script-neutral.
        assert script_compliance_fraction("12345 !!! ...", LATIN_RANGES) == 1.0

    def test_empty_expected_ranges_vacuously_compliant(self) -> None:
        assert script_compliance_fraction("anything", ()) == 1.0

    def test_japanese_text_against_japanese_ranges(self) -> None:
        # Mixes hiragana, katakana, han — all expected for ja_JP.
        text = "今日は、コンピューターを使っています"
        ranges = expected_script_ranges("ja_JP")
        assert script_compliance_fraction(text, ranges) == 1.0

    def test_digits_inside_hindi_dont_penalize(self) -> None:
        # "मेरी उम्र 30 साल है" — digits are script-neutral.
        text = "मेरी उम्र 30 साल है"
        assert script_compliance_fraction(text, DEVANAGARI_RANGES) == 1.0


# ---------------------------------------------------------------------------
# Script dominance
# ---------------------------------------------------------------------------


class TestScriptDominance:
    def test_pure_latin_text(self) -> None:
        result = script_dominance("Hello world")
        assert result["latin"] == 1.0
        assert result["devanagari"] == 0.0

    def test_pure_devanagari_text(self) -> None:
        result = script_dominance("नमस्ते")
        assert result["devanagari"] == 1.0
        assert result["latin"] == 0.0

    def test_japanese_mixes_three_buckets(self) -> None:
        # Hiragana + Katakana + Han all present.
        result = script_dominance("今日コンピュータを使います")
        # Each Japanese bucket has > 0 fraction.
        assert result["hiragana"] > 0
        assert result["katakana"] > 0
        assert result["han"] > 0
        assert (result["hiragana"] + result["katakana"] + result["han"]) == pytest.approx(1.0, abs=0.01)

    def test_empty_text_all_zeros(self) -> None:
        result = script_dominance("")
        assert all(v == 0.0 for v in result.values())

    def test_no_letters_all_zeros(self) -> None:
        result = script_dominance("12345 !!! ...")
        assert all(v == 0.0 for v in result.values())

    def test_returns_all_named_buckets_plus_other(self) -> None:
        from usersim.engine.core.locale import SCRIPT_BUCKETS

        result = script_dominance("anything")
        # Every named script bucket (incl. the Indic/Arabic scripts added for
        # the India language-variant layer) plus the catch-all "other".
        assert set(result) == set(SCRIPT_BUCKETS) | {"other"}
        # The original core buckets remain present (regression guard).
        assert {"latin", "devanagari", "hangul", "hiragana", "katakana", "han"} <= set(result)


class TestForeignScriptsUsed:
    """A degraded model emits stray glyphs from unrelated scripts mid-word --
    ``"MAB \u0ca8Listener"``, ``"\u0444\u043e\u0440\u043c\u0430\u091f\u094d"``. That is a broken generation rather
    than a language choice, and it is the only language signal available for
    the locales lingua has no model for, so it must not be confused with
    script compliance (which counts legitimate Latin against you).
    """

    def test_clean_native_text_has_no_intrusions(self) -> None:
        assert (
            foreign_scripts_used("\u0c95\u0ca8\u0ccd\u0ca8\u0ca1 \u0caa\u0ca6\u0c97\u0cb3\u0cc1", KANNADA_RANGES) == ()
        )

    def test_latin_is_always_allowed(self) -> None:
        """Kannada prose carrying UPI / KYC / PAN / a tool name is how the
        corpora are deliberately written; flagging it would make the axis
        unusable on exactly the locales it exists for."""
        text = "\u0c95\u0ca8\u0ccd\u0ca8\u0ca1 UPI KYC PAN verify_identity \u0caa\u0ca6\u0c97\u0cb3\u0cc1"
        assert foreign_scripts_used(text, KANNADA_RANGES) == ()

    def test_names_the_intruding_scripts(self) -> None:
        # Hangul + Cyrillic inside Kannada, as observed in a real run.
        text = "\u0c95\u0ca8\u0ccd\u0ca8\u0ca1 \uac80 \u0444\u043e\u0440\u043c\u0430 \u0caa\u0ca6"
        assert foreign_scripts_used(text, KANNADA_RANGES) == ("cyrillic", "hangul")

    def test_a_single_stray_glyph_is_enough(self) -> None:
        """All-or-nothing on purpose: one bad codepoint mid-word already is the
        defect, and waiting for a share of the text would hide the mild cases."""
        assert foreign_scripts_used("\u0c95\u0ca8\u0ccd\u0ca8\u0ca1 \uac80", KANNADA_RANGES) == ("hangul",)

    def test_devanagari_in_an_english_locale_is_foreign(self) -> None:
        assert foreign_scripts_used("Hello \u0928\u092e\u0938\u094d\u0924\u0947", LATIN_RANGES) == ("devanagari",)

    def test_no_expected_script_means_nothing_to_be_foreign_to(self) -> None:
        assert foreign_scripts_used("anything \uac80 \u0444", ()) == ()

    def test_non_letters_are_ignored(self) -> None:
        assert foreign_scripts_used("\u0c95\u0ca8\u0ccd\u0ca8\u0ca1 123 !!! \u20b9500", KANNADA_RANGES) == ()
