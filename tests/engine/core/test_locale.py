# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``core/locale.py`` — locale → language + script mappings.

The module is plain data; tests assert the invariants downstream
scorers rely on (every supported locale has both a language and a
script set; the script set is non-empty; lingua's expected language
names match what we declare).
"""

from __future__ import annotations

import pytest

from usersim.engine.core.locale import (
    DEVANAGARI_RANGES,
    HAN_RANGES,
    HANGUL_RANGES,
    HIRAGANA_RANGES,
    KATAKANA_RANGES,
    LATIN_RANGES,
    LOCALE_TO_EXPECTED_SCRIPT_RANGES,
    LOCALE_TO_LANGUAGE_DISPLAY,
    LOCALE_TO_LANGUAGE_NAME,
    SCRIPT_BUCKETS,
    SHIPPED_LOCALES,
    expected_language_display,
    expected_language_name,
    expected_script_ranges,
    persona_dataset_locale,
    supported_locales,
)


class TestSupportedLocales:
    def test_shipped_locales_are_unique(self) -> None:
        assert len(SHIPPED_LOCALES) == len(set(SHIPPED_LOCALES))

    def test_shipped_locales_supported(self) -> None:
        assert set(supported_locales()) == set(SHIPPED_LOCALES)

    def test_supported_locales_returns_sorted(self) -> None:
        assert list(supported_locales()) == sorted(supported_locales())

    def test_language_and_script_maps_have_same_keys(self) -> None:
        # Adding a locale to one map without the other = config bug.
        assert set(LOCALE_TO_LANGUAGE_NAME) == set(LOCALE_TO_EXPECTED_SCRIPT_RANGES)

    def test_display_map_covers_shipped_locales(self) -> None:
        assert set(LOCALE_TO_LANGUAGE_DISPLAY) == set(SHIPPED_LOCALES)

    def test_language_map_covers_shipped_locales(self) -> None:
        assert set(LOCALE_TO_LANGUAGE_NAME) == set(SHIPPED_LOCALES)


class TestExpectedLanguage:
    @pytest.mark.parametrize(
        "locale, expected",
        [
            ("en_US", "ENGLISH"),
            ("en_IN", "ENGLISH"),
            ("en_SG", "ENGLISH"),
            ("pt_BR", "PORTUGUESE"),
            ("fr_FR", "FRENCH"),
            ("ja_JP", "JAPANESE"),
            ("hi_Deva_IN", "HINDI"),
            ("hi_Latn_IN", None),
            ("ko_KR", "KOREAN"),
        ],
    )
    def test_each_locale_maps_to_expected_lingua_language(
        self,
        locale: str,
        expected: str,
    ) -> None:
        assert expected_language_name(locale) == expected

    def test_unknown_locale_returns_none(self) -> None:
        assert expected_language_name("xx_YY") is None

    def test_human_readable_language_labels(self) -> None:
        assert expected_language_display("ko_KR") == "Korean"
        assert expected_language_display("en_SG") == "English"

    def test_lingua_language_names_resolve(self) -> None:
        """Every name we declare must be a real lingua Language enum."""
        from lingua import Language

        for name in LOCALE_TO_LANGUAGE_NAME.values():
            if name is None:
                # Romanized locales (e.g. hi_Latn_IN) deliberately map to
                # None: lingua cannot detect them, so the deterministic
                # language scorers skip them. Nothing to resolve.
                continue
            assert hasattr(Language, name), (
                f"language name {name!r} declared in LOCALE_TO_LANGUAGE_NAME but not present in lingua.Language enum"
            )


class TestPersonaDatasetLocale:
    @pytest.mark.parametrize(
        "locale, expected",
        [
            # Base locales with their own NGC persona dataset.
            ("en_US", "en_US"),
            ("en_IN", "en_IN"),
            ("en_SG", "en_SG"),
            ("pt_BR", "pt_BR"),
            ("fr_FR", "fr_FR"),
            ("ja_JP", "ja_JP"),
            ("ko_KR", "ko_KR"),
            # Hindi is NOT an India language-variant: Nemotron-Personas-India
            # ships English + Hindi (Devanagari) + Hindi (Latin), so both Hindi
            # locales sample their OWN native panel. Guards against re-introducing
            # a blanket "endswith('_IN') -> en_IN" rule, which would silently
            # swap native Hindi personas for the en_IN English panel.
            ("hi_Deva_IN", "hi_Deva_IN"),
            ("hi_Latn_IN", "hi_Latn_IN"),
            # Registered India language-variants have no panel of their own -> en_IN.
            ("ta_Taml_IN", "en_IN"),
            ("bn_Beng_IN", "en_IN"),
        ],
    )
    def test_resolves_without_datasets_dir(self, locale: str, expected: str) -> None:
        assert persona_dataset_locale(locale) == expected

    def test_native_parquet_takes_precedence(self, tmp_path) -> None:
        # If a native parquet is dropped in, a variant uses its own panel.
        (tmp_path / "ta_Taml_IN.parquet").write_bytes(b"")
        assert persona_dataset_locale("ta_Taml_IN", datasets_dir=tmp_path) == "ta_Taml_IN"
        # Without the native file, it falls back to en_IN.
        assert persona_dataset_locale("bn_Beng_IN", datasets_dir=tmp_path) == "en_IN"


class TestExpectedScript:
    def test_latin_locales_get_latin_ranges(self) -> None:
        for locale in ("en_US", "en_IN", "en_SG", "pt_BR", "fr_FR", "hi_Latn_IN"):
            assert expected_script_ranges(locale) == LATIN_RANGES

    def test_japanese_includes_three_scripts(self) -> None:
        ranges = expected_script_ranges("ja_JP")
        # Should include all of hiragana / katakana / han ranges.
        assert HIRAGANA_RANGES[0] in ranges
        assert KATAKANA_RANGES[0] in ranges
        assert HAN_RANGES[0] in ranges

    def test_hindi_is_only_devanagari(self) -> None:
        assert expected_script_ranges("hi_Deva_IN") == DEVANAGARI_RANGES

    def test_korean_is_hangul(self) -> None:
        assert expected_script_ranges("ko_KR") == HANGUL_RANGES

    def test_unknown_locale_returns_empty_tuple(self) -> None:
        assert expected_script_ranges("xx_YY") == ()


class TestScriptBuckets:
    def test_buckets_cover_all_supported_locales(self) -> None:
        # Every range present in a locale's expected_script_ranges
        # must be one of the named bucket ranges. (Spot-check the
        # forward direction; the reverse test in TestSampleCodepoints
        # picks up surprises.)
        all_bucket_ranges = set()
        for ranges in SCRIPT_BUCKETS.values():
            for r in ranges:
                all_bucket_ranges.add(r)
        for locale in supported_locales():
            for r in expected_script_ranges(locale):
                assert r in all_bucket_ranges, (
                    f"{locale} declares range {r} that is not in any named SCRIPT_BUCKETS; reviewer must classify it"
                )


class TestSampleCodepoints:
    @pytest.mark.parametrize(
        "char, expected_locales",
        [
            # Latin (hi_Latn_IN is romanized Hindi -> Latin ranges)
            ("a", ["en_US", "en_IN", "en_SG", "pt_BR", "fr_FR", "hi_Latn_IN"]),
            ("Z", ["en_US", "en_IN", "en_SG", "pt_BR", "fr_FR", "hi_Latn_IN"]),
            ("é", ["en_US", "en_IN", "en_SG", "pt_BR", "fr_FR", "hi_Latn_IN"]),
            # Devanagari
            ("क", ["hi_Deva_IN"]),
            ("ष", ["hi_Deva_IN"]),
            # Hiragana
            ("あ", ["ja_JP"]),
            # Katakana
            ("カ", ["ja_JP"]),
            # Han
            ("漢", ["ja_JP"]),
            ("字", ["ja_JP"]),
            # Hangul
            ("한", ["ko_KR"]),
            ("ᄀ", ["ko_KR"]),
        ],
    )
    def test_known_chars_in_only_expected_locales(
        self,
        char: str,
        expected_locales: list[str],
    ) -> None:
        cp = ord(char)
        for locale in supported_locales():
            ranges = expected_script_ranges(locale)
            in_locale = any(lo <= cp <= hi for (lo, hi) in ranges)
            should_match = locale in expected_locales
            assert in_locale == should_match, (
                f"char {char!r} (U+{cp:04X}) "
                f"{'should' if should_match else 'should NOT'} match locale "
                f"{locale} (got: in_locale={in_locale})"
            )
