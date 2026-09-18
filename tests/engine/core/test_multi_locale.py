# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for multi-locale pipeline configuration."""

import pytest

from usersim.engine.config import ConversationSimulatorConfig
from usersim.engine.core.behavioral import get_conversation_language, LOCALE_LANGUAGE_MAP
from usersim.engine.core.locale import SHIPPED_LOCALES
from usersim.engine.core.seeds import load_seeds
from usersim.engine.core._assets import packaged_assets_dir


LOCALES = SHIPPED_LOCALES
ASSETS_DIR = packaged_assets_dir()


class TestMultiLocaleLoop:
    """Verify the for-loop over locales produces correct configs."""

    def test_each_locale_gets_correct_config(self):
        for locale in LOCALES:
            cfg = ConversationSimulatorConfig(
                name="conversation_messages",
                locale=locale,
            )
            assert cfg.locale == locale
            lang = get_conversation_language(locale)
            assert isinstance(lang, str)
            assert len(lang) > 0

    def test_locale_language_consistency(self):
        assert get_conversation_language("en_US") == "English"
        assert get_conversation_language("pt_BR") == "Portuguese"
        assert get_conversation_language("ja_JP") == "Japanese"
        assert get_conversation_language("en_IN") == "Indian English"
        assert get_conversation_language("hi_Deva_IN") == "Hindi Devanagari"

    def test_all_supported_locales_have_mapping(self):
        for locale in LOCALES:
            assert locale in LOCALE_LANGUAGE_MAP


class TestLocaleSeeds:
    """Verify seed loading works for each locale."""

    @pytest.fixture(autouse=True)
    def check_assets(self):
        # Fail rather than skip: the seeds are package data, so their absence
        # is a packaging defect, not an environment the test can excuse.
        base = ASSETS_DIR / "general_open_ended" / "topics" / "base"
        assert base.is_dir(), f"packaged seeds missing at {base}"

    def test_base_seeds_available_for_all_locales(self):
        for locale in LOCALES:
            topics = load_seeds(locale, "general_open_ended", "topics", ASSETS_DIR)
            assert len(topics) >= 15  # base has 15

            subjects = load_seeds(locale, "general_educational", "subjects", ASSETS_DIR)
            assert len(subjects) >= 10  # base has 10

    def test_locale_seed_counts_preserve_localized_replacements(self):
        base_count = len(load_seeds("en_US", "general_open_ended", "topics", ASSETS_DIR))
        pt_count = len(load_seeds("pt_BR", "general_open_ended", "topics", ASSETS_DIR))
        assert base_count == 48
        assert pt_count == 20

        base_ed = len(load_seeds("en_US", "general_educational", "subjects", ASSETS_DIR))
        ja_ed = len(load_seeds("ja_JP", "general_educational", "subjects", ASSETS_DIR))
        assert base_ed == 40
        assert ja_ed == 14

    def test_seed_kinds_have_required_fields(self):
        for locale in LOCALES:
            for probe, kind in [
                ("general_open_ended", "topics"),
                ("general_educational", "subjects"),
            ]:
                seeds = load_seeds(locale, probe, kind, ASSETS_DIR)
                for seed in seeds:
                    assert "type" in seed, f"Missing 'type' in {probe}/{kind} for {locale}"
                    assert "description" in seed, f"Missing 'description' in {probe}/{kind} for {locale}"
