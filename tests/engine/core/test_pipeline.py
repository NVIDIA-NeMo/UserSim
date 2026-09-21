# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Integration test: end-to-end pipeline build and config validation.

These tests verify the pipeline configuration is valid without requiring
LLM API calls. Full end-to-end generation tests require an NVIDIA API key
and are marked with @pytest.mark.integration.
"""

import pytest

from usersim.engine.config import ConversationSimulatorConfig
from usersim.engine.core._assets import packaged_assets_dir
from usersim.engine.core.behavioral import compute_behavioral_profile, get_conversation_language
from usersim.engine.core.locale import SHIPPED_LOCALES
from usersim.engine.core.seeds import load_seeds
from usersim.engine.evaluator.axes import (
    UNIVERSAL_SCORES,
    axes_for_probe,
)
from usersim.engine.generator import _make_failed_result


class TestPipelineConfig:
    """Verify the pipeline config produces valid structures."""

    def test_config_for_tool_calling(self):
        cfg = ConversationSimulatorConfig(
            name="conversation_messages",
            persona_column="persona",
            probe_type_column="probe_type",
            theme_column="theme",
            tools_column="tools",
            toolset_name_column="toolset_name",
            locale="en_US",
        )
        assert "conversation_messages" in cfg.side_effect_columns
        assert "simulation_outcome" in cfg.side_effect_columns
        assert "trajectory_id" in cfg.side_effect_columns
        assert cfg.locale == "en_US"

    def test_config_for_open_ended(self):
        cfg = ConversationSimulatorConfig(
            name="conversation_messages",
            locale="pt_BR",
        )
        assert cfg.tools_column is None
        assert cfg.locale == "pt_BR"

    def test_locale_language_mapping(self):
        assert get_conversation_language("en_US") == "English"
        assert get_conversation_language("pt_BR") == "Portuguese"
        assert get_conversation_language("ja_JP") == "Japanese"
        assert get_conversation_language("en_IN") == "Indian English"
        assert get_conversation_language("hi_Deva_IN") == "Hindi Devanagari"
        assert get_conversation_language("fr_FR") == "French"


class TestEndToEndStructure:
    """Verify that all components produce consistent output structures."""

    def test_failed_result_has_simulation_columns(self):
        result = _make_failed_result("test")
        simulation_columns = [
            "user_query",
            "conversation_messages",
            "conversation_metadata",
            "conversation_status",
            "simulation_outcome",
            "simulation_traces",
            "num_turns",
            "num_tool_calls",
            "tool_subset",
            "disclosure_style",
            "user_interaction_style",
        ]
        for col in simulation_columns:
            assert col in result, f"Missing column: {col}"

    def test_behavioral_profile_structure(self, full_persona):
        profile = compute_behavioral_profile(full_persona)
        assert isinstance(profile, dict)
        for key in ["patience", "verbosity", "cooperativeness", "tech_literacy", "error_proneness"]:
            assert key in profile
            assert 0.0 <= profile[key] <= 1.0
        assert "ocean" in profile
        for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            assert trait in profile["ocean"]

    def test_eval_axes_coverage(self):
        universal_names = {s.name for s in UNIVERSAL_SCORES}

        for probe in ["tool_calling", "general_open_ended", "general_educational"]:
            probe_axis_names = {s.name for s in axes_for_probe(probe)}
            for name in universal_names:
                assert name in probe_axis_names, f"Missing universal axis {name} for {probe}"

        tc_names = {s.name for s in axes_for_probe("tool_calling")}
        assert "tool_use_correctness" in tc_names

        ed_names = {s.name for s in axes_for_probe("general_educational")}
        assert "pedagogical_quality" in ed_names

        oe_names = {s.name for s in axes_for_probe("general_open_ended")}
        assert "tool_use_correctness" not in oe_names
        assert "pedagogical_quality" not in oe_names


class TestMultiLocaleConsistency:
    """Verify multi-locale support is consistent."""

    def test_seed_loading_per_locale(self):
        assets = packaged_assets_dir()
        if not (assets / "general_open_ended" / "topics" / "base").exists():
            pytest.skip("Assets directory not found")

        for locale in SHIPPED_LOCALES:
            topics = load_seeds(locale, "general_open_ended", "topics", assets)
            assert len(topics) > 0
            assert all(isinstance(t, dict) for t in topics)
            assert all("type" in t and "description" in t for t in topics)

    def test_pt_br_keeps_localized_topic_replacement(self):
        assets = packaged_assets_dir()
        if not (assets / "general_open_ended" / "topics" / "base").exists():
            pytest.skip("Assets directory not found")

        base = load_seeds("en_US", "general_open_ended", "topics", assets)
        extended = load_seeds("pt_BR", "general_open_ended", "topics", assets)
        assert len(base) == 48
        assert len(extended) == 20

    def test_all_locales_have_language_mapping(self):
        for locale in SHIPPED_LOCALES:
            lang = get_conversation_language(locale)
            assert lang != "English" or locale.startswith("en")
