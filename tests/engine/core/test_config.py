# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Tests for the unified ConversationSimulatorConfig."""

import pytest
from pydantic import ValidationError

from usersim.engine.config import ConversationSimulatorConfig


class TestConversationSimulatorConfig:
    def test_default_values(self):
        cfg = ConversationSimulatorConfig(name="test")
        assert cfg.locale == "en_US"
        assert cfg.tools_column is None
        assert cfg.toolset_name_column is None
        assert cfg.max_turns == 5
        assert cfg.max_tools == 5
        assert cfg.max_steps == 10
        assert cfg.random_seed is None

    def test_custom_locale(self):
        cfg = ConversationSimulatorConfig(name="test", locale="pt_BR")
        assert cfg.locale == "pt_BR"

    def test_random_seed_can_be_set(self):
        cfg = ConversationSimulatorConfig(name="test", random_seed=42)
        assert cfg.random_seed == 42

    def test_compression_window_must_be_positive(self):
        with pytest.raises(ValidationError):
            ConversationSimulatorConfig(name="test", compression_window=0)

    def test_incremental_disclosure_ratio_default(self):
        cfg = ConversationSimulatorConfig(name="test")
        assert cfg.incremental_disclosure_ratio == 0.6

    def test_incremental_disclosure_ratio_custom(self):
        cfg = ConversationSimulatorConfig(name="test", incremental_disclosure_ratio=0.8)
        assert cfg.incremental_disclosure_ratio == 0.8

    def test_side_effect_columns(self):
        cfg = ConversationSimulatorConfig(name="test")
        side_effects = cfg.side_effect_columns
        assert "conversation_messages" in side_effects
        assert "simulation_outcome" in side_effects
        assert "simulation_traces" in side_effects
        assert "trajectory_id" in side_effects
        assert "persona_uuid" in side_effects
        assert "behavioral_profile" in side_effects
        assert "persona_religion_language_context" in side_effects
        assert "locale" in side_effects
        assert "conversation_language" in side_effects
        assert "tool_subset" in side_effects
        assert "num_tool_calls" in side_effects
        assert "num_turns" in side_effects
        assert "disclosure_style" in side_effects
        assert "user_interaction_style" in side_effects

    def test_required_columns(self):
        cfg = ConversationSimulatorConfig(name="test")
        required = cfg.required_columns
        assert "persona" in required
        assert "probe_type" in required
        assert "theme" in required

    def test_tools_column_optional(self):
        cfg = ConversationSimulatorConfig(name="test", tools_column="tools")
        assert cfg.tools_column == "tools"
