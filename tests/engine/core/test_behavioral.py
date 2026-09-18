# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for core/behavioral.py -- OCEAN->behavioral parameter computation."""

from usersim.engine.core.behavioral import (
    compute_behavioral_profile,
    compute_disclosure_style,
    compute_frustration_level,
    compute_user_interaction_style,
    format_behavioral_profile_for_prompt,
    format_disclosure_instructions,
    format_interaction_style_instructions,
    get_conversation_language,
    get_frustration_prompt,
    _education_ordinal,
    _extract_ocean,
    _extract_ocean_metadata,
    _infer_tech_literacy,
    _is_tech_occupation,
    _parse_ocean_value,
)


class TestParseOceanValue:
    """DD returns OCEAN as dicts with t_scores; plain floats also supported."""

    def test_dict_with_t_score(self):
        val = _parse_ocean_value({"label": "high", "t_score": 57, "description": "..."})
        assert 0.5 < val < 0.7

    def test_dict_t_score_50_is_midpoint(self):
        val = _parse_ocean_value({"t_score": 50})
        assert abs(val - 0.5) < 0.01

    def test_dict_t_score_20_is_zero(self):
        assert _parse_ocean_value({"t_score": 20}) == 0.0

    def test_dict_t_score_80_is_one(self):
        assert _parse_ocean_value({"t_score": 80}) == 1.0

    def test_plain_float(self):
        assert _parse_ocean_value(0.7) == 0.7

    def test_plain_int(self):
        assert _parse_ocean_value(1) == 1.0

    def test_none_defaults_to_midpoint(self):
        assert _parse_ocean_value(None) == 0.5

    def test_clamps_extreme_t_scores(self):
        assert _parse_ocean_value({"t_score": 100}) == 1.0
        assert _parse_ocean_value({"t_score": 10}) == 0.0


class TestExtractOcean:
    def test_extracts_all_traits_from_plain_floats(self, full_persona):
        ocean = _extract_ocean(full_persona)
        assert set(ocean.keys()) == {"openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"}
        assert ocean["openness"] == 0.7
        assert ocean["neuroticism"] == 0.3

    def test_extracts_from_dd_dict_format(self):
        persona = {
            "openness": {"label": "high", "t_score": 57, "description": "Curious"},
            "conscientiousness": {"label": "average", "t_score": 48, "description": "Balanced"},
            "extraversion": {"label": "high", "t_score": 59, "description": "Sociable"},
            "agreeableness": {"label": "average", "t_score": 47, "description": "Cooperative"},
            "neuroticism": {"label": "low", "t_score": 42, "description": "Stable"},
        }
        ocean = _extract_ocean(persona)
        assert all(0.0 <= v <= 1.0 for v in ocean.values())
        assert ocean["openness"] > ocean["neuroticism"]

    def test_defaults_to_midpoint(self):
        ocean = _extract_ocean({})
        assert all(v == 0.5 for v in ocean.values())


class TestEducationOrdinal:
    def test_en_us_exact_values(self):
        assert _education_ordinal("graduate", "en_US") == 1.0
        assert _education_ordinal("less_than_9th", "en_US") == 0.0
        assert _education_ordinal("bachelors", "en_US") == 0.7
        assert _education_ordinal("some_college", "en_US") == 0.45

    def test_pt_br_exact_values(self):
        assert _education_ordinal("Superior completo", "pt_BR") == 1.0
        assert _education_ordinal("Sem instrução e fundamental incompleto", "pt_BR") == 0.0

    def test_en_in_exact_values(self):
        assert _education_ordinal("Graduate & above", "en_IN") == 1.0
        assert _education_ordinal("Illiterate", "en_IN") == 0.0
        assert _education_ordinal("Primary", "en_IN") == 0.2

    def test_keyword_fallback(self):
        assert _education_ordinal("Bachelor's degree", "xx_UNKNOWN") >= 0.7
        assert _education_ordinal("PhD", "xx_UNKNOWN") >= 0.7

    def test_unknown_defaults_to_midpoint(self):
        assert _education_ordinal("something_random", "en_US") == 0.5


class TestIsTechOccupation:
    def test_exact_match(self):
        assert _is_tech_occupation("software_developer")
        assert _is_tech_occupation("computer_programmer")
        assert _is_tech_occupation("web_developer")

    def test_keyword_fallback_for_test_fixtures(self):
        assert _is_tech_occupation("Software Engineer")

    def test_rejects_non_tech(self):
        assert not _is_tech_occupation("nurse_practitioner")
        assert not _is_tech_occupation("janitor_or_building_cleaner")
        assert not _is_tech_occupation("waiter_or_waitress")
        assert not _is_tech_occupation("accountant_or_auditor")
        assert not _is_tech_occupation("pharmacy_technician")
        assert not _is_tech_occupation("locomotive_engineer_or_operator")

    def test_no_false_positive_on_it_substring(self):
        assert not _is_tech_occupation("writer_or_author")
        assert not _is_tech_occupation("editor")
        assert not _is_tech_occupation("architect")


class TestInferTechLiteracy:
    def test_tech_occupation_boosts_score(self, full_persona):
        score = _infer_tech_literacy(full_persona)
        assert score > 0.6

    def test_elderly_low_ed_lowers_score(self, minimal_persona):
        score = _infer_tech_literacy(minimal_persona)
        assert score < 0.5

    def test_en_us_real_data(self):
        persona = {"education_level": "graduate", "occupation": "software_developer", "age": 28}
        assert _infer_tech_literacy(persona, "en_US") >= 0.9

    def test_age_over_75_penalty(self):
        persona = {"education_level": "high_school", "occupation": "not_in_workforce", "age": 80}
        assert _infer_tech_literacy(persona, "en_US") < 0.35

    def test_pt_br_locale(self):
        persona = {"education_level": "Superior completo", "occupation": "Diretor ou gerente", "age": 40}
        assert _infer_tech_literacy(persona, "pt_BR") > 0.5


class TestExtractOceanMetadata:
    def test_extracts_labels_and_descriptions(self):
        persona = {
            "openness": {"label": "high", "t_score": 60, "description": "Curious and creative."},
            "conscientiousness": {"label": "average", "t_score": 50, "description": "Balanced."},
            "extraversion": 0.5,
            "agreeableness": {"label": "low", "t_score": 35, "description": "Independent."},
            "neuroticism": {"label": "very low", "t_score": 25, "description": "Calm."},
        }
        labels, descs = _extract_ocean_metadata(persona)
        assert labels["openness"] == "high"
        assert descs["openness"] == "Curious and creative."
        assert labels["extraversion"] == "?"
        assert descs["extraversion"] == ""


class TestComputeBehavioralProfile:
    def test_returns_all_fields(self, full_persona):
        profile = compute_behavioral_profile(full_persona)
        assert "patience" in profile
        assert "verbosity" in profile
        assert "cooperativeness" in profile
        assert "tech_literacy" in profile
        assert "error_proneness" in profile
        assert "ocean" in profile

    def test_values_in_range(self, full_persona):
        profile = compute_behavioral_profile(full_persona)
        for key in ["patience", "verbosity", "cooperativeness", "tech_literacy", "error_proneness"]:
            assert 0.0 <= profile[key] <= 1.0

    def test_high_agreeableness_means_cooperative(self, full_persona):
        profile = compute_behavioral_profile(full_persona)
        assert profile["cooperativeness"] > 0.5

    def test_low_tech_literacy_for_elderly(self, minimal_persona):
        profile = compute_behavioral_profile(minimal_persona)
        assert profile["tech_literacy"] < 0.5


class TestFormatBehavioralProfile:
    def test_produces_nonempty_string(self, full_persona):
        profile = compute_behavioral_profile(full_persona)
        text = format_behavioral_profile_for_prompt(profile)
        assert len(text) > 50
        assert "Additional traits" in text

    def test_no_patience_verbosity_coop_in_prompt(self, full_persona):
        profile = compute_behavioral_profile(full_persona)
        text = format_behavioral_profile_for_prompt(profile)
        assert "patient" not in text.lower()
        assert "verbose" not in text.lower()
        assert "cooperative" not in text.lower()


class TestGetConversationLanguage:
    def test_known_locales(self):
        assert get_conversation_language("en_US") == "English"
        assert get_conversation_language("pt_BR") == "Portuguese"
        assert get_conversation_language("ja_JP") == "Japanese"
        assert get_conversation_language("fr_FR") == "French"

    def test_unknown_defaults_to_english(self):
        assert get_conversation_language("xx_XX") == "English"


# ---------------------------------------------------------------------------
# Sim2Real: Disclosure style tests
# ---------------------------------------------------------------------------


class TestComputeDisclosureStyle:
    def test_returns_valid_style(self):
        style = compute_disclosure_style(persona_seed=42)
        assert style in ("incremental", "upfront")

    def test_deterministic_with_seed(self):
        s1 = compute_disclosure_style(persona_seed=123)
        s2 = compute_disclosure_style(persona_seed=123)
        assert s1 == s2

    def test_ratio_zero_always_upfront(self):
        for seed in range(20):
            assert compute_disclosure_style(persona_seed=seed, incremental_ratio=0.0) == "upfront"

    def test_ratio_one_always_incremental(self):
        for seed in range(20):
            assert compute_disclosure_style(persona_seed=seed, incremental_ratio=1.0) == "incremental"

    def test_approximate_ratio(self):
        results = [compute_disclosure_style(persona_seed=i) for i in range(1000)]
        inc_frac = sum(1 for r in results if r == "incremental") / len(results)
        assert 0.5 < inc_frac < 0.7


class TestFormatDisclosureInstructions:
    def test_incremental_returns_block(self):
        text = format_disclosure_instructions("incremental")
        assert "INFORMATION_DISCLOSURE" in text
        assert "Do NOT provide all relevant details" in text

    def test_upfront_returns_empty(self):
        assert format_disclosure_instructions("upfront") == ""

    def test_unknown_returns_empty(self):
        assert format_disclosure_instructions("unknown") == ""

    def test_pacing_intent_is_preserved(self):
        """The style exists to stop the user front-loading the opening message."""
        text = format_disclosure_instructions("incremental")
        assert "Do NOT provide all relevant details in your first message" in text
        assert "wait for the assistant to ask" in text
        assert "Keep your initial message short" in text

    def test_no_blanket_licence_to_stall(self):
        """Regression. The block used to end with "It is OK to say 'I'm not sure' or
        'let me check' when appropriate", which contradicted "reveal ... only when
        ASKED" and let the user-sim refuse any direct question. On the
        financial_services probe that cost 0.424 vs 0.667 tool selection
        (`incremental` vs `upfront`, n=24/22): the assistant asked for what it needed,
        the user declined, and the task could never complete.

        Being unsure must stay scoped to things a customer plausibly would not know.
        """
        text = format_disclosure_instructions("incremental")
        assert "It is OK to say 'I'm not sure'" not in text
        # ...and the replacement obliges an answer for what the customer does know.
        assert "answer it" in text
        assert "would not know offhand" in text


# ---------------------------------------------------------------------------
# Sim2Real: User interaction style tests
# ---------------------------------------------------------------------------


class TestComputeUserInteractionStyle:
    def test_high_neuroticism_low_agreeableness_is_confrontational(self):
        profile = {
            "patience": 0.2,
            "ocean": {
                "neuroticism": 0.8,
                "agreeableness": 0.3,
                "openness": 0.5,
                "conscientiousness": 0.5,
                "extraversion": 0.5,
            },
        }
        assert compute_user_interaction_style(profile) == "confrontational"

    def test_high_neuroticism_low_patience_is_frustrated(self):
        profile = {
            "patience": 0.3,
            "ocean": {
                "neuroticism": 0.7,
                "agreeableness": 0.5,
                "openness": 0.5,
                "conscientiousness": 0.5,
                "extraversion": 0.5,
            },
        }
        assert compute_user_interaction_style(profile) == "frustrated"

    def test_high_agreeableness_low_neuroticism_is_cooperative(self):
        profile = {
            "patience": 0.8,
            "ocean": {
                "neuroticism": 0.2,
                "agreeableness": 0.8,
                "openness": 0.5,
                "conscientiousness": 0.5,
                "extraversion": 0.5,
            },
        }
        assert compute_user_interaction_style(profile) == "cooperative"

    def test_low_agreeableness_is_impatient(self):
        profile = {
            "patience": 0.5,
            "ocean": {
                "neuroticism": 0.4,
                "agreeableness": 0.3,
                "openness": 0.5,
                "conscientiousness": 0.5,
                "extraversion": 0.5,
            },
        }
        assert compute_user_interaction_style(profile) == "impatient"

    def test_moderate_is_neutral(self):
        profile = {
            "patience": 0.5,
            "ocean": {
                "neuroticism": 0.5,
                "agreeableness": 0.5,
                "openness": 0.5,
                "conscientiousness": 0.5,
                "extraversion": 0.5,
            },
        }
        assert compute_user_interaction_style(profile) == "neutral"

    def test_returns_valid_style_for_full_persona(self, full_persona):
        profile = compute_behavioral_profile(full_persona)
        style = compute_user_interaction_style(profile)
        assert style in ("cooperative", "neutral", "impatient", "frustrated", "confrontational")


class TestFormatInteractionStyleInstructions:
    def test_confrontational_returns_block(self):
        text = format_interaction_style_instructions("confrontational")
        assert "INTERACTION_STYLE" in text
        assert "confrontational" in text

    def test_neutral_returns_empty(self):
        assert format_interaction_style_instructions("neutral") == ""

    def test_cooperative_returns_block(self):
        text = format_interaction_style_instructions("cooperative")
        assert "patient" in text


# ---------------------------------------------------------------------------
# Sim2Real: Frustration level and prompts
# ---------------------------------------------------------------------------


class TestComputeFrustrationLevel:
    def test_cooperative_stays_calm(self):
        level = compute_frustration_level("cooperative", turn_idx=2, assistant_failures=1, patience=0.8, max_turns=3)
        assert level <= 1

    def test_confrontational_escalates_fast(self):
        level = compute_frustration_level(
            "confrontational", turn_idx=1, assistant_failures=1, patience=0.3, max_turns=3
        )
        assert level >= 2

    def test_no_failures_low_frustration(self):
        level = compute_frustration_level("neutral", turn_idx=0, assistant_failures=0, patience=0.5, max_turns=3)
        assert level == 0

    def test_clamped_to_three(self):
        level = compute_frustration_level(
            "confrontational", turn_idx=5, assistant_failures=5, patience=0.1, max_turns=8
        )
        assert level == 3

    def test_frustrated_starts_elevated(self):
        level = compute_frustration_level("frustrated", turn_idx=0, assistant_failures=0, patience=0.5, max_turns=3)
        assert level >= 1

    def test_proportional_thresholds_3_turns(self):
        level_early = compute_frustration_level(
            "impatient", turn_idx=0, assistant_failures=0, patience=0.3, max_turns=3
        )
        level_mid = compute_frustration_level("impatient", turn_idx=1, assistant_failures=0, patience=0.3, max_turns=3)
        level_late = compute_frustration_level("impatient", turn_idx=2, assistant_failures=0, patience=0.3, max_turns=3)
        assert level_late >= level_mid >= level_early

    def test_proportional_thresholds_8_turns(self):
        level_early = compute_frustration_level(
            "impatient", turn_idx=0, assistant_failures=0, patience=0.3, max_turns=8
        )
        level_mid = compute_frustration_level("impatient", turn_idx=4, assistant_failures=0, patience=0.3, max_turns=8)
        level_late = compute_frustration_level("impatient", turn_idx=6, assistant_failures=0, patience=0.3, max_turns=8)
        assert level_late >= level_mid >= level_early


class TestGetFrustrationPrompt:
    def test_level_zero_empty(self):
        assert get_frustration_prompt(0) == ""

    def test_level_one_has_content(self):
        prompt = get_frustration_prompt(1)
        assert len(prompt) > 20
        assert "impatient" in prompt.lower() or "short" in prompt.lower()

    def test_level_two_more_intense(self):
        prompt = get_frustration_prompt(2)
        assert "patience" in prompt.lower() or "direct" in prompt.lower()

    def test_level_three_most_intense(self):
        prompt = get_frustration_prompt(3)
        assert "displeased" in prompt.lower() or "blunt" in prompt.lower()

    def test_above_three_clamps(self):
        assert get_frustration_prompt(5) == get_frustration_prompt(3)
