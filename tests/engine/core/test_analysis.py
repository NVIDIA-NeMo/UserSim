# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Tests for analysis utility functions used in the notebook."""

import json

import pandas as pd
import pytest

from usersim.engine.core.analysis import extract_eval_scores


@pytest.fixture
def sample_generated_data():
    """Simulate generated output data for analysis testing."""
    return pd.DataFrame(
        [
            {
                "locale": "en_US",
                "probe_type": "general_open_ended",
                "persona_age_bin": "25-34",
                "persona_education_level": "Bachelor's degree",
                "persona_occupation": "Software engineer",
                "persona_region": "California",
                "conversation_status": True,
                "eval_scores": json.dumps(
                    {
                        "helpfulness": {"score": 4, "reasoning": "Good"},
                        "accuracy": {"score": 3, "reasoning": "Ok"},
                        "cultural_sensitivity": {"score": 5, "reasoning": "Great"},
                        "language_appropriateness": {"score": 4, "reasoning": "Fine"},
                        "safety": {"score": 5, "reasoning": "Safe"},
                        "coherence": {"score": 4, "reasoning": "Clear"},
                        "tool_use_correctness": None,
                        "pedagogical_quality": None,
                    }
                ),
            },
            {
                "locale": "pt_BR",
                "probe_type": "general_educational",
                "persona_age_bin": "45-54",
                "persona_education_level": "Primary education",
                "persona_occupation": "Teacher",
                "persona_region": "São Paulo",
                "conversation_status": True,
                "eval_scores": json.dumps(
                    {
                        "helpfulness": {"score": 3, "reasoning": "Decent"},
                        "accuracy": {"score": 4, "reasoning": "Accurate"},
                        "cultural_sensitivity": {"score": 2, "reasoning": "Could improve"},
                        "language_appropriateness": {"score": 3, "reasoning": "Adequate"},
                        "safety": {"score": 5, "reasoning": "Safe"},
                        "coherence": {"score": 3, "reasoning": "Ok"},
                        "tool_use_correctness": None,
                        "pedagogical_quality": {"score": 3, "reasoning": "Decent pedagogy"},
                    }
                ),
            },
            {
                "locale": "en_US",
                "probe_type": "tool_calling",
                "persona_age_bin": "65+",
                "persona_education_level": "High school",
                "persona_occupation": "Retired",
                "persona_region": "Texas",
                "conversation_status": False,
                "eval_scores": None,
            },
        ]
    )


class TestExtractEvalScores:
    def test_extracts_scores_from_valid_rows(self, sample_generated_data):
        scores_df = extract_eval_scores(sample_generated_data)
        assert len(scores_df) == 2

    def test_score_columns_present(self, sample_generated_data):
        scores_df = extract_eval_scores(sample_generated_data)
        assert "score_helpfulness" in scores_df.columns
        assert "score_accuracy" in scores_df.columns
        assert "score_safety" in scores_df.columns

    def test_none_axes_produce_none_values(self, sample_generated_data):
        scores_df = extract_eval_scores(sample_generated_data)
        row0 = scores_df.iloc[0]
        assert row0["score_tool_use_correctness"] is None or pd.isna(row0["score_tool_use_correctness"])

    def test_preserves_demographics(self, sample_generated_data):
        scores_df = extract_eval_scores(sample_generated_data)
        assert "locale" in scores_df.columns
        assert "probe_type" in scores_df.columns
        assert "persona_age_bin" in scores_df.columns
        assert "persona_occupation" in scores_df.columns
        # ``persona_country`` replaced ``persona_region`` in the
        # tag set (different locales use different subdivision
        # concepts; country is universal). Backward-compat fallback
        # in ``analysis.extract_eval_scores`` reads from
        # ``persona_region`` when ``persona_country`` is missing,
        # so older-format fixtures (like this one) still surface a value.
        assert "persona_country" in scores_df.columns
        # Older-format fixture has ``persona_region`` populated; the
        # fallback should pull the value forward into the new
        # ``persona_country`` slot.
        assert "California" in scores_df["persona_country"].tolist()

    def test_can_compute_mean_scores(self, sample_generated_data):
        scores_df = extract_eval_scores(sample_generated_data)
        score_cols = [c for c in scores_df.columns if c.startswith("score_")]
        means = scores_df[score_cols].mean()
        assert means["score_helpfulness"] == 3.5
        assert means["score_safety"] == 5.0

    def test_can_group_by_locale(self, sample_generated_data):
        scores_df = extract_eval_scores(sample_generated_data)
        grouped = scores_df.groupby("locale")["score_helpfulness"].mean()
        assert "en_US" in grouped.index
        assert "pt_BR" in grouped.index

    def test_empty_dataframe(self):
        empty = pd.DataFrame(columns=["eval_scores"])
        scores_df = extract_eval_scores(empty)
        assert len(scores_df) == 0

    def test_malformed_json_scores(self):
        df = pd.DataFrame([{"eval_scores": "not valid json"}])
        scores_df = extract_eval_scores(df)
        assert len(scores_df) == 0

    def test_scores_as_dict_not_string(self, sample_generated_data):
        df = sample_generated_data.copy()
        df.at[0, "eval_scores"] = {
            "helpfulness": {"score": 4, "reasoning": "Good"},
            "accuracy": {"score": 3, "reasoning": "Ok"},
        }
        scores_df = extract_eval_scores(df)
        assert len(scores_df) == 2
