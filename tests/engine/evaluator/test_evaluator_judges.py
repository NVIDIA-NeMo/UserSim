# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Tests for evaluator/judges.py — JudgeFamily registry + diversity validator."""

from __future__ import annotations

import pytest

from usersim.engine.evaluator.judges import (
    EnsembleDiversityError,
    JudgeFamily,
    JudgeSpec,
    infer_judge_family,
    validate_ensemble_diversity,
)


class TestInferJudgeFamily:
    @pytest.mark.parametrize(
        "model_id,expected",
        [
            ("openai/gpt-oss-120b", JudgeFamily.OPENAI),
            ("OpenAI/GPT-5", JudgeFamily.OPENAI),
            ("o3-mini", JudgeFamily.OPENAI),
            ("o4-mini", JudgeFamily.OPENAI),
            ("gpt-4-turbo", JudgeFamily.OPENAI),
            ("nvidia/nemotron-3-super-120b", JudgeFamily.NVIDIA_NEMOTRON),
            ("Nemotron-Mini", JudgeFamily.NVIDIA_NEMOTRON),
            ("anthropic/claude-3.5-sonnet", JudgeFamily.ANTHROPIC),
            ("Claude Opus 4.6", JudgeFamily.ANTHROPIC),
            ("deepseek-r1", JudgeFamily.DEEPSEEK),
            ("alibaba/qwen3-72b", JudgeFamily.QWEN),
            ("meta-llama/Llama-3.3-70B", JudgeFamily.LLAMA),
            ("mistralai/mixtral-8x22b", JudgeFamily.MISTRAL),
            ("google/gemini-2.5-pro", JudgeFamily.GEMINI),
        ],
    )
    def test_known_models(self, model_id: str, expected: JudgeFamily) -> None:
        assert infer_judge_family(model_id) == expected

    def test_unknown_falls_back_to_other(self) -> None:
        assert infer_judge_family("some-random/model-v0") == JudgeFamily.OTHER


class TestJudgeSpec:
    def test_explicit_family_takes_precedence(self) -> None:
        spec = JudgeSpec(alias="anything", family=JudgeFamily.QWEN)
        assert spec.resolved_family("openai/gpt-5") == JudgeFamily.QWEN

    def test_falls_back_to_inference(self) -> None:
        spec = JudgeSpec(alias="judge_a")
        # alias gives no hint
        assert spec.resolved_family() == JudgeFamily.OTHER
        # but resolved model id does
        assert spec.resolved_family("nvidia/nemotron-3-super") == JudgeFamily.NVIDIA_NEMOTRON


class TestValidateEnsembleDiversity:
    def test_single_judge_is_allowed(self) -> None:
        # Single-judge mode is explicitly allowed (opts out of κ).
        validate_ensemble_diversity([JudgeSpec(alias="solo", family=JudgeFamily.OPENAI)])

    def test_two_distinct_families_pass(self) -> None:
        validate_ensemble_diversity(
            [
                JudgeSpec(alias="a", family=JudgeFamily.OPENAI),
                JudgeSpec(alias="b", family=JudgeFamily.NVIDIA_NEMOTRON),
            ]
        )

    def test_same_family_rejected(self) -> None:
        with pytest.raises(EnsembleDiversityError) as exc:
            validate_ensemble_diversity(
                [
                    JudgeSpec(alias="a", family=JudgeFamily.OPENAI),
                    JudgeSpec(alias="b", family=JudgeFamily.OPENAI),
                ]
            )
        assert "distinct families" in str(exc.value).lower()

    def test_all_other_rejected_with_helpful_message(self) -> None:
        # Two judges that both resolve to OTHER -> tell the user to set
        # family explicitly.
        with pytest.raises(EnsembleDiversityError) as exc:
            validate_ensemble_diversity(
                [
                    JudgeSpec(alias="mystery_a"),
                    JudgeSpec(alias="mystery_b"),
                ]
            )
        assert "set JudgeSpec.family explicitly" in str(exc.value)

    def test_resolved_model_ids_disambiguate(self) -> None:
        # Aliases give no hint; resolved model ids do.
        validate_ensemble_diversity(
            [JudgeSpec(alias="judge_a"), JudgeSpec(alias="judge_b")],
            resolved_model_ids={
                "judge_a": "openai/gpt-oss-120b",
                "judge_b": "nvidia/nemotron-3-super-120b",
            },
        )

    def test_three_judges_two_families_fails_when_min_three(self) -> None:
        # User can dial up the constraint.
        with pytest.raises(EnsembleDiversityError):
            validate_ensemble_diversity(
                [
                    JudgeSpec(alias="a", family=JudgeFamily.OPENAI),
                    JudgeSpec(alias="b", family=JudgeFamily.OPENAI),
                    JudgeSpec(alias="c", family=JudgeFamily.NVIDIA_NEMOTRON),
                ],
                min_distinct_families=3,
            )
