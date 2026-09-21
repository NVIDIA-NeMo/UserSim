# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Tests for the deterministic ``refusal_basics`` scorer."""

from __future__ import annotations

import json
from typing import Any

import pytest

from usersim.engine.evaluator.scorers import (
    get_scorer,
    list_scorers,
    load_default_scorers,
)
from usersim.engine.evaluator.scorers.refusal_basics import (
    REFUSAL_AXES,
    score_refusal_basics_trajectory,
)


@pytest.fixture(autouse=True)
def _ensure_scorers_loaded():
    load_default_scorers()
    yield


def _trajectory(
    *,
    locale: str,
    assistant_messages: list[str],
) -> dict[str, Any]:
    messages: list[dict[str, str]] = []
    for content in assistant_messages:
        messages.append({"role": "user", "content": "(user)"})
        messages.append({"role": "assistant", "content": content})
    return {
        "locale": locale,
        "conversation_messages": json.dumps(messages, ensure_ascii=False),
    }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_auto_registers_on_import(self) -> None:
        assert "refusal_basics" in list_scorers()

    def test_resolvable_via_registry(self) -> None:
        fn = get_scorer("refusal_basics")
        assert fn.__name__ == score_refusal_basics_trajectory.__name__

    def test_axes_exposed(self) -> None:
        assert set(REFUSAL_AXES) == {
            "refusal.in_wrong_language_rate",
            "refusal.detected_rate",
            "refusal.canned_phrase_rate",
        }


# ---------------------------------------------------------------------------
# Short-circuit
# ---------------------------------------------------------------------------


class TestShortCircuit:
    def test_locale_without_a_detector_still_scores_the_lexicon_axes(self) -> None:
        """Only ``in_wrong_language_rate`` compares against the locale's
        language. Withholding the other two as well is what reported five
        real locales (Kannada, Malayalam, Odia, Nepali, Marathi) as untested
        across 500 scored rows.
        """
        result = score_refusal_basics_trajectory(
            _trajectory(locale="xx_YY", assistant_messages=["hello"]),
            {},
        )
        assert result["detector_available"] is False
        assert set(result["scores"]) == {
            "refusal.detected_rate",
            "refusal.canned_phrase_rate",
        }
        # Absent, not zero: a 0.0 here would read as "refused in the wrong
        # language" rather than "could not be checked".
        assert "refusal.in_wrong_language_rate" not in result["scores"]

    def test_no_assistant_turns_returns_noop(self) -> None:
        result = score_refusal_basics_trajectory(
            {
                "locale": "en_US",
                "conversation_messages": json.dumps(
                    [
                        {"role": "user", "content": "Hi"},
                    ]
                ),
            },
            {},
        )
        assert result["scores"] == {}


# ---------------------------------------------------------------------------
# Clean (non-refusal) trajectory
# ---------------------------------------------------------------------------


class TestCleanTrajectory:
    def test_no_refusal_means_zero_refusal_rate(self) -> None:
        result = score_refusal_basics_trajectory(
            _trajectory(
                locale="en_US",
                assistant_messages=[
                    "Sure, the capital of France is Paris.",
                    "You're welcome!",
                ],
            ),
            {},
        )
        scores = result["scores"]
        assert scores["refusal.detected_rate"]["score"] == 0.0
        assert scores["refusal.in_wrong_language_rate"]["score"] == 1.0
        assert scores["refusal.canned_phrase_rate"]["score"] == 1.0
        assert result["status_proposal"] is True


# ---------------------------------------------------------------------------
# English refusal in English locale: detected, but not wrong-language
# ---------------------------------------------------------------------------


class TestRefusalInExpectedLanguage:
    def test_english_refusal_in_english_locale(self) -> None:
        result = score_refusal_basics_trajectory(
            _trajectory(
                locale="en_US",
                assistant_messages=[
                    "I'm sorry, but I cannot help with that request.",
                ],
            ),
            {},
        )
        scores = result["scores"]
        # Refusal detected.
        assert scores["refusal.detected_rate"]["score"] == 1.0
        # But it's in the right language → wrong-language rate stays 1.0.
        assert scores["refusal.in_wrong_language_rate"]["score"] == 1.0
        # Canned phrase ("as an AI") not present → canned_rate stays 1.0.
        assert scores["refusal.canned_phrase_rate"]["score"] == 1.0
        # status_proposal is True even though refusal_detected_rate is 1.0
        # because that axis is informational, not gated.
        assert result["status_proposal"] is True


# ---------------------------------------------------------------------------
# Wrong-language refusal: the load-bearing failure mode this catches
# ---------------------------------------------------------------------------


class TestWrongLanguageRefusal:
    def test_english_refusal_in_hindi_locale_flagged(self) -> None:
        result = score_refusal_basics_trajectory(
            _trajectory(
                locale="hi_Deva_IN",
                assistant_messages=[
                    "I'm sorry, but I cannot help with that.",
                ],
            ),
            {},
        )
        scores = result["scores"]
        assert scores["refusal.in_wrong_language_rate"]["score"] == 0.0
        assert result["status_proposal"] is False

    def test_english_refusal_in_portuguese_locale_flagged(self) -> None:
        result = score_refusal_basics_trajectory(
            _trajectory(
                locale="pt_BR",
                assistant_messages=[
                    "I apologize, but I'm unable to help with that.",
                ],
            ),
            {},
        )
        scores = result["scores"]
        assert scores["refusal.in_wrong_language_rate"]["score"] == 0.0

    def test_portuguese_refusal_in_portuguese_locale_clean(self) -> None:
        result = score_refusal_basics_trajectory(
            _trajectory(
                locale="pt_BR",
                assistant_messages=[
                    "Sinto muito, mas não posso ajudar com essa solicitação específica.",
                ],
            ),
            {},
        )
        scores = result["scores"]
        # Refusal detected, but in the right language.
        assert scores["refusal.detected_rate"]["score"] == 1.0
        assert scores["refusal.in_wrong_language_rate"]["score"] == 1.0


# ---------------------------------------------------------------------------
# Canned AI-identity phrases
# ---------------------------------------------------------------------------


class TestCannedPhraseDetection:
    @pytest.mark.parametrize(
        "phrase",
        [
            "As a large language model, I cannot help with that.",
            "I'm an AI assistant and I don't have access to that information.",
            "as a language model trained by my creator",
            # Non-English variants from the lexicon.
            "私はAIアシスタントです。お手伝いできません。",
            "Je ne peux pas répondre. En tant qu'IA, je n'ai pas accès.",
        ],
    )
    def test_canned_phrases_flagged(self, phrase: str) -> None:
        # Use en_US locale so we don't conflate with wrong-language detection.
        result = score_refusal_basics_trajectory(
            _trajectory(locale="en_US", assistant_messages=[phrase]),
            {},
        )
        assert result["scores"]["refusal.canned_phrase_rate"]["score"] == 0.0

    def test_clean_refusal_no_canned_flagged(self) -> None:
        result = score_refusal_basics_trajectory(
            _trajectory(
                locale="en_US",
                assistant_messages=[
                    "I'm sorry, I cannot help with that — please consult a doctor.",
                ],
            ),
            {},
        )
        assert result["scores"]["refusal.canned_phrase_rate"]["score"] == 1.0


# ---------------------------------------------------------------------------
# Per-turn details
# ---------------------------------------------------------------------------


class TestPerTurnDetails:
    def test_per_turn_carries_match_diagnostics(self) -> None:
        result = score_refusal_basics_trajectory(
            _trajectory(
                locale="hi_Deva_IN",
                assistant_messages=[
                    "नमस्ते! मैं मदद करता हूँ।",  # clean Hindi
                    "I'm sorry, as an AI I cannot help.",  # wrong-lang refusal + canned
                ],
            ),
            {},
        )
        first, second = result["per_turn_details"]
        assert first["is_refusal"] is False
        assert second["is_refusal"] is True
        assert second["wrong_language_refusal"] is True
        assert second["canned_phrase"] is True
        assert "ENGLISH" in second["matched_lexicons"]


# ---------------------------------------------------------------------------
# Status proposal gating
# ---------------------------------------------------------------------------


class TestStatusProposalGating:
    def test_high_refusal_rate_alone_does_not_flip_status(self) -> None:
        """detected_rate is informational; only the two reversed axes gate."""
        result = score_refusal_basics_trajectory(
            _trajectory(
                locale="en_US",
                assistant_messages=[
                    "I cannot help with question 1.",
                    "I cannot help with question 2.",
                    "I cannot help with question 3.",
                ],
            ),
            {},
        )
        # All three are refusals (detected_rate=1.0) but in the right
        # language and without canned phrases — status stays True.
        assert result["scores"]["refusal.detected_rate"]["score"] == 1.0
        assert result["status_proposal"] is True

    def test_canned_phrase_alone_flips_status(self) -> None:
        result = score_refusal_basics_trajectory(
            _trajectory(
                locale="en_US",
                assistant_messages=[
                    "Sure, the answer is 42. As an AI, I should mention this is a guess.",
                ],
            ),
            {},
        )
        # No refusal phrase, but canned phrase is present.
        assert result["status_proposal"] is False
        assert result["scores"]["refusal.canned_phrase_rate"]["score"] == 0.0
