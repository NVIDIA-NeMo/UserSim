# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the deterministic ``language_compliance`` scorer.

No live LLM calls anywhere — the scorer doesn't use one. Lingua's
detector is shared process-wide; tests amortize the cold-start cost.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from usersim.engine.evaluator.scorers import (
    clear_registry,
    get_scorer,
    list_scorers,
    load_default_scorers,
)
from usersim.engine.evaluator.scorers.language_compliance import (
    LANGUAGE_AXES,
    score_language_compliance_trajectory,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _ensure_scorers_loaded():
    load_default_scorers()
    yield


def _trajectory(
    *,
    locale: str,
    assistant_messages: List[str],
) -> Dict[str, Any]:
    """Build a synthetic trajectory with given locale + assistant turns."""
    messages: List[Dict[str, str]] = []
    for content in assistant_messages:
        messages.append({"role": "user", "content": "(synthetic user turn)"})
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
        assert "language_compliance" in list_scorers()

    def test_resolvable_via_registry(self) -> None:
        fn = get_scorer("language_compliance")
        assert fn.__name__ == score_language_compliance_trajectory.__name__
        assert fn.__module__ == score_language_compliance_trajectory.__module__

    def test_reload_picks_up_registration(self) -> None:
        clear_registry()
        load_default_scorers()
        assert "language_compliance" in list_scorers()


# ---------------------------------------------------------------------------
# Axis definitions
# ---------------------------------------------------------------------------


class TestAxes:
    def test_four_axes_exposed(self) -> None:
        assert set(LANGUAGE_AXES) == {
            "language.requested_language_match_rate",
            "language.script_compliance_rate",
            "language.first_turn_match",
            "language.script_integrity_rate",
        }

    def test_namespace_prefixed(self) -> None:
        for axis in LANGUAGE_AXES:
            assert axis.startswith("language.")


# ---------------------------------------------------------------------------
# Short-circuit / no-op envelope
# ---------------------------------------------------------------------------


class TestDetectorlessLocales:
    """Five supported locales have no lingua model -- Kannada, Malayalam, Odia
    and Nepali are absent from lingua, and Marathi is excluded deliberately
    because it collides with Hindi. They still declare a native script, so the
    two script axes are computable. Skipping them wholesale reported 500 scored
    rows as untested.
    """

    # 100+ chars of Kannada so the turn clears the script-integrity length floor.
    _KANNADA = "\u0C97\u0CCD\u0CB0\u0CBE\u0CB9\u0C95\u0CB0 \u0C96\u0CBE\u0CA4\u0CC6\u0CAF \u0CB5\u0CBF\u0CB5\u0CB0 \u0CAA\u0CA1\u0CC6\u0CAF\u0CB2\u0CC1 \u0CA8\u0CBE\u0CB5\u0CC1 \u0CAA\u0CB0\u0CBF\u0CB6\u0CC0\u0CB2\u0CBF\u0CB8\u0CBF \u0CA8\u0CBF\u0CAE\u0C97\u0CC6 " * 4

    def test_scores_the_script_axes_without_a_detector(self) -> None:
        result = score_language_compliance_trajectory(
            _trajectory(locale="kn_Knda_IN", assistant_messages=[self._KANNADA]),
            {},
        )
        assert result["detector_available"] is False
        assert set(result["scores"]) == {
            "language.script_compliance_rate",
            "language.script_integrity_rate",
        }

    def test_language_identity_axes_are_absent_not_zero(self) -> None:
        """A 0.0 would read as "answered in the wrong language"; absent is
        skipped by the capability aggregation, which is the truth."""
        result = score_language_compliance_trajectory(
            _trajectory(locale="ml_Mlym_IN", assistant_messages=[self._KANNADA]),
            {},
        )
        assert "language.requested_language_match_rate" not in result["scores"]
        assert "language.first_turn_match" not in result["scores"]

    def test_intrusion_is_caught_without_a_detector(self) -> None:
        result = score_language_compliance_trajectory(
            _trajectory(
                locale="kn_Knda_IN",
                assistant_messages=[self._KANNADA + " \uAC80 \u0444\u043E\u0440\u043C\u0430"],
            ),
            {},
        )
        cell = result["scores"]["language.script_integrity_rate"]
        assert cell["score"] == 0.0
        assert "cyrillic" in cell["reasoning"] and "hangul" in cell["reasoning"]

    def test_romanized_locale_is_still_skipped(self) -> None:
        """Its expected script IS Latin, so both script axes come back a
        meaningless 1.0 for an answer that is plain English -- scoring it would
        manufacture a pass. ``hi_Latn_IN`` in particular ships as a first-class
        locale and is absent from ``INDIA_VARIANT_LOCALES``, so a romanized
        FLAG lookup misses it; the guard tests the script ranges instead.
        """
        for locale in ("hi_Latn_IN", "te_Latn_IN"):
            result = score_language_compliance_trajectory(
                _trajectory(
                    locale=locale,
                    assistant_messages=["Your account is closed. " * 10],
                ),
                {},
            )
            assert result["scores"] == {}, locale
            assert "expected script is Latin" in result["error"], locale


class TestScriptIntegrity:
    def test_legitimate_latin_identifiers_do_not_count(self) -> None:
        """Marathi answers carry UPI / KYC / PAN / NEFT / tool names because
        that is how Indians write. Script COMPLIANCE penalizes them (it floors
        near 0.83 on clean Devanagari); integrity must not, or the axis is
        unusable on the locales it exists for.
        """
        text = (
            "\u0924\u0941\u092E\u0939\u093E\u0932\u093E UPI \u0906\u0923\u093F KYC \u092A\u0942\u0930\u094D\u0923 \u0915\u0930\u093E\u0935\u0947 \u0932\u093E\u0917\u0947\u0932. PAN \u0915\u094D\u0930\u092E\u093E\u0902\u0915 \u0926\u094D\u092F\u093E. "
            "NEFT \u0915\u093F\u0902\u0935\u093E IMPS \u0935\u093E\u092A\u0930\u0942\u0928 \u092A\u0948\u0938\u0947 \u092A\u093E\u0920\u0935\u093E. verify_identity \u0915\u0930\u0942. "
        ) * 2
        result = score_language_compliance_trajectory(
            _trajectory(locale="mr_Deva_IN", assistant_messages=[text]), {},
        )
        integrity = result["scores"]["language.script_integrity_rate"]["score"]
        compliance = result["scores"]["language.script_compliance_rate"]["score"]
        assert integrity == 1.0
        # The contrast is the whole point of having both axes.
        assert compliance < 0.95

    def test_short_turns_are_excluded_from_the_rate(self) -> None:
        """A stray glyph is the entire signal, so on a 2-character turn one bad
        codepoint would sink the turn on noise."""
        result = score_language_compliance_trajectory(
            _trajectory(locale="kn_Knda_IN", assistant_messages=["\uAC80"]), {},
        )
        cell = result["scores"]["language.script_integrity_rate"]
        assert cell["score"] is None
        assert cell["n"] == 0

    def test_no_scorable_turn_does_not_fail_status(self) -> None:
        """``None`` is absence of evidence, not a failure."""
        result = score_language_compliance_trajectory(
            _trajectory(locale="en_US", assistant_messages=["ok"]), {},
        )
        assert result["scores"]["language.script_integrity_rate"]["score"] is None
        assert result["status_proposal"] is True


class TestShortCircuit:
    def test_unknown_locale_returns_noop(self) -> None:
        result = score_language_compliance_trajectory(
            _trajectory(locale="xx_YY", assistant_messages=["hello"]),
            {},
        )
        assert result["scores"] == {}
        assert "not registered" in result["error"]
        assert result["status_proposal"] is True

    def test_missing_locale_returns_noop(self) -> None:
        result = score_language_compliance_trajectory(
            {
                "conversation_messages": json.dumps([
                    {"role": "assistant", "content": "hi"},
                ]),
            },
            {},
        )
        assert result["scores"] == {}
        assert "not registered" in result["error"]

    def test_no_assistant_turns_returns_noop(self) -> None:
        result = score_language_compliance_trajectory(
            {
                "locale": "pt_BR",
                "conversation_messages": json.dumps([
                    {"role": "user", "content": "Olá!"},
                ]),
            },
            {},
        )
        assert result["scores"] == {}
        assert "no assistant turns" in result["error"]


# ---------------------------------------------------------------------------
# Happy path: clean Hindi conversation
# ---------------------------------------------------------------------------


class TestCleanHindiTrajectory:
    def test_perfect_hindi_response_scores_one(self) -> None:
        result = score_language_compliance_trajectory(
            _trajectory(
                locale="hi_Deva_IN",
                assistant_messages=[
                    "नमस्ते! आधार एक 12-अंकों की पहचान संख्या है।",
                ],
            ),
            {},
        )
        assert result["scores"]["language.requested_language_match_rate"]["score"] == 1.0
        assert result["scores"]["language.script_compliance_rate"]["score"] == 1.0
        assert result["scores"]["language.first_turn_match"]["score"] == 1.0
        assert result["status_proposal"] is True


# ---------------------------------------------------------------------------
# Failure: model dropped into English mid-conversation
# ---------------------------------------------------------------------------


class TestEnglishFallbackInHindiTrajectory:
    def test_turn_2_in_english_drops_match_rate(self) -> None:
        result = score_language_compliance_trajectory(
            _trajectory(
                locale="hi_Deva_IN",
                assistant_messages=[
                    "नमस्ते! आधार के बारे में बताता हूँ।",  # OK
                    "I'm sorry, but I cannot provide more details.",  # English bleed
                ],
            ),
            {},
        )
        scores = result["scores"]
        assert scores["language.requested_language_match_rate"]["score"] == 0.5
        assert scores["language.script_compliance_rate"]["score"] < 1.0
        # First turn was correct → first_turn_match should still be 1.
        assert scores["language.first_turn_match"]["score"] == 1.0
        # Status flips to False (any rate < 1.0 is a fail).
        assert result["status_proposal"] is False

    def test_turn_1_in_english_drops_first_turn_match(self) -> None:
        result = score_language_compliance_trajectory(
            _trajectory(
                locale="hi_Deva_IN",
                assistant_messages=[
                    "Sure, I can help you with that.",
                ],
            ),
            {},
        )
        scores = result["scores"]
        assert scores["language.requested_language_match_rate"]["score"] == 0.0
        assert scores["language.first_turn_match"]["score"] == 0.0


# ---------------------------------------------------------------------------
# Script-only failure: Hindi-in-Latin-transliteration
# ---------------------------------------------------------------------------


class TestHindiInLatinTransliteration:
    def test_namaste_in_latin_fails_script_check(self) -> None:
        # Lingua may detect this as English — that's the whole point:
        # the script check is the load-bearing one for this case.
        result = score_language_compliance_trajectory(
            _trajectory(
                locale="hi_Deva_IN",
                assistant_messages=[
                    "namaste, main aapki kaise madad kar sakta hu?",
                ],
            ),
            {},
        )
        scores = result["scores"]
        # Script compliance should be ~0 (Latin chars in a Devanagari locale).
        assert scores["language.script_compliance_rate"]["score"] == 0.0
        assert result["status_proposal"] is False


# ---------------------------------------------------------------------------
# Japanese mixes three scripts naturally → all three count as expected
# ---------------------------------------------------------------------------


class TestJapaneseMixedScripts:
    def test_japanese_with_hiragana_katakana_han_all_compliant(self) -> None:
        result = score_language_compliance_trajectory(
            _trajectory(
                locale="ja_JP",
                assistant_messages=[
                    "今日はコンピュータを使ってメールを書きます。",
                ],
            ),
            {},
        )
        scores = result["scores"]
        assert scores["language.requested_language_match_rate"]["score"] == 1.0
        # All three scripts are co-expected for ja_JP — total compliance == 1.
        assert scores["language.script_compliance_rate"]["score"] == 1.0


# ---------------------------------------------------------------------------
# Per-turn details
# ---------------------------------------------------------------------------


class TestPerTurnDetails:
    def test_per_turn_carries_detected_and_compliance(self) -> None:
        result = score_language_compliance_trajectory(
            _trajectory(
                locale="pt_BR",
                assistant_messages=[
                    "Olá! Como posso ajudar?",
                    "Sure, I can help with that.",
                ],
            ),
            {},
        )
        assert len(result["per_turn_details"]) == 2
        first, second = result["per_turn_details"]
        assert first["detected_language"] == "PORTUGUESE"
        assert first["language_match"] is True
        assert second["detected_language"] == "ENGLISH"
        assert second["language_match"] is False

    def test_script_dominance_present_when_compliance_failed(self) -> None:
        result = score_language_compliance_trajectory(
            _trajectory(
                locale="hi_Deva_IN",
                assistant_messages=["hello world"],
            ),
            {},
        )
        per_turn = result["per_turn_details"][0]
        # Script compliance was 0 → dominance dict surfaces what was used.
        assert per_turn["script_dominance"] is not None
        assert per_turn["script_dominance"]["latin"] == 1.0

    def test_script_dominance_omitted_when_compliance_perfect(self) -> None:
        result = score_language_compliance_trajectory(
            _trajectory(
                locale="hi_Deva_IN",
                assistant_messages=["नमस्ते"],
            ),
            {},
        )
        per_turn = result["per_turn_details"][0]
        # No need to surface dominance — compliance was perfect.
        assert per_turn["script_dominance"] is None
