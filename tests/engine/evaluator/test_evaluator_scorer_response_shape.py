# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the deterministic ``response_shape`` scorer.

Pattern-and-heuristic checks; no LLM, no lingua. Tests cover the
classification helpers (``_is_empty_or_trivial``, ``_is_over_formatted``,
``_looks_truncated``) and the trajectory-level aggregation.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from usersim.engine.evaluator.scorers import (
    get_scorer,
    list_scorers,
    load_default_scorers,
)
from usersim.engine.evaluator.scorers.response_shape import (
    SHAPE_AXES,
    _is_empty_or_trivial,
    _is_over_formatted,
    _looks_truncated,
    score_response_shape_trajectory,
)


@pytest.fixture(autouse=True)
def _ensure_scorers_loaded():
    load_default_scorers()
    yield


def _trajectory(assistant_messages: List[str]) -> Dict[str, Any]:
    messages: List[Dict[str, str]] = []
    for content in assistant_messages:
        messages.append({"role": "user", "content": "(user)"})
        messages.append({"role": "assistant", "content": content})
    return {
        "locale": "en_US",
        "conversation_messages": json.dumps(messages, ensure_ascii=False),
    }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_auto_registers_on_import(self) -> None:
        assert "response_shape" in list_scorers()

    def test_resolvable_via_registry(self) -> None:
        fn = get_scorer("response_shape")
        assert fn.__name__ == score_response_shape_trajectory.__name__

    def test_axes_exposed(self) -> None:
        assert set(SHAPE_AXES) == {
            "shape.empty_or_trivial_rate",
            "shape.over_formatted_rate",
            "shape.truncation_suspicion_rate",
        }


# ---------------------------------------------------------------------------
# Per-turn classifiers
# ---------------------------------------------------------------------------


class TestEmptyOrTrivial:
    @pytest.mark.parametrize(
        "text", ["", " ", "   \n  ", "ok", "...", "hi", "👍", "✓"],
    )
    def test_trivial_strings_flagged(self, text: str) -> None:
        assert _is_empty_or_trivial(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "Hello, how can I help you today?",
            "Sure thing.",
            "नमस्ते, मैं आपकी मदद करूँगा।",
            "Bien sûr, voici la réponse.",
        ],
    )
    def test_substantive_strings_not_flagged(self, text: str) -> None:
        assert _is_empty_or_trivial(text) is False


class TestOverFormatted:
    def test_two_headers_flagged(self) -> None:
        text = "## Section A\nbody\n## Section B\nbody"
        assert _is_over_formatted(text) is True

    def test_one_header_not_flagged(self) -> None:
        text = "## Title\nA short paragraph follows here."
        assert _is_over_formatted(text) is False

    def test_horizontal_rule_flagged(self) -> None:
        text = "Some text\n\n---\n\nMore text"
        assert _is_over_formatted(text) is True

    def test_pipe_table_flagged(self) -> None:
        text = "| col1 | col2 |\n| a | b |\n| c | d |"
        assert _is_over_formatted(text) is True

    def test_long_numbered_list_flagged(self) -> None:
        text = "\n".join(f"{i+1}. item {i+1}" for i in range(7))
        assert _is_over_formatted(text) is True

    def test_short_numbered_list_not_flagged(self) -> None:
        text = "1. one\n2. two\n3. three"
        assert _is_over_formatted(text) is False

    def test_long_bulleted_list_flagged(self) -> None:
        text = "\n".join(f"- bullet {i+1}" for i in range(7))
        assert _is_over_formatted(text) is True

    def test_long_code_block_flagged(self) -> None:
        text = "```python\n" + "\n".join(f"line {i}" for i in range(8)) + "\n```"
        assert _is_over_formatted(text) is True

    def test_short_code_block_not_flagged(self) -> None:
        text = "```python\nprint('hi')\n```"
        assert _is_over_formatted(text) is False

    def test_plain_paragraph_not_flagged(self) -> None:
        text = (
            "Sure, here's a quick answer. The Empire State Building is in "
            "New York City and was completed in 1931. Let me know if you'd "
            "like more detail."
        )
        assert _is_over_formatted(text) is False


class TestLooksTruncated:
    @pytest.mark.parametrize(
        "text",
        [
            "This sentence is complete.",
            "Of course!",
            "बहुत खुशी हुई।",  # Devanagari danda terminator
            "了解しました。",  # Japanese full-stop
            "Voici la réponse?",
            "(complete with parens)",
            '"clean closing quote"',
        ],
    )
    def test_clean_terminators_not_flagged(self, text: str) -> None:
        assert _looks_truncated(text) is False

    @pytest.mark.parametrize(
        "text",
        [
            "This sentence ends abru",
            "And then the cat",
            "(unclosed paren",
            "[unclosed bracket",
        ],
    )
    def test_truncated_strings_flagged(self, text: str) -> None:
        assert _looks_truncated(text) is True


# ---------------------------------------------------------------------------
# Trajectory-level aggregation
# ---------------------------------------------------------------------------


class TestTrajectoryScoring:
    def test_clean_trajectory_all_perfect(self) -> None:
        result = score_response_shape_trajectory(
            _trajectory([
                "Sure, here's the answer. The capital of France is Paris.",
                "You're welcome!",
            ]),
            {},
        )
        for axis in SHAPE_AXES:
            assert result["scores"][axis]["score"] == 1.0
        assert result["status_proposal"] is True

    def test_one_trivial_turn_drops_rate(self) -> None:
        result = score_response_shape_trajectory(
            _trajectory(["Sure, here's a real answer for you.", "ok"]),
            {},
        )
        assert result["scores"]["shape.empty_or_trivial_rate"]["score"] == 0.5
        assert result["status_proposal"] is False

    def test_one_over_formatted_turn_drops_rate(self) -> None:
        long_list = "\n".join(f"{i+1}. item {i+1}" for i in range(8))
        result = score_response_shape_trajectory(
            _trajectory([
                "A short clean reply.",
                long_list,
            ]),
            {},
        )
        assert result["scores"]["shape.over_formatted_rate"]["score"] == 0.5
        assert result["status_proposal"] is False

    def test_one_truncated_turn_drops_rate(self) -> None:
        result = score_response_shape_trajectory(
            _trajectory([
                "Clean response.",
                "And then the user said abru",
            ]),
            {},
        )
        assert result["scores"]["shape.truncation_suspicion_rate"]["score"] == 0.5
        assert result["status_proposal"] is False

    def test_no_assistant_turns_short_circuits(self) -> None:
        result = score_response_shape_trajectory(
            {
                "locale": "en_US",
                "conversation_messages": json.dumps([
                    {"role": "user", "content": "Hi"},
                ]),
            },
            {},
        )
        assert result["scores"] == {}
        assert "no assistant turns" in result["error"]
        # Vacuously True — no evidence to fail on.
        assert result["status_proposal"] is True

    def test_trivial_turn_not_double_counted_as_truncated(self) -> None:
        # "ok" is trivial but the truncation check should not also fire
        # (would be a misleading double-count). The implementation
        # short-circuits truncation/over-formatted on trivial turns.
        result = score_response_shape_trajectory(
            _trajectory(["ok", "Substantive reply here."]),
            {},
        )
        per_turn = result["per_turn_details"]
        assert per_turn[0]["empty_or_trivial"] is True
        assert per_turn[0]["truncation_suspicion"] is False
        assert per_turn[0]["over_formatted"] is False
