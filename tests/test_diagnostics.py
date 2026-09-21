# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Tests for reporting/diagnostics.py — D1–D4 + MATTR."""

from __future__ import annotations

import json

import pytest

from usersim.reporting import (
    compute_diagnostics_frame,
    compute_front_loading_ratio,
    compute_frustration_marker_fraction,
    compute_mattr,
    compute_polite_turn_fraction,
    compute_verbosity_cv,
)
from usersim.reporting.diagnostics import assistant_tokens_for_mattr


def _msgs(*pairs):
    out = []
    for role, content in pairs:
        out.append({"role": role, "content": content})
    return out


class TestFrontLoadingRatio:
    def test_balanced_turns_near_one(self) -> None:
        # All user turns ~5 words.
        m = _msgs(
            ("user", "one two three four five"),
            ("assistant", "ok"),
            ("user", "six seven eight nine ten"),
            ("assistant", "ok"),
            ("user", "eleven twelve thirteen fourteen fifteen"),
        )
        assert compute_front_loading_ratio(m) == pytest.approx(1.0)

    def test_front_loaded_returns_high(self) -> None:
        m = _msgs(
            ("user", " ".join(["w"] * 50)),
            ("assistant", "ok"),
            ("user", "two words"),
            ("assistant", "ok"),
            ("user", "three little words"),
        )
        ratio = compute_front_loading_ratio(m)
        assert ratio is not None and ratio > 10

    def test_short_first_returns_low(self) -> None:
        m = _msgs(
            ("user", "hi"),
            ("assistant", "ok"),
            ("user", " ".join(["w"] * 20)),
            ("assistant", "ok"),
            ("user", " ".join(["w"] * 20)),
        )
        ratio = compute_front_loading_ratio(m)
        assert ratio is not None and ratio < 1.0

    def test_single_user_turn_returns_none(self) -> None:
        m = _msgs(("user", "hi"), ("assistant", "hello"))
        assert compute_front_loading_ratio(m) is None

    def test_accepts_json_string(self) -> None:
        m = _msgs(
            ("user", "five words here are five"),
            ("assistant", "ok"),
            ("user", "five more words count this"),
        )
        assert compute_front_loading_ratio(json.dumps(m)) == pytest.approx(1.0)

    def test_handles_empty_or_garbage(self) -> None:
        assert compute_front_loading_ratio(None) is None
        assert compute_front_loading_ratio("not json") is None
        assert compute_front_loading_ratio([]) is None


class TestPoliteTurnFraction:
    def test_no_polite_markers(self) -> None:
        m = _msgs(
            ("user", "Where is the file"),
            ("assistant", "Here it is"),
            ("user", "I want a different one"),
        )
        assert compute_polite_turn_fraction(m) == 0.0

    def test_all_polite(self) -> None:
        m = _msgs(
            ("user", "Thanks for that"),
            ("assistant", "Sure"),
            ("user", "I appreciate it, please continue"),
        )
        assert compute_polite_turn_fraction(m) == 1.0

    def test_partial(self) -> None:
        m = _msgs(
            ("user", "Just give me a number"),
            ("assistant", "10"),
            ("user", "Thanks"),
        )
        assert compute_polite_turn_fraction(m) == pytest.approx(0.5)

    def test_no_user_turns_returns_none(self) -> None:
        m = _msgs(("assistant", "hello"))
        assert compute_polite_turn_fraction(m) is None


class TestVerbosityCV:
    def test_uniform_lengths_low_cv(self) -> None:
        m = _msgs(
            ("user", "one two three"),
            ("assistant", "ok"),
            ("user", "one two three"),
            ("assistant", "ok"),
            ("user", "one two three"),
        )
        assert compute_verbosity_cv(m) == 0.0

    def test_varied_lengths_high_cv(self) -> None:
        m = _msgs(
            ("user", "hi"),
            ("assistant", "ok"),
            ("user", " ".join(["w"] * 50)),
            ("assistant", "ok"),
            ("user", " ".join(["w"] * 100)),
        )
        cv = compute_verbosity_cv(m)
        assert cv is not None and cv > 0.5

    def test_single_user_turn_returns_none(self) -> None:
        m = _msgs(("user", "hi"))
        assert compute_verbosity_cv(m) is None


class TestFrustrationMarkers:
    def test_zero(self) -> None:
        m = _msgs(("user", "Hi there"), ("assistant", "Hello"))
        assert compute_frustration_marker_fraction(m) == 0.0

    def test_hits(self) -> None:
        m = _msgs(
            ("user", "I already told you what I need"),
            ("assistant", "Sorry"),
            ("user", "This is taking forever, terrible"),
        )
        assert compute_frustration_marker_fraction(m) == 1.0


class TestMATTR:
    def test_empty_returns_zero(self) -> None:
        assert compute_mattr([]) == 0.0

    def test_short_falls_back_to_global_ttr(self) -> None:
        # 3 unique tokens out of 4 -> TTR 0.75.
        assert compute_mattr(["a", "b", "c", "a"], window_size=50) == pytest.approx(0.75)

    def test_full_diversity_window(self) -> None:
        # Every window of 5 has 5 unique tokens.
        toks = list("abcdefghij")
        assert compute_mattr(toks, window_size=5) == 1.0

    def test_low_diversity_window(self) -> None:
        # Every window of 5 has 1 unique token.
        toks = ["a"] * 10
        assert compute_mattr(toks, window_size=5) == pytest.approx(0.2)

    def test_assistant_tokens_extracted(self) -> None:
        m = _msgs(
            ("user", "ignore me"),
            ("assistant", "Hello World HELLO"),
            ("user", "ignore me too"),
            ("assistant", "another response"),
        )
        toks = assistant_tokens_for_mattr(m)
        # Lowercased + whitespace split, only assistant turns.
        assert toks == ["hello", "world", "hello", "another", "response"]


class TestDiagnosticsFrame:
    def test_adds_columns(self, trajectory_df) -> None:
        out = compute_diagnostics_frame(trajectory_df)
        for col in [
            "d1_front_loading",
            "d2_polite_fraction",
            "d3_verbosity_cv",
            "d4_frustration_markers",
            "mattr_assistant",
        ]:
            assert col in out.columns
        # Confrontational row (idx 2) carries frustration markers.
        assert out.iloc[2]["d4_frustration_markers"] is not None
        assert out.iloc[2]["d4_frustration_markers"] > 0.0

    def test_requires_messages_column(self) -> None:
        pd = pytest.importorskip("pandas")
        with pytest.raises(ValueError, match="conversation_messages"):
            compute_diagnostics_frame(pd.DataFrame({"x": [1, 2]}))


# ---------------------------------------------------------------------------
# compute_response_length_by_locale
# ---------------------------------------------------------------------------


def _length_row(locale, assistant_chars, total_tokens=None, n_calls=None):
    """Build a synthetic trajectory row for length-by-locale tests.

    Each entry in ``assistant_chars`` becomes one assistant turn whose
    content is ``"x" * n``. ``total_tokens`` + ``n_calls`` populate the
    simulation_outcome's ``per_model_*["assistant_model"]`` so the
    tokens side has data to aggregate.
    """
    msgs = []
    for n in assistant_chars:
        msgs.append({"role": "user", "content": "u"})
        msgs.append({"role": "assistant", "content": "x" * n})
    outcome = {"per_model_calls": {}, "per_model_output_tokens": {}}
    if total_tokens is not None:
        outcome["per_model_output_tokens"]["assistant_model"] = total_tokens
    if n_calls is not None:
        outcome["per_model_calls"]["assistant_model"] = n_calls
    return {
        "locale": locale,
        "conversation_messages": json.dumps(msgs),
        "simulation_outcome": json.dumps(outcome),
    }


class TestResponseLengthByLocale:
    def _build_df(self, rows):
        pd = pytest.importorskip("pandas")
        return pd.DataFrame(rows)

    def test_single_locale_aggregates_correctly(self) -> None:
        from usersim.reporting import compute_response_length_by_locale

        df = self._build_df(
            [
                _length_row("en_US", [100, 200], total_tokens=80, n_calls=2),
                _length_row("en_US", [150], total_tokens=60, n_calls=1),
            ]
        )
        result = compute_response_length_by_locale(df)
        assert len(result) == 1
        row = result[0]
        assert row["locale"] == "en_US"
        assert row["n_trajectories"] == 2
        # Char distribution: 100, 200, 150 → mean 150, median 150.
        assert row["mean_chars_per_turn"] == 150.0
        assert row["median_chars_per_turn"] == 150.0
        # Token distribution: 80/2 = 40, 60/1 = 60 → mean 50.
        assert row["mean_tokens_per_turn"] == 50.0
        # Total tokens per trajectory: mean of 80 and 60 → 70.
        assert row["mean_total_tokens_per_trajectory"] == 70.0

    def test_two_locales_appear_separately(self) -> None:
        from usersim.reporting import compute_response_length_by_locale

        df = self._build_df(
            [
                _length_row("en_US", [200, 200], total_tokens=200, n_calls=2),
                _length_row("hi_Deva_IN", [50, 50], total_tokens=50, n_calls=2),
            ]
        )
        result = compute_response_length_by_locale(df)
        assert len(result) == 2
        # Result is sorted by locale name.
        en, hi = result
        assert en["locale"] == "en_US"
        assert hi["locale"] == "hi_Deva_IN"
        # Equity gap surfaces as a 4x difference in chars per turn.
        assert en["mean_chars_per_turn"] == 200.0
        assert hi["mean_chars_per_turn"] == 50.0
        assert en["mean_chars_per_turn"] / hi["mean_chars_per_turn"] == 4.0

    def test_missing_token_data_doesnt_fail(self) -> None:
        from usersim.reporting import compute_response_length_by_locale

        # No total_tokens / n_calls — only chars.
        df = self._build_df(
            [
                _length_row("en_US", [100, 150]),
            ]
        )
        result = compute_response_length_by_locale(df)
        row = result[0]
        # Char side computed.
        assert row["mean_chars_per_turn"] == 125.0
        # Token side gracefully None.
        assert row["mean_tokens_per_turn"] is None
        assert row["mean_total_tokens_per_trajectory"] is None

    def test_missing_locale_skipped(self) -> None:
        from usersim.reporting import compute_response_length_by_locale

        df = self._build_df(
            [
                _length_row("en_US", [100], total_tokens=50, n_calls=1),
                {
                    # Row with no locale.
                    "locale": None,
                    "conversation_messages": json.dumps(
                        [
                            {"role": "assistant", "content": "x"},
                        ]
                    ),
                    "simulation_outcome": json.dumps({}),
                },
            ]
        )
        result = compute_response_length_by_locale(df)
        assert len(result) == 1
        assert result[0]["locale"] == "en_US"
        # The row without locale doesn't contribute.
        assert result[0]["n_trajectories"] == 1

    def test_empty_df_returns_empty_list(self) -> None:
        from usersim.reporting import compute_response_length_by_locale

        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(columns=["locale", "conversation_messages", "simulation_outcome"])
        assert compute_response_length_by_locale(df) == []

    def test_missing_required_column_raises(self) -> None:
        from usersim.reporting import compute_response_length_by_locale

        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"locale": ["en_US"]})  # missing the other two cols
        with pytest.raises(ValueError, match="conversation_messages"):
            compute_response_length_by_locale(df)

    def test_p95_uses_pandas_quantile(self) -> None:
        from usersim.reporting import compute_response_length_by_locale

        df = self._build_df(
            [
                _length_row("en_US", [100, 200, 300, 400, 500], total_tokens=500, n_calls=5),
            ]
        )
        result = compute_response_length_by_locale(df)
        row = result[0]
        # P95 of [100, 200, 300, 400, 500] via pandas linear interp = 480.
        assert row["p95_chars_per_turn"] == 480.0
