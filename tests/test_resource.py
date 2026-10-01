# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for reporting/resource.py — per-bundle aggregation."""

from __future__ import annotations

import json

import pytest

from usersim.reporting import ResourceProfile, aggregate_resource_profile
from usersim.reporting.resource import aggregate_from_dataframe


def _outcome(*, calls=None, in_tokens=None, out_tokens=None, wall_by_alias=None, wall=10.0):
    return json.dumps(
        {
            "per_model_calls": calls or {},
            "per_model_input_tokens": in_tokens or {},
            "per_model_output_tokens": out_tokens or {},
            "wall_clock_s_by_alias": wall_by_alias or {},
            "wall_clock_s": wall,
        }
    )


class TestAggregateResourceProfile:
    def test_empty(self) -> None:
        p = aggregate_resource_profile([])
        assert p.n_trajectories == 0
        assert p.total_calls == 0
        assert p.total_tokens == 0
        assert p.total_wall_clock_s == 0.0

    def test_skips_unparseable_and_none(self) -> None:
        # All three decode to an empty dict, which the aggregator
        # treats as "no outcome to count" — n_trajectories stays 0.
        p = aggregate_resource_profile([None, "not json", "{}"])
        assert p.n_trajectories == 0
        assert p.total_calls == 0

    def test_sums_across_rows_and_aliases(self) -> None:
        rows = [
            _outcome(
                calls={"user_model": 3, "assistant_model": 3},
                in_tokens={"user_model": 100, "assistant_model": 80},
                out_tokens={"user_model": 50, "assistant_model": 90},
                wall_by_alias={"user_model": 4.0, "assistant_model": 6.0},
                wall=12.0,
            ),
            _outcome(
                calls={"user_model": 2, "judge_model": 5},
                in_tokens={"user_model": 50, "judge_model": 200},
                out_tokens={"user_model": 25, "judge_model": 30},
                wall_by_alias={"user_model": 2.0, "judge_model": 1.5},
                wall=8.0,
            ),
        ]
        p = aggregate_resource_profile(rows)
        assert p.n_trajectories == 2
        assert p.n_calls_by_alias == {"user_model": 5, "assistant_model": 3, "judge_model": 5}
        assert p.input_tokens_by_alias == {"user_model": 150, "assistant_model": 80, "judge_model": 200}
        assert p.output_tokens_by_alias == {"user_model": 75, "assistant_model": 90, "judge_model": 30}
        assert p.wall_clock_s_by_alias["user_model"] == pytest.approx(6.0)
        assert p.total_wall_clock_s == pytest.approx(20.0)
        assert p.total_calls == 13
        assert p.total_input_tokens == 430
        assert p.total_output_tokens == 195
        assert p.total_tokens == 625

    def test_to_dict_round_trip(self) -> None:
        p = aggregate_resource_profile(
            [
                _outcome(calls={"a": 1}, in_tokens={"a": 10}, out_tokens={"a": 5}),
            ]
        )
        d = p.to_dict()
        assert d["total_calls"] == 1
        assert d["n_calls_by_alias"] == {"a": 1}
        assert d["total_tokens"] == 15
        # `conversation_output_tokens` ships in the dict so the dashboard
        # JS layer can read it directly off the JSON manifest.
        assert "conversation_output_tokens" in d


class TestConversationOutputTokens:
    """``conversation_output_tokens`` narrows ``total_output_tokens`` to
    just the two parties IN the conversation -- ``user_model`` +
    ``assistant_model`` output. The dashboard's hero token figure.

    Excludes ``api_response_model`` (tool-response synthesiser; output
    is consumed by the assistant's next turn as input rather than as a
    conversation participant) and the in-sim scaffolding aliases
    (judge / summary)."""

    def test_sums_only_user_and_assistant_aliases(self) -> None:
        p = aggregate_resource_profile(
            [
                _outcome(
                    out_tokens={
                        "user_model": 100,
                        "assistant_model": 200,
                        "api_response_model": 50,  # excluded -- tool synthesiser
                        "judge_model": 999,  # excluded -- in-sim scaffolding
                        "summary_model": 999,  # excluded -- in-sim scaffolding
                        "evaluator_model": 9999,  # excluded (out-of-sim, shouldn't be here anyway)
                    }
                ),
            ]
        )
        # 100 + 200 = 300; api_response / judge / summary / evaluator absent.
        assert p.conversation_output_tokens == 300

    def test_excludes_api_response_model(self) -> None:
        """Regression lock: api_response_model output must NEVER count
        toward conversation tokens. The synthesiser stands up a tool
        backend so tool_calling probes have something to call -- its
        output is consumed by the assistant as INPUT to the assistant's
        next turn, not as a turn the synthesiser itself takes in the
        conversation. Counting it would conflate tool-call cost with
        conversation cost."""
        p = aggregate_resource_profile(
            [
                _outcome(
                    out_tokens={
                        "user_model": 50,
                        "assistant_model": 75,
                        "api_response_model": 1000,  # large, should NOT inflate conv
                    }
                ),
            ]
        )
        assert p.conversation_output_tokens == 125  # NOT 1125
        # `total_output_tokens` does include api_response (it's still
        # in-sim spend, just not conversation spend).
        assert p.total_output_tokens == 1125

    def test_zero_when_no_conversation_aliases_present(self) -> None:
        p = aggregate_resource_profile(
            [
                _outcome(out_tokens={"judge_model": 200, "summary_model": 100}),
            ]
        )
        assert p.conversation_output_tokens == 0
        # `total_output_tokens` still surfaces the broader figure.
        assert p.total_output_tokens == 300

    def test_handles_missing_aliases_individually(self) -> None:
        # Trajectories that don't fire api_response (non-tool_calling probes)
        # still report a non-zero conversation_output_tokens from
        # user_model + assistant_model.
        p = aggregate_resource_profile(
            [
                _outcome(out_tokens={"user_model": 50, "assistant_model": 75}),
            ]
        )
        assert p.conversation_output_tokens == 125

    def test_empty_profile_is_zero(self) -> None:
        assert ResourceProfile().conversation_output_tokens == 0


class TestAssistantOutputTokens:
    """``assistant_output_tokens`` reports just the model under test's
    output tokens -- a subset of conversation_output_tokens, surfaced
    as its own card so reviewers can compare the model's verbosity
    contribution to the rest of the conversation."""

    def test_returns_only_assistant_model_output(self) -> None:
        p = aggregate_resource_profile(
            [
                _outcome(
                    out_tokens={
                        "assistant_model": 200,
                        "user_model": 100,
                        "api_response_model": 50,
                        "judge_model": 999,
                    }
                ),
            ]
        )
        assert p.assistant_output_tokens == 200
        # And it's <= conversation_output_tokens (which sums user + assistant).
        assert p.assistant_output_tokens <= p.conversation_output_tokens

    def test_zero_when_no_assistant_alias_present(self) -> None:
        p = aggregate_resource_profile(
            [
                _outcome(out_tokens={"user_model": 50, "judge_model": 75}),
            ]
        )
        assert p.assistant_output_tokens == 0

    def test_serialized_in_to_dict(self) -> None:
        p = aggregate_resource_profile(
            [
                _outcome(out_tokens={"assistant_model": 12}),
            ]
        )
        d = p.to_dict()
        assert d["assistant_output_tokens"] == 12

    def test_empty_profile_is_zero(self) -> None:
        assert ResourceProfile().assistant_output_tokens == 0


class TestAssistantTotalTokens:
    """``assistant_total_tokens`` is the gross billable spend on the
    model under test alone (input + output, including any reasoning
    since reasoning rolls into output on the bill). Different shape
    from ``conversation_output_tokens`` -- single alias, both sides."""

    def test_sums_input_plus_output_for_assistant_alias(self) -> None:
        p = aggregate_resource_profile(
            [
                _outcome(
                    in_tokens={"assistant_model": 200, "user_model": 100},
                    out_tokens={"assistant_model": 300, "user_model": 50},
                ),
            ]
        )
        assert p.assistant_total_tokens == 500

    def test_zero_when_no_assistant_alias_present(self) -> None:
        p = aggregate_resource_profile(
            [
                _outcome(
                    in_tokens={"user_model": 100},
                    out_tokens={"user_model": 50},
                ),
            ]
        )
        assert p.assistant_total_tokens == 0

    def test_serialised_in_to_dict(self) -> None:
        p = aggregate_resource_profile(
            [
                _outcome(
                    in_tokens={"assistant_model": 50},
                    out_tokens={"assistant_model": 75},
                ),
            ]
        )
        d = p.to_dict()
        assert d["assistant_total_tokens"] == 125

    def test_empty_profile_is_zero(self) -> None:
        assert ResourceProfile().assistant_total_tokens == 0


class TestAssistantReasoningTokens:
    """``assistant_reasoning_tokens`` mirrors the global
    ``total_reasoning_tokens`` but scoped to the model under test, so
    reviewers can answer 'how much thinking did the assistant alone do?'
    without subtracting user_model / api_response_model contributions."""

    def test_pulls_assistant_alias_from_reasoning_dict(self) -> None:
        """Property semantics: ``assistant_reasoning_tokens`` reads
        ``reasoning_tokens_by_alias["assistant_model"]`` (typically
        populated by Phase B in ``aggregate_from_dataframe``)."""
        p = ResourceProfile(
            reasoning_tokens_by_alias={"assistant_model": 60, "user_model": 40},
            output_tokens_by_alias={"assistant_model": 200, "user_model": 80},
        )
        # Global sum is 100 (assistant + user), assistant-only is 60.
        assert p.total_reasoning_tokens == 100
        assert p.assistant_reasoning_tokens == 60

    def test_zero_when_no_assistant_capture(self) -> None:
        p = ResourceProfile(
            reasoning_tokens_by_alias={"user_model": 40},
        )
        assert p.assistant_reasoning_tokens == 0
        # And it's strictly <= the global total (subset relationship).
        assert p.assistant_reasoning_tokens <= p.total_reasoning_tokens

    def test_serialised_in_to_dict(self) -> None:
        p = ResourceProfile(
            reasoning_tokens_by_alias={"assistant_model": 30},
        )
        d = p.to_dict()
        assert d["assistant_reasoning_tokens"] == 30

    def test_empty_profile_is_zero(self) -> None:
        assert ResourceProfile().assistant_reasoning_tokens == 0


class TestAssistantConversationTokens:
    """``assistant_conversation_tokens`` is the visible-output complement
    of ``assistant_reasoning_tokens`` -- ``assistant_output -
    assistant_reasoning``. The dashboard's Assistant Tokens row uses it
    to show "what was actually said by the assistant" distinct from
    "thinking content"."""

    def test_output_minus_reasoning(self) -> None:
        p = ResourceProfile(
            output_tokens_by_alias={"assistant_model": 200},
            reasoning_tokens_by_alias={"assistant_model": 80},
        )
        assert p.assistant_conversation_tokens == 120  # 200 - 80

    def test_clamped_to_non_negative(self) -> None:
        """Defensive: if a future estimator bug pushes reasoning above
        output, the visible-output complement must not go negative."""
        p = ResourceProfile(
            output_tokens_by_alias={"assistant_model": 100},
            reasoning_tokens_by_alias={"assistant_model": 250},
        )
        assert p.assistant_conversation_tokens == 0

    def test_serialised_in_to_dict(self) -> None:
        p = ResourceProfile(
            output_tokens_by_alias={"assistant_model": 100},
            reasoning_tokens_by_alias={"assistant_model": 30},
        )
        assert p.to_dict()["assistant_conversation_tokens"] == 70


class TestUserTokens:
    """User-side decomposition: ``user_total`` (input + output),
    ``user_output``, ``user_reasoning`` (subset), ``user_conversation``
    (visible). Same algebra as the Assistant Tokens row, scoped to
    user_model alone."""

    def _user_outcome(self, *, in_tokens=0, out_tokens=0):
        return json.dumps(
            {
                "per_model_calls": {"user_model": 1},
                "per_model_input_tokens": {"user_model": in_tokens},
                "per_model_output_tokens": {"user_model": out_tokens},
                "wall_clock_s_by_alias": {},
                "wall_clock_s": 1.0,
            }
        )

    def test_total_is_input_plus_output(self) -> None:
        p = aggregate_resource_profile(
            [
                self._user_outcome(in_tokens=300, out_tokens=120),
            ]
        )
        assert p.user_total_tokens == 420
        assert p.input_tokens_by_alias.get("user_model", 0) == 300

    def test_output_isolates_user_alias(self) -> None:
        # user_model output must NOT include other aliases' output.
        outcome = json.dumps(
            {
                "per_model_calls": {"user_model": 1, "assistant_model": 1},
                "per_model_input_tokens": {"user_model": 50, "assistant_model": 80},
                "per_model_output_tokens": {"user_model": 30, "assistant_model": 200},
                "wall_clock_s_by_alias": {},
                "wall_clock_s": 1.0,
            }
        )
        p = aggregate_resource_profile([outcome])
        assert p.user_output_tokens == 30  # not 230

    def test_reasoning_isolates_user_alias(self) -> None:
        # Property test: user_reasoning_tokens reads only the user_model
        # entry of reasoning_tokens_by_alias.
        p = ResourceProfile(
            output_tokens_by_alias={"user_model": 30, "assistant_model": 200},
            reasoning_tokens_by_alias={"user_model": 5, "assistant_model": 60},
        )
        assert p.user_reasoning_tokens == 5  # not 65

    def test_conversation_is_output_minus_reasoning(self) -> None:
        p = ResourceProfile(
            output_tokens_by_alias={"user_model": 50},
            reasoning_tokens_by_alias={"user_model": 15},
        )
        assert p.user_conversation_tokens == 35  # 50 - 15

    def test_conversation_clamped_to_non_negative(self) -> None:
        p = ResourceProfile(
            output_tokens_by_alias={"user_model": 20},
            reasoning_tokens_by_alias={"user_model": 50},
        )
        assert p.user_conversation_tokens == 0

    def test_serialised_in_to_dict(self) -> None:
        p = ResourceProfile(
            input_tokens_by_alias={"user_model": 100},
            output_tokens_by_alias={"user_model": 80},
            reasoning_tokens_by_alias={"user_model": 12},
        )
        d = p.to_dict()
        assert d["user_total_tokens"] == 180
        assert d["user_output_tokens"] == 80
        assert d["user_reasoning_tokens"] == 12
        assert d["user_conversation_tokens"] == 68

    def test_empty_profile_is_zero(self) -> None:
        p = ResourceProfile()
        assert p.user_total_tokens == 0
        assert p.user_output_tokens == 0
        assert p.user_reasoning_tokens == 0
        assert p.user_conversation_tokens == 0


class TestSupportActorTokens:
    """Per-alias decompositions for api_response_model / judge_model /
    summary_model -- the support actors that round out the dashboard's
    Sim Health rows alongside User and Assistant. Same algebra:

        total = input + output
        output = reasoning + conversation     (visible)

    judge_model and summary_model don't preserve their output verbatim
    on the trajectory, so Phase B (tiktoken) reasoning estimation is
    unavailable -- their reasoning is typically 0 unless Phase A
    captured a value, and ``conversation`` collapses to ``output`` in
    that case. api_response_model has visible content (tool-response
    role) so Phase B does work for it.
    """

    @staticmethod
    def _outcome(out_tokens=None, in_tokens=None):
        return json.dumps(
            {
                "per_model_calls": {a: 1 for a in (out_tokens or {})},
                "per_model_input_tokens": in_tokens or {},
                "per_model_output_tokens": out_tokens or {},
                "wall_clock_s_by_alias": {},
                "wall_clock_s": 1.0,
            }
        )

    def test_api_response_decomposes_total_output_reasoning_conversation(self) -> None:
        # Property-level test: a profile with per-alias output and
        # reasoning entries decomposes correctly. (Phase B estimation
        # against a real conversation is exercised in TestReasoningTokens.)
        p = ResourceProfile(
            input_tokens_by_alias={"api_response_model": 200},
            output_tokens_by_alias={"api_response_model": 500},
            reasoning_tokens_by_alias={"api_response_model": 75},
        )
        assert p.api_response_total_tokens == 700  # 200 + 500
        assert p.api_response_output_tokens == 500
        assert p.api_response_reasoning_tokens == 75
        assert p.api_response_conversation_tokens == 425  # 500 - 75
        # Algebra invariant.
        assert p.api_response_output_tokens == (p.api_response_reasoning_tokens + p.api_response_conversation_tokens)

    def test_judge_decomposes_with_unavailable_reasoning(self) -> None:
        # Typical run: judge_model has no Phase B fallback (no visible
        # content on the trajectory) -> reasoning=0, conversation
        # collapses to output.
        p = aggregate_resource_profile(
            [
                self._outcome(
                    in_tokens={"judge_model": 1000},
                    out_tokens={"judge_model": 100},
                ),
            ]
        )
        assert p.judge_total_tokens == 1100
        assert p.judge_output_tokens == 100
        assert p.judge_reasoning_tokens == 0
        assert p.judge_conversation_tokens == 100  # equals output

    def test_summary_decomposes_with_unavailable_reasoning(self) -> None:
        p = aggregate_resource_profile(
            [
                self._outcome(
                    in_tokens={"summary_model": 5000},
                    out_tokens={"summary_model": 200},
                ),
            ]
        )
        assert p.summary_total_tokens == 5200
        assert p.summary_output_tokens == 200
        assert p.summary_reasoning_tokens == 0
        assert p.summary_conversation_tokens == 200

    def test_per_alias_decompositions_are_isolated(self) -> None:
        """Mixed profile -- each alias's decomposition reads only its
        own per_model_* entries."""
        p = ResourceProfile(
            input_tokens_by_alias={
                "api_response_model": 100,
                "judge_model": 200,
                "summary_model": 300,
                "assistant_model": 50,
            },
            output_tokens_by_alias={
                "api_response_model": 40,
                "judge_model": 30,
                "summary_model": 20,
                "assistant_model": 10,
            },
            reasoning_tokens_by_alias={
                "api_response_model": 5,
                "judge_model": 3,
                "summary_model": 2,
                "assistant_model": 1,
            },
        )
        # Each alias's "total" and "output" must NOT bleed across.
        assert p.api_response_total_tokens == 140
        assert p.judge_total_tokens == 230
        assert p.summary_total_tokens == 320
        assert p.api_response_output_tokens == 40
        assert p.judge_output_tokens == 30
        assert p.summary_output_tokens == 20

    def test_clamped_when_reasoning_exceeds_output(self) -> None:
        """Defensive: a malformed estimate that exceeds output for a
        support alias must clamp ``conversation`` to 0, not negative."""
        p = ResourceProfile(
            output_tokens_by_alias={"api_response_model": 100},
            reasoning_tokens_by_alias={"api_response_model": 200},
        )
        assert p.api_response_conversation_tokens == 0

    def test_serialised_in_to_dict(self) -> None:
        p = aggregate_resource_profile(
            [
                self._outcome(
                    in_tokens={"api_response_model": 50, "judge_model": 80, "summary_model": 30},
                    out_tokens={"api_response_model": 25, "judge_model": 12, "summary_model": 8},
                ),
            ]
        )
        d = p.to_dict()
        for alias_prefix, total, output in [
            ("api_response", 75, 25),
            ("judge", 92, 12),
            ("summary", 38, 8),
        ]:
            assert d[f"{alias_prefix}_total_tokens"] == total
            assert d[f"{alias_prefix}_output_tokens"] == output
            assert d[f"{alias_prefix}_reasoning_tokens"] == 0
            assert d[f"{alias_prefix}_conversation_tokens"] == output

    def test_empty_profile_is_zero_for_all_support_aliases(self) -> None:
        p = ResourceProfile()
        for alias_prefix in ("api_response", "judge", "summary"):
            assert getattr(p, f"{alias_prefix}_total_tokens") == 0
            assert getattr(p, f"{alias_prefix}_output_tokens") == 0
            assert getattr(p, f"{alias_prefix}_reasoning_tokens") == 0
            assert getattr(p, f"{alias_prefix}_conversation_tokens") == 0


class TestThreeRowAlgebraInvariant:
    """The dashboard's three-row Sim Health layout pivots on a single
    algebraic identity per row. These tests lock the identities so a
    future change to one of the underlying properties can't silently
    break the dashboard's claimed math."""

    def test_total_equals_input_plus_output_for_overall(self) -> None:
        p = aggregate_resource_profile(
            [
                json.dumps(
                    {
                        "per_model_calls": {"assistant_model": 1, "judge_model": 1},
                        "per_model_input_tokens": {"assistant_model": 100, "judge_model": 200},
                        "per_model_output_tokens": {"assistant_model": 50, "judge_model": 30},
                        "wall_clock_s_by_alias": {},
                        "wall_clock_s": 1.0,
                    }
                ),
            ]
        )
        assert p.total_tokens == p.total_input_tokens + p.total_output_tokens

    def test_user_total_equals_user_input_plus_user_output(self) -> None:
        p = aggregate_resource_profile(
            [
                json.dumps(
                    {
                        "per_model_calls": {"user_model": 1},
                        "per_model_input_tokens": {"user_model": 200},
                        "per_model_output_tokens": {"user_model": 80},
                        "wall_clock_s_by_alias": {},
                        "wall_clock_s": 1.0,
                    }
                ),
            ]
        )
        user_input = p.input_tokens_by_alias.get("user_model", 0)
        assert p.user_total_tokens == user_input + p.user_output_tokens

    def test_user_output_equals_user_reasoning_plus_user_conversation(self) -> None:
        # Property-level invariant: even when a profile is constructed
        # directly with per-alias output and reasoning entries (as the
        # Phase B aggregator builds it from estimates), the conversation
        # column equals output minus reasoning.
        p = ResourceProfile(
            output_tokens_by_alias={"user_model": 100},
            reasoning_tokens_by_alias={"user_model": 30},
        )
        assert p.user_output_tokens == (p.user_reasoning_tokens + p.user_conversation_tokens)

    def test_assistant_total_equals_assistant_input_plus_assistant_output(self) -> None:
        p = aggregate_resource_profile(
            [
                json.dumps(
                    {
                        "per_model_calls": {"assistant_model": 1},
                        "per_model_input_tokens": {"assistant_model": 400},
                        "per_model_output_tokens": {"assistant_model": 200},
                        "wall_clock_s_by_alias": {},
                        "wall_clock_s": 1.0,
                    }
                ),
            ]
        )
        assistant_input = p.input_tokens_by_alias.get("assistant_model", 0)
        assert p.assistant_total_tokens == (assistant_input + p.assistant_output_tokens)

    def test_assistant_output_equals_assistant_reasoning_plus_assistant_conversation(self) -> None:
        p = ResourceProfile(
            input_tokens_by_alias={"assistant_model": 100},
            output_tokens_by_alias={"assistant_model": 500},
            reasoning_tokens_by_alias={"assistant_model": 200},
        )
        assert p.assistant_output_tokens == (p.assistant_reasoning_tokens + p.assistant_conversation_tokens)


class TestReasoningTokens:
    """Reasoning-token estimation: post-hoc tiktoken pass at report build,
    with a 5% per-alias noise floor to swallow tokenizer-mismatch drift.

    DataDesigner's normalised ``Usage`` strips provider-specific reasoning
    telemetry, so there's no Phase A capture path -- everything goes
    through Phase B (``output_tokens - tiktoken(visible_content)``)
    followed by the noise-floor clamp.
    """

    @staticmethod
    def _outcome(*, out_tokens):
        return json.dumps(
            {
                "per_model_calls": {a: 1 for a in out_tokens},
                "per_model_input_tokens": {a: 10 for a in out_tokens},
                "per_model_output_tokens": out_tokens,
                "wall_clock_s_by_alias": {},
                "wall_clock_s": 1.0,
            }
        )

    def test_estimates_above_noise_floor_register(self) -> None:
        """When the diff between provider's output_tokens and tiktoken's
        count of visible content is well above the 5% noise floor (true
        reasoning), the alias surfaces with source='estimated' and a
        non-zero token count."""
        import pandas as pd

        df = pd.DataFrame(
            [
                {
                    "simulation_outcome": self._outcome(
                        out_tokens={"assistant_model": 1000},  # gross billable
                    ),
                    "conversation_messages": json.dumps(
                        [
                            {"role": "user", "content": "hi"},
                            # ~5 visible tokens; provider charged 1000 -> ~995
                            # reasoning -> 99.5% of output, clears the 5% floor.
                            {"role": "assistant", "content": "hello there"},
                        ]
                    ),
                }
            ]
        )
        p = aggregate_from_dataframe(df)
        assert p.reasoning_tokens_by_alias["assistant_model"] > 0
        assert p.reasoning_source_by_alias["assistant_model"] == "estimated"

    def test_below_noise_floor_zeros_out(self) -> None:
        """When the aggregate reasoning estimate is below 5% of the
        alias's output, the diff is dominated by tokenizer-mismatch
        drift (cl100k_base vs the provider's tokenizer), not real
        reasoning. Zero out and mark unavailable -- this is the
        gemma-instruct case."""
        import pandas as pd

        # Output 1000, visible content tokenizes to ~975 -> diff 25 ->
        # 25/1000 = 2.5% < 5% floor.
        df = pd.DataFrame(
            [
                {
                    "simulation_outcome": self._outcome(
                        out_tokens={"assistant_model": 1000},
                    ),
                    "conversation_messages": json.dumps(
                        [
                            {
                                "role": "assistant",
                                # Long content so tiktoken count approaches 1000.
                                "content": (
                                    "This is a fairly long assistant response that should "
                                    "tokenize to roughly the same number of tokens as the "
                                    "provider's output count, leaving only a small diff that "
                                    "falls below the noise floor. " * 30
                                ),
                            }
                        ]
                    ),
                }
            ]
        )
        p = aggregate_from_dataframe(df)
        # Below floor: zeroed out, marked unavailable.
        assert p.reasoning_tokens_by_alias.get("assistant_model", 0) == 0
        assert p.reasoning_source_by_alias["assistant_model"] == "unavailable"

    def test_negative_diff_clamps_to_zero(self) -> None:
        """Per-row negative diff (tiktoken over-counts vs provider)
        clamps to 0 silently and accumulates as 'estimated'. The
        post-aggregation noise floor decides whether the alias's total
        is real."""
        import pandas as pd

        df = pd.DataFrame(
            [
                {
                    "simulation_outcome": self._outcome(
                        out_tokens={"assistant_model": 5},  # tiny
                    ),
                    "conversation_messages": json.dumps(
                        [
                            {
                                "role": "assistant",
                                "content": "this is much longer than five tokens by far",
                            }
                        ]
                    ),
                }
            ]
        )
        p = aggregate_from_dataframe(df)
        # Single negative-diff row: total reasoning is 0, falls below
        # the 5% floor, gets flagged unavailable.
        assert p.reasoning_tokens_by_alias.get("assistant_model", 0) == 0
        assert p.reasoning_source_by_alias["assistant_model"] == "unavailable"

    def test_judge_summary_always_unavailable(self) -> None:
        """judge_model / summary_model never have visible output to
        tokenize. They're flagged unavailable regardless of input."""
        import pandas as pd

        df = pd.DataFrame(
            [
                {
                    "simulation_outcome": self._outcome(
                        out_tokens={"judge_model": 50, "summary_model": 30},
                    ),
                    "conversation_messages": json.dumps([]),
                }
            ]
        )
        p = aggregate_from_dataframe(df)
        assert p.reasoning_source_by_alias["judge_model"] == "unavailable"
        assert p.reasoning_source_by_alias["summary_model"] == "unavailable"
        assert p.reasoning_tokens_by_alias.get("judge_model", 0) == 0
        assert p.reasoning_source_by_alias.get("summary_model") == "unavailable"

    def test_missing_conversation_messages_column_short_circuits(self) -> None:
        """Synthetic frames without a conversation_messages column (e.g.
        the smoke-test fixture) must not trip the aggregator. Without
        visible content to tokenize, the finalisation pass flips every
        alias with output > 0 to ``unavailable`` (the noise floor sees
        reasoning=0 against output>0 and clamps)."""
        import pandas as pd

        df = pd.DataFrame(
            [
                {
                    "simulation_outcome": self._outcome(
                        out_tokens={"assistant_model": 100},
                    ),
                    # no conversation_messages column
                }
            ]
        )
        p = aggregate_from_dataframe(df)
        # All aliases land on unavailable -- assistant_model because
        # 0 reasoning / 100 output = 0% which is below the floor;
        # judge / summary because they're always unavailable.
        assert p.reasoning_source_by_alias["assistant_model"] == "unavailable"
        assert p.reasoning_source_by_alias["judge_model"] == "unavailable"
        assert p.reasoning_source_by_alias["summary_model"] == "unavailable"
        assert p.reasoning_tokens_by_alias.get("assistant_model", 0) == 0

    def test_tool_calls_counted_toward_visible(self) -> None:
        """Correctness lock: assistant ``tool_calls`` JSON counts toward
        visible content (provider charges it via completion_tokens).
        Omitting it would over-attribute to 'reasoning' for tool_calling
        probes -- a real gap caught during plan review."""
        import pandas as pd

        msg_no_tools = json.dumps(
            [
                {"role": "assistant", "content": "hi"},
            ]
        )
        msg_with_tools = json.dumps(
            [
                {
                    "role": "assistant",
                    "content": "hi",
                    "tool_calls": [
                        {
                            "id": "c_1",
                            "type": "function",
                            "function": {"name": "lookup_weather", "arguments": '{"city": "Tokyo", "units": "metric"}'},
                        }
                    ],
                },
            ]
        )
        df_no = pd.DataFrame(
            [
                {
                    "simulation_outcome": self._outcome(out_tokens={"assistant_model": 100}),
                    "conversation_messages": msg_no_tools,
                }
            ]
        )
        df_with = pd.DataFrame(
            [
                {
                    "simulation_outcome": self._outcome(out_tokens={"assistant_model": 100}),
                    "conversation_messages": msg_with_tools,
                }
            ]
        )
        p_no = aggregate_from_dataframe(df_no)
        p_with = aggregate_from_dataframe(df_with)
        # Same gross output_tokens (100), but tool_calls add visible
        # content -> reasoning estimate drops.
        no_tools_reason = p_no.reasoning_tokens_by_alias.get("assistant_model", 0)
        with_tools_reason = p_with.reasoning_tokens_by_alias.get("assistant_model", 0)
        assert with_tools_reason < no_tools_reason

    def test_legacy_per_model_reasoning_tokens_field_is_ignored(self) -> None:
        """Backwards compat: parquets from the abandoned dual-source
        design carry an empty ``per_model_reasoning_tokens`` dict on
        every outcome. The aggregator ignores it cleanly -- we don't
        re-introduce Phase A capture by accident, and we don't crash."""
        import pandas as pd

        legacy_outcome = json.dumps(
            {
                "per_model_calls": {"assistant_model": 1},
                "per_model_input_tokens": {"assistant_model": 10},
                "per_model_output_tokens": {"assistant_model": 1000},
                # Legacy Phase A field. Should be ignored.
                "per_model_reasoning_tokens": {"assistant_model": 42},
                "wall_clock_s_by_alias": {},
                "wall_clock_s": 1.0,
            }
        )
        df = pd.DataFrame(
            [
                {
                    "simulation_outcome": legacy_outcome,
                    "conversation_messages": json.dumps(
                        [
                            {"role": "assistant", "content": "hello"},
                        ]
                    ),
                }
            ]
        )
        p = aggregate_from_dataframe(df)
        # The 42 from the legacy field MUST NOT appear in the profile;
        # the value is whatever Phase B produced (large diff -> well
        # above floor -> 'estimated').
        assert p.reasoning_source_by_alias["assistant_model"] == "estimated"
        assert p.reasoning_tokens_by_alias["assistant_model"] != 42


class TestAggregateFromDataframe:
    def test_uses_simulation_outcome_column(self, trajectory_df) -> None:
        p = aggregate_from_dataframe(trajectory_df)
        assert p.n_trajectories == 6
        # Spot-check: every fixture row writes per_model_calls with at
        # least user_model + assistant_model.
        assert "user_model" in p.n_calls_by_alias
        assert "assistant_model" in p.n_calls_by_alias

    def test_missing_column_returns_empty(self) -> None:
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"x": [1, 2]})
        p = aggregate_from_dataframe(df)
        assert p.n_trajectories == 0

    def test_rejects_non_dataframe(self) -> None:
        with pytest.raises(TypeError):
            aggregate_from_dataframe([{"x": 1}])
