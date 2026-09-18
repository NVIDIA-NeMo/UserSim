# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the top-level reporting test suite.

Provides synthetic trajectory + eval frames matching the schemas
written by ``src/usersim/engine/`` so the
reporting layer (``reporting/``) can be exercised without a live
simulator run. Real end-to-end fixtures arrive with the smoke-test
deliverable.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest


class _WordSplitEncoder:
    """Deterministic stand-in for a ``tiktoken`` encoding."""

    @staticmethod
    def encode(text: str) -> List[int]:
        return [0] * len(text.split())


@pytest.fixture(autouse=True, scope="session")
def deterministic_token_encoder():
    """Stub the reasoning estimator's tokenizer for the whole session.

    ``tiktoken.get_encoding`` fetches its vocabulary over HTTPS on first
    use -- tiktoken ships the tokenizer, not the vocabulary -- so leaving
    it live makes every reasoning-token assertion depend on network
    egress. That is what took out 40+ tests across five files behind a
    proxy.

    Stubbing is the right level of mocking here because these tests
    exercise the *aggregation* logic (noise-floor clamp, negative-diff
    clamp, source promotion), not tiktoken's correctness, and every
    assertion is tolerance-based rather than an exact token count. A word
    splitter clears or misses the 5% floor in exactly the same cases.

    Autouse rather than opt-in so a new test cannot silently reintroduce
    the network dependency.
    """
    import usersim.reporting._reasoning as reasoning

    original = reasoning._encoder
    reasoning._encoder = lambda name: _WordSplitEncoder()
    try:
        yield
    finally:
        reasoning._encoder = original


def _outcome(
    *,
    status: str = "ok",
    failure_class: str | None = None,
    failure_attribution: str | None = None,
    n_turns: int = 3,
    n_tool_calls: int = 0,
    n_user_query_attempts: int = 1,
    n_user_followup_retries: int = 0,
    n_assistant_inline_failures: int = 0,
    n_api_response_rerolls: int = 0,
    n_fourth_wall_triggers: int = 0,
    n_user_role_violations: int = 0,
    early_stop: bool = False,
    per_model_calls: Dict[str, int] | None = None,
    per_model_input_tokens: Dict[str, int] | None = None,
    per_model_output_tokens: Dict[str, int] | None = None,
    wall_clock_s_by_alias: Dict[str, float] | None = None,
    wall_clock_s: float = 12.34,
    code_sha: str | None = "abc123",
    nemotron_personas_version: str | None = "2026-04",
    scenario_prompt_version: str | None = "v1.0",
) -> str:
    return json.dumps(
        {
            "status": status,
            "failure_class": failure_class,
            "failure_attribution": failure_attribution,
            "failure_detail": "",
            "n_turns": n_turns,
            "n_tool_calls": n_tool_calls,
            "n_user_query_attempts": n_user_query_attempts,
            "n_user_followup_retries": n_user_followup_retries,
            "n_assistant_inline_failures": n_assistant_inline_failures,
            "n_api_response_rerolls": n_api_response_rerolls,
            "n_fourth_wall_triggers": n_fourth_wall_triggers,
            "n_user_role_violations": n_user_role_violations,
            "warnings": [],
            "early_stop": early_stop,
            "per_model_input_tokens": per_model_input_tokens or {"user_model": 100, "assistant_model": 80},
            "per_model_output_tokens": per_model_output_tokens or {"user_model": 50, "assistant_model": 90},
            "per_model_calls": per_model_calls or {"user_model": 3, "assistant_model": 3},
            "wall_clock_s_by_alias": wall_clock_s_by_alias or {"user_model": 4.0, "assistant_model": 6.0},
            "wall_clock_s": wall_clock_s,
            "provenance": {
                "nemotron_personas_version": nemotron_personas_version,
                "scenario_prompt_version": scenario_prompt_version,
                "code_sha": code_sha,
                "bank_version": {},
            },
        }
    )


def _messages(user_turns: list[str], assistant_turns: list[str]) -> str:
    """Interleave user/assistant turns into a JSON-encoded conversation."""
    msgs: List[Dict[str, Any]] = []
    for u, a in zip(user_turns, assistant_turns):
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    if len(user_turns) > len(assistant_turns):
        msgs.append({"role": "user", "content": user_turns[-1]})
    return json.dumps(msgs)


@pytest.fixture
def trajectory_df():
    """Six-row synthetic trajectory frame spanning 2 probes x 3 outcomes."""
    pd = pytest.importorskip("pandas")

    rows = [
        {
            # OK row, en_US, general_open_ended, cooperative.
            "trajectory_id": "t000000000000001",
            "persona_uuid": "p00000000000001",
            "probe_family": "general_open_ended",
            "probe_variant": "default",
            "locale": "en_US",
            "conversation_language": "English",
            "user_interaction_style": "cooperative",
            "disclosure_style": "incremental",
            "persona_grounding": True,
            "conversation_messages": _messages(
                ["Hi! I have a question about something.", "Thanks, that helps!"],
                ["Sure, what is it?", "Glad I could help."],
            ),
            "simulation_outcome": _outcome(
                status="ok",
                n_turns=2,
                per_model_calls={"user_model": 2, "assistant_model": 2, "judge_model": 2},
                per_model_input_tokens={"user_model": 100, "assistant_model": 80, "judge_model": 200},
                per_model_output_tokens={"user_model": 30, "assistant_model": 60, "judge_model": 40},
                wall_clock_s_by_alias={"user_model": 2.0, "assistant_model": 3.5, "judge_model": 1.0},
                wall_clock_s=8.0,
                early_stop=True,
            ),
        },
        {
            # Failure, user-model attribution.
            "trajectory_id": "t000000000000002",
            "persona_uuid": "p00000000000002",
            "probe_family": "general_open_ended",
            "probe_variant": "default",
            "locale": "pt_BR",
            "conversation_language": "Portuguese",
            "user_interaction_style": "neutral",
            "disclosure_style": "upfront",
            "persona_grounding": False,
            "conversation_messages": _messages(["Olá."], ["Olá! Como posso ajudar?"]),
            "simulation_outcome": _outcome(
                status="failed",
                failure_class="user_query_gate_exhausted",
                failure_attribution="user_model",
                n_turns=0,
                n_user_query_attempts=3,
            ),
        },
        {
            # Confrontational frustrated.
            "trajectory_id": "t000000000000003",
            "persona_uuid": "p00000000000003",
            "probe_family": "general_open_ended",
            "probe_variant": "default",
            "locale": "en_US",
            "conversation_language": "English",
            "user_interaction_style": "confrontational",
            "disclosure_style": "upfront",
            "persona_grounding": True,
            "conversation_messages": _messages(
                ["I already told you what I need!", "This is a waste of time. You're terrible."],
                ["I understand your frustration.", "Let me try again."],
            ),
            "simulation_outcome": _outcome(
                status="completed_with_warnings",
                n_turns=2,
                n_assistant_inline_failures=1,
                n_user_role_violations=1,
            ),
        },
        {
            # tool_calling OK row.
            "trajectory_id": "t000000000000004",
            "persona_uuid": "p00000000000004",
            "probe_family": "tool_calling",
            "probe_variant": "default",
            "locale": "en_US",
            "conversation_language": "English",
            "user_interaction_style": "cooperative",
            "disclosure_style": "incremental",
            "persona_grounding": True,
            "conversation_messages": _messages(
                ["What is the weather in Tokyo?", "Thanks!"],
                ["It is 22 degrees in Tokyo.", "Anytime."],
            ),
            "simulation_outcome": _outcome(
                status="ok",
                n_turns=2,
                n_tool_calls=1,
                early_stop=True,
                per_model_calls={"user_model": 2, "assistant_model": 2, "api_response_model": 1},
                per_model_input_tokens={"user_model": 80, "assistant_model": 100, "api_response_model": 120},
                per_model_output_tokens={"user_model": 30, "assistant_model": 70, "api_response_model": 25},
            ),
        },
        {
            # tool_calling sim-quality failure (zero tool calls).
            "trajectory_id": "t000000000000005",
            "persona_uuid": "p00000000000005",
            "probe_family": "tool_calling",
            "probe_variant": "default",
            "locale": "en_US",
            "conversation_language": "English",
            "user_interaction_style": "neutral",
            "disclosure_style": "upfront",
            "persona_grounding": False,
            "conversation_messages": _messages(
                ["What's the weather?"],
                ["I can't help with that."],
            ),
            "simulation_outcome": _outcome(
                status="ok",
                n_turns=1,
                n_tool_calls=0,
            ),
        },
        {
            # API-response invalid path.
            "trajectory_id": "t000000000000006",
            "persona_uuid": "p00000000000006",
            "probe_family": "tool_calling",
            "probe_variant": "default",
            "locale": "ja_JP",
            "conversation_language": "Japanese",
            "user_interaction_style": "impatient",
            "disclosure_style": "upfront",
            "persona_grounding": True,
            "conversation_messages": _messages(
                ["京都の天気を教えて。", "わかった、ありがとう。"],
                ["京都は晴れです。", "どういたしまして。"],
            ),
            "simulation_outcome": _outcome(
                status="ok",
                n_turns=2,
                n_tool_calls=1,
                n_api_response_rerolls=1,
                code_sha="def456",  # different SHA -> tests provenance drift detection
            ),
        },
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def evaluator_df(trajectory_df):
    """Add an `assistant_eval` column matching the TrajectoryEvaluator wide envelope.

    Built on top of ``trajectory_df`` so trajectory_id / probe_family /
    locale / user_interaction_style stay aligned for stratified aggregation.
    Three of the six rows have full ensemble scores; one is skipped
    (envelope_match); one short-circuits (no_assistant_messages); one
    has a registered scorer output.
    """
    pytest.importorskip("pandas")

    df = trajectory_df.copy()

    base_envelope = {
        "judge_aliases": ["judge_a", "judge_b"],
        "judge_families": ["openai", "nvidia_nemotron"],
        "axes": ["helpfulness", "accuracy"],
        "scorers": [],
        "prompt_version": "v1.0",
        "evaluator_version": "v1.0",
    }

    eval_cells: list[str | None] = []
    for idx in range(len(df)):
        if idx == 0:
            cell = {
                "envelope": base_envelope,
                "axes": {
                    "helpfulness": {
                        "judge_a": {"score": 5, "reasoning": "great"},
                        "judge_b": {"score": 4, "reasoning": "good"},
                    },
                    "accuracy": {
                        "judge_a": {"score": 5, "reasoning": "accurate"},
                        "judge_b": {"score": 5, "reasoning": "accurate"},
                    },
                },
                "scorers": {},
                "skipped": False,
                "skipped_reason": None,
            }
        elif idx == 1:
            # Short-circuit because the failed trajectory has no assistant turns.
            cell = {
                "envelope": base_envelope,
                "axes": {},
                "scorers": {},
                "skipped": True,
                "skipped_reason": "no_assistant_messages",
            }
        elif idx == 2:
            # Confrontational row scored lower.
            cell = {
                "envelope": base_envelope,
                "axes": {
                    "helpfulness": {
                        "judge_a": {"score": 2, "reasoning": "lost the user"},
                        "judge_b": {"score": 3, "reasoning": "weak recovery"},
                    },
                    "accuracy": {
                        "judge_a": {"score": 4, "reasoning": "still accurate"},
                        "judge_b": {"score": 4, "reasoning": "still accurate"},
                    },
                },
                "scorers": {},
                "skipped": False,
                "skipped_reason": None,
            }
        elif idx == 3:
            # Tool-calling OK with a tool_use scorer output.
            envelope = dict(base_envelope, scorers=["tool_use"])
            cell = {
                "envelope": envelope,
                "axes": {
                    "helpfulness": {
                        "judge_a": {"score": 5, "reasoning": "good answer"},
                        "judge_b": {"score": 5, "reasoning": "good answer"},
                    },
                    "accuracy": {
                        "judge_a": {"score": 5, "reasoning": "correct"},
                        "judge_b": {"score": 4, "reasoning": "correct"},
                    },
                },
                "scorers": {
                    "tool_use": {
                        "judge_alias": "judge_a",
                        "scores": {"overall": {"score": 5, "reasoning": "excellent"}},
                        "status_proposal": True,
                    }
                },
                "skipped": False,
                "skipped_reason": None,
            }
        elif idx == 4:
            # Skipped via envelope_match (the partial-re-run path).
            cell = {
                "envelope": base_envelope,
                "axes": {
                    "helpfulness": {
                        "judge_a": {"score": 1, "reasoning": "didn't help"},
                        "judge_b": {"score": 2, "reasoning": "weak"},
                    },
                    "accuracy": {
                        "judge_a": {"score": 3, "reasoning": "ok"},
                        "judge_b": {"score": 3, "reasoning": "ok"},
                    },
                },
                "scorers": {},
                "skipped": True,
                "skipped_reason": "envelope_match",
            }
        else:
            # Last row: tool_use scorer with an error.
            envelope = dict(base_envelope, scorers=["tool_use"])
            cell = {
                "envelope": envelope,
                "axes": {
                    "helpfulness": {
                        "judge_a": {"score": 4, "reasoning": "ok"},
                        "judge_b": {"score": 4, "reasoning": "ok"},
                    },
                    "accuracy": {
                        "judge_a": {"score": 4, "reasoning": "ok"},
                        "judge_b": {"score": 4, "reasoning": "ok"},
                    },
                },
                "scorers": {
                    "tool_use": {
                        "judge_alias": "judge_a",
                        "scores": {},
                        "status_proposal": True,
                        "error": "RuntimeError: boom",
                    }
                },
                "skipped": False,
                "skipped_reason": None,
            }
        eval_cells.append(json.dumps(cell))

    df["assistant_eval"] = eval_cells
    return df
