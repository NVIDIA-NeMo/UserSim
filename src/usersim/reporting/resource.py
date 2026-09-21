# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Resource accounting — calls / tokens / wall-clock by model alias.

The framework reports resources, not dollars (SLURM-first deployment +
un-priced model catalogs). This module aggregates the per-row
``simulation_outcome.per_model_*`` fields into a per-bundle profile so
the ``SimHealthBundle`` and the capability dashboard can surface
"what did this report cost (in calls and tokens) to produce."

The CLI dry-run estimator reuses the same aggregation against
historical baselines stored under ``artifacts/resource_baselines/``;
that helper lands with the CLI deliverable. This file is the analysis
side that consumers always want.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

# The "conversation aliases" — the two parties IN the conversation:
# the simulated user and the assistant under test.
#
# - ``user_model``: simulated user turns (the user side of the dialogue).
# - ``assistant_model``: assistant turns (the model under test's responses
#   to the user, including any tool_calls the assistant emitted).
#
# Excludes:
# - ``api_response_model``: tool-response SYNTHESISER. The assistant may
#   incorporate this output into its next turn, but the synthesiser
#   itself isn't a conversation participant -- it's a tool-call backend
#   the simulator stands up so tool_calling probes have something to
#   call. Counted under ``total_tokens`` but not as conversation cost.
# - ``judge_model`` / ``summary_model``: in-sim scaffolding (capitulation
#   classifier, context compressor) that participate in the simulator's
#   loop but never produce visible conversation turns.
_CONVERSATION_ALIASES = ("user_model", "assistant_model")


def _decode_outcome(raw: Any) -> dict[str, Any]:
    """Best-effort decode of the ``simulation_outcome`` cell."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


@dataclass
class ResourceProfile:
    """Per-bundle aggregate of model-call resource usage.

    Per-alias dicts keyed by model alias (``user_model``, ``assistant_model``,
    etc.). Wall-clock-by-alias is the sum of attributable wall-clock
    across rows; ``total_wall_clock_s`` is the sum of end-to-end row
    elapsed (always >= per-alias because it includes overhead between
    calls).

    All fields default to empty / 0 so an empty trajectory store
    produces a structurally-valid (and unambiguously empty) profile.

    ``reasoning_tokens_by_alias`` + ``reasoning_source_by_alias`` carry
    the dual-source reasoning-token signal (provider-captured at sim time
    when exposed, tiktoken-estimated post-hoc otherwise -- see
    ``reporting/_reasoning.py``). Source labels per alias:
    ``captured`` / ``estimated`` / ``unreliable_estimate`` / ``unavailable``.
    """

    n_trajectories: int = 0
    n_calls_by_alias: dict[str, int] = field(default_factory=dict)
    input_tokens_by_alias: dict[str, int] = field(default_factory=dict)
    output_tokens_by_alias: dict[str, int] = field(default_factory=dict)
    reasoning_tokens_by_alias: dict[str, int] = field(default_factory=dict)
    reasoning_source_by_alias: dict[str, str] = field(default_factory=dict)
    wall_clock_s_by_alias: dict[str, float] = field(default_factory=dict)
    total_wall_clock_s: float = 0.0

    @property
    def total_calls(self) -> int:
        return sum(self.n_calls_by_alias.values())

    @property
    def total_input_tokens(self) -> int:
        return sum(self.input_tokens_by_alias.values())

    @property
    def total_output_tokens(self) -> int:
        return sum(self.output_tokens_by_alias.values())

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def conversation_output_tokens(self) -> int:
        """Output tokens from the two parties IN the conversation:
        ``user_model`` + ``assistant_model``.

        Excludes ``api_response_model`` (tool-response synthesiser; the
        assistant consumes that output as input to its next turn but
        the synthesiser itself isn't a conversation participant) and
        the in-sim scaffolding aliases (``judge_model`` /
        ``summary_model``). This is the headline number for "what did
        this conversation actually cost on the output side"; the
        broader in-sim resource bill is ``total_tokens``.
        """
        return sum(self.output_tokens_by_alias.get(alias, 0) for alias in _CONVERSATION_ALIASES)

    @property
    def assistant_output_tokens(self) -> int:
        """Output tokens from ``assistant_model`` only -- what the model
        under test produced. Subset of ``conversation_output_tokens``.
        Surfaced as a separate dashboard card so reviewers can compare
        the model's verbosity contribution to the total conversation cost.
        """
        return self.output_tokens_by_alias.get("assistant_model", 0)

    @property
    def assistant_total_tokens(self) -> int:
        """Gross billable spend on ``assistant_model`` alone (input +
        output, including any reasoning since reasoning rolls into output
        on the bill). The dashboard's "Assistant Tokens" row pairs this
        with ``assistant_output_tokens`` / ``assistant_reasoning_tokens``
        / ``assistant_conversation_tokens`` so reviewers can decompose:

        ``assistant_total = input + output``
        ``assistant_output = reasoning + conversation`` (visible)
        """
        return self.input_tokens_by_alias.get("assistant_model", 0) + self.output_tokens_by_alias.get(
            "assistant_model", 0
        )

    @property
    def assistant_reasoning_tokens(self) -> int:
        """Reasoning subset for ``assistant_model`` only. Already counted
        inside ``assistant_total_tokens`` and ``assistant_output_tokens``;
        the visible-output complement is ``assistant_conversation_tokens``
        (= output - reasoning).
        """
        return self.reasoning_tokens_by_alias.get("assistant_model", 0)

    @property
    def assistant_conversation_tokens(self) -> int:
        """Visible output the model under test produced -- output tokens
        with reasoning subtracted out. The "what was actually said by the
        assistant" number, distinct from gross output (which folds in
        invisible thinking content for reasoning models). Clamped to
        non-negative so a slightly-overcounted reasoning estimate can't
        make this go below zero.
        """
        return max(
            0,
            self.output_tokens_by_alias.get("assistant_model", 0)
            - self.reasoning_tokens_by_alias.get("assistant_model", 0),
        )

    @property
    def user_total_tokens(self) -> int:
        """Gross billable spend on ``user_model`` (the simulated-user
        agent) -- input + output, including any reasoning. Mirrors
        ``assistant_total_tokens`` so the dashboard's "User Tokens" row
        is structurally identical to the Assistant Tokens row.
        """
        return self.input_tokens_by_alias.get("user_model", 0) + self.output_tokens_by_alias.get("user_model", 0)

    @property
    def user_output_tokens(self) -> int:
        """Output tokens from ``user_model`` (the simulated-user agent's
        contribution to the conversation, gross -- includes reasoning
        when the user-agent model is a reasoning model).
        """
        return self.output_tokens_by_alias.get("user_model", 0)

    @property
    def user_reasoning_tokens(self) -> int:
        """Reasoning subset for ``user_model`` only. Estimates can be
        coarse here -- some probes inject the turn-1 user message
        verbatim from an asset bank rather than generating it via
        ``user_model``, so the per-row visible-text count under-counts
        what the provider actually charged. The estimator flags this
        with the ``estimated`` source label.
        """
        return self.reasoning_tokens_by_alias.get("user_model", 0)

    @property
    def user_conversation_tokens(self) -> int:
        """Visible user-agent output -- ``user_output - user_reasoning``.
        Clamped to non-negative as a defensive guard."""
        return max(0, self.user_output_tokens - self.user_reasoning_tokens)

    # ── Support actors (api_response / judge / summary) ──────────────
    # Mirror the User / Assistant decomposition for the support aliases.
    # judge_model and summary_model produce no visible conversation
    # turns -- they're in-sim scaffolding -- but the dashboard still
    # surfaces ``total / output / reasoning / conversation`` cards for
    # symmetry with the User and Assistant rows. ``conversation`` here
    # means "non-reasoning output" rather than "conversation turn"
    # (judge/summary reasoning is typically 0 anyway since Phase B
    # estimation is unavailable for aliases that don't preserve their
    # output verbatim on the trajectory).

    @property
    def api_response_total_tokens(self) -> int:
        """Gross billable spend on ``api_response_model`` -- the tool-
        response synthesiser the simulator stands up so tool_calling
        probes have something to call. Output is consumed by the
        assistant's NEXT turn as input, not as a conversation turn.
        """
        return self.input_tokens_by_alias.get("api_response_model", 0) + self.output_tokens_by_alias.get(
            "api_response_model", 0
        )

    @property
    def api_response_output_tokens(self) -> int:
        return self.output_tokens_by_alias.get("api_response_model", 0)

    @property
    def api_response_reasoning_tokens(self) -> int:
        return self.reasoning_tokens_by_alias.get("api_response_model", 0)

    @property
    def api_response_conversation_tokens(self) -> int:
        """Visible (non-reasoning) api_response_model output."""
        return max(
            0,
            self.api_response_output_tokens - self.api_response_reasoning_tokens,
        )

    @property
    def judge_total_tokens(self) -> int:
        """Gross billable spend on ``judge_model`` -- the in-sim user-LLM
        gate / capitulation classifier. Not a conversation participant.
        """
        return self.input_tokens_by_alias.get("judge_model", 0) + self.output_tokens_by_alias.get("judge_model", 0)

    @property
    def judge_output_tokens(self) -> int:
        return self.output_tokens_by_alias.get("judge_model", 0)

    @property
    def judge_reasoning_tokens(self) -> int:
        return self.reasoning_tokens_by_alias.get("judge_model", 0)

    @property
    def judge_conversation_tokens(self) -> int:
        """Visible (non-reasoning) judge_model output. Phase B estimation
        is unavailable for ``judge_model`` (no visible content on the
        trajectory), so this typically equals ``judge_output_tokens``
        unless the provider exposed a Phase A capture."""
        return max(0, self.judge_output_tokens - self.judge_reasoning_tokens)

    @property
    def summary_total_tokens(self) -> int:
        """Gross billable spend on ``summary_model`` -- in-sim context
        compressor for long conversations. Not a conversation participant.
        """
        return self.input_tokens_by_alias.get("summary_model", 0) + self.output_tokens_by_alias.get("summary_model", 0)

    @property
    def summary_output_tokens(self) -> int:
        return self.output_tokens_by_alias.get("summary_model", 0)

    @property
    def summary_reasoning_tokens(self) -> int:
        return self.reasoning_tokens_by_alias.get("summary_model", 0)

    @property
    def summary_conversation_tokens(self) -> int:
        """Visible (non-reasoning) summary_model output. Same Phase B
        unavailability caveat as ``judge_conversation_tokens``."""
        return max(0, self.summary_output_tokens - self.summary_reasoning_tokens)

    @property
    def total_reasoning_tokens(self) -> int:
        """Sum of reasoning tokens across every alias, regardless of
        source. Already included in ``total_output_tokens`` (gross billing
        convention -- ``output_tokens`` IS reasoning + visible). The
        reasoning-tokens dashboard card surfaces this as the invisible
        subset.
        """
        return sum(self.reasoning_tokens_by_alias.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_trajectories": self.n_trajectories,
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "conversation_output_tokens": self.conversation_output_tokens,
            "assistant_output_tokens": self.assistant_output_tokens,
            "assistant_total_tokens": self.assistant_total_tokens,
            "assistant_reasoning_tokens": self.assistant_reasoning_tokens,
            "assistant_conversation_tokens": self.assistant_conversation_tokens,
            "user_total_tokens": self.user_total_tokens,
            "user_output_tokens": self.user_output_tokens,
            "user_reasoning_tokens": self.user_reasoning_tokens,
            "user_conversation_tokens": self.user_conversation_tokens,
            "api_response_total_tokens": self.api_response_total_tokens,
            "api_response_output_tokens": self.api_response_output_tokens,
            "api_response_reasoning_tokens": self.api_response_reasoning_tokens,
            "api_response_conversation_tokens": self.api_response_conversation_tokens,
            "judge_total_tokens": self.judge_total_tokens,
            "judge_output_tokens": self.judge_output_tokens,
            "judge_reasoning_tokens": self.judge_reasoning_tokens,
            "judge_conversation_tokens": self.judge_conversation_tokens,
            "summary_total_tokens": self.summary_total_tokens,
            "summary_output_tokens": self.summary_output_tokens,
            "summary_reasoning_tokens": self.summary_reasoning_tokens,
            "summary_conversation_tokens": self.summary_conversation_tokens,
            "total_reasoning_tokens": self.total_reasoning_tokens,
            "total_wall_clock_s": round(self.total_wall_clock_s, 3),
            "n_calls_by_alias": dict(self.n_calls_by_alias),
            "input_tokens_by_alias": dict(self.input_tokens_by_alias),
            "output_tokens_by_alias": dict(self.output_tokens_by_alias),
            "reasoning_tokens_by_alias": dict(self.reasoning_tokens_by_alias),
            "reasoning_source_by_alias": dict(self.reasoning_source_by_alias),
            "wall_clock_s_by_alias": {k: round(v, 3) for k, v in self.wall_clock_s_by_alias.items()},
        }


def _accumulate_alias_dict(target: dict[str, Any], source: Any) -> None:
    """Merge ``source`` (dict-or-decodable) into ``target``, summing values.

    Tolerant of unparseable / wrong-type sources (treats them as empty).
    """
    if isinstance(source, str):
        try:
            source = json.loads(source)
        except (json.JSONDecodeError, TypeError):
            return
    if not isinstance(source, dict):
        return
    for k, v in source.items():
        if not isinstance(k, str):
            continue
        if isinstance(v, (int, float)):
            target[k] = target.get(k, 0) + v


def aggregate_resource_profile(
    outcomes: Iterable[Any],
) -> ResourceProfile:
    """Aggregate ``simulation_outcome`` cells into a single ResourceProfile.

    ``outcomes`` is any iterable of cells (dict-or-JSON-string-or-None).
    Typical caller passes ``df["simulation_outcome"].tolist()``.
    Skips rows whose outcome cell is missing / unparseable.

    Reasoning-token estimation is NOT done here -- that lives on
    ``aggregate_from_dataframe`` since it needs the conversation
    messages, not just the outcome dict. This function only handles the
    sim-captured per-alias call/token/wall-clock totals.
    """
    profile = ResourceProfile()
    for raw in outcomes:
        outcome = _decode_outcome(raw)
        if not outcome:
            continue
        profile.n_trajectories += 1
        _accumulate_alias_dict(profile.n_calls_by_alias, outcome.get("per_model_calls"))
        _accumulate_alias_dict(profile.input_tokens_by_alias, outcome.get("per_model_input_tokens"))
        _accumulate_alias_dict(profile.output_tokens_by_alias, outcome.get("per_model_output_tokens"))
        _accumulate_alias_dict(profile.wall_clock_s_by_alias, outcome.get("wall_clock_s_by_alias"))
        wc = outcome.get("wall_clock_s")
        if isinstance(wc, (int, float)):
            profile.total_wall_clock_s += float(wc)
    return profile


def aggregate_from_dataframe(df: Any, column: str = "simulation_outcome") -> ResourceProfile:
    """Convenience wrapper: pull the column out of a DataFrame and aggregate.

    DataFrame-aware version of ``aggregate_resource_profile`` that also
    runs the post-hoc reasoning-token estimator (``output_tokens -
    tiktoken(visible_content)``) per row, then applies a 5% per-alias
    noise-floor clamp. Aliases with reasoning estimates below 5% of
    their output tokens are zeroed out and marked ``unavailable`` -- the
    estimate is indistinguishable from cl100k_base tokenizer drift
    against the provider's actual tokenizer at that scale.

    Aliases that don't preserve their output verbatim on the trajectory
    (``judge_model`` / ``summary_model``) are always ``unavailable``;
    Phase B can't help them.
    """
    import pandas as pd  # lazy import

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"aggregate_from_dataframe expects a pandas DataFrame, got {type(df).__name__}")
    if column not in df.columns:
        # No outcomes -> empty profile.
        return ResourceProfile()
    profile = aggregate_resource_profile(df[column].tolist())
    if "conversation_messages" not in df.columns:
        # No content available for tiktoken estimation. Mark
        # judge/summary unavailable as usual; visible-output aliases
        # have no signal so they stay absent from the source dict.
        return _finalize_reasoning_sources(profile)

    from usersim.reporting._reasoning import (
        _VISIBLE_OUTPUT_ALIASES,
        estimate_reasoning_for_row,
    )

    # Per-row pass: accumulate raw reasoning estimates per alias.
    for _, row in df.iterrows():
        outcome = _decode_outcome(row[column])
        if not outcome:
            continue
        outputs = outcome.get("per_model_output_tokens") or {}
        if isinstance(outputs, str):
            try:
                outputs = json.loads(outputs)
            except (json.JSONDecodeError, TypeError):
                outputs = {}
        if not isinstance(outputs, dict):
            outputs = {}
        for alias in _VISIBLE_OUTPUT_ALIASES:
            n_out = outputs.get(alias, 0)
            if not isinstance(n_out, (int, float)) or n_out <= 0:
                continue
            tokens, source = estimate_reasoning_for_row(
                row.get("conversation_messages"),
                alias,
                int(n_out),
            )
            if tokens > 0:
                profile.reasoning_tokens_by_alias[alias] = profile.reasoning_tokens_by_alias.get(alias, 0) + tokens
            # ``estimated`` always wins over the default ``unavailable``
            # for visible-output aliases. The noise-floor pass below
            # decides whether the accumulated estimate is real.
            if alias not in profile.reasoning_source_by_alias:
                profile.reasoning_source_by_alias[alias] = source
    return _finalize_reasoning_sources(profile)


def _finalize_reasoning_sources(profile: ResourceProfile) -> ResourceProfile:
    """Mark judge/summary as ``unavailable`` and apply the noise-floor
    clamp to visible-output aliases.

    Per-alias noise floor: if the aggregate reasoning estimate is below
    ``_REASONING_NOISE_FLOOR_RATIO`` of the alias's output tokens, the
    estimate is dominated by tokenizer drift between ``cl100k_base`` and
    whatever the provider actually used -- not real reasoning content.
    Zero out the alias and flag ``unavailable``.

    Real reasoning models clear the floor by 4-10x in practice; non-
    reasoning models stay under it.
    """
    from usersim.reporting._reasoning import (
        _REASONING_NOISE_FLOOR_RATIO,
        _UNAVAILABLE_ALIASES,
        _VISIBLE_OUTPUT_ALIASES,
    )

    for alias in _UNAVAILABLE_ALIASES:
        profile.reasoning_source_by_alias.setdefault(alias, "unavailable")

    for alias in _VISIBLE_OUTPUT_ALIASES:
        out_n = profile.output_tokens_by_alias.get(alias, 0)
        reason_n = profile.reasoning_tokens_by_alias.get(alias, 0)
        if out_n <= 0:
            continue
        if reason_n / out_n < _REASONING_NOISE_FLOOR_RATIO:
            profile.reasoning_tokens_by_alias.pop(alias, None)
            profile.reasoning_source_by_alias[alias] = "unavailable"
    return profile
