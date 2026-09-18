# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral-realism diagnostics — D1 / D2 / D3 / D4 + MATTR.

Relocated from ``usersim.ipynb`` Section 13 into a proper module so
the ``SimHealthBundle`` (and any other consumer) can compute them
without duplicating logic. Inspired by Zhou et al. (2026), "Mind the
Sim2Real Gap in User Simulation for Agentic Tasks".

The four diagnostics:

- **D1 — front-loading ratio**: word count of first user turn vs.
  mean of subsequent user turns. Ratio >> 1.0 means the user dumps
  everything upfront. Real users typically sit closer to 1.0.
- **D2 — polite turn fraction**: fraction of user turns containing a
  polite/agreeable marker. Real users << 0.3.
- **D3 — verbosity CV**: coefficient of variation of user turn word
  counts. Low CV = unnaturally uniform (an LLM artifact). Real users > 0.5.
- **D4 — frustration marker fraction**: fraction of user turns
  containing frustration / pushback markers. Stratified by interaction
  style; ``confrontational`` should be visibly higher than ``cooperative``.

MATTR is the moving-average type-token ratio — a stable lexical
diversity metric for varying document lengths. Tracked alongside the
diagnostics because mode collapse in the user agent shows up as
suddenly-flat MATTR.

All functions accept either a list-of-message-dicts or a JSON-encoded
string (which is how trajectories are stored), and return ``None`` for
trajectories that cannot be measured (e.g., < 2 user turns for D1/D3).
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Marker dictionaries — kept here (not in __init__.py) so additions are
# trivially reviewable. English-only for now; locale-specific marker
# packs land alongside the corresponding non-English probes.
# ---------------------------------------------------------------------------

POLITE_MARKERS = re.compile(
    r"\b(thank you|thanks|please|appreciate|sorry|no problem|of course|"
    r"sure thing|happy to|glad to|certainly|absolutely)\b",
    re.IGNORECASE,
)

FRUSTRATION_MARKERS = re.compile(
    r"\b(frustrated|annoyed|unacceptable|ridiculous|already told you|"
    r"already said|this is taking|waste of time|terrible|awful|"
    r"I already|you asked me this|can you just)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Message-list normalization
# ---------------------------------------------------------------------------


def _decode_messages(raw: Any) -> list[dict[str, Any]]:
    """Accept either a list-of-dicts or a JSON-encoded string."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []
    return []


def _user_turns(messages: Any) -> list[str]:
    """Extract text content of every user-role message, preserving order."""
    msgs = _decode_messages(messages)
    out: list[str] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str) and content.strip():
            out.append(content)
    return out


def _word_count(text: str) -> int:
    """Whitespace word count. CJK is undercounted but the diagnostics are
    normalized comparisons within a single trajectory, so the CJK bias
    is consistent and does not corrupt the ratios."""
    return len(text.split())


# ---------------------------------------------------------------------------
# D1 — front-loading ratio
# ---------------------------------------------------------------------------


def compute_front_loading_ratio(messages: Any) -> float | None:
    """First-turn word count / mean of subsequent-turn word counts.

    Returns ``None`` when there are < 2 user turns (the metric is
    undefined) or when the subsequent turns are empty (division by zero).
    """
    turns = _user_turns(messages)
    if len(turns) < 2:
        return None
    first_len = _word_count(turns[0])
    rest_lengths = [_word_count(t) for t in turns[1:]]
    rest_mean = sum(rest_lengths) / len(rest_lengths)
    if rest_mean == 0:
        return None
    return round(first_len / rest_mean, 3)


# ---------------------------------------------------------------------------
# D2 — polite turn fraction
# ---------------------------------------------------------------------------


def compute_polite_turn_fraction(messages: Any) -> float | None:
    """Fraction of user turns containing a polite-marker hit."""
    turns = _user_turns(messages)
    if not turns:
        return None
    polite = sum(1 for t in turns if POLITE_MARKERS.search(t))
    return round(polite / len(turns), 3)


# ---------------------------------------------------------------------------
# D3 — verbosity CV
# ---------------------------------------------------------------------------


def compute_verbosity_cv(messages: Any) -> float | None:
    """Coefficient of variation of user-turn word counts. None if undefined."""
    turns = _user_turns(messages)
    if len(turns) < 2:
        return None
    lengths = [_word_count(t) for t in turns]
    n = len(lengths)
    mean_len = sum(lengths) / n
    if mean_len == 0:
        return None
    # Population stddev (n divisor) — matches numpy's default and the
    # original notebook implementation.
    var = sum((x - mean_len) ** 2 for x in lengths) / n
    std = var**0.5
    return round(std / mean_len, 3)


# ---------------------------------------------------------------------------
# D4 — frustration marker fraction
# ---------------------------------------------------------------------------


def compute_frustration_marker_fraction(messages: Any) -> float | None:
    """Fraction of user turns containing a frustration-marker hit."""
    turns = _user_turns(messages)
    if not turns:
        return None
    frust = sum(1 for t in turns if FRUSTRATION_MARKERS.search(t))
    return round(frust / len(turns), 3)


# ---------------------------------------------------------------------------
# MATTR — moving-average type-token ratio
# ---------------------------------------------------------------------------


def compute_mattr(tokens: Iterable[str], window_size: int = 50) -> float:
    """Moving-average type-token ratio.

    Slides a fixed-size window across the token sequence and averages
    the type-token ratio within each window. More stable than global
    TTR for varying document lengths. Returns 0.0 for empty input;
    returns the global TTR when ``len(tokens) < window_size``.
    """
    toks = list(tokens)
    if not toks:
        return 0.0
    if len(toks) < window_size:
        return len(set(toks)) / len(toks)
    ttrs: list[float] = []
    for i in range(len(toks) - window_size + 1):
        window = toks[i : i + window_size]
        ttrs.append(len(set(window)) / window_size)
    return sum(ttrs) / len(ttrs)


def assistant_tokens_for_mattr(messages: Any) -> list[str]:
    """Lowercase whitespace tokens for every assistant turn in a trajectory.

    Concatenated across turns; MATTR sliding window then runs over the
    resulting per-trajectory token sequence.
    """
    msgs = _decode_messages(messages)
    out: list[str] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        if m.get("role") != "assistant":
            continue
        content = m.get("content")
        if isinstance(content, str):
            out.extend(content.lower().split())
    return out


# ---------------------------------------------------------------------------
# Frame builder
# ---------------------------------------------------------------------------


def compute_diagnostics_frame(df: Any) -> Any:
    """Add D1–D4 + per-trajectory MATTR columns to a trajectory frame.

    Pure-pandas; the SimHealthBundle calls this and then aggregates.
    Returns a copy so callers don't mutate their input.

    Required column: ``conversation_messages``. Missing column → ValueError.
    """
    import pandas as pd  # lazy import — keeps top-level free for tests

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"compute_diagnostics_frame expects a pandas DataFrame, got {type(df).__name__}")
    if "conversation_messages" not in df.columns:
        raise ValueError(
            "compute_diagnostics_frame: input must have a 'conversation_messages' column (the trajectory store schema)"
        )
    out = df.copy()
    out["d1_front_loading"] = out["conversation_messages"].apply(compute_front_loading_ratio)
    out["d2_polite_fraction"] = out["conversation_messages"].apply(compute_polite_turn_fraction)
    out["d3_verbosity_cv"] = out["conversation_messages"].apply(compute_verbosity_cv)
    out["d4_frustration_markers"] = out["conversation_messages"].apply(compute_frustration_marker_fraction)
    out["mattr_assistant"] = out["conversation_messages"].apply(
        lambda raw: compute_mattr(assistant_tokens_for_mattr(raw))
    )
    return out


# ---------------------------------------------------------------------------
# Response length by locale
# ---------------------------------------------------------------------------
#
# Equity diagnostic: does the model give shorter responses in non-English
# locales for matched workloads? The token side reads the simulator's
# already-recorded ``per_model_output_tokens["assistant_model"]`` from
# ``simulation_outcome``; the character side computes from the raw
# ``conversation_messages`` (more meaningful for non-Latin scripts where
# token-to-character ratios diverge sharply).


def _assistant_chars_per_turn(messages: Any) -> list[int]:
    """Character count of every assistant turn, in order. Empty / non-string
    contents contribute 0 (so they appear in the per-trajectory mean as a
    real 0, not as a missing value)."""
    msgs = _decode_messages(messages)
    out: list[int] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        if m.get("role") != "assistant":
            continue
        content = m.get("content")
        if isinstance(content, str):
            out.append(len(content))
        else:
            out.append(0)
    return out


def _outcome_assistant_tokens(raw: Any) -> int | None:
    """Pull total assistant-model output tokens from a simulation_outcome
    cell (JSON string or dict). Returns ``None`` if missing — the
    simulator only writes this when the assistant_model was actually
    invoked, so failed-startup trajectories may have no count."""
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            outcome = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
    elif isinstance(raw, dict):
        outcome = raw
    else:
        return None
    per_model = outcome.get("per_model_output_tokens")
    if not isinstance(per_model, dict):
        return None
    val = per_model.get("assistant_model")
    if isinstance(val, (int, float)):
        return int(val)
    return None


def _outcome_n_assistant_turns(raw: Any) -> int | None:
    """Pull ``per_model_calls["assistant_model"]`` from a simulation_outcome
    cell — this is the number of assistant turns the simulator actually
    invoked the model for. Used as a per-turn-token denominator."""
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            outcome = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
    elif isinstance(raw, dict):
        outcome = raw
    else:
        return None
    per_model = outcome.get("per_model_calls")
    if not isinstance(per_model, dict):
        return None
    val = per_model.get("assistant_model")
    if isinstance(val, (int, float)):
        return int(val)
    return None


def compute_response_length_by_locale(df: Any) -> list[dict[str, Any]]:
    """Aggregate assistant-response length per locale.

    Returns a list of dicts, one per locale present in ``df``, with
    fields:

    - ``locale``
    - ``n_trajectories``
    - ``mean_chars_per_turn``     — averaged across all assistant turns in the locale
    - ``median_chars_per_turn``
    - ``p95_chars_per_turn``
    - ``mean_tokens_per_turn``    — from ``per_model_output_tokens["assistant_model"]`` / ``per_model_calls["assistant_model"]``
    - ``median_tokens_per_turn``
    - ``p95_tokens_per_turn``
    - ``mean_total_tokens_per_trajectory``  — convenience: total assistant tokens / trajectory

    Required columns: ``locale``, ``conversation_messages``,
    ``simulation_outcome``. Missing column → ``ValueError``.

    Trajectories whose ``simulation_outcome`` lacks ``per_model_output_tokens`` (e.g.,
    the assistant_model was never invoked) contribute 0 to per-trajectory counts but
    are still counted in ``n_trajectories``. The character side is computed from
    ``conversation_messages`` directly so it works even when the simulator's
    token-counting path failed.
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"compute_response_length_by_locale expects a pandas DataFrame, got {type(df).__name__}")
    for col in ("locale", "conversation_messages", "simulation_outcome"):
        if col not in df.columns:
            raise ValueError(
                f"compute_response_length_by_locale: missing required column "
                f"{col!r} (expected on the trajectory store schema)"
            )

    # Per-trajectory expansion: every assistant turn contributes one
    # row to the chars-per-turn distribution; tokens are averaged
    # within the trajectory (we don't have per-turn token counts in
    # the simulation_outcome — only the totals).
    char_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    n_per_locale: dict[str, int] = {}

    for _, row in df.iterrows():
        locale = row.get("locale")
        if not isinstance(locale, str) or not locale:
            continue
        n_per_locale[locale] = n_per_locale.get(locale, 0) + 1

        # Char distribution: every assistant turn is one observation.
        chars = _assistant_chars_per_turn(row.get("conversation_messages"))
        for n_chars in chars:
            char_rows.append({"locale": locale, "n_chars": n_chars})

        # Token distribution: total assistant tokens / number of
        # assistant turns (per the simulator's recorded call count).
        # If either is missing we skip this trajectory's contribution
        # to the token side rather than guess.
        total_tokens = _outcome_assistant_tokens(row.get("simulation_outcome"))
        n_calls = _outcome_n_assistant_turns(row.get("simulation_outcome"))
        if total_tokens is not None and n_calls is not None and n_calls > 0:
            mean_tokens_per_turn_in_traj = total_tokens / n_calls
            token_rows.append(
                {
                    "locale": locale,
                    "tokens_per_turn": mean_tokens_per_turn_in_traj,
                    "total_tokens": total_tokens,
                }
            )

    chars_df = pd.DataFrame(char_rows)
    tokens_df = pd.DataFrame(token_rows)

    out: list[dict[str, Any]] = []
    for locale in sorted(n_per_locale):
        loc_chars = chars_df[chars_df["locale"] == locale] if len(chars_df) else chars_df
        loc_tokens = tokens_df[tokens_df["locale"] == locale] if len(tokens_df) else tokens_df

        record: dict[str, Any] = {
            "locale": locale,
            "n_trajectories": n_per_locale[locale],
        }
        if len(loc_chars) > 0:
            record["mean_chars_per_turn"] = round(float(loc_chars["n_chars"].mean()), 2)
            record["median_chars_per_turn"] = float(loc_chars["n_chars"].median())
            record["p95_chars_per_turn"] = float(loc_chars["n_chars"].quantile(0.95))
        else:
            record["mean_chars_per_turn"] = None
            record["median_chars_per_turn"] = None
            record["p95_chars_per_turn"] = None

        if len(loc_tokens) > 0:
            record["mean_tokens_per_turn"] = round(float(loc_tokens["tokens_per_turn"].mean()), 2)
            record["median_tokens_per_turn"] = round(float(loc_tokens["tokens_per_turn"].median()), 2)
            record["p95_tokens_per_turn"] = round(float(loc_tokens["tokens_per_turn"].quantile(0.95)), 2)
            record["mean_total_tokens_per_trajectory"] = round(float(loc_tokens["total_tokens"].mean()), 2)
        else:
            record["mean_tokens_per_turn"] = None
            record["median_tokens_per_turn"] = None
            record["p95_tokens_per_turn"] = None
            record["mean_total_tokens_per_trajectory"] = None

        out.append(record)

    return out
