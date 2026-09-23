# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Response-shape scorer — deterministic, no LLM call.

Checks for cheap structural failure modes that the LLM-judge
``response_conciseness`` axis is bad at catching reliably (judges are
inconsistent on "is this too formatted for chat?" — humans are too,
which is why we want a deterministic anchor):

================================================ ==========================
axis                                             interpretation (1.0 best)
================================================ ==========================
``shape.empty_or_trivial_rate``                  reversed: 1.0 = no empty/trivial turns
``shape.over_formatted_rate``                    reversed: 1.0 = no chat-inappropriate formatting
``shape.truncation_suspicion_rate``              reversed: 1.0 = no truncation-shaped tells
================================================ ==========================

All three axes are reversed-scale (1.0 = best, low = bad) so the
scorer's ``status_proposal`` and the bundle's per-axis aggregation
follow the same higher-is-better convention as the other deterministic
axes. The per-turn details carry the raw flag bits so a reviewer can
drill in.

Empty / trivial = empty string, whitespace-only, single emoji, or
strings shorter than ``MIN_TRIVIAL_CHARS=3`` after whitespace strip
(catches "ok", "...", and the like).

Over-formatted = at least one of: 2+ markdown headers (``##``),
horizontal rules (``---``), pipe-delimited tables (``|``-style row),
6+ numbered list items, 6+ bulleted list items, fenced code blocks
spanning > 5 lines. Calibrated for **chat reply length** — a
multi-section essay-shaped response is the failure mode this catches.

Truncation suspicion = the response ends in a way that suggests the
generator got cut off mid-thought: no terminal punctuation, unmatched
opening bracket / parenthesis / quote, or trailing whitespace after a
mid-sentence fragment.

These checks are language-aware where the rules differ. Japanese
sentence terminators (。!?) are recognised alongside Latin
``.!?…``; Hindi uses ``।`` (devanagari danda) plus the Latin set.

``status_proposal``: ``False`` if any of the three rates is below
``1.0`` (strict). The bundle layer applies the configurable threshold.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from usersim.engine.evaluator.scorers import register_scorer

logger = logging.getLogger("usersim.engine")


SHAPE_AXES: tuple[str, ...] = (
    "shape.empty_or_trivial_rate",
    "shape.over_formatted_rate",
    "shape.truncation_suspicion_rate",
)


# ---------------------------------------------------------------------------
# Tunable thresholds (deliberately conservative)
# ---------------------------------------------------------------------------

# Below this character count after whitespace-strip, a response is
# considered trivial / non-substantive. "ok", "yes", "..." all fall
# under this. Two-character responses might be "Hi" or "OK"; we don't
# want to over-flag, so 3 is the floor.
MIN_TRIVIAL_CHARS: int = 3

# 2+ markdown headers (``##``) on their own line is the load-bearing
# signal. Single header is acceptable for a response that genuinely
# has structure.
MAX_HEADERS: int = 1

# Numbered list items above this count = chat-inappropriate
# enumeration (an exhaustive 8-item how-to should be a doc, not a
# chat reply).
MAX_LIST_ITEMS: int = 5

# Code blocks spanning more than this many lines hint at a
# document-shaped response. Code blocks themselves are fine; long
# ones are the signal.
MAX_CODE_BLOCK_LINES: int = 5


# Sentence terminators we accept as "this turn ended on a clean
# boundary". Combines Latin terminators, Japanese full stop and
# question/exclamation marks, and the Devanagari danda used in Hindi
# and other Indian scripts.
_SENTENCE_TERMINATORS: frozenset[str] = frozenset([".", "!", "?", "…", "。", "！", "？", "।"])

# Closing-side characters that we expect to be balanced 1:1 with their
# opening counterparts. Imbalance suggests truncation.
_PAIR_OPENERS: dict[str, str] = {"(": ")", "[": "]", "{": "}", "「": "」", "『": "』"}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def score_response_shape_trajectory(
    trajectory: dict[str, Any],
    models: dict[str, Any],  # unused — deterministic scorer, no LLM call
) -> dict[str, Any]:
    """Compute per-trajectory shape rates over assistant turns."""
    assistant_messages = _extract_assistant_messages(
        trajectory.get("conversation_messages"),
    )
    if not assistant_messages:
        return _noop(
            error=("no assistant turns in conversation_messages — response_shape scorer skipped"),
        )

    n = len(assistant_messages)
    n_trivial = 0
    n_over_formatted = 0
    n_truncation_suspicion = 0
    per_turn: list[dict[str, Any]] = []

    for idx, content in enumerate(assistant_messages):
        text = content or ""
        is_trivial = _is_empty_or_trivial(text)
        is_over_formatted = (not is_trivial) and _is_over_formatted(text)
        is_truncation_suspicion = (not is_trivial) and _looks_truncated(text)
        if is_trivial:
            n_trivial += 1
        if is_over_formatted:
            n_over_formatted += 1
        if is_truncation_suspicion:
            n_truncation_suspicion += 1
        per_turn.append(
            {
                "turn_idx": idx,
                "n_chars": len(text),
                "empty_or_trivial": is_trivial,
                "over_formatted": is_over_formatted,
                "truncation_suspicion": is_truncation_suspicion,
            }
        )

    # Reversed-scale rates: 1.0 = no failures detected.
    empty_rate = 1.0 - (n_trivial / n)
    over_formatted_rate = 1.0 - (n_over_formatted / n)
    truncation_rate = 1.0 - (n_truncation_suspicion / n)

    scores = {
        "shape.empty_or_trivial_rate": _score_cell(
            round(empty_rate, 4),
            n,
            _summarize_shape_axis(per_turn, n, "empty_or_trivial", "empty/trivial"),
        ),
        "shape.over_formatted_rate": _score_cell(
            round(over_formatted_rate, 4),
            n,
            _summarize_shape_axis(per_turn, n, "over_formatted", "over-formatted (chat-inappropriate)"),
        ),
        "shape.truncation_suspicion_rate": _score_cell(
            round(truncation_rate, 4),
            n,
            _summarize_shape_axis(per_turn, n, "truncation_suspicion", "suspected truncation"),
        ),
    }
    status_proposal = all(c["score"] is not None and c["score"] >= 1.0 for c in scores.values())

    return {
        "scorer_kind": "deterministic",
        "n_assistant_turns": n,
        "scores": scores,
        "per_turn_details": per_turn,
        "status_proposal": status_proposal,
    }


# ---------------------------------------------------------------------------
# Per-turn classifiers
# ---------------------------------------------------------------------------


def _is_empty_or_trivial(text: str) -> bool:
    """True if the turn is empty, whitespace-only, contains no
    alphanumeric content (pure punctuation / emoji / symbols), or is
    shorter than :data:`MIN_TRIVIAL_CHARS` after stripping whitespace.

    Catches: ``""``, ``"   "``, ``"ok"``, ``"hi"``, ``"..."``,
    ``"👍"``, ``"✓"``, ``"!!!"``, and pure-emoji strings of any
    length. Substantive replies that happen to contain emoji
    (``"Sure, here's the answer 👍"``) are NOT flagged because they
    contain alphanumeric letters.
    """
    if not text:
        return True
    stripped = text.strip()
    if not stripped:
        return True
    if len(stripped) < MIN_TRIVIAL_CHARS:
        return True
    # Pure punctuation / emoji / symbols — no alphanumeric content
    # anywhere. Length-independent: a long string of only emoji is
    # also non-substantive.
    if not any(c.isalnum() for c in stripped):
        return True
    return False


# Pre-compiled regexes used by ``_is_over_formatted``.
_RE_HEADER = re.compile(r"^\s{0,3}#{1,6}\s+\S", re.MULTILINE)
_RE_HORIZONTAL_RULE = re.compile(r"^\s{0,3}(?:-{3,}|\*{3,}|_{3,})\s*$", re.MULTILINE)
_RE_PIPE_TABLE_ROW = re.compile(r"^\s*\|.+\|\s*$", re.MULTILINE)
_RE_NUMBERED_ITEM = re.compile(r"^\s{0,3}\d+[.)]\s+\S", re.MULTILINE)
_RE_BULLET_ITEM = re.compile(r"^\s{0,3}[-*+]\s+\S", re.MULTILINE)
_RE_FENCED_CODE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


def _is_over_formatted(text: str) -> bool:
    """Heuristic chat-formatting check.

    Triggers if any one of: more than :data:`MAX_HEADERS` markdown
    headers; any horizontal rule; any pipe-delimited table row; more
    than :data:`MAX_LIST_ITEMS` numbered or bulleted items; any fenced
    code block longer than :data:`MAX_CODE_BLOCK_LINES` lines.
    """
    if len(_RE_HEADER.findall(text)) > MAX_HEADERS:
        return True
    if _RE_HORIZONTAL_RULE.search(text):
        return True
    if _RE_PIPE_TABLE_ROW.search(text):
        return True
    if len(_RE_NUMBERED_ITEM.findall(text)) > MAX_LIST_ITEMS:
        return True
    if len(_RE_BULLET_ITEM.findall(text)) > MAX_LIST_ITEMS:
        return True
    for match in _RE_FENCED_CODE.findall(text):
        if match.count("\n") + 1 > MAX_CODE_BLOCK_LINES:
            return True
    return False


def _looks_truncated(text: str) -> bool:
    """Heuristic mid-sentence-cutoff check.

    Triggers if the rstripped text either:
    - ends in a non-terminator non-closing character (mid-word / mid-clause);
    - has more openers than closers for any bracket/quote pair.
    """
    stripped = text.rstrip()
    if not stripped:
        return False
    last_char = stripped[-1]
    # If it doesn't end in a terminator and doesn't end in a closing
    # bracket / quote, we suspect truncation.
    closers = set(_PAIR_OPENERS.values()) | {'"', "'", "”", "’", "»"}
    ends_clean = (last_char in _SENTENCE_TERMINATORS) or (last_char in closers)
    if not ends_clean:
        return True
    # Bracket-pair imbalance check (cheap).
    for opener, closer in _PAIR_OPENERS.items():
        if stripped.count(opener) > stripped.count(closer):
            return True
    return False


# ---------------------------------------------------------------------------
# Helpers (shared shape with language_compliance)
# ---------------------------------------------------------------------------


def _extract_assistant_messages(raw: Any) -> list[str]:
    """Pull the assistant turns out of ``conversation_messages``.

    Skips tool-call envelopes (``tool_calls`` set) and empty /
    whitespace-only content — both are infrastructure messages with no
    natural-language shape to score and would otherwise be falsely
    counted as ``empty_or_trivial`` rate hits, dragging the cell to Gap
    even when the assistant's actual reply was fine.
    """
    messages = _normalize_conversation(raw)
    out: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        if m.get("role") != "assistant":
            continue
        if m.get("tool_calls"):
            continue
        content = m.get("content")
        if content is None:
            continue
        text = str(content)
        if not text.strip():
            continue
        out.append(text)
    return out


def _normalize_conversation(raw: Any) -> list[dict[str, Any]]:
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


def _summarize_shape_axis(
    per_turn: list[dict[str, Any]],
    n_turns: int,
    flag_key: str,
    label: str,
) -> str:
    """Narrate per-turn structural failures.

    For deterministic shape axes there is no LLM judge reasoning to
    surface — the only data we have is the per-turn flag bits. This
    helper produces a human-readable diagnostic that says *which* turns
    were flagged and how long they were, so a reviewer can jump straight
    to the offending turn in the conversation panel below instead of
    decoding ``truncation_suspicion/turns = 4/4`` style counters.
    """
    flagged = [d for d in per_turn if d.get(flag_key)]
    if not flagged:
        return f"All {n_turns} assistant turn(s) cleared the {label} check."
    chunks = ", ".join(f"#{d['turn_idx'] + 1} ({d['n_chars']} chars)" for d in flagged[:3])
    suffix = "" if len(flagged) <= 3 else f" + {len(flagged) - 3} more"
    return (
        f"{len(flagged)}/{n_turns} assistant turn(s) flagged for {label}. "
        f"Flagged turns: {chunks}{suffix}. (deterministic check, not an LLM judge.)"
    )


def _score_cell(score: float | None, n: int, reasoning: str) -> dict[str, Any]:
    return {
        "score": score,
        "reasoning": reasoning,
        "n": n,
    }


def _noop(*, error: str) -> dict[str, Any]:
    return {
        "scorer_kind": "deterministic",
        "n_assistant_turns": 0,
        "scores": {},
        "per_turn_details": [],
        "status_proposal": True,
        "error": error,
    }


# Auto-register on import.
register_scorer("response_shape", score_response_shape_trajectory)
