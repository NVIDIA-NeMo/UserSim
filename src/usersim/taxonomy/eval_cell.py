# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure parsers for the trajectory-evaluator's wide ``assistant_eval`` cell.

The evaluator plugin emits one JSON cell per trajectory shaped like::

    {
      "envelope": {...},
      "axes":    {"<axis>": {"<judge_alias>": {"score": 1-5, "reasoning": ...}}},
      "scorers": {"<scorer>": {"scorer_kind": ..., "scores": {"<axis>": {"score": ...}}}},
      "skipped": false,
      "skipped_reason": null
    }

These helpers decode that cell and pull per-axis scores out of it. They
are deliberately dependency-free (``json`` + ``math`` only) so both the
capability dashboard (:mod:`reporting.capability_report`) and the
trajectory-selection layer (:mod:`selection`) can read eval cells through
one source of truth — keeping score extraction, scorer-state semantics,
and the rate-vs-1-5 scale convention from drifting between the two.

``reporting.capability_report`` imports these under their historical
private names; new code should import the public names directly.
"""

from __future__ import annotations

import json
import math
from typing import Any


def decode_cell(raw: Any) -> dict[str, Any]:
    """Decode a possibly-stringified JSON cell into a dict (``{}`` on failure)."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def scorer_state(block: Any) -> str:
    """Classify a scorer block as ``ok`` / ``error`` / ``not_applicable``.

    A scorer that ran but recorded an ``error`` string is ``error``,
    unless the error is the "no <...> scorer skipped" sentinel — that
    means the scorer legitimately did not apply to this row, which is
    ``not_applicable``. An empty/absent block is ``not_applicable``.
    """
    if not isinstance(block, dict) or not block:
        return "not_applicable"
    err = block.get("error")
    if isinstance(err, str) and err:
        if err.startswith("no ") and "scorer skipped" in err:
            return "not_applicable"
        return "error"
    return "ok"


def axis_scale(axis: str) -> str:
    """Return ``"rate"`` (0-1, higher=better) or ``"score_1_5"`` for an axis.

    Rate axes are deterministic-scorer outputs identifiable by their
    namespaced prefixes / ``_rate`` suffix; everything else is a 1-5
    judge axis.
    """
    if axis.startswith(("language.", "shape.", "refusal.")) or axis.endswith("_rate"):
        return "rate"
    return "score_1_5"


def normalize_axis_score(axis: str, score: float) -> float:
    """Map a raw axis score onto the 0-1 scale (1-5 axes divided by 5)."""
    if axis_scale(axis) == "score_1_5":
        return float(score) / 5.0
    return float(score)


def score_from_axis(axis_cell: Any) -> float | None:
    """Pull a numeric score from an axis/judge cell (``{"score": ...}`` or scalar)."""
    if isinstance(axis_cell, dict):
        val = axis_cell.get("score")
    else:
        val = axis_cell
    if isinstance(val, (int, float)) and not math.isnan(float(val)):
        return float(val)
    return None


def score_from_eval_cell(cell: dict[str, Any], axis: str) -> float | None:
    """Best-effort score for ``axis``: judge-ensemble mean, then scorer blocks."""
    # First check universal axis results: axis -> judge_alias -> cell.
    axis_block = (cell.get("axes") or {}).get(axis)
    if isinstance(axis_block, dict):
        scores = [score_from_axis(v) for v in axis_block.values() if isinstance(v, dict)]
        scores = [s for s in scores if s is not None]
        if scores:
            return sum(scores) / len(scores)
    # Then check scorer blocks: scorers -> name -> scores -> axis.
    for block in (cell.get("scorers") or {}).values():
        if not isinstance(block, dict) or scorer_state(block) != "ok":
            continue
        score = score_from_axis((block.get("scores") or {}).get(axis))
        if score is not None:
            return score
    return None


def score_for_source(
    cell: dict[str, Any],
    scorer: str | None,
    axis: str,
) -> float | None:
    """Score for ``axis`` from a specific source (``None`` scorer = judge axes)."""
    if scorer is None:
        axis_block = (cell.get("axes") or {}).get(axis)
        if isinstance(axis_block, dict):
            scores = [score_from_axis(v) for v in axis_block.values() if isinstance(v, dict)]
            scores = [s for s in scores if s is not None]
            if scores:
                return sum(scores) / len(scores)
        return None
    block = (cell.get("scorers") or {}).get(scorer)
    if not isinstance(block, dict) or scorer_state(block) != "ok":
        return None
    return score_from_axis((block.get("scores") or {}).get(axis))
