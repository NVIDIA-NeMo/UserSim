# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Shared analysis utilities used by both the notebook and tests.

Extracted here to avoid duplication and ensure consistent behavior.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd


def extract_eval_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Extract per-axis evaluation scores into a flat DataFrame for analysis.

    Reads the ``eval_scores`` column (JSON string or dict), unpacks
    each axis score into ``score_{axis}`` columns, and preserves
    demographic columns for stratified analysis.

    Returns an empty DataFrame if no valid scores are found.
    """
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        scores_raw = row.get("eval_scores")
        if not scores_raw:
            continue
        try:
            scores = json.loads(scores_raw) if isinstance(scores_raw, str) else scores_raw
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(scores, dict):
            continue

        # Stratification tag set. ``persona_country`` is universally
        # populated; falls back to ``persona_region`` for parquets
        # that don't carry the new column, so cohort reports group
        # correctly on either schema.
        entry = {
            "locale": row.get("locale", ""),
            "probe_type": row.get("probe_type", ""),
            "persona_age_bin": row.get("persona_age_bin", ""),
            "persona_education_level": row.get("persona_education_level", ""),
            "persona_occupation": row.get("persona_occupation", ""),
            "persona_country": (row.get("persona_country") or row.get("persona_region", "")),
        }
        for axis, val in scores.items():
            if val is not None and isinstance(val, dict):
                entry[f"score_{axis}"] = val.get("score")
            else:
                entry[f"score_{axis}"] = None
        rows.append(entry)

    return pd.DataFrame(rows) if rows else pd.DataFrame()
