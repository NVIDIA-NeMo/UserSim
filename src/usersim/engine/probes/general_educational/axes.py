# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The axes the evaluator's judges score ``general_educational`` rows on, beside the universal ones.

Definitions only: the evaluator imports this module, and it imports no probe code.
"""

from __future__ import annotations

from data_designer.config.column_configs import Score

JUDGE_AXES: dict[str, list[Score]] = {
    "general_educational": [
        Score(
            name="pedagogical_quality",
            description=(
                "Did the response teach effectively at the right level? "
                "Consider explanation clarity, scaffolding, encouragement, "
                "and adaptation to the learner's level."
            ),
            options={
                1: "Very poor — confusing, condescending, or ineffective teaching",
                2: "Poor — unclear explanations or wrong level for the learner",
                3: "Adequate — understandable but lacks scaffolding or engagement",
                4: "Good — clear, appropriately leveled with minor pedagogical gaps",
                5: "Excellent — masterful teaching adapted to the learner",
            },
        ),
    ],
}
