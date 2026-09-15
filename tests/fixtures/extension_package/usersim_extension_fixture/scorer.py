# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from usersim.engine.evaluator.scorers import register_scorer


def score(row, models):
    return {"fixture_axis": 1.0}


register_scorer("fixture_scorer", score)
