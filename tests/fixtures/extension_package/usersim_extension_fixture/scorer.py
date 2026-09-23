# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

from usersim.engine.evaluator.scorers import register_scorer


async def score(row, models):
    return {"fixture_axis": 1.0}


register_scorer("fixture_scorer", score)
