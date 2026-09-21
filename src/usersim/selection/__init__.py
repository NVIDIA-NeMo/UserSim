# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Trajectory selection: curate high-quality trajectories from a run.

Stage between evaluation and training-data extraction. Joins a run's
trajectories + evaluations, applies a configurable :class:`SelectionProfile`
of quality gates (reusing the capability registry's structure with our own
thresholds), and writes the passing subset as a curated, partitioned parquet.

The CLI (``usersim select``) and ``notebooks/03_extract_training_data.ipynb``
both drive this package, so the two flows stay in lockstep — exactly as
``cli/_pipeline.py`` keeps simulation + evaluation aligned.

Typical use::

    from usersim.selection import load_run, select_trajectories, write_curated_dataset
    from usersim.selection.profiles import GENEROUS

    traj_df, eval_df, run_id = load_run("output/trajectories", "output/evaluations")
    result = select_trajectories(traj_df, eval_df, profile=GENEROUS)
    write_curated_dataset(
        result.curated, "output/curated", run_id, GENEROUS, summary=result.summary
    )
"""

from __future__ import annotations

from usersim.selection.engine import (
    ADDED_COLUMNS,
    SelectionResult,
    SelectionSummary,
    SelectionVerdict,
    evaluate_row,
    select_trajectories,
)
from usersim.selection.io import (
    MANIFEST_NAME,
    build_manifest,
    load_run,
    load_runs,
    parse_repo_id,
    push_to_hf,
    resolve_generation_model,
    write_curated_dataset,
)
from usersim.selection.profiles import (
    GENEROUS,
    STRICT,
    AxisGate,
    SelectionProfile,
    applicable_gates,
    available_profiles,
    get_profile,
)
from usersim.selection.schemas import (
    available_schemas,
    project,
    resolve_columns,
)

__all__ = [
    "ADDED_COLUMNS",
    "MANIFEST_NAME",
    "AxisGate",
    "GENEROUS",
    "STRICT",
    "SelectionProfile",
    "SelectionResult",
    "SelectionSummary",
    "SelectionVerdict",
    "applicable_gates",
    "available_profiles",
    "available_schemas",
    "build_manifest",
    "resolve_generation_model",
    "evaluate_row",
    "get_profile",
    "load_run",
    "load_runs",
    "parse_repo_id",
    "project",
    "push_to_hf",
    "resolve_columns",
    "select_trajectories",
    "write_curated_dataset",
]
