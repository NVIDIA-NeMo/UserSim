# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trajectory-id-based idempotency helpers for resumable simulator runs.

The simulator's discipline: a ``trajectory_id`` whose row already
exists in the output dataset is skipped on re-runs, so resumed runs
only fill missing rows.

This module supplies the candidate-filtering helpers; the actual
on-disk lookup is in :mod:`usersim.engine.core.storage`
(``existing_trajectory_ids`` walks either a single-file parquet or
a Hive-partitioned dataset directory). The simulator-driver glue
(notebook or CLI) loads the existing dataset, calls
``existing_trajectory_ids``, filters its seed dataset to skip
duplicates, and runs only the remainder.
"""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

# Re-export for back-compat — most callers import from here.
from usersim.engine.core.storage import existing_trajectory_ids  # noqa: F401

logger = logging.getLogger("usersim.engine")


def is_duplicate_trajectory_id(candidate: str, existing: set[str] | Iterable[str]) -> bool:
    """O(1) check that ``candidate`` is already present in ``existing``.

    Accepts either a set (preferred) or any iterable; the latter
    materializes once if needed. Centralised so all callers share the
    same string-coercion semantics.
    """
    if not isinstance(existing, set):
        existing = set(existing)
    return str(candidate) in existing


def filter_to_missing_trajectory_ids(
    candidates: Sequence[str],
    existing: set[str] | Iterable[str],
) -> list[str]:
    """Return the subset of ``candidates`` not in ``existing``, order-preserved.

    Used by the simulator-driver to filter a planned batch down to only
    the trajectories that have not yet been written. Order-preserving
    so the resumed run keeps the same scheduling as a fresh run.
    """
    if not isinstance(existing, set):
        existing = set(existing)
    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        s = str(c)
        if s in existing or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out
