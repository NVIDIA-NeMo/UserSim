# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for core/idempotency.py — trajectory_id-based resumability.

These tests pin the helper contract that the simulator-driver glue
(notebook / CLI) consumes for resumable runs.

Covers:
- ``existing_trajectory_ids`` returns an empty set for a missing path
  (the common fresh-run case).
- ``existing_trajectory_ids`` returns an empty set when the parquet
  exists but lacks the ``trajectory_id`` column (must not crash the
  loader).
- ``existing_trajectory_ids`` returns a populated set when the
  column is present.
- ``is_duplicate_trajectory_id`` is correct for present / absent ids,
  and accepts both sets and arbitrary iterables.
- ``filter_to_missing_trajectory_ids`` preserves order, drops dupes
  in both ``existing`` and within the candidate list itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from usersim.engine.core.idempotency import (
    existing_trajectory_ids,
    filter_to_missing_trajectory_ids,
    is_duplicate_trajectory_id,
)

pd = pytest.importorskip("pandas")


# ── existing_trajectory_ids ────────────────────────────────────────────


class TestExistingTrajectoryIds:
    def test_missing_file_returns_empty_set(self, tmp_path: Path) -> None:
        result = existing_trajectory_ids(tmp_path / "nope.parquet")
        assert result == set()

    def test_parquet_without_trajectory_id_column_returns_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "no_traj_id.parquet"
        pd.DataFrame({"some_other_col": ["a", "b"]}).to_parquet(path)
        assert existing_trajectory_ids(path) == set()

    def test_returns_populated_set(self, tmp_path: Path) -> None:
        path = tmp_path / "trajs.parquet"
        ids = ["abc1234567890def", "deadbeefcafe1234", "feedfacecafe5678"]
        pd.DataFrame({"trajectory_id": ids, "other": [1, 2, 3]}).to_parquet(path)
        assert existing_trajectory_ids(path) == set(ids)

    def test_drops_nulls(self, tmp_path: Path) -> None:
        path = tmp_path / "with_nulls.parquet"
        pd.DataFrame({"trajectory_id": ["abc1234567890def", None, "feedfacecafe5678"]}).to_parquet(path)
        result = existing_trajectory_ids(path)
        assert result == {"abc1234567890def", "feedfacecafe5678"}


# ── is_duplicate_trajectory_id ─────────────────────────────────────────


class TestIsDuplicate:
    def test_present_returns_true(self) -> None:
        existing = {"abc", "def"}
        assert is_duplicate_trajectory_id("abc", existing) is True

    def test_absent_returns_false(self) -> None:
        existing = {"abc", "def"}
        assert is_duplicate_trajectory_id("xyz", existing) is False

    def test_accepts_iterable(self) -> None:
        # Not just sets — generators, lists, etc.
        assert is_duplicate_trajectory_id("abc", iter(["abc", "def"])) is True
        assert is_duplicate_trajectory_id("xyz", ["abc", "def"]) is False

    def test_string_coercion(self) -> None:
        # Whatever we hand in, we coerce to str for the membership test.
        existing = {"123"}
        assert is_duplicate_trajectory_id(123, existing) is True


# ── filter_to_missing_trajectory_ids ───────────────────────────────────


class TestFilterToMissing:
    def test_drops_existing(self) -> None:
        existing = {"a", "b"}
        candidates = ["a", "b", "c", "d"]
        assert filter_to_missing_trajectory_ids(candidates, existing) == ["c", "d"]

    def test_preserves_order(self) -> None:
        existing: set[str] = set()
        candidates = ["d", "b", "c", "a"]
        assert filter_to_missing_trajectory_ids(candidates, existing) == [
            "d",
            "b",
            "c",
            "a",
        ]

    def test_drops_intra_candidate_duplicates(self) -> None:
        # Two seeds expanding to the same trajectory_id should only run
        # once, not twice.
        existing: set[str] = set()
        candidates = ["a", "b", "a", "c", "b"]
        assert filter_to_missing_trajectory_ids(candidates, existing) == [
            "a",
            "b",
            "c",
        ]

    def test_empty_returns_empty(self) -> None:
        assert filter_to_missing_trajectory_ids([], set()) == []

    def test_all_duplicates(self) -> None:
        existing = {"a", "b", "c"}
        assert filter_to_missing_trajectory_ids(["a", "b", "c"], existing) == []
