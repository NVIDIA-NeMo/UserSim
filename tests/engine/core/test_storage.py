# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``usersim.engine.core.storage``.

Covers:

- Round-trip write + read of partitioned datasets.
- Polymorphic reader (single-file vs partitioned-directory paths).
- Per-locale atomic writes (a crash mid-run leaves prior partitions).
- Schema unification across partitions written by different probes.
- ``existing_trajectory_ids`` walks both single-file and partitioned
  inputs.
- Selective column reads.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from usersim.engine.core.storage import (
    DEFAULT_PARTITION_COLS,
    LEGACY_RUN_ID,
    existing_trajectory_ids,
    is_partitioned_directory,
    latest_run_id,
    list_partition_keys,
    list_runs,
    materialize_to_temp_file,
    new_run_id,
    read_partitioned_dataset,
    resolve_run,
    resolve_run_or_raise,
    run_subroot,
    write_locale_partition,
    write_partitioned_dataset,
    write_run_locales_partition,
    write_run_partition,
)


def _trajectory_frame(*rows):
    return pd.DataFrame(list(rows))


# ─── Round-trip write + read ─────────────────────────────────────────


class TestRoundTrip:
    def test_basic_round_trip_preserves_rows(self, tmp_path: Path) -> None:
        df = _trajectory_frame(
            {
                "trajectory_id": "T-001",
                "locale": "en_US",
                "probe_family": "general_open_ended",
                "score": 4.5,
            },
            {
                "trajectory_id": "T-002",
                "locale": "pt_BR",
                "probe_family": "safety_chat_pressure",
                "score": 3.2,
            },
        )
        root = tmp_path / "trajs"
        write_partitioned_dataset(df, root)
        out = read_partitioned_dataset(root)
        assert sorted(out["trajectory_id"].tolist()) == ["T-001", "T-002"]
        assert set(out["locale"].astype(str)) == {"en_US", "pt_BR"}
        assert set(out["probe_family"].astype(str)) == {
            "general_open_ended",
            "safety_chat_pressure",
        }

    def test_list_columns_stay_readable_by_plain_pandas(self, tmp_path: Path) -> None:
        """A probe emitting a list column must not produce an unreadable file.

        Frames arriving here are pyarrow-backed, and pandas cannot parse the
        dtype string it records for a nested pyarrow column. Writing one
        verbatim yields a trajectory that scores as written and then raises
        ``TypeError`` on every read, including a read of unrelated scalar
        columns, because the failure is in the metadata.
        """
        import pyarrow as pa

        df = _trajectory_frame(
            {
                "trajectory_id": "T-001",
                "locale": "ja_JP",
                "probe_family": "sovereign_ai",
                "categories_explored": ["policy", "language"],
            }
        ).astype({"categories_explored": pd.ArrowDtype(pa.list_(pa.string()))})

        root = tmp_path / "trajs"
        write_partitioned_dataset(df, root)

        written = next(root.rglob("*.parquet"))
        # Selecting scalar columns is the read the evaluator performs.
        assert pd.read_parquet(written, columns=["trajectory_id"])["trajectory_id"].tolist() == ["T-001"]
        # The list survives as a sequence, and on-disk it is still a list type.
        assert list(pd.read_parquet(written)["categories_explored"][0]) == ["policy", "language"]
        assert pa.types.is_list(pq.read_schema(written).field("categories_explored").type)

        out = read_partitioned_dataset(root)
        assert list(out["categories_explored"][0]) == ["policy", "language"]

    def test_creates_hive_partition_directories(self, tmp_path: Path) -> None:
        df = _trajectory_frame(
            {
                "trajectory_id": "T-001",
                "locale": "en_US",
                "probe_family": "general_open_ended",
            }
        )
        root = tmp_path / "trajs"
        write_partitioned_dataset(df, root)
        # The output uses Hive layout: locale=X/probe_family=Y/...
        assert (root / "locale=en_US" / "probe_family=general_open_ended").is_dir()

    def test_partition_columns_live_only_in_directory_names(
        self,
        tmp_path: Path,
    ) -> None:
        """The leaf parquet files must NOT contain the partition columns.

        Partition values are encoded in the ``key=value/`` directory
        names; storing them again as row-group columns inside each file
        would duplicate every value across every row (a real bytes-on-
        disk waste, especially for low-cardinality columns like
        ``locale``). The reader reconstructs the partition columns
        from the directory paths transparently.

        We verify by reading the leaf parquet's physical schema via
        ``pq.read_metadata`` (which does NOT auto-promote partition
        columns the way ``pq.read_table`` does).
        """
        df = _trajectory_frame(
            {
                "trajectory_id": "T-001",
                "locale": "en_US",
                "probe_family": "general_open_ended",
                "score": 4.5,
            }
        )
        root = tmp_path / "trajs"
        write_partitioned_dataset(df, root)
        leaf = next(root.rglob("*.parquet"))
        physical_schema_names = pq.read_metadata(leaf).schema.names
        assert "locale" not in physical_schema_names
        assert "probe_family" not in physical_schema_names
        # Non-partition columns are still in the file.
        assert "trajectory_id" in physical_schema_names
        assert "score" in physical_schema_names

    def test_returns_resolved_root(self, tmp_path: Path) -> None:
        df = _trajectory_frame(
            {
                "trajectory_id": "T-001",
                "locale": "en_US",
                "probe_family": "general_open_ended",
            }
        )
        root = tmp_path / "trajs"
        returned = write_partitioned_dataset(df, root)
        assert returned == root.resolve()


# ─── Validation: missing partition columns ───────────────────────────


class TestPartitionColumnValidation:
    def test_raises_when_partition_column_missing(self, tmp_path: Path) -> None:
        # No probe_family column.
        df = _trajectory_frame({"trajectory_id": "T-001", "locale": "en_US"})
        with pytest.raises(ValueError, match="probe_family"):
            write_partitioned_dataset(df, tmp_path / "trajs")

    def test_default_partition_cols_are_locale_and_probe_family(self) -> None:
        assert DEFAULT_PARTITION_COLS == ("locale", "probe_family")


# ─── Polymorphic reader ──────────────────────────────────────────────


class TestPolymorphicRead:
    def test_reads_single_file_parquet(self, tmp_path: Path) -> None:
        # Single-file fixture (not partitioned). The reader auto-detects
        # via Path.is_file() and falls through to pq.read_table.
        df = _trajectory_frame(
            {
                "trajectory_id": "T-001",
                "locale": "en_US",
                "probe_family": "general_open_ended",
            }
        )
        path = tmp_path / "single.parquet"
        df.to_parquet(path, index=False)
        out = read_partitioned_dataset(path)
        assert len(out) == 1
        assert out["trajectory_id"].iloc[0] == "T-001"

    def test_reads_partitioned_directory(self, tmp_path: Path) -> None:
        df = _trajectory_frame(
            {
                "trajectory_id": "T-001",
                "locale": "en_US",
                "probe_family": "general_open_ended",
            },
            {
                "trajectory_id": "T-002",
                "locale": "ja_JP",
                "probe_family": "safety_chat_pressure",
            },
        )
        root = tmp_path / "trajs"
        write_partitioned_dataset(df, root)
        out = read_partitioned_dataset(root)
        assert len(out) == 2

    def test_raises_filenotfound_on_missing_path(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_partitioned_dataset(tmp_path / "does_not_exist")

    def test_columns_projection(self, tmp_path: Path) -> None:
        df = _trajectory_frame(
            {
                "trajectory_id": "T-001",
                "locale": "en_US",
                "probe_family": "general_open_ended",
                "extra_col": "junk",
            }
        )
        root = tmp_path / "trajs"
        write_partitioned_dataset(df, root)
        # Project to a subset of columns; locale + probe_family come
        # for free as Hive partition columns regardless.
        out = read_partitioned_dataset(root, columns=["trajectory_id"])
        assert "trajectory_id" in out.columns
        assert "extra_col" not in out.columns

    def test_recovers_from_cross_run_schema_drift(self, tmp_path: Path) -> None:
        """Two runs land in the same partition: the older one wrote
        ``side_channel_x`` as all-NaN (PyArrow infers ``double``);
        the newer one wrote it as a real string. The reader must
        recover and surface a single ``string``-typed column with
        the older run's cells null-filled.

        This is the storage-side analog of the bug class that
        manifested when the broken ``cli/simulate.py`` toolset path
        produced 17 failed ``tool_calling`` trajectories with
        side-channel columns left None (PyArrow inferred ``double``
        from the all-NaN column), and the subsequent healthy run
        wrote 17 trajectories with the same column populated as
        strings — and ``pa.unify_schemas`` rejected the partition
        with ``Unable to merge: Field side_channel_x has incompatible
        types: double vs string``. The defensive
        ``_unify_fragment_schemas_robust`` + per-fragment cast lets
        the read complete cleanly.
        """
        import pandas as pd

        # Older run: side_channel_x is all-NaN (object dtype with no
        # real string values, written as ``double`` by pyarrow's
        # NaN-inference behaviour).
        old_df = pd.DataFrame(
            [
                {
                    "trajectory_id": "OLD-001",
                    "locale": "en_US",
                    "probe_family": "tool_calling",
                    "side_channel_x": float("nan"),
                },
            ]
        )
        # Newer run: side_channel_x is a real string.
        new_df = pd.DataFrame(
            [
                {
                    "trajectory_id": "NEW-001",
                    "locale": "en_US",
                    "probe_family": "tool_calling",
                    "side_channel_x": "real_value",
                },
            ]
        )
        root = tmp_path / "trajs"
        write_partitioned_dataset(old_df, root)
        write_partitioned_dataset(new_df, root)

        out = read_partitioned_dataset(root)
        # Both rows surface, the column lives as a string-typed
        # column with the older row's cell as <NA>.
        assert len(out) == 2
        assert set(out["trajectory_id"].astype(str)) == {"OLD-001", "NEW-001"}
        assert "side_channel_x" in out.columns
        assert out.loc[out["trajectory_id"] == "NEW-001", "side_channel_x"].iloc[0] == "real_value"
        # The older row's cell is null (pandas-pyarrow ``<NA>``).
        assert out.loc[out["trajectory_id"] == "OLD-001", "side_channel_x"].isna().iloc[0]


# ─── Per-locale atomic writes ────────────────────────────────────────


class TestWriteLocalePartition:
    def test_writes_only_the_named_locale(self, tmp_path: Path) -> None:
        df = _trajectory_frame(
            {
                "trajectory_id": "T-001",
                "locale": "en_US",
                "probe_family": "general_open_ended",
            },
            {
                "trajectory_id": "T-002",
                "locale": "pt_BR",
                "probe_family": "safety_chat_pressure",
            },
        )
        root = tmp_path / "trajs"
        write_locale_partition(df, root, locale="en_US", run_id="test_run")
        out = read_partitioned_dataset(root)
        assert sorted(out["trajectory_id"].tolist()) == ["T-001"]
        # pt_BR partition was NOT written.
        assert not (root / "locale=pt_BR").exists()

    def test_writes_nothing_when_locale_absent(self, tmp_path: Path) -> None:
        df = _trajectory_frame(
            {
                "trajectory_id": "T-001",
                "locale": "en_US",
                "probe_family": "general_open_ended",
            }
        )
        root = tmp_path / "trajs"
        # Filtering by an absent locale produces no output but does
        # not raise — the helper short-circuits cleanly.
        write_locale_partition(df, root, locale="ja_JP", run_id="test_run")
        assert not (root / "locale=ja_JP").exists()

    def test_per_locale_writes_compose(self, tmp_path: Path) -> None:
        # Writing two locales sequentially should leave both partitions
        # on disk (this is the "per-locale atomic write" semantic the
        # simulator depends on for resumability).
        df = _trajectory_frame(
            {
                "trajectory_id": "T-001",
                "locale": "en_US",
                "probe_family": "general_open_ended",
            },
            {
                "trajectory_id": "T-002",
                "locale": "pt_BR",
                "probe_family": "general_open_ended",
            },
        )
        root = tmp_path / "trajs"
        write_locale_partition(df, root, locale="en_US", run_id="test_run")
        write_locale_partition(df, root, locale="pt_BR", run_id="test_run")
        out = read_partitioned_dataset(root)
        assert sorted(out["trajectory_id"].tolist()) == ["T-001", "T-002"]
        assert (root / "run=test_run" / "locale=en_US").is_dir()
        assert (root / "run=test_run" / "locale=pt_BR").is_dir()

    def test_raises_when_locale_column_missing(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "trajectory_id": ["T-001"],
                "probe_family": ["general_open_ended"],
            }
        )
        with pytest.raises(ValueError, match="locale"):
            write_locale_partition(df, tmp_path / "trajs", locale="en_US", run_id="test_run")


# ─── Schema unification across probes ────────────────────────────────


class TestSchemaUnion:
    def test_partitions_with_different_columns_union_on_read(
        self,
        tmp_path: Path,
    ) -> None:
        # Write two partitions whose probes wrote different
        # side-channel columns. The unified read should surface the
        # union with None-filled cells where the column did not exist.
        sov_df = _trajectory_frame(
            {
                "trajectory_id": "T-001",
                "locale": "en_US",
                "probe_family": "sov_ai_facts",
                "facts_probed": ["FACT-001"],
            }
        )
        write_partitioned_dataset(sov_df, tmp_path / "trajs")

        agentic_df = _trajectory_frame(
            {
                "trajectory_id": "T-002",
                "locale": "en_US",
                "probe_family": "safety_agentic",
                "action_request_id": "AR-001",
            }
        )
        write_partitioned_dataset(agentic_df, tmp_path / "trajs")

        out = read_partitioned_dataset(tmp_path / "trajs")
        assert len(out) == 2
        # Both probe-specific columns surface in the unified frame.
        assert "facts_probed" in out.columns
        assert "action_request_id" in out.columns
        # The sov_ai row has no action_request_id; the agentic row
        # has no facts_probed. Both should read as None / NaN /
        # missing — the exact semantics depend on the pyarrow type
        # mapping, but neither should raise.
        sov_row = out[out["trajectory_id"] == "T-001"].iloc[0]
        agentic_row = out[out["trajectory_id"] == "T-002"].iloc[0]
        # action_request_id is missing for the sov_ai row.
        assert pd.isna(sov_row["action_request_id"])
        # facts_probed is a list-of-string column; missing-cell
        # representation may be None / NA / empty list depending on
        # the pyarrow → pandas type mapping.
        assert (
            pd.isna(agentic_row["facts_probed"])
            or agentic_row["facts_probed"] is None
            or (isinstance(agentic_row["facts_probed"], list) and len(agentic_row["facts_probed"]) == 0)
        )


# ─── existing_trajectory_ids ─────────────────────────────────────────


class TestExistingTrajectoryIds:
    def test_returns_empty_set_for_missing_path(self, tmp_path: Path) -> None:
        assert existing_trajectory_ids(tmp_path / "missing") == set()

    def test_walks_single_file(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "trajectory_id": ["T-001", "T-002", "T-003"],
                "other": ["a", "b", "c"],
            }
        )
        path = tmp_path / "single.parquet"
        df.to_parquet(path, index=False)
        assert existing_trajectory_ids(path) == {"T-001", "T-002", "T-003"}

    def test_walks_partitioned_dataset(self, tmp_path: Path) -> None:
        df = _trajectory_frame(
            {
                "trajectory_id": "T-001",
                "locale": "en_US",
                "probe_family": "general_open_ended",
            },
            {
                "trajectory_id": "T-002",
                "locale": "pt_BR",
                "probe_family": "safety_chat_pressure",
            },
            {
                "trajectory_id": "T-003",
                "locale": "ja_JP",
                "probe_family": "sov_ai_facts",
            },
        )
        root = tmp_path / "trajs"
        write_partitioned_dataset(df, root)
        assert existing_trajectory_ids(root) == {"T-001", "T-002", "T-003"}

    def test_returns_empty_when_column_missing(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"some_other_col": ["a", "b"]})
        path = tmp_path / "no_id.parquet"
        df.to_parquet(path, index=False)
        assert existing_trajectory_ids(path) == set()

    def test_partitioned_resume_after_partial_write(
        self,
        tmp_path: Path,
    ) -> None:
        # Simulate a crash: only en_US's partition landed.
        df_en = _trajectory_frame(
            {
                "trajectory_id": "T-001",
                "locale": "en_US",
                "probe_family": "general_open_ended",
            }
        )
        root = tmp_path / "trajs"
        write_locale_partition(df_en, root, locale="en_US", run_id="test_run")
        # Resume sees the en_US trajectory but no pt_BR.
        ids = existing_trajectory_ids(root)
        assert ids == {"T-001"}


# ─── Auxiliary helpers ──────────────────────────────────────────────


class TestAuxiliaryHelpers:
    def test_is_partitioned_directory_true_for_dir(self, tmp_path: Path) -> None:
        df = _trajectory_frame(
            {
                "trajectory_id": "T-001",
                "locale": "en_US",
                "probe_family": "general_open_ended",
            }
        )
        root = tmp_path / "trajs"
        write_partitioned_dataset(df, root)
        assert is_partitioned_directory(root) is True

    def test_is_partitioned_directory_false_for_file(self, tmp_path: Path) -> None:
        path = tmp_path / "single.parquet"
        pd.DataFrame({"x": [1]}).to_parquet(path)
        assert is_partitioned_directory(path) is False

    def test_is_partitioned_directory_false_for_missing(self, tmp_path: Path) -> None:
        assert is_partitioned_directory(tmp_path / "absent") is False

    def test_list_partition_keys_returns_locales(self, tmp_path: Path) -> None:
        df = _trajectory_frame(
            {
                "trajectory_id": "T-001",
                "locale": "en_US",
                "probe_family": "general_open_ended",
            },
            {
                "trajectory_id": "T-002",
                "locale": "pt_BR",
                "probe_family": "general_open_ended",
            },
        )
        root = tmp_path / "trajs"
        write_partitioned_dataset(df, root)
        keys = list_partition_keys(root, partition_col="locale")
        assert sorted(keys) == ["en_US", "pt_BR"]

    def test_list_partition_keys_empty_for_missing(self, tmp_path: Path) -> None:
        assert list_partition_keys(tmp_path / "absent") == []

    def test_materialize_to_temp_file_round_trips(self, tmp_path: Path) -> None:
        df = _trajectory_frame(
            {
                "trajectory_id": "T-001",
                "locale": "en_US",
                "probe_family": "general_open_ended",
            },
            {
                "trajectory_id": "T-002",
                "locale": "pt_BR",
                "probe_family": "safety_chat_pressure",
            },
        )
        root = tmp_path / "trajs"
        write_partitioned_dataset(df, root)
        temp = materialize_to_temp_file(root)
        try:
            assert temp.exists()
            assert temp.suffix == ".parquet"
            # Re-read the materialized file via vanilla pyarrow to
            # confirm it is a single-file parquet (not a directory).
            table = pq.read_table(temp)
            assert sorted(table.column("trajectory_id").to_pylist()) == [
                "T-001",
                "T-002",
            ]
        finally:
            temp.unlink()


# ─── Run-partition layer ─────────────────────────────────────────────


class TestRunHelpers:
    """Cover the small helper API: new_run_id / list_runs /
    latest_run_id / resolve_run / run_subroot."""

    def test_new_run_id_is_epoch_seconds_string(self) -> None:
        rid = new_run_id(now=1714074853.5)
        assert rid == "1714074853"
        assert rid.isdigit()

    def test_new_run_id_default_is_current_epoch(self) -> None:
        import time

        before = int(time.time())
        rid = new_run_id()
        after = int(time.time())
        assert before <= int(rid) <= after

    def test_list_runs_empty_for_missing_root(self, tmp_path: Path) -> None:
        assert list_runs(tmp_path / "does_not_exist") == []

    def test_list_runs_empty_for_empty_root(self, tmp_path: Path) -> None:
        (tmp_path / "trajs").mkdir()
        assert list_runs(tmp_path / "trajs") == []

    def test_list_runs_returns_run_subdirs_oldest_first(
        self,
        tmp_path: Path,
    ) -> None:
        root = tmp_path / "trajs"
        root.mkdir()
        for rid in ["1714074853", "1700000000", "1800000000"]:
            (root / f"run={rid}").mkdir()
        # ``sorted(iterdir())`` yields the lexicographic order which
        # matches numeric order for same-length integer strings.
        assert list_runs(root) == ["1700000000", "1714074853", "1800000000"]

    def test_list_runs_returns_legacy_for_bare_layout(
        self,
        tmp_path: Path,
    ) -> None:
        """A directory with bare ``locale=*/`` subdirs (the older flat
        layout, pre-run-partitioning) surfaces as a single synthetic
        ``"legacy"`` run."""
        root = tmp_path / "trajs"
        (root / "locale=en_US" / "probe_family=tool_calling").mkdir(parents=True)
        assert list_runs(root) == [LEGACY_RUN_ID]

    def test_list_runs_prefers_real_runs_over_legacy(
        self,
        tmp_path: Path,
    ) -> None:
        """When both ``run=*`` and bare ``locale=*/`` subdirs coexist
        (a partial migration), the real runs are returned and the
        legacy bare layout is ignored — the migration helper is
        expected to either fold the bare data into a run or remove it."""
        root = tmp_path / "trajs"
        (root / "run=1714074853").mkdir(parents=True)
        (root / "locale=en_US").mkdir(parents=True)
        assert list_runs(root) == ["1714074853"]

    def test_latest_run_id_returns_largest(self, tmp_path: Path) -> None:
        root = tmp_path / "trajs"
        for rid in ["1714074853", "1800000000", "1700000000"]:
            (root / f"run={rid}").mkdir(parents=True)
        assert latest_run_id(root) == "1800000000"

    def test_latest_run_id_falls_back_to_legacy(self, tmp_path: Path) -> None:
        root = tmp_path / "trajs"
        (root / "locale=en_US").mkdir(parents=True)
        assert latest_run_id(root) == LEGACY_RUN_ID

    def test_resolve_run_default_is_latest(self, tmp_path: Path) -> None:
        root = tmp_path / "trajs"
        (root / "run=1714074853").mkdir(parents=True)
        assert resolve_run(root, None) == "1714074853"
        assert resolve_run(root, "latest") == "1714074853"

    def test_resolve_run_passes_through_explicit_id(
        self,
        tmp_path: Path,
    ) -> None:
        # Note: resolve_run does NOT validate; the reader does. This
        # lets the caller pin a future-but-not-yet-written run id
        # for write-side use.
        root = tmp_path / "trajs"
        (root / "run=1714074853").mkdir(parents=True)
        assert resolve_run(root, "1800000000") == "1800000000"

    def test_run_subroot_for_real_run(self, tmp_path: Path) -> None:
        assert run_subroot(tmp_path, "1714074853") == tmp_path / "run=1714074853"

    def test_run_subroot_for_legacy_returns_root_itself(
        self,
        tmp_path: Path,
    ) -> None:
        # The bare-layout legacy data lives directly under root, so
        # the "subroot" for the legacy sentinel is root itself.
        assert run_subroot(tmp_path, LEGACY_RUN_ID) == tmp_path


class TestRunScopedReadAndWrite:
    """End-to-end behaviour: write to a run, read back, scope across
    multiple runs, default-to-latest, cross-run reads with run column."""

    def test_write_locale_partition_scopes_to_run_id(
        self,
        tmp_path: Path,
    ) -> None:
        df = _trajectory_frame(
            {
                "trajectory_id": "T-001",
                "locale": "en_US",
                "probe_family": "tool_calling",
            }
        )
        root = tmp_path / "trajs"
        write_locale_partition(df, root, locale="en_US", run_id="1714074853")
        assert (root / "run=1714074853" / "locale=en_US" / "probe_family=tool_calling").is_dir()
        # The bare locale=*/ directory (legacy layout) is NOT created.
        assert not (root / "locale=en_US").is_dir()

    def test_read_default_picks_latest_run(self, tmp_path: Path) -> None:
        root = tmp_path / "trajs"
        old = _trajectory_frame(
            {
                "trajectory_id": "OLD",
                "locale": "en_US",
                "probe_family": "tool_calling",
            }
        )
        new = _trajectory_frame(
            {
                "trajectory_id": "NEW",
                "locale": "en_US",
                "probe_family": "tool_calling",
            }
        )
        write_locale_partition(old, root, locale="en_US", run_id="1700000000")
        write_locale_partition(new, root, locale="en_US", run_id="1800000000")
        df = read_partitioned_dataset(root)
        assert set(df["trajectory_id"].astype(str)) == {"NEW"}

    def test_read_pinned_run_id(self, tmp_path: Path) -> None:
        root = tmp_path / "trajs"
        old = _trajectory_frame(
            {
                "trajectory_id": "OLD",
                "locale": "en_US",
                "probe_family": "tool_calling",
            }
        )
        new = _trajectory_frame(
            {
                "trajectory_id": "NEW",
                "locale": "en_US",
                "probe_family": "tool_calling",
            }
        )
        write_locale_partition(old, root, locale="en_US", run_id="1700000000")
        write_locale_partition(new, root, locale="en_US", run_id="1800000000")
        df = read_partitioned_dataset(root, run="1700000000")
        assert set(df["trajectory_id"].astype(str)) == {"OLD"}

    def test_read_all_runs_includes_run_column(self, tmp_path: Path) -> None:
        root = tmp_path / "trajs"
        write_locale_partition(
            _trajectory_frame(
                {
                    "trajectory_id": "OLD",
                    "locale": "en_US",
                    "probe_family": "tool_calling",
                }
            ),
            root,
            locale="en_US",
            run_id="1700000000",
        )
        write_locale_partition(
            _trajectory_frame(
                {
                    "trajectory_id": "NEW",
                    "locale": "en_US",
                    "probe_family": "tool_calling",
                }
            ),
            root,
            locale="en_US",
            run_id="1800000000",
        )
        df = read_partitioned_dataset(root, run="all")
        assert "run" in df.columns
        # Both runs present, each tagged with its source id.
        assert sorted(df["trajectory_id"].astype(str).tolist()) == ["NEW", "OLD"]
        run_to_traj = dict(zip(df["trajectory_id"].astype(str), df["run"].astype(str)))
        assert run_to_traj["OLD"] == "1700000000"
        assert run_to_traj["NEW"] == "1800000000"

    def test_read_unknown_run_id_raises(self, tmp_path: Path) -> None:
        root = tmp_path / "trajs"
        write_locale_partition(
            _trajectory_frame(
                {
                    "trajectory_id": "T-001",
                    "locale": "en_US",
                    "probe_family": "tool_calling",
                }
            ),
            root,
            locale="en_US",
            run_id="1714074853",
        )
        with pytest.raises(FileNotFoundError, match="run="):
            read_partitioned_dataset(root, run="9999999999")

    def test_read_empty_root_returns_empty_dataframe(
        self,
        tmp_path: Path,
    ) -> None:
        # Empty root with no runs and no legacy layout should
        # short-circuit to an empty DataFrame so notebooks and CLI
        # entry points don't crash on a first-invocation setup.
        empty_root = tmp_path / "trajs"
        empty_root.mkdir()
        df = read_partitioned_dataset(empty_root)
        assert df.empty

    def test_legacy_bare_layout_readable_through_reader(
        self,
        tmp_path: Path,
    ) -> None:
        """Older datasets (no ``run=*`` partitions) keep working
        without manual migration."""
        root = tmp_path / "trajs"
        # Write the OLD way: directly under root, no run partition.
        legacy_df = _trajectory_frame(
            {
                "trajectory_id": "LEGACY-001",
                "locale": "en_US",
                "probe_family": "tool_calling",
            }
        )
        write_partitioned_dataset(legacy_df, root)
        df = read_partitioned_dataset(root)
        assert set(df["trajectory_id"].astype(str)) == {"LEGACY-001"}

    def test_existing_trajectory_ids_scopes_to_run(
        self,
        tmp_path: Path,
    ) -> None:
        root = tmp_path / "trajs"
        write_locale_partition(
            _trajectory_frame(
                {
                    "trajectory_id": "OLD",
                    "locale": "en_US",
                    "probe_family": "tool_calling",
                }
            ),
            root,
            locale="en_US",
            run_id="1700000000",
        )
        write_locale_partition(
            _trajectory_frame(
                {
                    "trajectory_id": "NEW",
                    "locale": "en_US",
                    "probe_family": "tool_calling",
                }
            ),
            root,
            locale="en_US",
            run_id="1800000000",
        )
        # Default = latest = NEW only.
        assert existing_trajectory_ids(root) == {"NEW"}
        # Pinned to the old run.
        assert existing_trajectory_ids(root, run="1700000000") == {"OLD"}
        # All runs.
        assert existing_trajectory_ids(root, run="all") == {"OLD", "NEW"}


class TestResolveRunOrRaise:
    """The friendly-error variant used by notebook + CLI entry points."""

    def test_resolves_latest_when_run_is_none(self, tmp_path: Path) -> None:
        root = tmp_path / "trajs"
        (root / "run=1714074853").mkdir(parents=True)
        assert resolve_run_or_raise(root, None) == "1714074853"

    def test_resolves_latest_when_run_is_latest_string(self, tmp_path: Path) -> None:
        root = tmp_path / "trajs"
        (root / "run=1714074853").mkdir(parents=True)
        assert resolve_run_or_raise(root, "latest") == "1714074853"

    def test_passes_through_all(self, tmp_path: Path) -> None:
        root = tmp_path / "trajs"
        (root / "run=1714074853").mkdir(parents=True)
        # `all` is sentinel; no existence check.
        assert resolve_run_or_raise(root, "all") == "all"

    def test_passes_through_existing_literal_id(self, tmp_path: Path) -> None:
        root = tmp_path / "trajs"
        (root / "run=1714074853").mkdir(parents=True)
        assert resolve_run_or_raise(root, "1714074853") == "1714074853"

    def test_coerces_int_typed_run_id(self, tmp_path: Path) -> None:
        """Common notebook foot-gun: ``RUN_ID = 1714074853`` (no quotes).
        Should be coerced to the string form, not raise."""
        root = tmp_path / "trajs"
        (root / "run=1714074853").mkdir(parents=True)
        assert resolve_run_or_raise(root, 1714074853) == "1714074853"

    def test_raises_with_label_when_root_empty(self, tmp_path: Path) -> None:
        root = tmp_path / "trajs"  # never created
        with pytest.raises(FileNotFoundError, match="SIM_RUN_ID"):
            resolve_run_or_raise(root, None, label="SIM_RUN_ID")

    def test_raises_with_available_list_on_missing_id(self, tmp_path: Path) -> None:
        root = tmp_path / "trajs"
        (root / "run=1714074853").mkdir(parents=True)
        (root / "run=1800000000").mkdir(parents=True)
        with pytest.raises(FileNotFoundError) as exc_info:
            resolve_run_or_raise(root, "9999999999", label="REPORT_RUN_ID")
        msg = str(exc_info.value)
        assert "REPORT_RUN_ID='9999999999'" in msg
        assert "1714074853" in msg and "1800000000" in msg


class TestWriteRunPartition:
    """Round-trip + overwrite semantics for the run-scoped write helper."""

    def test_round_trip(self, tmp_path: Path) -> None:
        root = tmp_path / "evals"
        df = pd.DataFrame(
            [
                {"trajectory_id": "T1", "locale": "en_US", "probe_family": "tool_calling", "score": 0.9},
            ]
        )
        subroot = write_run_partition(df, root, "1714074853")
        assert subroot == root / "run=1714074853"
        roundtripped = read_partitioned_dataset(root, run="1714074853")
        assert set(roundtripped["trajectory_id"]) == {"T1"}

    def test_overwrite_replaces_existing_run(self, tmp_path: Path) -> None:
        root = tmp_path / "evals"
        write_run_partition(
            pd.DataFrame(
                [
                    {"trajectory_id": "OLD", "locale": "en_US", "probe_family": "tool_calling"},
                ]
            ),
            root,
            "1714074853",
        )
        write_run_partition(
            pd.DataFrame(
                [
                    {"trajectory_id": "NEW", "locale": "en_US", "probe_family": "tool_calling"},
                ]
            ),
            root,
            "1714074853",
            overwrite=True,
        )
        # Only the new partition survives -- OLD has been blasted.
        roundtripped = read_partitioned_dataset(root, run="1714074853")
        assert set(roundtripped["trajectory_id"]) == {"NEW"}

    def test_overwrite_false_against_missing_run_is_fine(self, tmp_path: Path) -> None:
        # The kwarg is only consequential when the subroot already
        # exists; against a missing run it's a no-op + fresh write.
        root = tmp_path / "evals"
        df = pd.DataFrame([{"trajectory_id": "T1", "locale": "en_US", "probe_family": "x"}])
        write_run_partition(df, root, "1714074853", overwrite=False)
        assert (root / "run=1714074853").exists()


class TestWriteRunLocalesPartition:
    """Per-locale partition rewrite for locale-targeted re-evaluation.

    Two workflows this helper supports:

    1. **Re-eval one locale, preserve others.** An infra issue (or
       evaluator change) requires re-running just en_US; the other 7
       locales' eval data on disk for the same run id must stay
       byte-identical. ``overwrite=True`` (default) wipes the target
       locale's subdir and writes fresh.
    2. **Add a new locale to an existing eval run.** A 7-locale run
       gains an 8th locale; existing data stays untouched, the new
       locale's partitions are added. ``overwrite=False`` for additive
       semantics (with a writer-id token in the basename so concurrent
       appends don't collide).
    """

    @staticmethod
    def _multi_locale_frame():
        """3 locales × 2 probes per locale = 6 rows."""
        rows = []
        for locale in ("en_US", "ja_JP", "pt_BR"):
            for probe in ("tool_calling", "general_open_ended"):
                rows.append(
                    {
                        "trajectory_id": f"T-{locale}-{probe}",
                        "locale": locale,
                        "probe_family": probe,
                        "score": 0.5,
                    }
                )
        return pd.DataFrame(rows)

    def test_round_trip_single_locale(self, tmp_path: Path) -> None:
        """Sanity: targeted write of one locale's worth of rows lands
        cleanly with no other partitions written."""
        root = tmp_path / "evals"
        df = pd.DataFrame(
            [
                {"trajectory_id": "T1", "locale": "en_US", "probe_family": "x"},
            ]
        )
        subroot = write_run_locales_partition(
            df,
            root,
            "1714074853",
            locales=["en_US"],
        )
        assert subroot == root / "run=1714074853"
        assert (subroot / "locale=en_US").exists()
        # No other locale partitions created.
        siblings = sorted(p.name for p in subroot.iterdir() if p.is_dir())
        assert siblings == ["locale=en_US"]

    def test_targeted_overwrite_preserves_other_locales(
        self,
        tmp_path: Path,
    ) -> None:
        """The headline use-case: re-eval en_US and confirm ja_JP /
        pt_BR partitions are byte-identical before and after."""
        root = tmp_path / "evals"
        run_id = "1714074853"

        # Initial whole-run write spanning 3 locales.
        write_run_partition(self._multi_locale_frame(), root, run_id)
        subroot = root / f"run={run_id}"

        # Capture other locales' partition contents BEFORE the targeted write.
        before_ja = sorted((subroot / "locale=ja_JP").rglob("*.parquet"))
        before_pt = sorted((subroot / "locale=pt_BR").rglob("*.parquet"))
        before_ja_bytes = {p.relative_to(subroot): p.read_bytes() for p in before_ja}
        before_pt_bytes = {p.relative_to(subroot): p.read_bytes() for p in before_pt}
        assert before_ja_bytes  # sanity
        assert before_pt_bytes

        # Re-eval just en_US with new content.
        en_us_only = pd.DataFrame(
            [
                {"trajectory_id": "T-EN-NEW-1", "locale": "en_US", "probe_family": "tool_calling", "score": 0.99},
                {"trajectory_id": "T-EN-NEW-2", "locale": "en_US", "probe_family": "general_open_ended", "score": 0.99},
            ]
        )
        write_run_locales_partition(
            en_us_only,
            root,
            run_id,
            locales=["en_US"],
        )

        # en_US has the NEW rows and ONLY the new rows (old en_US gone).
        rt = read_partitioned_dataset(root, run=run_id)
        en_us_ids = sorted(rt[rt["locale"] == "en_US"]["trajectory_id"].tolist())
        assert en_us_ids == ["T-EN-NEW-1", "T-EN-NEW-2"]
        # ja_JP and pt_BR rows survive unchanged.
        assert set(rt[rt["locale"] == "ja_JP"]["trajectory_id"]) == {
            "T-ja_JP-tool_calling",
            "T-ja_JP-general_open_ended",
        }
        assert set(rt[rt["locale"] == "pt_BR"]["trajectory_id"]) == {
            "T-pt_BR-tool_calling",
            "T-pt_BR-general_open_ended",
        }

        # Strongest invariant: ja_JP / pt_BR parquet files are byte-
        # identical to before the targeted write. No accidental rewrite,
        # no metadata churn, no `_common_metadata` mutation.
        after_ja_bytes = {p.relative_to(subroot): p.read_bytes() for p in (subroot / "locale=ja_JP").rglob("*.parquet")}
        after_pt_bytes = {p.relative_to(subroot): p.read_bytes() for p in (subroot / "locale=pt_BR").rglob("*.parquet")}
        assert after_ja_bytes == before_ja_bytes, "ja_JP partitions must be byte-identical after en_US re-eval"
        assert after_pt_bytes == before_pt_bytes, "pt_BR partitions must be byte-identical after en_US re-eval"

    def test_additive_new_locale_keeps_existing(self, tmp_path: Path) -> None:
        """Workflow 2: an 8th locale gets added to a 7-locale run with
        ``overwrite=False``. Existing locales' partitions stay
        byte-identical; the new locale's partitions are appended."""
        root = tmp_path / "evals"
        run_id = "1714074853"

        # Initial run has en_US + ja_JP only.
        initial = pd.DataFrame(
            [
                {"trajectory_id": "T-EN", "locale": "en_US", "probe_family": "tool_calling", "score": 0.5},
                {"trajectory_id": "T-JA", "locale": "ja_JP", "probe_family": "tool_calling", "score": 0.6},
            ]
        )
        write_run_partition(initial, root, run_id)
        subroot = root / f"run={run_id}"
        before_en = {p.relative_to(subroot): p.read_bytes() for p in (subroot / "locale=en_US").rglob("*.parquet")}
        before_ja = {p.relative_to(subroot): p.read_bytes() for p in (subroot / "locale=ja_JP").rglob("*.parquet")}

        # Add pt_BR (new locale) without touching en_US / ja_JP.
        pt_br_only = pd.DataFrame(
            [
                {"trajectory_id": "T-PT", "locale": "pt_BR", "probe_family": "tool_calling", "score": 0.7},
            ]
        )
        write_run_locales_partition(
            pt_br_only,
            root,
            run_id,
            locales=["pt_BR"],
            overwrite=False,
        )

        rt = read_partitioned_dataset(root, run=run_id)
        assert set(rt["trajectory_id"]) == {"T-EN", "T-JA", "T-PT"}

        # Existing en_US + ja_JP partitions unchanged.
        after_en = {p.relative_to(subroot): p.read_bytes() for p in (subroot / "locale=en_US").rglob("*.parquet")}
        after_ja = {p.relative_to(subroot): p.read_bytes() for p in (subroot / "locale=ja_JP").rglob("*.parquet")}
        assert after_en == before_en
        assert after_ja == before_ja

    def test_rejects_df_with_locale_outside_target(
        self,
        tmp_path: Path,
    ) -> None:
        """Pre-condition: df rows must all be in the target locale set.
        Otherwise the helper would write rows that escape the targeted-
        overwrite contract -- e.g., a stray ja_JP row would land in
        the ja_JP partition of a "just en_US" call. Raise loudly."""
        root = tmp_path / "evals"
        df_mixed = pd.DataFrame(
            [
                {"trajectory_id": "T-EN", "locale": "en_US", "probe_family": "x", "score": 0.5},
                {"trajectory_id": "T-JA", "locale": "ja_JP", "probe_family": "x", "score": 0.5},
            ]
        )
        with pytest.raises(ValueError, match="not in target locales"):
            write_run_locales_partition(
                df_mixed,
                root,
                "r",
                locales=["en_US"],
            )

    def test_empty_locales_raises(self, tmp_path: Path) -> None:
        """Empty locales is a programming error -- there's a separate
        helper (``write_run_partition``) for whole-run writes; reroute
        the user there rather than silently writing nothing."""
        root = tmp_path / "evals"
        df = pd.DataFrame([{"trajectory_id": "T", "locale": "en_US", "probe_family": "x"}])
        with pytest.raises(ValueError, match="non-empty sequence"):
            write_run_locales_partition(df, root, "r", locales=[])

    def test_missing_locale_column_raises(self, tmp_path: Path) -> None:
        root = tmp_path / "evals"
        df_no_locale = pd.DataFrame([{"trajectory_id": "T", "probe_family": "x", "score": 0.5}])
        with pytest.raises(ValueError, match="missing 'locale' column"):
            write_run_locales_partition(
                df_no_locale,
                root,
                "r",
                locales=["en_US"],
            )

    def test_empty_df_is_no_op(self, tmp_path: Path) -> None:
        """An empty df shouldn't rmtree anything -- caller may have
        legitimately filtered to zero rows and expects existing
        partitions to stay intact."""
        root = tmp_path / "evals"
        run_id = "1714074853"
        write_run_partition(self._multi_locale_frame(), root, run_id)
        subroot = root / f"run={run_id}"
        before_en = {p.relative_to(subroot): p.read_bytes() for p in (subroot / "locale=en_US").rglob("*.parquet")}

        # Empty df targeting en_US.
        empty = pd.DataFrame(columns=["trajectory_id", "locale", "probe_family", "score"])
        write_run_locales_partition(empty, root, run_id, locales=["en_US"])

        # en_US partition untouched.
        after_en = {p.relative_to(subroot): p.read_bytes() for p in (subroot / "locale=en_US").rglob("*.parquet")}
        assert after_en == before_en

    def test_multi_locale_targeted_overwrite(self, tmp_path: Path) -> None:
        """Multiple locales targeted at once: re-eval en_US AND ja_JP,
        keep pt_BR untouched."""
        root = tmp_path / "evals"
        run_id = "1714074853"
        write_run_partition(self._multi_locale_frame(), root, run_id)
        subroot = root / f"run={run_id}"
        before_pt = {p.relative_to(subroot): p.read_bytes() for p in (subroot / "locale=pt_BR").rglob("*.parquet")}

        new_data = pd.DataFrame(
            [
                {"trajectory_id": "T-EN-V2", "locale": "en_US", "probe_family": "tool_calling", "score": 0.99},
                {"trajectory_id": "T-JA-V2", "locale": "ja_JP", "probe_family": "tool_calling", "score": 0.99},
            ]
        )
        write_run_locales_partition(
            new_data,
            root,
            run_id,
            locales=["en_US", "ja_JP"],
        )

        rt = read_partitioned_dataset(root, run=run_id)
        # en_US and ja_JP each have only their new V2 row.
        assert set(rt[rt["locale"] == "en_US"]["trajectory_id"]) == {"T-EN-V2"}
        assert set(rt[rt["locale"] == "ja_JP"]["trajectory_id"]) == {"T-JA-V2"}
        # pt_BR survives unchanged.
        assert set(rt[rt["locale"] == "pt_BR"]["trajectory_id"]) == {
            "T-pt_BR-tool_calling",
            "T-pt_BR-general_open_ended",
        }
        after_pt = {p.relative_to(subroot): p.read_bytes() for p in (subroot / "locale=pt_BR").rglob("*.parquet")}
        assert after_pt == before_pt
