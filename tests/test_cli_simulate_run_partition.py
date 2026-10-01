# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``cli/simulate.py``'s run-id wiring.

Pins the bug class that motivated the run-partition layer:

- Old behaviour: ``RESUME=False`` / ``--no-skip-existing`` printed
  ``"will be OVERWRITTEN"`` but actually appended new files
  alongside the old ones (because filenames included a fresh
  pid+uuid token, never colliding). Stale failed-trajectory files
  silently mixed with healthy ones in the evaluator + reporter
  downstream, contaminating every metric.
- Current behaviour: ``--no-skip-existing`` mints a new
  ``run=<epoch>`` partition, isolating the new run's output.
  ``--skip-existing`` (default) resumes into the most-recent
  existing run. Both paths scope ``existing_trajectory_ids`` to
  the resolved run id so cross-run skipping doesn't happen.

This test verifies the wiring without invoking real LLMs — it
calls into ``cli.simulate._simulate_to_parquet`` with a stubbed
DataDesigner pipeline, then asserts:

1. ``--no-skip-existing`` lands writes under a NEW
   ``run=<epoch>/`` partition each time.
2. ``--skip-existing`` resumes into the latest existing run (no
   new partition).
3. ``existing_trajectory_ids`` scoping prevents cross-run
   accidental skipping (a trajectory_id in run A doesn't suppress
   a write in run B even when both runs target the same
   ``trajectory_id``).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from usersim.engine.core._assets import packaged_assets_dir
from usersim.engine.core.storage import (
    list_runs,
    new_run_id,
    read_partitioned_dataset,
    write_locale_partition,
)

# ---------------------------------------------------------------------------
# Fixtures: synthetic trajectory frame + minimal models config
# ---------------------------------------------------------------------------


def _synthetic_trajectory(trajectory_id: str = "T-001") -> pd.DataFrame:
    """Single-row frame matching the simulator's output schema enough
    to exercise ``write_locale_partition`` + ``read_partitioned_dataset``."""
    outcome = json.dumps(
        {
            "status": "ok",
            "failure_class": None,
            "failure_attribution": None,
            "failure_detail": "",
            "n_turns": 1,
            "warnings": [],
        }
    )
    return pd.DataFrame(
        [
            {
                "trajectory_id": trajectory_id,
                "persona_uuid": f"p-{trajectory_id}",
                "locale": "en_US",
                "probe_family": "general_open_ended",
                "probe_variant": "default",
                "simulation_outcome": outcome,
                "conversation_messages": "[]",
            }
        ]
    )


def _synthetic_context_failure() -> pd.DataFrame:
    frame = _synthetic_trajectory("CTX-FAIL")
    frame.loc[0, "simulation_outcome"] = json.dumps(
        {
            "status": "failed",
            "failure_class": "infrastructure_error",
            "failure_attribution": "assistant_model",
            "failure_detail": "assistant_model context window exceeded",
            "n_turns": 1,
            "warnings": [],
        }
    )
    frame.loc[0, "conversation_messages"] = json.dumps(
        [
            {"role": "user", "content": "help"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "function": {"name": "kb_search"}}],
            },
            {
                "role": "tool",
                "content": '{"results":[{"id":"d1","body":"evidence"}]}',
                "tool_call_id": "c1",
            },
        ]
    )
    frame.loc[0, "finance_task_id"] = "TASK-CTX"
    frame.loc[0, "retrieved_document_ids"] = '["d1"]'
    return frame


# ---------------------------------------------------------------------------
# Direct round-trip via the storage helpers (catches the bug at the
# layer where it lives, not the CLI dispatch layer)
# ---------------------------------------------------------------------------


class TestRunIsolation:
    """Two consecutive `RESUME=False` runs land in DISTINCT
    ``run=*`` partitions and the default reader sees only the
    latest one."""

    def test_two_fresh_runs_isolate_into_separate_partitions(
        self,
        tmp_path: Path,
    ) -> None:
        root = tmp_path / "trajectories"

        # Run 1 (some seconds ago): mint a run id and write.
        run1 = new_run_id(now=1700000000)
        write_locale_partition(
            _synthetic_trajectory("OLD-001"),
            root,
            locale="en_US",
            run_id=run1,
        )
        # Run 2 (now): mint a fresh id and write.
        run2 = new_run_id(now=1800000000)
        write_locale_partition(
            _synthetic_trajectory("NEW-001"),
            root,
            locale="en_US",
            run_id=run2,
        )

        assert sorted(list_runs(root)) == sorted([run1, run2])
        assert (root / f"run={run1}" / "locale=en_US").is_dir()
        assert (root / f"run={run2}" / "locale=en_US").is_dir()

        # Default reader picks the latest run only.
        df = read_partitioned_dataset(root)
        assert set(df["trajectory_id"].astype(str)) == {"NEW-001"}

        # Pinning to run1 returns OLD-001 only.
        df_old = read_partitioned_dataset(root, run=run1)
        assert set(df_old["trajectory_id"].astype(str)) == {"OLD-001"}

        # run="all" surfaces both with a ``run`` column.
        df_all = read_partitioned_dataset(root, run="all")
        assert set(df_all["trajectory_id"].astype(str)) == {"OLD-001", "NEW-001"}
        run_to_traj = dict(
            zip(
                df_all["trajectory_id"].astype(str),
                df_all["run"].astype(str),
            )
        )
        assert run_to_traj["OLD-001"] == run1
        assert run_to_traj["NEW-001"] == run2

    def test_resume_into_latest_run_appends_to_same_partition(
        self,
        tmp_path: Path,
    ) -> None:
        """RESUME semantics: writing into an existing run id with a
        new trajectory_id appends to that run's partition rather
        than creating a new one."""
        root = tmp_path / "trajectories"
        run = new_run_id(now=1700000000)
        write_locale_partition(
            _synthetic_trajectory("FIRST"),
            root,
            locale="en_US",
            run_id=run,
        )
        # Resume: another write with the same run id.
        write_locale_partition(
            _synthetic_trajectory("SECOND"),
            root,
            locale="en_US",
            run_id=run,
        )
        assert list_runs(root) == [run]
        df = read_partitioned_dataset(root)
        assert set(df["trajectory_id"].astype(str)) == {"FIRST", "SECOND"}


class TestNewRunIdWiring:
    """The simulate CLI helper must mint a fresh run id when
    ``--no-skip-existing`` is set, even if a previous run exists."""

    def test_new_run_id_changes_on_back_to_back_calls(self) -> None:
        # Without injecting a clock, two back-to-back calls might
        # get the same epoch second. Inject monotonic now() values
        # to verify the wiring deterministically.
        rid_a = new_run_id(now=1700000000)
        rid_b = new_run_id(now=1700000001)
        assert rid_a != rid_b
        assert int(rid_b) > int(rid_a)

    def test_simulate_helper_mints_new_run_when_skip_existing_false(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """``cli.simulate._simulate_to_parquet`` with ``skip_existing=False``
        must call ``new_run_id`` and route writes under that new
        partition, leaving the previous run intact.

        We stub the LLM-driven DataDesigner pipeline by patching
        ``build_simulator_config_builder`` to return a fake
        DataDesigner whose ``create().load_dataset()`` returns our
        synthetic frame. This isolates the run-id wiring without
        needing API keys."""
        from usersim.cli._models import default_models_path, load_models_config
        from usersim.cli.simulate import _simulate_to_parquet

        out = tmp_path / "trajs"
        out.mkdir()
        # Pre-existing "old" run on disk.
        write_locale_partition(
            _synthetic_trajectory("OLD"),
            out,
            locale="en_US",
            run_id="1700000000",
        )

        # Fake DataDesigner: returns our synthetic frame from
        # ``data_designer.create(builder).load_dataset()``.
        class _FakeResult:
            def load_dataset(self):
                return _synthetic_trajectory("FRESH")

        class _FakeDD:
            def create(self, builder, num_records=None):
                return _FakeResult()

        captured_builder_kwargs = {}

        def _fake_builder(**kwargs):
            captured_builder_kwargs.update(kwargs)
            return _FakeDD(), object()

        models = load_models_config(default_models_path())
        with patch("usersim.cli._pipeline.build_simulator_config_builder", _fake_builder):
            rc = _simulate_to_parquet(
                models=models,
                models_path=default_models_path(),
                panel=None,
                locales=["en_US"],
                num_rows=1,
                out=out,
                assets_dir=packaged_assets_dir(),
                max_turns=1,
                probe_mix={"general_open_ended": 1.0},
                random_seed=42,
                skip_existing=False,  # FRESH run
                store_reasoning=False,
            )
        assert rc == 0
        assert captured_builder_kwargs["toolset_seed_path"] is None
        assert captured_builder_kwargs["store_reasoning"] is False

        # Two distinct runs on disk now: the seeded "1700000000"
        # plus a freshly minted one (>= 1714074853 — present-day
        # epoch). The fresh write didn't clobber the old run's
        # files.
        runs = list_runs(out)
        assert "1700000000" in runs, runs
        fresh_runs = [r for r in runs if r != "1700000000"]
        assert len(fresh_runs) == 1, f"expected exactly one new run id; got {fresh_runs}"
        new_id = fresh_runs[0]
        assert int(new_id) > 1700000000, f"new run id {new_id} should be a current epoch second"

        # The fresh run contains FRESH; the old run still contains OLD.
        df_new = read_partitioned_dataset(out, run=new_id)
        assert set(df_new["trajectory_id"].astype(str)) == {"FRESH"}
        df_old = read_partitioned_dataset(out, run="1700000000")
        assert set(df_old["trajectory_id"].astype(str)) == {"OLD"}

        # Manifest is written for the fresh run with the schema-version
        # stamped and the resolved assistant_model identity present. The
        # old run (which predates manifest support) has no manifest —
        # that's the "metadata not captured" backstop.
        from usersim.engine.core.manifest import load_run_manifest

        fresh_manifest = load_run_manifest(new_id, out)
        assert fresh_manifest is not None
        assert fresh_manifest.manifest_schema_version == "v1"
        assert fresh_manifest.run_id == new_id
        assert "assistant_model" in fresh_manifest.models
        assert fresh_manifest.scope.toolset_seed_path is None
        assert fresh_manifest.scope.toolset_content_hash is None
        assert fresh_manifest.simulation_config.store_reasoning is False
        assert load_run_manifest("1700000000", out) is None

    def test_simulate_helper_resumes_into_latest_when_skip_existing_true(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """``skip_existing=True`` resumes into the most-recent
        existing run; the FRESH trajectory lands in that same run
        partition, not a new one."""
        from usersim.cli._models import default_models_path, load_models_config
        from usersim.cli.simulate import _simulate_to_parquet

        out = tmp_path / "trajs"
        out.mkdir()
        existing_run = "1700000000"
        write_locale_partition(
            _synthetic_trajectory("OLD"),
            out,
            locale="en_US",
            run_id=existing_run,
        )

        class _FakeResult:
            def load_dataset(self):
                return _synthetic_trajectory("RESUMED")

        class _FakeDD:
            def create(self, builder, num_records=None):
                return _FakeResult()

        def _fake_builder(**kwargs):
            return _FakeDD(), object()

        models = load_models_config(default_models_path())
        with patch("usersim.cli._pipeline.build_simulator_config_builder", _fake_builder):
            rc = _simulate_to_parquet(
                models=models,
                models_path=default_models_path(),
                panel=None,
                locales=["en_US"],
                num_rows=1,
                out=out,
                assets_dir=packaged_assets_dir(),
                max_turns=1,
                probe_mix={"general_open_ended": 1.0},
                random_seed=42,
                skip_existing=True,  # RESUME
            )
        assert rc == 0

        # Still exactly one run on disk — RESUMED appended to
        # existing_run rather than minting a new one.
        runs = list_runs(out)
        assert runs == [existing_run]
        df = read_partitioned_dataset(out)
        assert set(df["trajectory_id"].astype(str)) == {"OLD", "RESUMED"}


def test_context_failure_row_survives_cli_storage(tmp_path: Path) -> None:
    """A structured partial failure is a row, not a Data Designer omission."""
    from usersim.cli._models import default_models_path, load_models_config
    from usersim.cli.simulate import _simulate_to_parquet

    out = tmp_path / "trajs"
    models = load_models_config(default_models_path())

    class _FakeResult:
        def load_dataset(self):
            return _synthetic_context_failure()

    class _FakeDD:
        def create(self, builder, num_records=None):
            assert num_records == 1
            return _FakeResult()

    with patch(
        "usersim.cli._pipeline.build_simulator_config_builder",
        lambda **kwargs: (_FakeDD(), object()),
    ):
        rc = _simulate_to_parquet(
            models=models,
            models_path=default_models_path(),
            panel=None,
            locales=["en_US"],
            num_rows=1,
            out=out,
            assets_dir=packaged_assets_dir(),
            max_turns=1,
            probe_mix={"financial_services": 1.0},
            random_seed=42,
            skip_existing=False,
        )

    stored = read_partitioned_dataset(out)
    assert rc == 0
    assert len(stored) == 1
    row = stored.iloc[0]
    outcome = json.loads(row["simulation_outcome"])
    assert outcome["failure_attribution"] == "assistant_model"
    assert row["finance_task_id"] == "TASK-CTX"
    assert json.loads(row["retrieved_document_ids"]) == ["d1"]
    assert "evidence" in row["conversation_messages"]
