# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How ``usersim eval`` sampled a run, and the report label that follows from it."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from usersim import cli
from usersim.cli import _pipeline
from usersim.cli._models import bundled_models_path
from usersim.engine.core.storage import read_partitioned_dataset, run_subroot, write_partitioned_dataset
from usersim.reporting import EVAL_STORE_MANIFEST_FILENAME, read_eval_sample_manifest, record_eval_sample_pass

RUN_ID = "1700000000"


class _StubDataDesigner:
    """Returns one empty evaluation cell per requested record."""

    def create(self, builder, num_records):
        return SimpleNamespace(load_dataset=lambda: pd.DataFrame({"assistant_eval": ["{}"] * num_records}))


def _evaluate(tmp_path, monkeypatch, trajectory_df, *flags):
    """Run ``usersim eval`` over ``trajectory_df`` and return the evaluations root and its recorded pass."""
    traj_root = tmp_path / "trajectories"
    write_partitioned_dataset(trajectory_df, run_subroot(traj_root, RUN_ID))
    monkeypatch.setattr(_pipeline, "build_evaluator_config_builder", lambda **_: (_StubDataDesigner(), object()))
    out = tmp_path / "evaluations"
    args = ["eval", "--trajectories", str(traj_root), "--out", str(out), "--models", str(bundled_models_path())]

    assert cli.main([*args, *flags]) == 0

    (recorded,) = read_eval_sample_manifest(run_subroot(out, RUN_ID) / EVAL_STORE_MANIFEST_FILENAME).passes
    return out, recorded


class TestEvalRecordsItsSampling:
    """``usersim eval`` records how it sampled the run, beside the evaluations."""

    def test_a_full_pass(self, tmp_path, monkeypatch, trajectory_df):
        out, recorded = _evaluate(tmp_path, monkeypatch, trajectory_df)

        assert (recorded.mode, recorded.n, recorded.n_selected, recorded.n_total) == ("full", None, 6, 6)
        assert sorted(recorded.trajectory_ids) == sorted(trajectory_df["trajectory_id"])
        assert len(read_partitioned_dataset(out, run=RUN_ID)) == 6

    def test_the_first_n_rows(self, tmp_path, monkeypatch, trajectory_df):
        _, recorded = _evaluate(tmp_path, monkeypatch, trajectory_df, "--num-records", "2")

        assert (recorded.mode, recorded.n, recorded.n_selected, recorded.n_total) == ("first_n", 2, 2, 6)

    def test_a_per_locale_sample(self, tmp_path, monkeypatch, trajectory_df):
        _, recorded = _evaluate(tmp_path, monkeypatch, trajectory_df, "--num-records-per-locale", "1")

        # One row from each of en_US, pt_BR and ja_JP.
        assert (recorded.mode, recorded.n, recorded.n_selected, recorded.n_total) == ("per_locale", 1, 3, 6)
        assert recorded.random_seed == 42


@pytest.mark.parametrize("recorded", ["per_locale", None], ids=["recorded", "not_recorded"])
def test_report_labels_the_run_by_its_recorded_pass(tmp_path, trajectory_df, evaluator_df, recorded):
    """A pass that recorded nothing about its sampling is not labelled full."""
    traj_root, eval_root = tmp_path / "trajectories", tmp_path / "evaluations"
    write_partitioned_dataset(trajectory_df, run_subroot(traj_root, RUN_ID))
    write_partitioned_dataset(evaluator_df, run_subroot(eval_root, RUN_ID))
    if recorded:
        record_eval_sample_pass(
            run_subroot(eval_root, RUN_ID) / EVAL_STORE_MANIFEST_FILENAME,
            run_id=RUN_ID,
            mode=recorded,
            n=1,
            random_seed=42,
            eval_locales=None,
            n_selected=3,
            n_total=len(trajectory_df),
            trajectory_ids=list(evaluator_df["trajectory_id"].head(3)),
        )
    out = tmp_path / "out"

    rc = cli.main(["report", "--trajectories", str(traj_root), "--evaluations", str(eval_root), "--out", str(out)])

    assert rc == 0
    manifest = json.loads((out / f"run={RUN_ID}" / "report_manifest.json").read_text())
    assert manifest["sample_mode"] == (recorded or "unknown")
