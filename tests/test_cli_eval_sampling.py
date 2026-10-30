# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The rows and scorers of a ``usersim eval`` pass, and the report label that follows from its sampling."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pandas as pd
import pytest

from usersim import cli
from usersim.cli import _pipeline
from usersim.cli._models import bundled_models_path
from usersim.engine.core.storage import read_partitioned_dataset, run_subroot, write_partitioned_dataset
from usersim.engine.evaluator.scorers import GENERIC_SCORERS
from usersim.reporting import EVAL_STORE_MANIFEST_FILENAME, read_eval_sample_manifest, record_eval_sample_pass

RUN_ID = "1700000000"


class _StubDataDesigner:
    """Returns one empty evaluation cell per requested record."""

    def create(self, builder, num_records):
        return SimpleNamespace(load_dataset=lambda: pd.DataFrame({"assistant_eval": ["{}"] * num_records}))


def _evaluate(tmp_path, monkeypatch, trajectory_df, *flags, built=None):
    """Run ``usersim eval`` over ``trajectory_df`` and return the evaluations root and its recorded pass.

    ``built``, when given, receives the arguments the evaluator is built with.
    """
    traj_root = tmp_path / "trajectories"
    write_partitioned_dataset(trajectory_df, run_subroot(traj_root, RUN_ID))

    def build(**kwargs):
        if built is not None:
            built.update(kwargs)
        return _StubDataDesigner(), object()

    monkeypatch.setattr(_pipeline, "build_evaluator_config_builder", build)
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


def test_auto_scorers_reach_the_evaluator_resolved_for_the_run(tmp_path, monkeypatch, trajectory_df, capsys):
    """``--scorers auto`` becomes the generic scorers plus the default_scorers of the run's probes."""
    run = trajectory_df.assign(probe_family=["tool_calling", "financial_services", *["general_open_ended"] * 4])
    built: dict = {}

    _evaluate(tmp_path, monkeypatch, run, "--scorers", "tool_use,auto", built=built)

    assert built["scorers"] == ["tool_use", *GENERIC_SCORERS, "financial_services"]
    assert f"scorers:       {', '.join(built['scorers'])}" in capsys.readouterr().out


class TestEvalWarnsOnRepeatedIds:
    """One ``trajectory_id`` is one conversation; a run that repeats one would score it more than once."""

    @staticmethod
    def _warnings(caplog):
        return [r.getMessage() for r in caplog.records if "more than once" in r.getMessage()]

    @pytest.mark.parametrize("flags", [(), ("--num-records", "1"), ("--num-records-per-locale", "1")])
    def test_a_repeated_id_is_named_whatever_rows_the_pass_picks(
        self, tmp_path, monkeypatch, trajectory_df, caplog, flags
    ):
        repeated = trajectory_df.copy()
        twice = repeated.loc[0, "trajectory_id"]
        repeated.loc[5, "trajectory_id"] = twice
        with caplog.at_level(logging.WARNING, logger="usersim.cli.eval"):
            out, _ = _evaluate(tmp_path, monkeypatch, repeated, *flags)

        (warning,) = self._warnings(caplog)
        assert f"{twice} (2 rows)" in warning
        assert repeated.loc[1, "trajectory_id"] not in warning
        if not flags:
            # Every row is still evaluated.
            assert len(read_partitioned_dataset(out, run=RUN_ID)) == len(repeated)

    def test_a_run_without_repeats_gives_no_warning(self, tmp_path, monkeypatch, trajectory_df, caplog):
        with caplog.at_level(logging.WARNING, logger="usersim.cli.eval"):
            _evaluate(tmp_path, monkeypatch, trajectory_df)

        assert self._warnings(caplog) == []


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
