# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Tests for the ``usersim smoke`` subcommand.

The smoke command itself is the production smoke test for everything
*else*; this test module verifies the smoke command itself works
end-to-end on a clean tree, plus exercises the individual checks
in isolation so a failure points to the right culprit.
"""

from __future__ import annotations

import json

import pytest

from usersim import cli
from usersim.cli.smoke import (
    SmokeReport,
    _check_capability_dashboard,
    _check_dry_runs,
    _check_evaluator_config,
    _check_imports,
    _check_models_config,
    _check_probe_registry,
    _check_scorer_registry,
    _check_simulator_config,
    _synthetic_smoke_frames,
)

# ---------------------------------------------------------------------------
# End-to-end CLI invocation
# ---------------------------------------------------------------------------


_DASHBOARD_ARTIFACTS = (
    "index.html",
    "report_manifest.json",
    "capability_matrix.json",
    "coverage_summary.json",
    "triage_queue.json",
)


class TestSmokeCommand:
    def test_offline_run_passes(self, capsys):
        rc = cli.main(["smoke"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "All checks passed." in out
        assert "[FAIL]" not in out

    def test_help_advertises_subcommand(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            cli.main(["--help"])
        assert excinfo.value.code == 0
        assert "smoke" in capsys.readouterr().out

    def test_with_fixtures_writes_artifacts(self, tmp_path, capsys):
        out_dir = tmp_path / "smoke_out"
        rc = cli.main(
            [
                "smoke",
                "--with-fixtures",
                "--fixtures-dir",
                str(out_dir),
            ]
        )
        assert rc == 0
        for name in _DASHBOARD_ARTIFACTS:
            assert (out_dir / name).exists(), f"missing {name}"
            assert (out_dir / name).stat().st_size > 0, f"empty {name}"
        # JSON artifacts parse and the manifest carries the run-level keys.
        manifest = json.loads((out_dir / "report_manifest.json").read_text())
        for key in ("run_id", "model_id", "metadata_source", "n_trajectories", "n_evaluated"):
            assert key in manifest, f"manifest missing {key}"
        # Trajectories and evaluations are partitioned datasets, not
        # single files. Assert the partition root is a directory and
        # contains at least one ``locale=*`` partition.
        for name in ("trajectories", "evaluations"):
            root = out_dir / name
            assert root.is_dir(), f"missing partition root {name}/"
            partitions = list(root.glob("locale=*"))
            assert partitions, f"{name}/ has no locale=* partitions: {list(root.iterdir())}"


# ---------------------------------------------------------------------------
# Individual check coverage — useful for pinpointing regressions.
# ---------------------------------------------------------------------------


class TestImportsCheck:
    def test_passes_in_normal_env(self):
        report = SmokeReport()
        _check_imports(report)
        assert report.all_passed
        assert report.results[-1].name == "core imports"


class TestModelsConfigCheck:
    def test_default_config_passes(self):
        report = SmokeReport()
        _check_models_config(report)
        assert report.all_passed


class TestSimulatorConfigCheck:
    def test_default_simulator_config_passes(self):
        report = SmokeReport()
        _check_simulator_config(report)
        assert report.all_passed


class TestEvaluatorConfigCheck:
    def test_single_judge_passes(self):
        report = SmokeReport()
        _check_evaluator_config(report)
        assert report.all_passed


class TestScorerRegistryCheck:
    def test_tool_use_present(self):
        report = SmokeReport()
        _check_scorer_registry(report)
        assert report.all_passed
        assert "tool_use" in report.results[-1].detail


class TestProbeRegistryCheck:
    def test_all_registered_probes_resolve(self):
        report = SmokeReport()
        _check_probe_registry(report)
        assert report.all_passed
        # The detail line should enumerate the registered probes so
        # a future regression makes the missing one visible immediately.
        detail = report.results[-1].detail
        for probe in (
            "tool_calling",
            "general_open_ended",
            "general_educational",
            "sov_ai_facts",
            "sov_ai_dynamic",
            "sov_ai_multilingual_parity",
            "safety_chat_pressure",
            "safety_agentic",
        ):
            assert probe in detail, f"missing {probe!r} in: {detail}"


class TestDryRunsCheck:
    def test_three_dry_runs_pass(self):
        report = SmokeReport()
        _check_dry_runs(report)
        assert report.all_passed


class TestCapabilityDashboardCheck:
    def test_in_process_build(self):
        report = SmokeReport()
        out = _check_capability_dashboard(report, write_fixtures=False, fixtures_dir=None)
        assert report.all_passed
        # ``write_fixtures=False`` still writes to a temp dir (so the
        # check can validate JSON parse), but it is reported as None to
        # the caller.
        assert out is None

    def test_with_fixtures_writes_files(self, tmp_path):
        report = SmokeReport()
        out = _check_capability_dashboard(
            report,
            write_fixtures=True,
            fixtures_dir=tmp_path,
        )
        assert report.all_passed
        assert out == tmp_path
        for name in _DASHBOARD_ARTIFACTS:
            assert (tmp_path / name).exists(), f"missing {name}"


# ---------------------------------------------------------------------------
# SmokeReport accumulator
# ---------------------------------------------------------------------------


class TestSmokeReport:
    def test_all_passed_with_zero_results(self):
        report = SmokeReport()
        assert report.all_passed is True
        assert report.render() == ""

    def test_render_aligns_check_names(self):
        report = SmokeReport()
        report.add("short", True, "ok")
        report.add("a longer name", True, "also ok")
        rendered = report.render()
        # Both lines should have the detail starting at the same column.
        line_a, line_b = rendered.splitlines()
        # Find where 'ok' appears — it should be at the same position.
        a_pos = line_a.index("ok")
        b_pos = line_b.rindex("also ok")
        assert a_pos == b_pos

    def test_failed_check_breaks_all_passed(self):
        report = SmokeReport()
        report.add("a", True)
        report.add("b", False, "boom")
        assert report.all_passed is False


# ---------------------------------------------------------------------------
# Synthetic fixture builder
# ---------------------------------------------------------------------------


class TestSyntheticFrames:
    def test_traj_and_eval_align_on_trajectory_id(self):
        pd = pytest.importorskip("pandas")
        traj_df, eval_df = _synthetic_smoke_frames(pd)
        assert len(traj_df) == 2
        assert len(eval_df) == 2
        assert list(traj_df["trajectory_id"]) == list(eval_df["trajectory_id"])
        assert "assistant_eval" in eval_df.columns

    def test_outcomes_carry_status(self):
        pd = pytest.importorskip("pandas")
        traj_df, _ = _synthetic_smoke_frames(pd)
        statuses = [json.loads(o)["status"] for o in traj_df["simulation_outcome"]]
        assert statuses == ["ok", "failed"]

    def test_eval_cell_has_envelope(self):
        pd = pytest.importorskip("pandas")
        _, eval_df = _synthetic_smoke_frames(pd)
        first = json.loads(eval_df.iloc[0]["assistant_eval"])
        assert "envelope" in first
        assert "axes" in first
        assert first["axes"]["helpfulness"]["judge_a"]["score"] == 5

    def test_aggregate_from_dataframe_handles_synthetic_smoke_frames(self):
        """Sanity lock: the smoke fixture's synthetic frames don't trip
        the report-time reasoning aggregator. Phase A field is absent, the
        second trajectory's ``conversation_messages`` is an empty array,
        and the second outcome carries an empty ``per_model_output_tokens``.

        Phase B should:
          1. Run cleanly (no crash).
          2. Surface ``estimated`` for assistant_model on the OK row
             (visible content present, output_tokens > 0).
          3. Mark judge_model / summary_model ``unavailable`` (no Phase A
             capture, Phase B can't help them by design).
        """
        pd = pytest.importorskip("pandas")
        from usersim.reporting.resource import aggregate_from_dataframe

        traj_df, _ = _synthetic_smoke_frames(pd)
        profile = aggregate_from_dataframe(traj_df)

        # Phase B ran for the OK row; the failed row has no per-alias
        # output tokens so it skips silently. Smoke fixture keeps the
        # numbers tiny -- assert the source label, not the value.
        assert profile.reasoning_source_by_alias.get("assistant_model") in {
            "estimated",
            "unreliable_estimate",
        }
        # judge_model / summary_model don't appear on the trajectory at all
        # in this fixture, but the finalizer still marks them unavailable.
        assert profile.reasoning_source_by_alias["judge_model"] == "unavailable"
        assert profile.reasoning_source_by_alias["summary_model"] == "unavailable"
