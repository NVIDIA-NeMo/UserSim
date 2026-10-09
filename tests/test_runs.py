# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for `reporting.print_available_runs` (on-disk run discovery + table).

Covers:

- empty trajectory_root prints the no-runs hint (no table).
- per-run row carries the right counts + presence flags.
- summary footer counts runs / runs-with-eval / runs-with-report.
- assistant_model surfaces from the manifest; missing-manifest renders ``—``.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

import pandas as pd
from rich.console import Console

from usersim.engine.core.storage import write_run_partition
from usersim.reporting import print_available_runs

_NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"


def _stage_trajectory_run(
    trajectory_root: Path,
    run_id: str,
    *,
    locales: list[str],
    n_per_locale: int = 2,
) -> None:
    """Write a tiny trajectory partition under
    ``trajectory_root/run=<run_id>/locale=*/probe_family=*/``.
    """
    rows = []
    for locale in locales:
        for i in range(n_per_locale):
            rows.append(
                {
                    "trajectory_id": f"{run_id}_{locale}_{i}",
                    "locale": locale,
                    "probe_family": "tool_calling",
                }
            )
    write_run_partition(pd.DataFrame(rows), trajectory_root, run_id)


def _stage_eval_run(eval_root: Path, run_id: str, cell: str = "{}") -> None:
    rows = [
        {
            "trajectory_id": f"{run_id}_en_US_0",
            "locale": "en_US",
            "probe_family": "tool_calling",
            "assistant_eval": cell,
        }
    ]
    write_run_partition(pd.DataFrame(rows), eval_root, run_id)


def _stage_report_run(report_root: Path, run_id: str) -> None:
    report_dir = report_root / f"run={run_id}"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "index.html").write_text("<html></html>")


def _capture(func, *args, **kwargs) -> str:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=240)
    func(*args, console=console, **kwargs)
    return buf.getvalue()


def test_empty_trajectory_root_prints_yellow_hint(tmp_path: Path) -> None:
    """No runs on disk -> friendly no-runs hint, no table rendered."""
    out = _capture(
        print_available_runs,
        tmp_path / "trajectories",
        tmp_path / "evaluations",
        tmp_path / "report",
    )
    assert "No trajectory runs under" in out
    assert "01_simulate.ipynb" in out
    assert "run_id" not in out  # no table header


def test_lists_runs_with_counts_and_flags(tmp_path: Path) -> None:
    trajectory_root = tmp_path / "trajectories"
    eval_root = tmp_path / "evaluations"
    report_root = tmp_path / "report"
    _stage_trajectory_run(
        trajectory_root,
        "1714074853",
        locales=["en_US", "fr_FR", "ja_JP"],
        n_per_locale=2,
    )
    _stage_trajectory_run(
        trajectory_root,
        "1714000000",
        locales=["en_US"],
        n_per_locale=4,
    )
    # Only the older run has eval data; only the newer run has a report.
    _stage_eval_run(eval_root, "1714000000")
    _stage_report_run(report_root, "1714074853")

    out = _capture(
        print_available_runs,
        trajectory_root,
        eval_root,
        report_root,
    )
    # Both run ids appear in the table.
    assert "1714074853" in out
    assert "1714000000" in out
    # Locale counts (3 and 1) appear.
    assert " 3 " in out or "│ 3 " in out
    assert " 1 " in out or "│ 1 " in out
    # Trajectory counts (3*2=6, 1*4=4) appear.
    assert "6" in out
    assert "4" in out
    # Footer counts.
    assert "2 run(s) on disk" in out
    assert "1 with evals" in out
    assert "1 with rendered report" in out


def test_models_block_surfaces_assistant_model_from_manifest(
    tmp_path: Path,
) -> None:
    trajectory_root = tmp_path / "trajectories"
    _stage_trajectory_run(
        trajectory_root,
        "1714074853",
        locales=["en_US"],
        n_per_locale=1,
    )
    # Hand-write a manifest so the helper can surface the assistant model.
    manifest_path = trajectory_root / "run=1714074853" / "_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "run_id": "1714074853",
                "timestamp": "2024-04-25T00:00:00Z",
                "models": {
                    "assistant_model": {
                        "alias": "assistant_model",
                        "model": "openai/gpt-oss-120b",
                        "provider": "nvidia",
                        "endpoint": None,
                        "api_key_env_var": "NVIDIA_API_KEY",
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "max_tokens": 8192,
                        "max_parallel_requests": 4,
                        "timeout": 300,
                        "extra_body": {"reasoning_effort": "high"},
                    },
                    "user_model": {
                        "alias": "user_model",
                        "model": "openai/gpt-oss-20b",
                        "provider": "nvidia",
                        "endpoint": None,
                        "api_key_env_var": "NVIDIA_API_KEY",
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "max_tokens": 1024,
                        "max_parallel_requests": 4,
                        "timeout": None,
                        "extra_body": None,
                    },
                },
            }
        )
    )

    out = _capture(
        print_available_runs,
        trajectory_root,
        tmp_path / "evaluations",
        tmp_path / "report",
    )
    # Assistant model surfaces with the "assistant" label.
    assert "assistant" in out
    assert "openai/gpt-oss-120b" in out
    # User model surfaces under its `_model`-stripped label.
    assert "openai/gpt-oss-20b" in out


def test_models_block_names_the_model_that_scored_the_run(tmp_path: Path) -> None:
    """The evaluator line comes from the evaluation rows, whatever the run manifest says."""
    trajectory_root = tmp_path / "trajectories"
    eval_root = tmp_path / "evaluations"
    _stage_trajectory_run(trajectory_root, "1714074853", locales=["en_US"], n_per_locale=1)
    envelope = {"judge_aliases": ["evaluator_model"], "judge_models": ["vendor/scoring-model"]}
    _stage_eval_run(eval_root, "1714074853", cell=json.dumps({"envelope": envelope}))

    out = _capture(print_available_runs, trajectory_root, eval_root, tmp_path / "report")
    assert re.search(r"evaluator\s*: vendor/scoring-model", out)


def test_missing_manifest_renders_em_dash(tmp_path: Path) -> None:
    trajectory_root = tmp_path / "trajectories"
    _stage_trajectory_run(
        trajectory_root,
        "1714074853",
        locales=["en_US"],
        n_per_locale=1,
    )
    # No manifest written -- the helper should render `—` in the models cell.
    out = _capture(
        print_available_runs,
        trajectory_root,
        tmp_path / "evaluations",
        tmp_path / "report",
    )
    assert "1714074853" in out
    # The em-dash sentinel appears (in the models cell + the dim eval/report
    # cells). At minimum the table renders without crashing.
    assert "\u2014" in out


class TestDisplayPath:
    """The helper renders short, repo-rooted display strings instead of
    blasting full ``/Users/<name>/...`` paths into the table title."""

    def test_path_inside_a_repo_starts_with_repo_name(self, tmp_path: Path) -> None:
        from usersim.reporting.runs import _display_path

        repo = tmp_path / "fake-repo"
        (repo / "output" / "trajectories").mkdir(parents=True)
        (repo / "pyproject.toml").write_text("")
        assert _display_path(repo / "output" / "trajectories") == ("fake-repo/output/trajectories")

    def test_path_with_git_marker_works_without_pyproject(self, tmp_path: Path) -> None:
        from usersim.reporting.runs import _display_path

        repo = tmp_path / "other-repo"
        (repo / ".git").mkdir(parents=True)
        (repo / "output").mkdir()
        assert _display_path(repo / "output") == "other-repo/output"

    def test_path_outside_any_repo_falls_back_to_last_two_components(
        self,
        tmp_path: Path,
    ) -> None:
        from usersim.reporting.runs import _display_path

        loose = tmp_path / "loose" / "deep" / "output"
        loose.mkdir(parents=True)
        # No pyproject.toml or .git anywhere up the tree from this leaf
        # (inside the tmp_path fixture). Fallback returns the last two
        # path components.
        assert _display_path(loose).endswith("deep/output")


# ── resuming a run you stopped part-way ──────────────────────────────────────


def _simulate_notebook_source() -> str:
    import json

    nb_path = _NOTEBOOKS / "01_simulate.ipynb"
    assert nb_path.is_file(), f"simulate notebook is missing: {nb_path}"
    nb = json.loads(nb_path.read_text())
    return "\n".join("".join(c.get("source", [])) for c in nb["cells"])


def test_simulate_notebook_can_pin_an_explicit_run_id() -> None:
    """RESUME only reaches the MOST RECENT run, so once a newer run exists an older
    partial is unreachable and its locales can never be filled in -- exactly what
    happened to run 1785990612, stopped at 7 of 13 locales with no manifest. A pinned
    SIM_RUN_ID is the way back to it, mirroring the eval notebook's knob of the same
    name.
    """
    src = _simulate_notebook_source()
    assert "SIM_RUN_ID" in src
    # Resolved through the shared helper, whose `label` names the knob to fix on a
    # bad id rather than failing somewhere downstream.
    assert 'resolve_run_or_raise(TRAJECTORY_PATH, SIM_RUN_ID, label="SIM_RUN_ID")' in src
    # It must take precedence over RESUME, not sit behind it.
    assert src.index("if SIM_RUN_ID is not None:") < src.index("elif RESUME:")


def test_simulate_notebook_warns_about_already_finished_locales() -> None:
    """The trajectory-id skip happens AFTER generation, so a locale left in LOCALES
    that the run already has is re-simulated at full LLM cost and then discarded.
    On a 100-row locale that is a long, silent waste; say so."""
    src = _simulate_notebook_source()
    assert "locales already on disk" in src
    assert "re-simulated at full LLM cost" in src


def test_simulate_notebook_writes_runs_as_the_cli_does() -> None:
    """The notebook saves through the same locked helpers as ``usersim simulate``.

    Writing partitions itself would let it resume a run of stored rows, share
    a run with another writer started in the same second, or save an id twice.
    """
    src = _simulate_notebook_source()
    assert "sampled_run_to_resume(TRAJECTORY_PATH)" in src
    assert "save_sampled_rows(" in src
    assert "write_locale_partition(" not in src and "new_run_id(" not in src
    # Each batch's Data Designer artifacts in a folder of its own.
    assert "artifact_path=_artifacts" in src
    # The bare legacy layout has no run folder to hold a manifest.
    assert "elif RUN_ID == LEGACY_RUN_ID:" in src


def test_resumed_manifest_describes_the_run_not_the_last_pass() -> None:
    """The manifest is first-writer-wins, so the pass that happens to complete a run
    writes the only manifest it will ever have. Recording `locales_requested=LOCALES`
    would describe a resumed run by the handful of locales that pass filled in, and
    there is no second chance to correct it."""
    src = _simulate_notebook_source()
    assert "_run_locales = sorted(" in src
    assert "locales_requested=_run_locales" in src
    assert "trajectories_requested={l: NUM_ROWS_PER_LOCALE for l in _run_locales}" in src
