# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``usersim report`` — render the capability dashboard.

This is the cheapest CLI subcommand: no LLM calls, just pandas + JSON +
embedded React/Vega HTML. Reads a trajectory parquet (mandatory) and an
optional evaluations parquet (output of `usersim eval`); writes the
five capability-dashboard artifacts into ``--out``::

    --out report/run=<id>/
        index.html             # static no-build React/Vega dashboard
        report_manifest.json   # run-level metadata + per-cell summaries
        capability_matrix.json # per (capability × locale) cell
        coverage_summary.json  # per (locale × probe_family) coverage
        triage_queue.json      # ranked failed/missing cells

The four JSON artifacts feed the React/Vega view embedded in
``index.html``; they ship together as a single deliverable.

Run-id handling mirrors `usersim simulate` / `eval`: directory inputs
default to the latest run under ``--trajectories``; pass ``--run <id>``
to report on a historical run, or ``--run all`` for cross-run.

An evaluation parquet may key the assistant score as ``eval_v1`` or
``assistant_eval_v2`` rather than the canonical ``assistant_eval``. All
three are accepted in ``build_capability_report``, so such a parquet
renders without re-evaluation or extra flags.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger("usersim.cli.report")


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "report",
        help="📊 Render the capability dashboard from trajectory + evaluation parquets.",
        description=(
            "Read a trajectory parquet (and optionally an evaluations "
            "parquet) and write the capability-dashboard artifacts: "
            "``index.html`` plus four JSON cells "
            "(``report_manifest`` / ``capability_matrix`` / "
            "``coverage_summary`` / ``triage_queue``)."
        ),
    )
    p.add_argument(
        "--trajectories",
        type=Path,
        required=True,
        help="Trajectory parquet (output of `usersim simulate`).",
    )
    p.add_argument(
        "--evaluations",
        type=Path,
        default=None,
        help=(
            "Evaluations parquet (output of `usersim eval`). When "
            "omitted, the report still writes a manifest and coverage "
            "but the capability cells render as ``not measured``."
        ),
    )
    p.add_argument(
        "--eval-column",
        type=str,
        default="assistant_eval",
        help=(
            "Name of the wide eval cell column in the evaluations "
            "parquet (default ``assistant_eval``). Legacy column names "
            "``eval_v1`` and ``assistant_eval_v2`` are auto-aliased."
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help=(
            "Output directory; created if missing. Reports for a "
            "specific run land under <out>/run=<id>/ so each run "
            "has its own self-contained artifact set."
        ),
    )
    p.add_argument(
        "--run",
        type=str,
        default=None,
        help=(
            "Run id to report on (must exist under --trajectories AND "
            "--evaluations). Defaults to the most-recent run under "
            "--trajectories. Pass an explicit Unix-epoch run id to "
            "report on a historical run, or ``all`` for cross-run."
        ),
    )
    p.add_argument(
        "--open",
        action="store_true",
        help=("Open the rendered report in the default browser. Opt-in, so CI and headless runs never spawn one."),
    )
    p.add_argument(
        "--list-runs",
        action="store_true",
        help=("List the run ids available under --trajectories and exit, without rendering anything."),
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    from usersim.reporting import build_capability_report, write_capability_dashboard_artifacts

    if not args.trajectories.exists():
        raise SystemExit(f"trajectory parquet not found: {args.trajectories}")

    from usersim.engine.core.storage import (
        list_runs,
        read_partitioned_dataset,
        resolve_run,
        run_subroot,
    )

    if getattr(args, "list_runs", False):
        return _print_runs(args.trajectories, list_runs)

    # Deferred until after --list-runs so merely asking what exists does
    # not create an output directory.
    base_out = Path(args.out).resolve()
    base_out.mkdir(parents=True, exist_ok=True)

    # Resolve which run to report on. For directory inputs, default to
    # the most-recent run under ``--trajectories``. For single-file
    # inputs (legacy / test fixtures), skip run resolution and report
    # into ``base_out`` directly — single files don't have the
    # run-partition concept.
    if args.trajectories.is_file():
        run_id = None
        out_dir = base_out
    else:
        run_id = resolve_run(args.trajectories, getattr(args, "run", None))
        if run_id is None:
            raise SystemExit(f"no runs found under {args.trajectories} — run `usersim simulate` first")
        out_dir = run_subroot(base_out, run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    if run_id is not None:
        print(f"reporting on run_id={run_id} → {out_dir}")

    logger.info(
        "loading trajectories from %s (run=%s)",
        args.trajectories,
        run_id,
    )
    traj_df = read_partitioned_dataset(args.trajectories, run=run_id)
    logger.info("loaded %d trajectory rows", len(traj_df))

    eval_df = None
    if args.evaluations is not None:
        if not args.evaluations.exists():
            raise SystemExit(f"evaluations parquet not found: {args.evaluations}")
        logger.info("loading evaluations from %s", args.evaluations)
        eval_df = read_partitioned_dataset(args.evaluations, run=run_id)

    report = build_capability_report(
        traj_df,
        eval_df,
        eval_column=args.eval_column,
        run_id=run_id,
        trajectory_root=args.trajectories,
    )

    write_capability_dashboard_artifacts(report, out_dir)

    # index.html is the one a person opens; the rest are machine-readable
    # cells. Listing all five identically leaves a first-time user guessing.
    data_artifacts = (
        "report_manifest.json",
        "capability_matrix.json",
        "coverage_summary.json",
        "triage_queue.json",
    )
    index = (out_dir / "index.html").resolve()

    print(f"wrote {len(data_artifacts) + 1} artifacts to {out_dir}/")
    # Both forms: the URI is clickable but percent-encodes ``run=``, which
    # makes it awkward to paste into a shell.
    print(f"\n  open this:  {index.as_uri()}")
    print(f"        path:  {index}")
    print("\n  data:")
    for name in data_artifacts:
        print(f"    {out_dir / name}")

    if getattr(args, "open", False):
        _open_in_browser(index.as_uri())

    return 0


def _print_runs(trajectories: Path, list_runs) -> int:
    """List available run ids, newest first, and say how to report on one.

    Answers "which runs do I have?" without rendering anything, so finding
    a run id does not mean building a report for the wrong one first.
    """
    if trajectories.is_file():
        print(f"{trajectories} is a single parquet, not a run-partitioned tree")
        return 0

    runs = list_runs(trajectories)
    if not runs:
        print(f"no runs under {trajectories} — run `usersim simulate` first")
        return 1

    print(f"{len(runs)} run(s) under {trajectories}, newest first:\n")
    for run_id in reversed(runs):
        print(f"  {run_id}")
    print(f"\nreport on one with:  usersim report --run <run_id> --trajectories {trajectories} --out output/report")
    return 0


def _open_in_browser(url: str) -> None:
    """Open ``url`` in the default browser, never failing the run.

    A headless or sandboxed environment has no browser to hand off to, and
    the report is already written by this point -- so a failure here is
    reported and ignored rather than turned into a non-zero exit.
    """
    import webbrowser

    try:
        webbrowser.open(url)
    except Exception as exc:  # pragma: no cover — platform-dependent
        logger.warning("could not open a browser (%s); open %s yourself", exc, url)
