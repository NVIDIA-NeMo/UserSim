# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``usersim select`` — curate high-quality trajectories from a run.

Joins a run's trajectories + evaluations, applies a named selection profile
of quality gates, and writes the passing subset as a curated, Hive-partitioned
parquet under ``--out/run=<id>/profile=<name>/``. No LLM calls — pure pandas +
the shared :mod:`selection` engine the notebook also drives.

Output is namespaced by profile so re-running with a different profile never
clobbers a prior curation. The run id is inherited from the trajectory run
being selected (so trajectories + evaluations + curated subsets group by run).
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

logger = logging.getLogger("usersim.cli.select")


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "select",
        help="✂️  Curate high-quality trajectories into a partitioned subset.",
        description=(
            "Apply a selection profile (quality gates over the evaluator's "
            "scores + simulator health) to a run's trajectories and write the "
            "passing subset, namespaced by profile, with a funnel manifest."
        ),
    )
    p.add_argument(
        "--trajectories",
        type=Path,
        required=True,
        help="Trajectory dataset root (output of `usersim simulate`).",
    )
    p.add_argument(
        "--evaluations",
        type=Path,
        required=True,
        help="Evaluations dataset root (output of `usersim eval`).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("output/curated"),
        help=(
            "Output root. Becomes "
            "<out>/run=<id>/profile=<name>/locale=X/probe_family=Y/part-*.parquet "
            "plus a _selection_manifest.json. Default: output/curated."
        ),
    )
    p.add_argument(
        "--run",
        type=str,
        default=None,
        help=(
            "Trajectory run id(s) to select from. Defaults to the most-recent "
            "run under --trajectories; pass an explicit id, 'latest', or a "
            "comma-separated list of ids to pull from multiple runs."
        ),
    )
    p.add_argument(
        "--profile",
        type=str,
        default="generous",
        help="Selection profile name (default: generous).",
    )
    p.add_argument(
        "--min-turns",
        type=int,
        default=None,
        help="Override the profile's min_turns gate (opt-in; default off).",
    )
    p.add_argument(
        "--holdout-fraction",
        type=float,
        default=None,
        help=(
            "Reserve this fraction of passing personas as a 'holdout' split "
            "(deterministic by persona_uuid) for Goodhart-safe evaluation."
        ),
    )
    p.add_argument(
        "--max-records",
        type=int,
        default=1024,
        help=(
            "Cap the curated subset at N highest-quality records (default "
            "1024). Pass 0 (or a negative) to keep every passing record."
        ),
    )
    p.add_argument(
        "--max-per-locale",
        type=int,
        default=None,
        help=(
            "Keep at most N highest-quality passing records PER LOCALE across "
            "the whole run (applied before --max-records, so both compose). "
            "Default: no per-locale cap."
        ),
    )
    p.add_argument(
        "--stratify-by",
        type=str,
        default="probe_family",
        help=(
            "Comma-separated columns to balance --max-records across, taking "
            "the highest-quality records within each stratum (default "
            "'probe_family'). Pass 'none' for a single global top-K."
        ),
    )
    p.add_argument(
        "--schema",
        type=str,
        default="full",
        help=(
            "Output column schema: a preset (full / compact), a comma-"
            "separated column/group list, or 'full' (default = all columns). "
            "Group tokens 'conversation' and 'persona' expand to their columns."
        ),
    )
    p.add_argument(
        "--hf-repo",
        type=str,
        default=None,
        help=(
            "If set, also upload the curated subset to this Hugging Face "
            "dataset repo (id or URL, e.g. nvidia/usersim-...). Private by "
            "default; auth via HF_TOKEN or a cached huggingface-cli login."
        ),
    )
    p.add_argument(
        "--hf-public",
        action="store_true",
        help="Make the uploaded HF dataset public (default: private).",
    )
    p.add_argument(
        "--generation-model",
        type=str,
        default=None,
        help=(
            "Name of the generation LLM (assistant under test) to record in "
            "the HF dataset card / manifest, e.g. 'gemma-4'. Not stored in the "
            "trajectory parquet, so supply it here for provenance."
        ),
    )
    p.add_argument(
        "--keep-all",
        action="store_true",
        help=(
            "Write every trajectory annotated with selection verdict columns "
            "(not just the passing subset), for funnel/threshold tuning."
        ),
    )
    p.add_argument(
        "--eval-column",
        type=str,
        default="assistant_eval",
        help="Name of the wide eval cell column (default assistant_eval).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved plan and exit.",
    )
    p.set_defaults(func=run)


def _resolve_profile(args: argparse.Namespace):
    from usersim.selection.profiles import get_profile

    profile = get_profile(args.profile)
    overrides: dict = {}
    if args.min_turns is not None:
        overrides["min_turns"] = int(args.min_turns)
    if args.holdout_fraction is not None:
        overrides["holdout_fraction"] = float(args.holdout_fraction)
    if overrides:
        profile = dataclasses.replace(profile, **overrides)
    return profile


def _print_funnel(summary) -> None:
    print(f"  profile:       {summary.profile}")
    print(f"  total:         {summary.total}")
    print(f"  passed:        {summary.passed}")
    print(f"  dropped:       {summary.dropped}")
    caps = []
    if getattr(summary, "max_per_locale", None) is not None:
        caps.append(f"<={summary.max_per_locale}/locale")
    if summary.max_records is not None:
        caps.append(f"<={summary.max_records} total")
    if caps and summary.selected < summary.passed:
        print(f"  selected:      {summary.selected} (capped: {', '.join(caps)})")
    else:
        print(f"  selected:      {summary.selected}")
    if summary.drop_reasons:
        print("  drop reasons:")
        for reason, count in sorted(
            summary.drop_reasons.items(), key=lambda kv: (-kv[1], kv[0])
        ):
            print(f"    {reason:40s} {count}")
    if summary.holdout:
        print(f"  holdout split: {summary.holdout}")


def run(args: argparse.Namespace) -> int:
    from usersim.selection import (
        build_manifest,
        load_runs,
        push_to_hf,
        select_trajectories,
        write_curated_dataset,
    )
    from usersim.selection.io import _provenance

    profile = _resolve_profile(args)
    max_records = args.max_records if args.max_records and args.max_records > 0 else None
    max_per_locale = (
        args.max_per_locale
        if args.max_per_locale and args.max_per_locale > 0
        else None
    )
    stratify_by = None
    if args.stratify_by and args.stratify_by.strip().lower() != "none":
        stratify_by = tuple(c.strip() for c in args.stratify_by.split(",") if c.strip())
    # --run accepts a single id, 'latest', or a comma-separated list.
    runs: object = args.run
    if isinstance(args.run, str) and "," in args.run:
        runs = [r.strip() for r in args.run.split(",") if r.strip()]

    if args.dry_run:
        print("usersim select — dry run")
        print(f"  trajectories:  {args.trajectories}")
        print(f"  evaluations:   {args.evaluations}")
        print(f"  run:           {args.run or 'latest'}")
        print(f"  profile:       {profile.label()}")
        print(f"  axis_floors:   {dict(profile.axis_floors)}")
        print(f"  min_turns:     {profile.min_turns}")
        print(f"  holdout:       {profile.holdout_fraction}")
        print(f"  max_records:   {max_records if max_records is not None else 'all'}")
        print(f"  max_per_locale:{max_per_locale if max_per_locale is not None else 'none'}")
        print(f"  stratify_by:   {list(stratify_by) if stratify_by else 'none (global top-K)'}")
        print(f"  schema:        {args.schema}")
        print(f"  keep_all:      {args.keep_all}")
        print(f"  eval_column:   {args.eval_column}")
        print(f"  out:           {args.out}")
        print(f"  hf_repo:       {args.hf_repo or '(none)'}"
              + ("" if not args.hf_repo else f"  private={not args.hf_public}"))
        return 0

    traj_df, eval_df, run_id = load_runs(
        args.trajectories, args.evaluations, runs=runs
    )
    print(f"  run_id:        {run_id}")
    logger.info("loaded %d trajectory rows, %d eval rows", len(traj_df),
                0 if eval_df is None else len(eval_df))

    result = select_trajectories(
        traj_df, eval_df, profile=profile, eval_column=args.eval_column,
        max_records=max_records, max_per_locale=max_per_locale,
        stratify_by=stratify_by,
    )
    _print_funnel(result.summary)

    to_write = result.all if args.keep_all else result.curated
    out_root = write_curated_dataset(
        to_write, args.out, run_id, profile,
        summary=result.summary, eval_column=args.eval_column,
        columns=args.schema,
    )
    label = "annotated" if args.keep_all else "curated"
    print(f"wrote {len(to_write)} {label} rows to {out_root} (schema={args.schema})")

    if args.hf_repo:
        manifest = build_manifest(
            profile=profile, summary=result.summary, source_run=str(run_id),
            eval_column=args.eval_column, provenance=_provenance(to_write, args.eval_column),
            generation_model=args.generation_model,
        )
        url = push_to_hf(
            to_write, args.hf_repo, private=not args.hf_public,
            manifest=manifest, profile=profile, columns=args.schema,
        )
        print(f"uploaded {len(to_write)} rows to {url} (private={not args.hf_public})")
    return 0
