# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``usersim eval`` — run the trajectory-evaluator over a trajectory parquet.

Drives the ``trajectory-evaluator`` Data Designer plugin. The judge
ensemble is configured via ``--judges`` (a comma-separated list of
DD model aliases, optionally suffixed with a ``:family`` hint to satisfy
the ensemble-diversity validator). Axes default to "all axes applicable
to each row's probe_family"; pass ``--axes`` to restrict.

Re-runs replace the evaluator partition for the selected run id before
writing fresh rows, so stale eval files cannot accumulate alongside
new ones.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Any

from usersim.engine.evaluator.prompts import EVAL_PROMPT_VERSION

logger = logging.getLogger("usersim.cli.eval")

#: Seed for ``--num-records-per-locale`` sampling, recorded with each pass.
_SAMPLE_SEED = 42

#: Trajectory columns reattached to every evaluator row for partitioning and joins.
_KEY_COLS = ("trajectory_id", "locale", "probe_family")


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "eval",
        help="⚖️  Run the trajectory-evaluator and write an evaluations parquet.",
        description=(
            "Score the assistant under test on a stored trajectory "
            "parquet using a multi-family judge ensemble + probe-specific "
            "scorers. Produces a wide JSON cell per row."
        ),
    )
    p.add_argument(
        "--trajectories",
        type=Path,
        required=True,
        help="Input trajectory parquet (output of `usersim simulate`).",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help=(
            "Output evaluator dataset root. Becomes a Hive-partitioned "
            "directory tree: "
            "<out>/run=<id>/locale=X/probe_family=Y/part-N.parquet. "
            "The run id is inherited from the trajectory run being "
            "evaluated (so evaluations + reports group naturally by run)."
        ),
    )
    p.add_argument(
        "--run",
        type=str,
        default=None,
        help=(
            "Trajectory run id to evaluate. Defaults to the most-recent "
            "run under --trajectories (the typical 'eval what I just "
            "simulated' case). Pass an explicit Unix-epoch run id "
            "(e.g. 1714074853) to re-evaluate a historical run, or "
            "'latest' as an explicit alias for the default."
        ),
    )
    p.add_argument(
        "--models",
        type=Path,
        default=None,
        help="Path to models TOML. Defaults to cli/models_default.toml.",
    )
    p.add_argument(
        "--judges",
        type=str,
        default="evaluator_model",
        help=(
            "Comma-separated DD model aliases to use as judges, optionally "
            "with ``:family`` (e.g., evaluator_model:openai,evaluator_secondary:anthropic). "
            "≥2 distinct families enforced for ensembles of size ≥2."
        ),
    )
    p.add_argument(
        "--axes",
        type=str,
        default=None,
        help=("Comma-separated axis names to score. Default: all axes applicable to each row's probe_family."),
    )
    p.add_argument(
        "--scorers",
        type=str,
        default=None,
        help=(
            "Comma-separated deterministic scorer names (registered in "
            "evaluator/scorers/). Default: empty (no scorers)."
        ),
    )
    p.add_argument(
        "--prompt-version",
        type=str,
        default=EVAL_PROMPT_VERSION,
        help="Bump to invalidate prior eval cells with a different version.",
    )
    p.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Force re-judging of all rows (overrides default skip_if_existing).",
    )
    p.add_argument(
        "--num-records",
        type=int,
        default=None,
        help=("Evaluate only the first N ordered trajectory rows. Default: evaluate the full run."),
    )
    p.add_argument(
        "--num-records-per-locale",
        type=int,
        default=None,
        help=(
            "Evaluate up to N sampled trajectory rows per locale. "
            "Default: evaluate the full run. Mutually exclusive with "
            "--num-records."
        ),
    )
    p.add_argument(
        "--output-column",
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


def _parse_judges(spec: str) -> list[dict]:
    out: list[dict] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            alias, family = token.split(":", 1)
            out.append({"alias": alias.strip(), "family": family.strip()})
        else:
            out.append({"alias": token})
    if not out:
        raise SystemExit("--judges is empty")
    return out


def _parse_csv_or_none(spec: str | None) -> list[str] | None:
    if spec is None:
        return None
    items = [x.strip() for x in spec.split(",") if x.strip()]
    return items or None


def _choose_rows(args: argparse.Namespace, seed_path: Path) -> tuple[Any, Any, int, dict[str, Any]]:
    """The trajectory rows this pass evaluates, and how they were chosen.

    Returns their key columns in seed order, the sampled rows Data Designer
    must be seeded from under ``--num-records-per-locale`` (None when it reads
    ``seed_path`` itself), the row count, and the pass record for the
    eval-sample manifest.
    """
    import pandas as pd

    if args.num_records is not None and args.num_records_per_locale is not None:
        raise ValueError("--num-records and --num-records-per-locale are mutually exclusive")
    if args.num_records_per_locale is None:
        keys = pd.read_parquet(seed_path, columns=list(_KEY_COLS))
        total = len(keys)
        num_records = total if args.num_records is None else min(int(args.num_records), total)
        if num_records <= 0:
            raise ValueError("--num-records must be positive when provided")
        if num_records < total:
            return keys, None, num_records, {"mode": "first_n", "n": num_records, "n_total": total}
        return keys, None, num_records, {"mode": "full", "n": None, "n_total": total}

    per_locale = int(args.num_records_per_locale)
    if per_locale <= 0:
        raise ValueError("--num-records-per-locale must be positive when provided")
    full = pd.read_parquet(seed_path)
    sampled = pd.concat(
        [
            group.sample(n=min(per_locale, len(group)), random_state=_SAMPLE_SEED)
            for _, group in full.groupby("locale", sort=True)
        ]
    ).sort_index()
    record = {"mode": "per_locale", "n": per_locale, "n_total": len(full)}
    return sampled[list(_KEY_COLS)].reset_index(drop=True), sampled, len(sampled), record


def run(args: argparse.Namespace) -> int:
    from usersim.cli._models import (
        default_models_path,
        load_models_config,
        warn_if_missing_api_key,
    )

    models_path = args.models or default_models_path()
    models = load_models_config(models_path)

    judges = _parse_judges(args.judges)
    axes = _parse_csv_or_none(args.axes)
    scorers = _parse_csv_or_none(args.scorers) or []

    # Confirm every requested judge alias is declared in the models config.
    declared = set(models.aliases())
    missing = [j["alias"] for j in judges if j["alias"] not in declared]
    if missing:
        raise SystemExit(f"--judges references aliases not in {models_path}: {missing}; declared: {sorted(declared)}")

    if args.dry_run:
        print("usersim eval — dry run")
        print(f"  trajectories:  {args.trajectories}")
        print(f"  judges:        {judges}")
        print(f"  axes:          {axes if axes is not None else 'auto (per probe_family)'}")
        print(f"  scorers:       {scorers}")
        print(f"  prompt_ver:    {args.prompt_version}")
        print(f"  skip_existing: {not args.no_skip_existing}")
        print(f"  num_records:   {args.num_records if args.num_records is not None else 'all'}")
        print(f"  per_locale:    {args.num_records_per_locale if args.num_records_per_locale is not None else 'off'}")
        print(f"  output column: {args.output_column}")
        print(f"  models config: {models_path}")
        print(f"  output:        {args.out}")
        return 0

    if (msg := warn_if_missing_api_key(models)) is not None:
        logger.warning(msg)

    from usersim.cli._pipeline import build_evaluator_config_builder
    from usersim.engine.core.storage import (
        is_partitioned_directory,
        materialize_to_temp_file,
        resolve_run,
        run_subroot,
        write_partitioned_dataset,
    )
    from usersim.reporting.eval_manifest import EVAL_STORE_MANIFEST_FILENAME, record_eval_sample_pass

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Resolve the trajectory run we'll evaluate. ``--run`` honours
    # the user's explicit pin; otherwise default to the latest run
    # under ``--trajectories``. Eval output inherits the same run id
    # so each run becomes the unit of "X's trajectories + X's evals
    # + X's reports".
    traj_path = Path(args.trajectories).resolve()
    run_id = resolve_run(traj_path, getattr(args, "run", None))
    if run_id is None:
        raise FileNotFoundError(f"no runs found under {traj_path} — run `usersim simulate` first")
    print(f"  run_id:        {run_id}")

    # Resolve the trajectory dataset for THIS run only. The
    # underlying read goes through DataDesigner's LocalFileSeedSource
    # which expects a single file; partitioned trees get materialized
    # to a temp parquet first.
    traj_run_root = run_subroot(traj_path, run_id) if is_partitioned_directory(traj_path) else traj_path
    temp_seed: Path | None = None
    temp_sample_seed: Path | None = None
    if is_partitioned_directory(traj_run_root):
        temp_seed = materialize_to_temp_file(traj_run_root)
        seed_source_path = temp_seed
    else:
        seed_source_path = traj_run_root

    try:
        import tempfile

        seed_df, sampled, num_records, sample_pass = _choose_rows(args, seed_source_path)
        if sampled is not None:
            tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False, prefix="usersim_eval_seed_")
            tmp.close()
            temp_sample_seed = seed_source_path = Path(tmp.name)
            sampled.to_parquet(temp_sample_seed, index=False)

        data_designer, builder = build_evaluator_config_builder(
            models=models,
            trajectory_parquet=seed_source_path,
            judges=judges,
            axes=axes,
            scorers=scorers,
            skip_if_existing=not args.no_skip_existing,
            output_column=args.output_column,
            prompt_version=args.prompt_version,
        )
        result = data_designer.create(builder, num_records=num_records)
        df = result.load_dataset()

        # Data Designer returns only generated columns for this plugin.
        # Reattach trajectory keys needed for partitioning and downstream
        # joins. The seed source is ordered, so row order is the join
        # contract here.
        if len(df) > len(seed_df):
            raise ValueError(
                f"Evaluator output row count {len(df)} exceeds "
                f"trajectory row count {len(seed_df)}; cannot safely reattach keys."
            )
        seed_df = seed_df.head(len(df)).reset_index(drop=True)
        df = df.reset_index(drop=True)
        for col in _KEY_COLS:
            if col not in df.columns:
                df[col] = seed_df[col]
    finally:
        if temp_sample_seed is not None and temp_sample_seed.exists():
            temp_sample_seed.unlink()
        if temp_seed is not None and temp_seed.exists():
            temp_seed.unlink()

    # Eval output lands under the SAME run id as the source
    # trajectories: ``output/evaluations/run=<id>/locale=<l>/probe_family=<f>/...``.
    eval_run_root = run_subroot(out, run_id)
    if eval_run_root.exists():
        logger.info("removing existing evaluator rows for run=%s: %s", run_id, eval_run_root)
        shutil.rmtree(eval_run_root)
    eval_run_root.mkdir(parents=True, exist_ok=True)
    write_partitioned_dataset(df, eval_run_root)
    # The report labels the run by how it was sampled, so a partial pass
    # cannot be shown as a full one.
    record_eval_sample_pass(
        eval_run_root / EVAL_STORE_MANIFEST_FILENAME,
        run_id=run_id,
        random_seed=_SAMPLE_SEED,
        eval_locales=None,
        n_selected=len(df),
        trajectory_ids=df["trajectory_id"].astype(str).tolist(),
        fresh=True,
        **sample_pass,
    )
    print(f"wrote {len(df)} evaluator rows to {out}")
    print(
        f"\nNext step: render the report\n"
        f"  usersim report --run {run_id} \\\n"
        f"    --trajectories {args.trajectories} \\\n"
        f"    --evaluations {out} --out output/report"
    )
    return 0
