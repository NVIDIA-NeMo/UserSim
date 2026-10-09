# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``usersim rescore`` — recompute deterministic scorers over an existing eval run.

Some scorers need no model. When one of them is fixed or gains an axis, the
signal for every already-evaluated trajectory is recoverable from the stored
conversation without asking an LLM anything. A full ``usersim eval`` would
re-run the judge ensemble too, which is where essentially all of the cost lives,
so this command exists to make "the scorer changed" cheap:

    usersim rescore --run 1786325828 --scorers language_compliance

What it does NOT do is touch the judge axes. It decodes each ``assistant_eval``
cell, replaces only ``cell["scorers"][<name>]``, and writes the frame back. The
``axes`` block, ``envelope``, and every scorer it was not asked about survive
byte-for-byte.

Two deliberate choices worth knowing about, because both are the kind of thing
that is invisible until it bites:

- **``envelope["scorers"]`` is left alone.** ``TrajectoryEvaluatorGenerator``
  skips a row when the config envelope matches the stored one, so rewriting the
  envelope here would make the next full eval re-judge every patched row at full
  LLM cost. The envelope keeps describing the run that produced the judge axes,
  which is what it is for.
- **Only genuinely model-free scorers are allowed.** ``financial_services`` and
  ``safety_agentic`` look deterministic (they emit mechanical axes) but branch
  into an LLM judge on some rows — ``task_tier == "dynamic"`` and certain
  ``sub_protocol`` values. Passing them here would silently spend money, so
  they are refused by name rather than trusted to behave.

The scorers' inputs come from the TRAJECTORY frame, not the eval frame: the eval
parquet carries only ``assistant_eval`` plus the three key columns, so
``conversation_messages`` and friends have to be re-joined on ``trajectory_id``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("usersim.cli.rescore")

_ROOT = Path(__file__).resolve().parents[1]

# Scorers that provably issue no LLM call: the only three whose envelopes carry
# ``scorer_kind="deterministic"``. Kept as an explicit allowlist rather than
# derived from that field, because the field is absent on the hybrids and a
# missing marker must not read as "safe".
DETERMINISTIC_SCORERS: tuple[str, ...] = (
    "language_compliance",
    "refusal_basics",
    "response_shape",
)

# Hybrids, named so the error can say WHY instead of "unknown scorer".
_HYBRID_SCORERS: dict[str, str] = {
    "financial_services": "calls an LLM judge on dynamic-tier rows",
    "safety_agentic": "calls an LLM judge on some sub-protocols",
    "health_disclosure_concealment": ("runs an LLM realized-behavior audit whenever a judge model is wired"),
}


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "rescore",
        help="♻️  Recompute deterministic scorers on an existing eval run (no LLM).",
        description=(
            "Recompute model-free scorers over an already-evaluated run and patch "
            "them into the stored eval cells, leaving the LLM judge axes untouched. "
            "Use after fixing a deterministic scorer or adding an axis to one, "
            "instead of paying for a full re-eval."
        ),
    )
    p.add_argument(
        "--evaluations",
        type=Path,
        default=_ROOT / "output" / "evaluations",
        help="Evaluations root (default output/evaluations).",
    )
    p.add_argument(
        "--trajectories",
        type=Path,
        default=_ROOT / "output" / "trajectories",
        help="Trajectories root, source of the scorer inputs (default output/trajectories).",
    )
    p.add_argument(
        "--run",
        default=None,
        help="Run id to patch. Defaults to the most recent run.",
    )
    p.add_argument(
        "--scorers",
        nargs="+",
        required=True,
        help=("Deterministic scorers to recompute. One or more of: " + ", ".join(DETERMINISTIC_SCORERS)),
    )
    p.add_argument(
        "--eval-column",
        default="assistant_eval",
        help="Name of the wide eval cell column (default assistant_eval).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing.",
    )
    p.set_defaults(func=run)


def _validate_scorers(names: list[str]) -> list[str]:
    """Reject anything that might reach an LLM, with a reason."""
    bad: list[str] = []
    for name in names:
        if name in DETERMINISTIC_SCORERS:
            continue
        if name in _HYBRID_SCORERS:
            bad.append(f"{name!r} ({_HYBRID_SCORERS[name]}) — refused: rescore is LLM-free")
        else:
            bad.append(f"{name!r} — not a known deterministic scorer")
    if bad:
        raise SystemExit("cannot rescore:\n  " + "\n  ".join(bad) + "\n\nallowed: " + ", ".join(DETERMINISTIC_SCORERS))
    return list(dict.fromkeys(names))


def run(args: argparse.Namespace) -> int:
    from usersim.cli._async import run_coroutine
    from usersim.engine.core.missing import none_for_missing
    from usersim.engine.core.storage import (
        read_partitioned_dataset,
        resolve_run_or_raise,
        write_run_partition,
    )
    from usersim.engine.evaluator.scorers import get_scorer

    scorers = _validate_scorers(args.scorers)
    run_id = resolve_run_or_raise(args.evaluations, args.run, label="--run")

    eval_df = read_partitioned_dataset(args.evaluations, run=run_id)
    traj_df = read_partitioned_dataset(args.trajectories, run=run_id)
    if eval_df is None or len(eval_df) == 0:
        raise SystemExit(f"no evaluation rows for run={run_id} under {args.evaluations}")
    if traj_df is None or len(traj_df) == 0:
        raise SystemExit(
            f"no trajectory rows for run={run_id} under {args.trajectories} — "
            "the scorers read conversation_messages from there, so the run's "
            "trajectories must still be on disk"
        )

    # The eval frame is the thing we rewrite, so keep exactly its columns. The
    # trajectory frame is joined in only to feed the scorers.
    eval_columns = list(eval_df.columns)
    traj_by_id = {str(r["trajectory_id"]): r for _, r in traj_df.iterrows() if r.get("trajectory_id") is not None}

    n_patched = 0
    n_unmatched = 0
    per_scorer_changed: dict[str, int] = {name: 0 for name in scorers}
    new_cells: list[Any] = []

    for _, row in eval_df.iterrows():
        raw = row.get(args.eval_column)
        cell = _decode(raw)
        tid = str(row.get("trajectory_id"))
        traj = traj_by_id.get(tid)
        if traj is None or not cell:
            n_unmatched += 1
            new_cells.append(raw)
            continue

        # Scorers take the whole trajectory row as a dict, with every empty
        # cell as None.
        traj_row = none_for_missing({k: traj[k] for k in traj.index})
        blocks = cell.setdefault("scorers", {})
        changed = False
        for name in scorers:
            before = blocks.get(name)
            try:
                blocks[name] = run_coroutine(get_scorer(name)(traj_row, {}))
            except Exception as exc:  # a bad row must not lose the whole run
                logger.warning("scorer %s failed on trajectory_id=%s: %s", name, tid, exc)
                continue
            if blocks[name] != before:
                per_scorer_changed[name] += 1
                changed = True
        if changed:
            # Says where these blocks came from, so a reader is not left
            # puzzling over a cell whose scorers disagree with its envelope.
            cell["rescored"] = sorted(set(cell.get("rescored") or []) | set(scorers))
            n_patched += 1
        new_cells.append(json.dumps(cell, ensure_ascii=False))

    print(f"run={run_id}: {len(eval_df)} eval row(s), {n_patched} patched")
    for name, count in per_scorer_changed.items():
        print(f"  {name}: {count} row(s) changed")
    if n_unmatched:
        print(f"  {n_unmatched} row(s) had no matching trajectory and were left as-is")

    if args.dry_run:
        print("--dry-run: nothing written")
        return 0

    out = eval_df.copy()
    out[args.eval_column] = new_cells
    out = out[eval_columns]
    subroot = write_run_partition(out, args.evaluations, run_id, overwrite=True)
    print(f"wrote {subroot}")
    return 0


def _decode(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}
