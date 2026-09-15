# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``usersim`` command-line interface.

Thin orchestration layer over the simulator + evaluator + reporting
packages. Mirrors the notebook flow so a user can drive an end-to-end
run from a script or CI without opening Jupyter:

    usersim panel    --locale en_US --num-personas 50  --out panel.parquet
    usersim simulate --panel panel.parquet            --out trajectories.parquet
    usersim eval     --trajectories trajectories.parquet --out evaluations.parquet
    usersim report   --trajectories trajectories.parquet --evaluations evaluations.parquet --out report/

Each subcommand is implemented in its own module under :mod:`cli`.
The shared pipeline-construction helpers live in :mod:`cli._pipeline`
(used by both the CLI and the notebooks under ``notebooks/``). Model
configuration is loaded from a TOML file via :mod:`cli._models`; a
sensible default ships at ``cli/models_default.toml``.

Reproducibility note: the CLI does not invent identity. ``persona_uuid``
and ``trajectory_id`` are computed by the simulator plugin from input
content + probe metadata + resolved model identities, so the same
panel + same model config + same plugin code yields the same trajectory
ids. The CLI can therefore safely be re-run; the simulator's idempotency
helpers skip rows whose ``trajectory_id`` already exists in the output
parquet.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

from usersim.cli._errors import ConfigError

__version__ = "0.1.0"

def _register_extension_commands(subparsers: argparse._SubParsersAction) -> None:
    """Let installed packages contribute subcommands.

    An entry point resolves to the same ``register(subparsers)`` callable the
    built-in command modules expose, so a contributed command is built the
    same way and gets the same ``--help`` treatment.

    A command that collides with a built-in is skipped: argparse raises on a
    duplicate name, and letting that propagate would mean one installed
    package could make the whole CLI refuse to start.
    """
    import logging

    from usersim.engine.core._extensions import COMMANDS, load_extensions

    for name, register in load_extensions(COMMANDS):
        if name in subparsers.choices:
            logging.getLogger("usersim.engine").warning(
                "  |-- extensions: command %r is already provided by this "
                "package; ignoring the contributed one.", name,
            )
            continue
        try:
            register(subparsers)
        except Exception as exc:
            logging.getLogger("usersim.engine").warning(
                "  |-- extensions: command %r failed to register and was "
                "skipped: %s", name, exc,
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="usersim",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "🎭  NeMo UserSim \u2014 Population-Grounded User Simulation\n"
            "\n"
            "Simulate a statistically representative, census-grounded population of\n"
            "users having multi-turn conversations with your LLM \u2014 then score the\n"
            "assistant and turn the trajectories into evaluation signal and shareable\n"
            "reports. Every step runs from the command line; no notebook required.\n"
        ),
        epilog=(
            "typical flow\n"
            "  👥 panel  \u2192  🗣 simulate  \u2192  \u2696\ufe0f  eval  \u2192  📊 report\n"
            "\n"
            "  usersim panel    --locale en_US --num-personas 50 --out panel.parquet\n"
            "  usersim simulate --panel panel.parquet            --out output/trajectories\n"
            "  usersim eval     --trajectories output/trajectories --out output/evaluations\n"
            "  usersim report   --trajectories output/trajectories \\\n"
            "                    --evaluations output/evaluations   --out output/report/\n"
            "\n"
            "  🔥 usersim smoke      verify the install offline (no LLM calls)\n"
            "\n"
            "Run `usersim <command> -h` for a command's options.\n"
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"usersim {__version__}"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (-v=INFO, -vv=DEBUG).",
    )

    subparsers = parser.add_subparsers(
        dest="command", required=True, metavar="<command>",
    )

    from usersim.cli import panel as _panel
    from usersim.cli import simulate as _simulate
    from usersim.cli import evaluate as _evaluate
    from usersim.cli import rescore as _rescore
    from usersim.cli import select as _select
    from usersim.cli import report as _report
    from usersim.cli import gen_assets as _gen_assets
    from usersim.cli import validate_assets as _validate_assets
    from usersim.cli import evaluate_assets as _evaluate_assets
    from usersim.cli import smoke as _smoke
    from usersim.cli import setup_ngc as _setup_ngc

    _panel.register(subparsers)
    _simulate.register(subparsers)
    _evaluate.register(subparsers)
    _rescore.register(subparsers)
    _select.register(subparsers)
    _report.register(subparsers)
    _gen_assets.register(subparsers)
    _validate_assets.register(subparsers)
    _evaluate_assets.register(subparsers)
    _smoke.register(subparsers)
    _setup_ngc.register(subparsers)

    _register_extension_commands(subparsers)

    return parser


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 2
    try:
        return int(func(args) or 0)
    except KeyboardInterrupt:
        logging.warning("interrupted")
        return 130
    except ConfigError as e:
        logging.error("%s", e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
