# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the live CLI pipeline, skipping only absent provider credentials."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def preflight(models: Path) -> list[str]:
    # Keep imports within the reported boundary: a broken installation fails,
    # rather than masquerading as an unconfigured provider.
    import dotenv  # noqa: F401 -- required by load_local_env, which tolerates its absence

    from usersim.cli._env import load_local_env, missing_keys
    from usersim.cli._models import (
        REQUIRED_EVALUATOR_ALIASES,
        REQUIRED_SIMULATOR_ALIASES,
        assistant_replays_reasoning,
        load_models_config,
        require_aliases,
        required_api_key_env_vars,
        to_model_configs,
    )

    load_local_env()
    config = load_models_config(models)
    require_aliases(
        config,
        required=REQUIRED_SIMULATOR_ALIASES + REQUIRED_EVALUATOR_ALIASES,
        context="health check",
    )
    assistant_replays_reasoning(config, store_reasoning=True)
    # Validate model parameters through the same offline conversion as the CLI.
    to_model_configs(config)
    return sorted(missing_keys(required_api_key_env_vars(config)))


def run_stage(stage: str, arguments: list[str], timeout: float) -> None:
    print(f"==> {stage}", flush=True)
    subprocess.run([sys.executable, "-m", "usersim.cli", stage, *arguments], check=True, timeout=timeout)


def pipeline(args: argparse.Namespace, output: Path, outcome: dict[str, str]) -> None:
    trajectories = str(output / "trajectories")
    evaluations = str(output / "evaluations")
    outcome["stage"] = "simulate"
    run_stage(
        "simulate",
        [
            "--locale",
            args.locale,
            "--num-rows",
            "1",
            "--max-turns",
            "2",
            "--probe-mix",
            f"{args.probe}=1.0",
            "--models",
            str(args.models),
            "--out",
            trajectories,
        ],
        args.timeout,
    )
    runs = sorted(path for path in Path(trajectories).glob("run=*") if path.is_dir())
    if len(runs) != 1:
        raise RuntimeError("simulate must produce exactly one run directory")
    run = runs[0].name.removeprefix("run=")
    outcome["stage"] = "eval"
    run_stage(
        "eval",
        ["--run", run, "--trajectories", trajectories, "--models", str(args.models), "--out", evaluations],
        args.timeout,
    )
    outcome["stage"] = "report"
    run_stage(
        "report",
        ["--run", run, "--trajectories", trajectories, "--evaluations", evaluations, "--out", str(output / "report")],
        args.timeout,
    )
    if not (output / "report" / f"run={run}" / "index.html").is_file():
        raise RuntimeError("report produced no index.html")


def error_detail(error: Exception) -> str:
    # Avoid printing subprocess arguments or credential-bearing exception text.
    if isinstance(error, subprocess.CalledProcessError):
        return f"command exited with status {error.returncode}"
    if isinstance(error, subprocess.TimeoutExpired):
        return f"command exceeded {error.timeout:g} seconds"
    if isinstance(error, RuntimeError):
        return str(error)
    return f"{type(error).__name__}; check configuration, dependencies and installation"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--locale", default="ja_JP")
    parser.add_argument("--probe", default="sov_ai_dynamic")
    parser.add_argument("--timeout", type=float, default=900, help="maximum seconds per CLI command (default: 900)")
    parser.add_argument("--summary-file", type=Path, default=os.environ.get("HEALTH_SUMMARY_FILE") or None)
    args = parser.parse_args(argv)
    outcome = {"status": "failure", "stage": "preflight", "detail": ""}
    try:
        if args.timeout <= 0 or not args.timeout < float("inf"):
            raise ValueError("timeout must be finite and positive")
        missing = preflight(args.models)
        if missing:
            outcome.update(status="skip", detail=f"missing credentials: {', '.join(missing)}")
        else:
            with tempfile.TemporaryDirectory(prefix="usersim-health-") as temporary:
                pipeline(args, Path(temporary), outcome)
            outcome.update(status="pass", stage="complete", detail="simulate, eval and report completed")
    except Exception as error:
        outcome["detail"] = error_detail(error)
    print(json.dumps(outcome), flush=True)
    if args.summary_file:
        args.summary_file.write_text(f"**{outcome['status']}** ({outcome['stage']}): {outcome['detail']}\n")
    return int(outcome["status"] == "failure")


if __name__ == "__main__":
    raise SystemExit(main())
