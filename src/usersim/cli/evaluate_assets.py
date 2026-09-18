# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``usersim evaluate-assets --domain <d>`` — DD-judge quality scorecard for a bank.

Generic, domain-registry-backed (see ``asset_gen/registry.py``). Runs the domain's
LLM-judge evaluation over a generated bank and prints a per-axis quality scorecard
(+ writes ``_eval_scorecard.parquet``). Non-blocking (a scorecard, not a gate);
network-gated (needs the judge model alias + credentials).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from usersim.asset_gen.registry import domain_names, get_domain
from usersim.engine.core._assets import packaged_assets_dir


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "evaluate-assets",
        help="⚖️  DD-judge quality scorecard for a generated asset bank (--domain).",
        description=(
            "Run a domain's LLM-judge evaluation over a generated bank and print a "
            "per-axis quality scorecard. Complements the deterministic validate-assets "
            "gate; non-blocking. Needs the judge model alias + network."
        ),
    )
    p.add_argument("--domain", required=True, choices=domain_names(),
                   help="Asset domain to evaluate (e.g. financial_services).")
    p.add_argument("--locale", required=True, help="Locale, e.g. en_US.")
    p.add_argument("--dir", type=Path, default=None,
                   help="Bank directory (defaults to assets/<domain>/<locale>).")
    p.add_argument("--region-spec", type=Path, default=None,
                   help="region_spec for authoritative-value context (defaults to bundled).")
    p.add_argument("--models", type=Path, default=None,
                   help="Models TOML (judge alias); defaults to the bundled config.")
    p.add_argument("--model", action="append", default=None, metavar="ALIAS=MODEL",
                   help="Override a model alias, e.g. --model asset_judge_model="
                        "openai/openai/gpt-5.4. Repeatable. Provider auto-routes via "
                        "cli/model_catalog.py.")
    p.add_argument("--max-parallel", action="append", default=None, metavar="ALIAS=N",
                   help="Override an alias's concurrency, e.g. --max-parallel "
                        "asset_judge_model=8. Repeatable. (Accepted for parity with "
                        "gen-assets so a shared flag set works across both.)")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    from usersim.cli._models import (
        ConfigError, apply_model_overrides, apply_parallel_overrides,
        default_models_path, load_models_config, parse_cli_model_overrides,
        parse_cli_parallel_overrides,
    )

    dom = get_domain(args.domain)
    if dom.evaluate is None:
        print(f"  domain {args.domain!r} has no evaluator")
        return 2
    bank_dir = args.dir or (packaged_assets_dir() / args.domain / args.locale)
    spec_path = args.region_spec or (dom.region_spec_dir / f"{args.locale}.yaml")

    print(f"usersim evaluate-assets [{args.domain}] — {args.locale}")
    print(f"  bank: {bank_dir}")

    spec = None
    if spec_path.exists():
        try:
            spec = dom.load_spec(spec_path)
        except ValueError as e:
            print(f"  (region_spec invalid; value context skipped): {e}")

    models = load_models_config(args.models or default_models_path())
    overrides = parse_cli_model_overrides(args.model)
    if overrides:
        models = apply_model_overrides(models, overrides)
    parallel = parse_cli_parallel_overrides(args.max_parallel)
    if parallel:
        models = apply_parallel_overrides(models, parallel)
    try:
        scorecard = dom.evaluate(bank_dir, spec, models)
    except ConfigError as e:
        print(f"  config error: {e}")
        return 2
    print(scorecard.to_text())
    return 0
