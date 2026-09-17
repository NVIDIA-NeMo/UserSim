# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``usersim gen-assets --domain <d>`` — generate an asset bank for a domain.

Generic, domain-registry-backed (see ``asset_gen/registry.py``). Expands a
validated region_spec into a committed asset bank (+ embeddings). Live generation
needs the doc-gen/embedding model aliases + network; ``--dry-run`` prints the plan
offline. Per-domain generation knobs are injected by the domain's ``add_gen_args``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from usersim.asset_gen.registry import domain_names, get_domain
from usersim.engine.core._assets import packaged_assets_dir


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "gen-assets",
        help="🏦 Generate an asset bank for a domain (--domain).",
        description=(
            "Expand a domain's region_spec into a committed asset bank (corpus + "
            "embeddings). Use --dry-run to preview the plan without any LLM calls. "
            "Per-domain flags are added based on --domain (see the domain's help)."
        ),
    )
    p.add_argument("--domain", required=True, choices=domain_names(),
                   help="Asset domain to generate (e.g. financial_services).")
    p.add_argument("--locale", required=True, help="Locale, e.g. en_US.")
    p.add_argument("--region-spec", type=Path, default=None,
                   help="Override the region_spec path (defaults to the bundled one).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output locale dir (defaults to assets/<domain>/<locale>).")
    p.add_argument("--models", type=Path, default=None,
                   help="Models TOML (only needed for live generation, not --dry-run).")
    p.add_argument("--model", action="append", default=None, metavar="ALIAS=MODEL",
                   help="Override a model alias, e.g. --model asset_judge_model="
                        "openai/openai/gpt-5.4. Repeatable. Provider auto-routes via "
                        "cli/model_catalog.py.")
    p.add_argument("--max-parallel", action="append", default=None, metavar="ALIAS=N",
                   help="Override an alias's concurrency, e.g. --max-parallel "
                        "doc_gen_model=8. Repeatable. Lower it if the hub throws "
                        "5xx/timeouts under load.")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate the region_spec and print the plan; no LLM calls.")
    for name in domain_names():
        get_domain(name).add_gen_args(p)
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    dom = get_domain(args.domain)
    spec_path = args.region_spec or (dom.region_spec_dir / f"{args.locale}.yaml")
    if not spec_path.exists():
        # Generation cannot proceed without a spec, so say what IS available and
        # how to add one rather than surfacing a file-not-found traceback.
        available = sorted(p.stem for p in dom.region_spec_dir.glob("*.yaml"))
        print(f"usersim gen-assets [{args.domain}] — no region_spec for "
              f"locale {args.locale!r}")
        print(f"  looked for: {spec_path}")
        print(f"  available locales: {', '.join(available) or '(none)'}")
        print("  to add one: author <locale>.yaml in that directory, or reuse an "
              "existing region's model with `extends: <base-locale>` and override "
              "just what differs (see assets/<domain>/SCHEMA.md).")
        return 2
    spec = dom.load_spec(spec_path)  # validates against the domain's spec contract

    print(f"usersim gen-assets [{args.domain}] — {args.locale}")
    print(f"  region_spec: {spec_path}")
    dom.describe_plan(spec, args)

    if args.dry_run:
        print("  (dry run — no assets written)")
        return 0

    out_dir = args.out or (packaged_assets_dir() / args.domain / args.locale)
    from usersim.cli._models import (
        apply_model_overrides, apply_parallel_overrides, default_models_path,
        load_models_config, parse_cli_model_overrides, parse_cli_parallel_overrides,
    )
    models = load_models_config(args.models or default_models_path())
    overrides = parse_cli_model_overrides(args.model)
    if overrides:
        models = apply_model_overrides(models, overrides)
    parallel = parse_cli_parallel_overrides(args.max_parallel)
    if parallel:
        models = apply_parallel_overrides(models, parallel)
    dom.generate(spec, models, out_dir, args)
    return 0
