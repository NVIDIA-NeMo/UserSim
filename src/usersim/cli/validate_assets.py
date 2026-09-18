# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``usersim validate-assets --domain <d>`` — vet a generated asset bank (offline).

Generic, domain-registry-backed (see ``asset_gen/registry.py``). Runs the domain's
deterministic validation gate over a generated bank and prints a per-check
PASS/WARN/FAIL report. Exits non-zero on any FAIL so it can gate a run in CI or a
shell loop. No LLM/network.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from usersim.asset_gen.registry import domain_names, get_domain
from usersim.engine.core._assets import packaged_assets_dir


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "validate-assets",
        help="🔎 Vet a generated asset bank (offline, --domain).",
        description=(
            "Run a domain's offline validation gate over a generated bank "
            "(e.g. numeric faithfulness, language/script, coverage, scope). "
            "Non-zero exit on FAIL."
        ),
    )
    p.add_argument("--domain", required=True, choices=domain_names(),
                   help="Asset domain to validate (e.g. financial_services).")
    p.add_argument("--locale", required=True, help="Locale, e.g. en_US.")
    p.add_argument("--dir", type=Path, default=None,
                   help="Bank directory (defaults to assets/<domain>/<locale>).")
    p.add_argument("--region-spec", type=Path, default=None,
                   help="region_spec for numeric-faithfulness ground truth (defaults to bundled).")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    dom = get_domain(args.domain)
    bank_dir = args.dir or (packaged_assets_dir() / args.domain / args.locale)
    spec_path = args.region_spec or (dom.region_spec_dir / f"{args.locale}.yaml")

    print(f"usersim validate-assets [{args.domain}] — {args.locale}")
    print(f"  bank: {bank_dir}")

    spec = None
    if spec_path.exists():
        try:
            spec = dom.load_spec(spec_path)
        except ValueError as e:  # RegionSpecError et al.
            print(f"  (region_spec invalid; ground-truth checks skipped): {e}")
    else:
        print(f"  (no region_spec at {spec_path}; ground-truth checks skipped)")

    report = dom.validate(bank_dir, spec)
    print(report.to_text())
    return 1 if report.failed else 0
