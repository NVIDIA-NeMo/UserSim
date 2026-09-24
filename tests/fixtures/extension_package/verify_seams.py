# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Assert all six extension seams work against a genuinely installed package.

Run inside a throwaway venv that has the project wheels plus the fixture
distribution installed, and nothing else. Unlike the in-repo seam tests,
nothing here is faked: entry points come from real installed metadata.

Exits non-zero with a list of every seam that failed, rather than stopping
at the first, so one run says how much is broken.
"""

from __future__ import annotations

import os
import sys

# Discovery is off inside the project's own suite; this script exists to
# exercise it, so make the expectation explicit rather than inherited.
os.environ.pop("USERSIM_DISABLE_EXTENSIONS", None)

failures: list[str] = []


def check(seam: str, condition: bool, detail: str = "") -> None:
    status = "ok" if condition else "FAILED"
    print(f"  {status:>6}  {seam}{'  — ' + detail if detail and not condition else ''}")
    if not condition:
        failures.append(seam)


print("verifying extension seams against an installed distribution")

# 1. Probe
import usersim.engine.generator  # noqa: E402,F401  (populates built-ins)
from usersim.engine.core.probes import known_probes  # noqa: E402

probes = known_probes()
check("usersim.probes", "fixture_probe" in probes, f"saw {probes}")

# 2. Scorer
from usersim.engine.evaluator.scorers import list_scorers  # noqa: E402

check("usersim.scorers", "fixture_scorer" in list_scorers())

# 3. Asset domain
from usersim.asset_gen.registry import domain_names  # noqa: E402

domains = domain_names()
check("usersim.asset_domains", "fixture_domain" in domains, f"saw {domains}")
check(
    "asset_domains: built-in survives",
    "financial_services" in domains,
    "the contributed domain displaced the shipped one",
)

# 4. Backend
from usersim.engine.core.backends import known_backends  # noqa: E402

backends = known_backends()
check("usersim.backends", "fixture_backend" in backends, f"saw {backends}")

# 5. Capability — and, crucially, that it reaches a rendered report cell
from usersim.reporting.capability_report import capability_order  # noqa: E402
from usersim.taxonomy.capabilities import capability_definitions  # noqa: E402

ids = [d.id for d in capability_definitions()]
check("usersim.capabilities", "fixture_capability" in ids)
check(
    "capability reaches the report",
    "fixture_capability" in [c for c, _ in capability_order()],
    "registered but absent from report ordering",
)

# 6. Command
from usersim import cli  # noqa: E402

choices = cli._build_parser()._subparsers._group_actions[0].choices
check("usersim.commands", "fixture-cmd" in choices, f"saw {sorted(choices)}")
check("commands: built-ins survive", "simulate" in choices)

# The seams must compose: a contributed probe's assets resolve through the
# search path, which is what makes the probe actually runnable.
from usersim.engine.core._assets import probe_assets_dir  # noqa: E402

fixture_assets = (
    os.path.dirname(sys.modules["usersim_extension_fixture"].__file__)
    if "usersim_extension_fixture" in sys.modules
    else None
)
if fixture_assets:
    os.environ["USERSIM_ASSET_PATH"] = os.path.join(fixture_assets, "assets")
check(
    "USERSIM_ASSET_PATH resolves a contributed bank",
    probe_assets_dir("fixture_probe").is_dir(),
    f"looked at {probe_assets_dir('fixture_probe')}",
)

# The published conformance suite must pass for a real external probe.
from usersim.testing import probe_conformance_problems  # noqa: E402

problems = probe_conformance_problems("fixture_probe")
check("usersim.testing conformance", not problems, str(problems))

# A probe built on a shipped one: its spec layer ships in this package and
# extends identity_disclosure's spec, which must be inside the installed wheel.
from usersim.engine.core.identity_spec import load_identity_spec_default, validate_spec  # noqa: E402

check("probe built on identity_disclosure", "fixture_identity" in known_probes(), f"saw {known_probes()}")
spec_problems = validate_spec("fixture_identity")
check("identity spec layer loads", not spec_problems, "; ".join(str(p) for p in spec_problems))
if not spec_problems:
    derived = load_identity_spec_default("fixture_identity")
    shipped = load_identity_spec_default("identity_disclosure")
    check(
        "identity layer extends the shipped spec",
        derived.layers == ("fixture_identity@1", f"identity_disclosure@{shipped.bank_version}"),
        f"layers {derived.layers}",
    )
    own = derived.resolve("fixture-labs/fixturelm-7b-instruct")
    check("identity layer adds its own model", own is not None and own.developers == ("fixture_labs",), str(own))
    inherited = derived.resolve("nvidia/nemotron-3-super-120b-a12b")
    check(
        "identity layer inherits the shipped rules",
        inherited is not None and inherited.developers == ("nvidia",),
        str(inherited),
    )
identity_problems = probe_conformance_problems("fixture_identity")
check("identity probe conformance", not identity_problems, str(identity_problems))

if failures:
    print(f"\n{len(failures)} seam(s) failed: {', '.join(failures)}")
    sys.exit(1)
print("\nall extension seams work against an installed distribution")
