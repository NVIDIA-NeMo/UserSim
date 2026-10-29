# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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

# The seams must compose: a contributed probe's assets resolve from the
# assets_root its class declares, with no USERSIM_ASSET_PATH set, which is
# what makes the probe runnable as installed. Its bank loader reads under
# assets_dir(), the public way to find them.
from usersim.engine.core.probes import resolve_probe  # noqa: E402

check("no asset path set", "USERSIM_ASSET_PATH" not in os.environ, os.environ.get("USERSIM_ASSET_PATH", ""))
fixture_dir = resolve_probe("fixture_probe").assets_dir()
check(
    "assets_root resolves a contributed bank",
    (fixture_dir / "en_US" / "sample.yaml").is_file() and "site-packages" in fixture_dir.parts,
    f"looked at {fixture_dir}",
)

# A contributed probe's output_columns join the columns every run keeps.
from usersim.engine.config import ConversationSimulatorConfig  # noqa: E402

kept = ConversationSimulatorConfig(name="conversation_messages").side_effect_columns
check("output_columns joins the run's columns", "fixture_probe_marker" in kept, f"saw {len(kept)} columns")

# The pre-flight checks a contributed probe's locale banks with the check its
# class declares, and a refusal links the probe's own authoring guide.
from usersim.cli._errors import ConfigError  # noqa: E402
from usersim.cli._pipeline import verify_probe_assets  # noqa: E402


def preflight_refusal(locales: list[str]) -> str:
    try:
        verify_probe_assets({"fixture_probe": 1.0}, locales)
    except ConfigError as error:
        return str(error)
    return ""


check("pre-flight passes a locale with a bank", not preflight_refusal(["en_US"]), preflight_refusal(["en_US"]))
refusal = preflight_refusal(["fr_FR"])
check(
    "pre-flight refuses a locale without one, linking the probe's guide",
    "fixture_probe can't run fr_FR: no bank" in refusal and "https://example.com/fixture-probe/locales" in refusal,
    refusal,
)

# A contributed probe's themes come from its installed assets, read as a run
# reads them, with the run's asset root installed.
from usersim.cli._pipeline import _probe_themes  # noqa: E402
from usersim.engine.core._assets import default_assets_dir  # noqa: E402

themes = _probe_themes("fixture_probe", "en_US", default_assets_dir())
check(
    "themes come from the contributed assets",
    themes == ['{"type": "fixture topic", "description": "A topic every fixture row draws."}'],
    str(themes),
)

# A model a contributed probe declares joins the models every row resolves.
from usersim.engine.core.probes import declared_model_aliases  # noqa: E402

declared = declared_model_aliases(ConversationSimulatorConfig(name="conversation_messages"))
check("model_aliases joins the run's models", "fixture_critic_model" in declared, str(declared))

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

# `usersim smoke` names the installed package and finds nothing wrong with it.
from usersim.cli.smoke import SmokeReport, _check_extensions  # noqa: E402

smoke = SmokeReport()
_check_extensions(smoke)
(extensions,) = smoke.results
check(
    "usersim smoke reports the package",
    extensions.ok and extensions.detail == "usersim-extension-fixture 0.1.0: 7 entry point(s)",
    extensions.detail,
)

if failures:
    print(f"\n{len(failures)} seam(s) failed: {', '.join(failures)}")
    sys.exit(1)
print("\nall extension seams work against an installed distribution")
