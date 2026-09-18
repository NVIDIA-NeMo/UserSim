# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Each registry must actually consult entry-point discovery.

The loader in ``core/_extensions.py`` is tested on its own; these tests assert
the wiring — that ``known_probes``, ``get_scorer``, and ``domain_names``
reach it, so an installed extension appears without editing this project.

Entry points are faked and the extension module is written to ``tmp_path``, so
nothing is installed into the developer's environment. The end-to-end proof
against a genuinely installed distribution is ``make check-extensions``.
"""

from __future__ import annotations

from importlib.metadata import EntryPoint

import pytest

from usersim.engine.core import _extensions
from usersim.engine.core._extensions import (
    ASSET_DOMAINS,
    BACKENDS,
    CAPABILITIES,
    COMMANDS,
    DISABLE_ENV_VAR,
    PROBES,
    SCORERS,
    clear_extension_cache,
)


@pytest.fixture
def extension(monkeypatch, tmp_path):
    """Write an importable module and advertise parts of it as entry points."""
    monkeypatch.delenv(DISABLE_ENV_VAR, raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))

    def _make(filename: str, source: str, *specs: tuple[str, str, str]):
        (tmp_path / filename).write_text(source, encoding="utf-8")
        eps = [EntryPoint(name=n, value=v, group=g) for n, v, g in specs]
        monkeypatch.setattr(
            _extensions,
            "entry_points",
            lambda group: [e for e in eps if e.group == group],
        )
        clear_extension_cache()

    yield _make
    clear_extension_cache()


@pytest.fixture
def probe_registry():
    """Restore the probe registry exactly, rather than clearing it.

    ``clear_registry()`` drops the built-ins as well, which would leave an
    empty registry for whatever test runs next in the same worker.
    """
    from usersim.engine.core import probes

    snapshot = dict(probes._PROBE_REGISTRY)
    yield
    probes._PROBE_REGISTRY.clear()
    probes._PROBE_REGISTRY.update(snapshot)


class TestProbeDiscovery:
    """A probe package must register through the same decorator a built-in uses."""

    SOURCE = """
from usersim.engine.core.probes import BaseProbe, register_probe


@register_probe(family="general", prompt_version="v1.0", variants=("default",))
class PartnerProbe(BaseProbe):
    label = "partner_probe"
"""

    def test_contributed_probe_appears(self, extension, probe_registry) -> None:
        from usersim.engine.core.probes import known_probes

        extension(
            "ext_probe_mod.py",
            self.SOURCE,
            ("partner_probe", "ext_probe_mod:PartnerProbe", PROBES),
        )
        assert "partner_probe" in known_probes()

    def test_contributed_probe_resolves(self, extension, probe_registry) -> None:
        from usersim.engine.core.probes import resolve_probe

        extension(
            "ext_probe_mod2.py",
            self.SOURCE.replace("PartnerProbe", "P2"),
            ("partner_probe", "ext_probe_mod2:P2", PROBES),
        )
        assert resolve_probe("partner_probe").label == "partner_probe"

    def test_absent_when_discovery_is_off(self, extension, probe_registry, monkeypatch) -> None:
        from usersim.engine.core.probes import known_probes

        extension(
            "ext_probe_mod3.py",
            self.SOURCE.replace("PartnerProbe", "P3"),
            ("partner_probe", "ext_probe_mod3:P3", PROBES),
        )
        monkeypatch.setenv(DISABLE_ENV_VAR, "1")
        assert "partner_probe" not in known_probes()


@pytest.fixture
def scorer_registry():
    """Restore the scorer registry so a fixture scorer does not leak."""
    from usersim.engine.evaluator import scorers

    snapshot = dict(scorers._REGISTRY)
    yield
    scorers._REGISTRY.clear()
    scorers._REGISTRY.update(snapshot)


class TestScorerDiscovery:
    SOURCE = """
from usersim.engine.evaluator.scorers import register_scorer


def _score(row, models):
    return {"score": 1.0}


register_scorer("partner_scorer", _score)
"""

    def test_contributed_scorer_resolves(self, extension, scorer_registry) -> None:
        from usersim.engine.evaluator.scorers import get_scorer

        extension(
            "ext_scorer_mod.py",
            self.SOURCE,
            ("partner_scorer", "ext_scorer_mod", SCORERS),
        )
        assert get_scorer("partner_scorer")({}, {}) == {"score": 1.0}

    def test_contributed_scorer_is_listed(self, extension, scorer_registry) -> None:
        from usersim.engine.evaluator.scorers import list_scorers

        extension(
            "ext_scorer_mod2.py",
            self.SOURCE,
            ("partner_scorer", "ext_scorer_mod2", SCORERS),
        )
        assert "partner_scorer" in list_scorers()


class TestAssetDomainDiscovery:
    """The registry whose docstring promised extensibility it did not have."""

    SOURCE = """
from usersim.asset_gen.registry import AssetDomain


def _noop(*a, **k):
    return None


DOMAIN = AssetDomain(
    name="partner_domain",
    region_spec_dir=None,
    add_gen_args=_noop,
    load_spec=_noop,
    describe_plan=_noop,
    generate=_noop,
    validate=_noop,
    evaluate=_noop,
)


def factory():
    return DOMAIN
"""

    def test_contributed_domain_appears(self, extension) -> None:
        from usersim.asset_gen.registry import clear_domain_cache, domain_names

        extension(
            "ext_domain_mod.py",
            self.SOURCE,
            ("partner_domain", "ext_domain_mod:DOMAIN", ASSET_DOMAINS),
        )
        clear_domain_cache()
        try:
            assert "partner_domain" in domain_names()
            # The built-in must survive the merge.
            assert "financial_services" in domain_names()
        finally:
            clear_domain_cache()

    def test_factory_form_is_accepted(self, extension) -> None:
        """Built-ins are built lazily; an extension may do the same."""
        from usersim.asset_gen.registry import clear_domain_cache, get_domain

        extension(
            "ext_domain_mod2.py",
            self.SOURCE,
            ("partner_domain", "ext_domain_mod2:factory", ASSET_DOMAINS),
        )
        clear_domain_cache()
        try:
            assert get_domain("partner_domain").name == "partner_domain"
        finally:
            clear_domain_cache()

    def test_builtin_is_not_displaced(self, extension) -> None:
        """Shadowing financial_services would silently change generated assets."""
        from usersim.asset_gen.registry import clear_domain_cache, get_domain

        extension(
            "ext_domain_mod3.py",
            self.SOURCE.replace('"partner_domain"', '"financial_services"'),
            ("financial_services", "ext_domain_mod3:DOMAIN", ASSET_DOMAINS),
        )
        clear_domain_cache()
        try:
            domain = get_domain("financial_services")
            assert domain.region_spec_dir is not None, "the extension displaced the built-in financial_services domain"
        finally:
            clear_domain_cache()


class TestCapabilityDiscovery:
    """A probe that simulates and evaluates must also be able to reach the report.

    The capability registry was a hardcoded tuple, so an out-of-tree probe
    produced trajectories that nothing ever asked about: no cell, no coverage
    row, no triage entry.
    """

    SOURCE = """
from usersim.taxonomy.capabilities import CapabilityDefinition, EvidenceSource

PARTNER = CapabilityDefinition(
    id="partner_capability",
    label="Partner Capability",
    description="Contributed by an out-of-tree package.",
    sources=(EvidenceSource(probe="partner_probe", scorer="partner_scorer",
                            axes=("accuracy",)),),
    threshold=0.8,
    scale="rate",
    aggregation_policy="mean",
)

PAIR = (PARTNER,)
"""

    def test_contributed_capability_appears(self, extension) -> None:
        from usersim.taxonomy.capabilities import capability_definitions

        extension(
            "ext_cap_mod.py",
            self.SOURCE,
            ("partner_capability", "ext_cap_mod:PARTNER", CAPABILITIES),
        )
        ids = [d.id for d in capability_definitions()]
        assert "partner_capability" in ids
        # Built-ins keep their order and position ahead of contributed ones.
        assert ids.index("partner_capability") == len(ids) - 1

    def test_iterable_form_is_accepted(self, extension) -> None:
        """A package adding a probe usually adds several capabilities at once."""
        from usersim.taxonomy.capabilities import capability_definitions

        extension(
            "ext_cap_mod2.py",
            self.SOURCE,
            ("partner_pair", "ext_cap_mod2:PAIR", CAPABILITIES),
        )
        assert "partner_capability" in [d.id for d in capability_definitions()]

    def test_contributed_capability_is_addressable_by_id(self, extension) -> None:
        """capability_by_id read the raw tuple and would not have found it."""
        from usersim.taxonomy.capabilities import capability_by_id

        extension(
            "ext_cap_mod3.py",
            self.SOURCE,
            ("partner_capability", "ext_cap_mod3:PARTNER", CAPABILITIES),
        )
        assert capability_by_id("partner_capability").label == "Partner Capability"

    def test_contributed_scorer_counts_as_required(self, extension) -> None:
        """Otherwise the contributed capability silently renders Untested."""
        from usersim.taxonomy.capabilities import (
            capabilities_missing_scorer,
            scorers_required_by_capabilities,
        )

        extension(
            "ext_cap_mod4.py",
            self.SOURCE,
            ("partner_capability", "ext_cap_mod4:PARTNER", CAPABILITIES),
        )
        assert "partner_scorer" in scorers_required_by_capabilities()
        missing = capabilities_missing_scorer(dispatched=[])
        assert "partner_capability" in missing.get("partner_scorer", ())

    def test_colliding_id_is_rejected(self, extension) -> None:
        """Duplicate ids would make the rendered cell depend on load order."""
        from usersim.taxonomy.capabilities import capability_definitions

        builtin_id = capability_definitions()[0].id
        extension(
            "ext_cap_mod5.py",
            self.SOURCE.replace('"partner_capability"', f'"{builtin_id}"'),
            (builtin_id, "ext_cap_mod5:PARTNER", CAPABILITIES),
        )
        ids = [d.id for d in capability_definitions()]
        assert ids.count(builtin_id) == 1


class TestLayering:
    """The taxonomy package exists to break a cycle; assert it stays broken."""

    def test_taxonomy_does_not_import_its_consumers(self) -> None:
        import ast
        import pathlib

        offenders = []
        from usersim import taxonomy

        taxonomy_dir = pathlib.Path(taxonomy.__file__).resolve().parent
        for path in taxonomy_dir.glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                names = []
                if isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                elif isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                for n in names:
                    # Everything now shares the ``usersim`` root, so the
                    # subpackage is the second component.
                    parts = n.split(".")
                    layer = parts[1] if parts[:1] == ["usersim"] else parts[0]
                    if layer in {"reporting", "selection", "cli"}:
                        offenders.append(f"{path}: {n}")
        assert not offenders, "taxonomy must not import the layers that depend on it: " + str(offenders)

    def test_config_time_construction_avoids_reporting(self) -> None:
        """Config-time construction must not import the reporting layer."""
        import pathlib

        from usersim import cli

        source = (pathlib.Path(cli.__file__).resolve().parent / "_pipeline.py").read_text()
        assert "usersim.taxonomy.capabilities" in source
        assert "usersim.reporting.capabilities" not in source


class TestCommandDiscovery:
    """An overlay must be able to add a subcommand without editing the CLI."""

    SOURCE = """
def register(subparsers):
    p = subparsers.add_parser("partner-cmd", help="Contributed by a package.")
    p.set_defaults(func=lambda args: 0)


def register_colliding(subparsers):
    p = subparsers.add_parser("smoke", help="Shadows a built-in.")
    p.set_defaults(func=lambda args: 0)


def register_broken(subparsers):
    raise RuntimeError("this command is broken")
"""

    def test_contributed_command_appears(self, extension) -> None:
        from usersim import cli

        extension(
            "ext_cmd_mod.py",
            self.SOURCE,
            ("partner-cmd", "ext_cmd_mod:register", COMMANDS),
        )
        assert "partner-cmd" in cli._build_parser()._subparsers._group_actions[0].choices

    def test_contributed_command_dispatches(self, extension) -> None:
        from usersim import cli

        extension(
            "ext_cmd_mod2.py",
            self.SOURCE,
            ("partner-cmd", "ext_cmd_mod2:register", COMMANDS),
        )
        assert cli.main(["partner-cmd"]) == 0

    def test_collision_with_builtin_is_skipped(self, extension) -> None:
        """argparse raises on a duplicate name; that must not kill the CLI."""
        from usersim import cli

        extension(
            "ext_cmd_mod3.py",
            self.SOURCE,
            ("smoke", "ext_cmd_mod3:register_colliding", COMMANDS),
        )
        parser = cli._build_parser()
        assert "smoke" in parser._subparsers._group_actions[0].choices

    def test_broken_command_does_not_break_help(self, extension) -> None:
        """One bad package must not make every other command unreachable."""
        from usersim import cli

        extension(
            "ext_cmd_mod4.py",
            self.SOURCE,
            ("partner-cmd", "ext_cmd_mod4:register_broken", COMMANDS),
        )
        choices = cli._build_parser()._subparsers._group_actions[0].choices
        assert "simulate" in choices and "partner-cmd" not in choices


class TestBackendDiscovery:
    SOURCE = """
from usersim.engine.core.backends import JobHandle, JobStatus


class PartnerBackend:
    name = "partner"

    def submit(self, spec, *, credentials=None):
        self.seen_credentials = credentials
        return JobHandle(backend=self.name, job_id="j1")

    def poll(self, handle):
        return JobStatus(state="succeeded", exit_code=0)

    def fetch(self, handle, dest):
        return dest
"""

    def test_local_backend_is_always_available(self) -> None:
        from usersim.engine.core.backends import known_backends

        assert "local" in known_backends()

    def test_contributed_backend_appears(self, extension) -> None:
        from usersim.engine.core.backends import known_backends

        extension(
            "ext_backend_mod.py",
            self.SOURCE,
            ("partner", "ext_backend_mod:PartnerBackend", BACKENDS),
        )
        assert "partner" in known_backends()

    def test_contributed_backend_is_offered_by_the_cli(self, extension) -> None:
        """--help must list what is installed, not a literal list."""
        from usersim import cli

        extension(
            "ext_backend_mod2.py",
            self.SOURCE,
            ("partner", "ext_backend_mod2:PartnerBackend", BACKENDS),
        )
        parser = cli._build_parser()
        simulate = parser._subparsers._group_actions[0].choices["simulate"]
        backend_action = next(a for a in simulate._actions if a.dest == "backend")
        assert "partner" in backend_action.choices

    def test_contributed_backend_satisfies_the_protocol(self, extension) -> None:
        from usersim.engine.core.backends import ExecutionBackend, get_backend

        extension(
            "ext_backend_mod3.py",
            self.SOURCE,
            ("partner", "ext_backend_mod3:PartnerBackend", BACKENDS),
        )
        assert isinstance(get_backend("partner"), ExecutionBackend)

    def test_unknown_backend_names_the_alternatives(self) -> None:
        from usersim.engine.core.backends import get_backend

        with pytest.raises(KeyError, match="local"):
            get_backend("no-such-backend")


class TestLocalBackend:
    def test_runs_the_job_and_reports_success(self) -> None:
        from usersim.engine.core.backends import JobSpec, LocalBackend

        ran = []
        backend = LocalBackend()
        handle = backend.submit(JobSpec(name="t", callable_=lambda: ran.append(1) or 0))
        assert ran == [1]
        status = backend.poll(handle)
        assert status.state == "succeeded" and status.done

    def test_nonzero_exit_is_a_failure(self) -> None:
        from usersim.engine.core.backends import JobSpec, LocalBackend

        backend = LocalBackend()
        handle = backend.submit(JobSpec(name="t", callable_=lambda: 3))
        status = backend.poll(handle)
        assert status.state == "failed" and status.exit_code == 3

    def test_exception_is_kept_for_the_caller_to_reraise(self) -> None:
        """The CLI re-raises it so a ConfigError still exits 2, not 1."""
        from usersim.engine.core.backends import JobSpec, LocalBackend

        def boom():
            raise ValueError("nope")

        backend = LocalBackend()
        status = backend.poll(backend.submit(JobSpec(name="t", callable_=boom)))
        assert isinstance(status.error, ValueError)

    def test_credentials_are_passed_per_call_not_held_as_state(self) -> None:
        """A backend holding keys as state invites baking them into config."""
        import dataclasses
        import inspect

        from usersim.engine.core.backends import (
            ExecutionBackend,
            JobHandle,
            JobSpec,
        )

        submit = inspect.signature(ExecutionBackend.submit)
        credentials = submit.parameters["credentials"]
        assert credentials.kind is inspect.Parameter.KEYWORD_ONLY

        # Nothing that outlives a single call may carry them.
        for carrier in (JobSpec, JobHandle):
            assert "credentials" not in {f.name for f in dataclasses.fields(carrier)}
