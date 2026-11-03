# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Asset-domain registry: the one generic layer over per-domain asset generation.

The `usersim gen-assets / validate-assets / evaluate-assets --domain <d>` CLIs
dispatch through here. Each domain registers an :class:`AssetDomain` carrying its
entry functions (load_spec / describe_plan / generate / validate / evaluate) + its
region_spec directory + an ``add_gen_args`` hook that injects domain-specific CLI
flags. The heavy domain modules are imported lazily inside the callables so
``usersim --help`` stays fast; ``add_gen_args`` is pure argparse (no import).

A built-in domain lives beside the probe whose banks it builds and is named in
``BUILTIN_DOMAINS``; an extension's domain registers under ``usersim.asset_domains``.
Neither adds a CLI subcommand. There is deliberately NO generic
``asset_gen/core/`` engine -- generation internals are domain-specific; only this
thin dispatch is shared.
"""

from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class AssetDomain:
    name: str
    #: directory holding ``<locale>.yaml`` region specs for this domain.
    region_spec_dir: Path
    #: add domain-specific generation flags to a ``gen-assets`` parser (argparse only).
    add_gen_args: Callable[[argparse.ArgumentParser], None]
    #: load + validate a region spec from a path -> a domain spec object.
    load_spec: Callable[[Any], Any]
    #: print an offline plan summary for ``gen-assets`` (also used by --dry-run).
    describe_plan: Callable[[Any, argparse.Namespace], None]
    #: run live generation: ``(spec, models, out_dir, args) -> out_dir``.
    generate: Callable[[Any, Any, Any, argparse.Namespace], Any]
    #: run the offline validation gate: ``(bank_dir, spec | None) -> ValidationReport``.
    validate: Callable[[Any, Any | None], Any]
    #: run the DD-judge evaluation scorecard (network-gated); set in the asset_eval bucket.
    evaluate: Callable[..., Any] | None = None


# ── built-in domains ─────────────────────────────────────────────────────

#: The built-in domains, each a ``module:function`` naming a zero-argument factory
#: that returns the domain. A built-in domain lives beside the probe whose banks
#: it builds, in a module that imports none of its generator.
BUILTIN_DOMAINS: tuple[str, ...] = ("usersim.engine.probes.financial_services.asset_domain:asset_domain",)


def _builtin_domains() -> dict[str, AssetDomain]:
    domains = {}
    for target in BUILTIN_DOMAINS:
        module, _, factory = target.partition(":")
        domain = getattr(importlib.import_module(module), factory)()
        domains[domain.name] = domain
    return domains


# ── registry access ──────────────────────────────────────────────────────

_DOMAINS: dict[str, AssetDomain] | None = None


def _domains() -> dict[str, AssetDomain]:
    """Built-in domains merged with any advertised under ``usersim.asset_domains``.

    An entry point may resolve to an ``AssetDomain`` or to a zero-argument
    factory returning one, as each built-in is a factory. A domain imports its
    pipeline inside its callables; building one eagerly at import time would drag
    those dependencies into every ``--help``.
    """
    global _DOMAINS
    if _DOMAINS is None:
        from usersim.engine.core._extensions import ASSET_DOMAINS, merge_extensions

        merged = merge_extensions(ASSET_DOMAINS, _builtin_domains(), what="asset domain")
        _DOMAINS = {name: obj if isinstance(obj, AssetDomain) else obj() for name, obj in merged.items()}
    return _DOMAINS


def clear_domain_cache() -> None:
    """Drop the memoized registry — for tests that toggle extension discovery."""
    global _DOMAINS
    _DOMAINS = None


def domain_names() -> tuple[str, ...]:
    return tuple(sorted(_domains()))


def get_domain(name: str) -> AssetDomain:
    domains = _domains()
    if name not in domains:
        raise KeyError(f"unknown asset domain {name!r}; known: {sorted(domains)}")
    return domains[name]
