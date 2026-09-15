# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Asset-domain registry: the one generic layer over per-domain asset generation.

The `usersim gen-assets / validate-assets / evaluate-assets --domain <d>` CLIs
dispatch through here. Each domain registers an :class:`AssetDomain` carrying its
entry functions (load_spec / describe_plan / generate / validate / evaluate) + its
region_spec directory + an ``add_gen_args`` hook that injects domain-specific CLI
flags. The heavy domain modules are imported lazily inside the callables so
``usersim --help`` stays fast; ``add_gen_args`` is pure argparse (no import).

Adding a new domain (e.g. healthcare) = a new ``asset_gen/<domain>/`` package + one
factory here; no new CLI subcommands. There is deliberately NO generic
``asset_gen/core/`` engine -- generation internals are domain-specific; only this
thin dispatch is shared.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

_ROOT = Path(__file__).resolve().parent  # asset_gen/


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
    validate: Callable[[Any, Optional[Any]], Any]
    #: run the DD-judge evaluation scorecard (network-gated); set in the asset_eval bucket.
    evaluate: Optional[Callable[..., Any]] = None


# ── financial_services domain ───────────────────────────────────────────

def _financial_services() -> AssetDomain:
    def add_gen_args(p: argparse.ArgumentParser) -> None:
        # Defaults are None so the single source of truth stays the pipeline
        # constants (DEFAULT_DOCS_PER_PRODUCT / DEFAULT_MAX_DOCS_PER_CLUSTER),
        # resolved in describe_plan / generate. Keeps add_gen_args import-free
        # (fast --help) while avoiding a drifting duplicate default.
        p.add_argument(
            "--docs-per-product", type=int, default=None,
            help="[financial_services] target docs per product cluster "
                 "(default: pipeline DEFAULT_DOCS_PER_PRODUCT).",
        )
        p.add_argument(
            "--max-docs-per-cluster", type=int, default=None,
            help="[financial_services] per-call cap; larger targets fan into "
                 "batches (default: pipeline DEFAULT_MAX_DOCS_PER_CLUSTER).",
        )

    def load_spec(path: Any) -> Any:
        from usersim.asset_gen.financial_services.spec import load_region_spec
        return load_region_spec(path)

    def describe_plan(spec: Any, args: argparse.Namespace) -> None:
        from collections import Counter

        from usersim.asset_gen.financial_services.pipeline import (
            DEFAULT_DOCS_PER_PRODUCT,
            DEFAULT_MAX_DOCS_PER_CLUSTER,
            _batches,
            build_product_doc_plan,
            build_shared_doc_plan,
        )
        dpp = args.docs_per_product if args.docs_per_product is not None else DEFAULT_DOCS_PER_PRODUCT
        cap = args.max_docs_per_cluster if args.max_docs_per_cluster is not None else DEFAULT_MAX_DOCS_PER_CLUSTER
        by_type: Counter = Counter()
        total_docs = total_clusters = 0
        print(f"  docs_per_product={dpp}  max_docs_per_cluster={cap}")
        print(f"  institutions: {len(spec.institutions)} "
              f"({', '.join(i.id for i in spec.institutions)})")
        for inst in spec.institutions:
            clusters = []
            for product in inst.products:
                for batch in _batches(build_product_doc_plan(product, dpp), cap):
                    clusters.append(len(batch))
                    by_type.update(d["document_type"] for d in batch)
            for batch in _batches(build_shared_doc_plan(inst), cap):
                clusters.append(len(batch))
                by_type.update(d["document_type"] for d in batch)
            inst_docs = sum(clusters)
            total_docs += inst_docs
            total_clusters += len(clusters)
            print(f"    - {inst.id}: 1 brief + {len(clusters)} cluster call(s) "
                  f"({len(inst.products)} product + shared) -> {inst_docs} docs")
        print(f"  total: {total_clusters} cluster calls -> {total_docs} documents")
        print(f"  by document_type: {dict(by_type)}")

    def generate(spec: Any, models: Any, out_dir: Any, args: argparse.Namespace) -> Any:
        from usersim.asset_gen.financial_services.pipeline import (
            DEFAULT_DOCS_PER_PRODUCT,
            DEFAULT_MAX_DOCS_PER_CLUSTER,
            generate_corpus,
        )
        return generate_corpus(
            spec, models, out_dir,
            docs_per_product=(args.docs_per_product
                              if args.docs_per_product is not None
                              else DEFAULT_DOCS_PER_PRODUCT),
            max_docs_per_cluster=(args.max_docs_per_cluster
                                  if args.max_docs_per_cluster is not None
                                  else DEFAULT_MAX_DOCS_PER_CLUSTER),
        )

    def validate(bank_dir: Any, spec: Optional[Any]) -> Any:
        from usersim.asset_gen.financial_services.validate import validate_bank
        return validate_bank(bank_dir, spec)

    def evaluate(bank_dir: Any, spec: Optional[Any], models: Any) -> Any:
        from usersim.asset_gen.financial_services.evaluate import evaluate_bank
        return evaluate_bank(bank_dir, spec, models)

    return AssetDomain(
        name="financial_services",
        region_spec_dir=_ROOT / "financial_services" / "region_spec",
        add_gen_args=add_gen_args,
        load_spec=load_spec,
        describe_plan=describe_plan,
        generate=generate,
        validate=validate,
        evaluate=evaluate,
    )


# ── registry access ──────────────────────────────────────────────────────

_DOMAINS: Optional[Dict[str, AssetDomain]] = None


def _domains() -> Dict[str, AssetDomain]:
    """Built-in domains merged with any advertised under ``usersim.asset_domains``.

    An entry point may resolve to an ``AssetDomain`` or to a zero-argument
    factory returning one, matching how the built-ins are constructed lazily
    here — building a domain eagerly at import time would drag its pipeline
    dependencies into every ``--help``.
    """
    global _DOMAINS
    if _DOMAINS is None:
        from usersim.engine.core._extensions import ASSET_DOMAINS, merge_extensions

        merged = merge_extensions(
            ASSET_DOMAINS,
            {"financial_services": _financial_services()},
            what="asset domain",
        )
        _DOMAINS = {
            name: obj if isinstance(obj, AssetDomain) else obj()
            for name, obj in merged.items()
        }
    return _DOMAINS


def clear_domain_cache() -> None:
    """Drop the memoized registry — for tests that toggle extension discovery."""
    global _DOMAINS
    _DOMAINS = None


def domain_names() -> Tuple[str, ...]:
    return tuple(sorted(_domains()))


def get_domain(name: str) -> AssetDomain:
    domains = _domains()
    if name not in domains:
        raise KeyError(f"unknown asset domain {name!r}; known: {sorted(domains)}")
    return domains[name]
