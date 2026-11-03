# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``financial_services`` asset domain: how ``usersim gen-assets`` builds its banks.

:func:`asset_domain` returns the :class:`AssetDomain` the asset commands dispatch
to. The generator, the validation gate and the judge scorecard live in
``asset_gen/`` beside this module and are imported inside the callables, so
listing the domains, as ``usersim --help`` does, imports none of them.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from usersim.extension import AssetDomain


def asset_domain() -> AssetDomain:
    def add_gen_args(p: argparse.ArgumentParser) -> None:
        # Defaults are None so the single source of truth stays the pipeline
        # constants (DEFAULT_DOCS_PER_PRODUCT / DEFAULT_MAX_DOCS_PER_CLUSTER),
        # resolved in describe_plan / generate. Keeps add_gen_args import-free
        # (fast --help) while avoiding a drifting duplicate default.
        p.add_argument(
            "--docs-per-product",
            type=int,
            default=None,
            help="[financial_services] target docs per product cluster (default: pipeline DEFAULT_DOCS_PER_PRODUCT).",
        )
        p.add_argument(
            "--max-docs-per-cluster",
            type=int,
            default=None,
            help="[financial_services] per-call cap; larger targets fan into "
            "batches (default: pipeline DEFAULT_MAX_DOCS_PER_CLUSTER).",
        )

    def load_spec(path: Any) -> Any:
        from usersim.engine.probes.financial_services.asset_gen.spec import load_region_spec

        return load_region_spec(path)

    def describe_plan(spec: Any, args: argparse.Namespace) -> None:
        from collections import Counter

        from usersim.engine.probes.financial_services.asset_gen.pipeline import (
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
        print(f"  institutions: {len(spec.institutions)} ({', '.join(i.id for i in spec.institutions)})")
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
            print(
                f"    - {inst.id}: 1 brief + {len(clusters)} cluster call(s) "
                f"({len(inst.products)} product + shared) -> {inst_docs} docs"
            )
        print(f"  total: {total_clusters} cluster calls -> {total_docs} documents")
        print(f"  by document_type: {dict(by_type)}")

    def generate(spec: Any, models: Any, out_dir: Any, args: argparse.Namespace) -> Any:
        from usersim.engine.probes.financial_services.asset_gen.pipeline import (
            DEFAULT_DOCS_PER_PRODUCT,
            DEFAULT_MAX_DOCS_PER_CLUSTER,
            generate_corpus,
        )

        return generate_corpus(
            spec,
            models,
            out_dir,
            docs_per_product=(args.docs_per_product if args.docs_per_product is not None else DEFAULT_DOCS_PER_PRODUCT),
            max_docs_per_cluster=(
                args.max_docs_per_cluster if args.max_docs_per_cluster is not None else DEFAULT_MAX_DOCS_PER_CLUSTER
            ),
        )

    def validate(bank_dir: Any, spec: Any | None) -> Any:
        from usersim.engine.probes.financial_services.asset_gen.validate import validate_bank

        return validate_bank(bank_dir, spec)

    def evaluate(bank_dir: Any, spec: Any | None, models: Any) -> Any:
        from usersim.engine.probes.financial_services.asset_gen.evaluate import evaluate_bank

        return evaluate_bank(bank_dir, spec, models)

    return AssetDomain(
        name="financial_services",
        region_spec_dir=Path(__file__).resolve().parent / "asset_gen" / "region_spec",
        add_gen_args=add_gen_args,
        load_spec=load_spec,
        describe_plan=describe_plan,
        generate=generate,
        validate=validate,
        evaluate=evaluate,
    )
