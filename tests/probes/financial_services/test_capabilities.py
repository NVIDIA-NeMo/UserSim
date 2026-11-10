# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""financial_services' capabilities: dynamic grounding, per-institution strata, and recall kept apart from task success."""

from __future__ import annotations

from usersim.taxonomy.capabilities import capability_by_id, capability_definitions


def test_dynamic_grounding_is_cross_domain_and_gated() -> None:
    """The cross-domain dynamic_grounding capability keys on domain-agnostic
    dynamic.* axes with numeric_faithfulness + no_fabrication as the must-pass
    gate; financial_services is the first (of potentially many) evidence
    sources on the SAME axes."""
    d = capability_by_id("dynamic_grounding")
    assert d.scale == "score_1_5"
    assert d.aggregation_policy == "critical_axis"
    assert set(d.critical_axes) == {
        "dynamic.numeric_faithfulness",
        "dynamic.no_fabrication",
    }
    probes = {s.probe for s in d.sources}
    assert "financial_services" in probes
    # All axes are the domain-agnostic dynamic.* namespace (so future domain
    # probes can add EvidenceSources on the same capability).
    for s in d.sources:
        assert all(a.startswith("dynamic.") for a in s.axes)


def test_finance_verifiable_capability_is_stratified_by_institution_type() -> None:
    """A per-institution-type verifiable capability exists for each shipped
    finance institution type, scoped via row_filter to that type but reusing
    the aggregate capability's axes/policy."""
    from usersim.engine.probes.financial_services.capabilities import FINANCE_CAPABILITY_INSTITUTION_TYPES

    ids = {d.id for d in capability_definitions()}
    assert "financial_task_success" in ids  # aggregate stays
    for itype in FINANCE_CAPABILITY_INSTITUTION_TYPES:
        d = capability_by_id(f"financial_task_success_{itype}")
        assert d.row_filter == {"institution_type": itype}
        assert d.scale == "rate" and d.aggregation_policy == "critical_axis"
        assert "finance.tool_selection_rate" in {a for s in d.sources for a in s.axes}
        # Stratified rows share the aggregate's probe/scorer (one probe).
        assert {s.probe for s in d.sources} == {"financial_services"}


def test_finance_retrieval_recall_is_decoupled_from_task_success() -> None:
    """Document recall is surfaced as its OWN diagnostic capability and is NOT
    folded into the verifiable task-success axes — so retrieval-access and
    reasoning/utilization stay disentangled (cf. tau-Knowledge)."""
    ids = {d.id for d in capability_definitions()}
    assert "financial_retrieval_recall" in ids

    recall = capability_by_id("financial_retrieval_recall")
    recall_axes = {a for s in recall.sources for a in s.axes}
    assert recall_axes == {"finance.document_recall_rate"}
    assert recall.scale == "rate"

    # The recall axis must NOT appear in task-success (aggregate or per-type).
    from usersim.engine.probes.financial_services.capabilities import FINANCE_CAPABILITY_INSTITUTION_TYPES

    success_ids = ["financial_task_success"] + [
        f"financial_task_success_{itype}" for itype in FINANCE_CAPABILITY_INSTITUTION_TYPES
    ]
    for cid in success_ids:
        axes = {a for s in capability_by_id(cid).sources for a in s.axes}
        assert "finance.document_recall_rate" not in axes, cid
